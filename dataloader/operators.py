"""Measured data and per-state NUFFT operators for the forward model."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import sigpy as sp
import torch
import torch.nn.functional as F

from dataloader.coils import (center_of_mass_3d, jsense_sensitivities, lowres_sensitivities,
                              pca_coil_compression, reconstruct_coil_images, select_coils)
from dataloader.raw_io import crop_to_matrix, load_raw_data
from utils.plotting import visualize_coils

logger = logging.getLogger(__name__)


class DataAndOperators:
    """Compressed k-space, coil maps, and per-state operators at the current scale.

    After :meth:`change_scale` the following are valid for state ``i``:

    * ``ksp_states[i]``  measured k-space ``(n_coils, n_views_i, n_samples, 2)`` (CPU, real view)
    * ``fwd_ops[i]``     torch autograd function, image -> k-space (NUFFT)
    * ``weight_ops[i]``  torch autograd function, multiplies by sqrt(dcf) + epsilon
    * ``csm_currscale``  coil maps ``(n_coils, N, N, N)`` on the torch device
    """

    def __init__(self, cfg, device: torch.device, rng: Optional[np.random.Generator] = None,
                 plot_dir: Optional[Path] = None):
        self.cfg = cfg
        self.device = device
        self.sp_device = device.index if device.index is not None else 0
        self.im_size = np.asarray(cfg.data.im_size)
        plot_dir = Path(plot_dir) if plot_dir is not None else None

        ksp, dcf, coord = load_raw_data(cfg.data.data_dir)
        ksp, coord, dcf = crop_to_matrix(ksp, coord, dcf, int(self.im_size[0]), cfg.data.zoom_factor)

        n_selected = 1
        if ksp.shape[0] > 1:
            ksp, selected = select_coils(ksp, coord, dcf, cfg.coils.selection,
                                         cfg.coils.outside_fov_threshold, plot_dir, self.sp_device)
            n_selected = len(selected)

        self.n_coils = min(cfg.coils.n_virtual_coils, n_selected)
        ksp = pca_coil_compression(ksp, self.n_coils, rng)
        self.n_readouts = ksp.shape[1]
        logger.info("Compressed k-space: %s", ksp.shape)

        logger.info("Reconstructing compressed coil images")
        self.coil_images = reconstruct_coil_images(ksp, coord, dcf, tuple(self.im_size), device=self.sp_device)
        if plot_dir is not None:
            visualize_coils(self.coil_images, center_of_mass_3d(self.coil_images),
                            plot_dir / "compressed_coil_images.png")

        logger.info("Estimating coil sensitivities (%s)", cfg.coils.sensitivity_method)
        if cfg.coils.sensitivity_method == "jsense":
            csm = jsense_sensitivities(ksp, coord, dcf, self.im_size, self.sp_device)
        else:
            csm = lowres_sensitivities(ksp, coord, dcf, self.im_size, cfg.coils.sense_process, self.sp_device)
        if self.n_coils == 1:
            csm[...] = 1
        if plot_dir is not None:
            visualize_coils(csm.numpy(), center_of_mass_3d(self.coil_images), plot_dir / "csm.png")

        self.ksp = ksp
        self.coord = coord
        self.dcf = dcf[None]          # (1, n_views, n_samples)
        self.csm = csm

    # Change scale
    def change_scale(self, size: int, bins: Sequence[np.ndarray], n_frames: int) -> None:
        """Rebuild operators for a ``size``^3 grid and the given state bins."""
        t0 = time.time()
        ops = self.cfg.operators
        n_states = len(bins)
        spokes_per_frame = self.n_readouts // n_frames
        dev = sp.Device(self.sp_device)
        grid = (1, size, size, size)

        radius = np.sqrt(np.sum(self.coord[1] ** 2, axis=1))
        mask = radius < size / 2

        csm_r = F.interpolate(self.csm.real.unsqueeze(0), size=(size,) * 3).squeeze(0)
        csm_i = F.interpolate(self.csm.imag.unsqueeze(0), size=(size,) * 3).squeeze(0)
        self.csm_currscale = (csm_r + 1j * csm_i).to(self.device)

        spoke_sets = [np.concatenate([np.arange(f * spokes_per_frame, (f + 1) * spokes_per_frame)
                                      for f in frames]) for frames in bins]

        # Largest eigenvalue of F^H W F over all states, used to normalise the DCF.
        max_eig = 0.0
        for spokes in spoke_sets:
            coord_d = sp.to_device(self.coord[spokes][:, mask, :], dev)
            dcf_d = sp.to_device(self.dcf[:, spokes][:, :, mask], dev)
            FT = sp.linop.NUFFT(grid, coord_d, oversamp=ops.nufft_oversamp)
            W = sp.linop.Multiply(FT.oshape, dcf_d)
            max_eig = max(max_eig, float(sp.app.MaxEig(FT.H * W * FT, max_iter=ops.maxeig_iter,
                                                       dtype=np.complex64, device=dev, show_pbar=False).run()))

        self.fwd_ops: List = []
        self.weight_ops: List = []
        ksp_states = []
        ksp_max = 0.0
        bad_states = []
        for i, spokes in enumerate(spoke_sets):
            coord_d = sp.to_device(self.coord[spokes][:, mask, :], dev)
            dcf_d = sp.to_device(self.dcf[:, spokes][:, :, mask] / max_eig, dev)
            ksp_i = self.ksp[:, spokes][:, :, mask]
            ksp_states.append(ksp_i)

            FT = sp.linop.NUFFT(grid, coord_d, oversamp=ops.nufft_oversamp)
            W = sp.linop.Multiply(FT.oshape, dev.xp.sqrt(dcf_d) + ops.dcf_epsilon)
            self.fwd_ops.append(sp.to_pytorch_function(FT, input_iscomplex=True, output_iscomplex=True))
            self.weight_ops.append(sp.to_pytorch_function(W, input_iscomplex=True, output_iscomplex=True))

            for ch in range(self.n_coils):
                recon = FT.H * W * W * sp.to_device(ksp_i[ch:ch + 1], dev)
                ksp_max = max(ksp_max, float(dev.xp.max(dev.xp.abs(recon))))

            if ops.probe_operators and self._probe_is_bad(FT, coord_d, len(spokes), size, i):
                bad_states.append(i)

        if ops.probe_operators:
            logger.info("[operator probe] N=%d: %d/%d states flagged%s", size, len(bad_states), n_states,
                        f"; first={bad_states[:20]}" if bad_states else "")

        self.ksp_states = [torch.view_as_real(torch.from_numpy(k / ksp_max)) for k in ksp_states]
        del ksp_states
        torch.cuda.empty_cache()
        logger.info("Operators for %d^3 built in %.1f s", size, time.time() - t0)

    def _probe_is_bad(self, FT, coord_d, n_spokes: int, size: int, state: int) -> bool:
        """Apply the forward NUFFT to a constant image and check the output is finite."""
        dev = sp.Device(self.sp_device)
        xp = dev.xp
        with dev:
            r = xp.sqrt((coord_d ** 2).sum(-1))
            rmax = float(r.max()) if r.size else float("nan")
            k = FT * (xp.ones(FT.ishape, dtype=xp.complex64) * self.cfg.operators.probe_value)
            finite = bool(xp.isfinite(k).all())
            kabs = float(xp.abs(k).max()) if finite else float("inf")
        n_samp = int(coord_d.shape[1])
        bad = (not finite) or n_spokes == 0 or n_samp == 0 or rmax >= size / 2
        if bad:
            logger.warning("[operator probe] state %d: finite=%s |k|max=%.3e nspokes=%d nsamp=%d "
                           "coord_rmax=%.2f (grid/2=%d)", state, finite, kabs, n_spokes, n_samp, rmax, size // 2)
        return bad
