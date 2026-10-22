"""Writing reconstruction outputs in a ``Results.h5`` file.

    Run<N>/
        attrs: RunStart, RunStop, OptimizeLatent
        ParametersJSON         legacy flat parameter dict (resolved siz_l)
        ConfigYAML             the full resolved config of this version
        GeneratorState/<name>  generator weights
        Template               (N, N, N, 2) real/imag
        FullLatent             navigator (per frame) or trained (per state) latent
        TrainingLatent         latent the generator was trained with
        OriginalLatent, TrainedBulkDimension   ('bulk' mode only)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch

from evaluation.render import render_volume, representative_state

logger = logging.getLogger(__name__)


def nifti_affine(im_size, fov: float) -> np.ndarray:
    """Affine for the output NIfTI files.

    Voxel size is FOV/matrix in mm. The origin puts voxel N//2, 
    the zero position of the centred FFT, at isocentre, so the 
    volume spans the FOV centred on 0.
    """
    n = np.asarray(im_size, dtype=float)
    voxel = fov / n
    affine = np.diag([voxel[0], voxel[1], voxel[2], 1.0])
    affine[:3, 3] = -(n // 2) * voxel
    return affine


def save_nifti(image: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Write ``image`` with ``affine``, spatial units recorded as mm."""
    img = nib.Nifti1Image(image, affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, path)


def save_results(cfg, output_dir: Path, G, template: torch.Tensor, latent: torch.Tensor, plan,
                 start: datetime, stop: datetime) -> Path:
    h5_path = output_dir / "Results.h5"
    with h5py.File(h5_path, "a") as f:
        n = 0
        while f"Run{n}" in f:
            n += 1
        run = f.create_group(f"Run{n}")
        run.attrs["RunStart"] = str(start)
        run.attrs["RunStop"] = str(stop)
        run.attrs["OptimizeLatent"] = str(cfg.latent.optimize_latent)

        run.create_dataset("ParametersJSON", data=json.dumps(cfg.to_legacy_dict(plan.siz_l)))
        run.create_dataset("ConfigYAML", data=cfg.to_yaml())

        gen = run.create_group("GeneratorState")
        for key, tensor in G.state_dict().items():
            gen.create_dataset(key, data=tensor.detach().cpu().numpy())

        run.create_dataset("Template", data=template.detach().cpu().numpy(),
                           compression="gzip", compression_opts=9)
        if cfg.latent.optimize_latent == "bulk":
            run.create_dataset("OriginalLatent", data=plan.z_init.squeeze().numpy())
            run.create_dataset("TrainedBulkDimension", data=plan.trained_bulk)
        if plan.full_latent is not None:
            run.create_dataset("FullLatent", data=plan.full_latent)
        run.create_dataset("TrainingLatent", data=latent.squeeze().detach().cpu().numpy())
    logger.info("Wrote %s (Run%d)", h5_path, n)

    # Image at the state nearest the median latent. State 0 holds the start-up
    # transient (an out-of-distribution latent), so it is not representative.
    state = representative_state(latent)
    with torch.no_grad():
        image = render_volume(G, template, latent[state:state + 1])
    nifti_path = output_dir / f"X_{state:04d}.nii.gz"
    save_nifti(image, nifti_affine(cfg.data.im_size, cfg.data.fov), nifti_path)
    logger.info("Wrote %s", nifti_path)
    
    return h5_path
