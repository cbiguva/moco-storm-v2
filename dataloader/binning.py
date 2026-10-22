"""
Grouping navigator frames into motion states and building the initial latent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

if TYPE_CHECKING:  # avoid a runtime dependency on the executor package
    from executor.config import LatentConfig

logger = logging.getLogger(__name__)


@dataclass
class LatentPlan:
    """Everything the reconstruction needs to know about motion states.

    bins[i] holds the frame indices that make up state i.
    z_init is the starting latent ``(n_states, siz_l, 1, 1, 1)``. In 'bulk'
    mode it holds only the fixed navigator part; the trainable part is
    ``siz_bulk_l`` extra columns appended during training.
    full_latent is the latent saved as ``FullLatent``: per frame for the
    navigator modes, per state once a latent has been trained.
    trained_bulk holds the learnt bulk columns after a 'bulk' run.
    """

    bins: List[np.ndarray]
    n_frames: int
    n_states: int
    siz_l: int                 # width the generator is built with
    nav_siz_l: int             # width of the navigator part
    z_init: torch.Tensor
    full_latent: Optional[np.ndarray]
    trained_bulk: Optional[np.ndarray] = None


def trim_navigator(latent: np.ndarray, cfg: LatentConfig) -> np.ndarray:
    """Keep the navigator columns the chosen binning uses."""
    if cfg.siz_l < 3:
        latent = latent[:, :cfg.siz_l]
    if cfg.resp_latent or not cfg.bulk_latent:
        latent = latent[:, :2]
    return latent


def bin_frames(zz: np.ndarray, cfg: LatentConfig, n_states: int) -> List[np.ndarray]:
    """Assign each frame to one of ``n_states`` equally sized states.

    ``zz`` must already be truncated to a multiple of ``n_states``.
    """
    n_frames = zz.shape[0]
    per_state = n_frames // n_states

    if cfg.resp_latent and not cfg.bulk_latent:
        if cfg.latent_type != "resp_cardiac":
            raise NotImplementedError("respiratory binning is only implemented for latent_type 'resp_cardiac'")
        # Sort by the cardiac signal, then each cardiac block by respiration.
        order = _sort_within_blocks(np.argsort(zz[:, 1]), cfg.cardiac_states, key=zz[:, 0])
    elif cfg.bulk_latent:
        # Sort by bulk motion, then each bulk block by |resp - second latent|.
        key = np.abs(zz[:, 0] - zz[:, 1])
        order = _sort_within_blocks(np.argsort(zz[:, 2]), cfg.bulk_states, key=key)
    else:
        order = np.arange(n_frames)          # bin by acquisition time

    return list(order.reshape(n_states, per_state).astype(int))


def _sort_within_blocks(primary_order: np.ndarray, n_blocks: int, key: np.ndarray) -> np.ndarray:
    n = len(primary_order)
    block = n // n_blocks
    out = np.zeros(n)
    for b in range(n_blocks):
        sl = slice(b * block, (b + 1) * block)
        members = primary_order[sl]
        out[sl] = members[np.argsort(key[members], axis=0)]
    return out


def plan_from_navigator(zz: np.ndarray, cfg: LatentConfig) -> LatentPlan:
    """Build a LatentPlan for optimize_latent 'none' or 'bulk'."""
    if zz.ndim == 1:
        zz = zz.reshape(-1, 1)
    n_states = cfg.n_states
    n_frames = zz.shape[0] - (zz.shape[0] % n_states)
    if n_frames == 0:
        raise ValueError(f"{zz.shape[0]} navigator frames cannot fill {n_states} states")
    if n_frames != zz.shape[0]:
        logger.info("Truncating %d frames to %d so they divide into %d states",
                    zz.shape[0], n_frames, n_states)
    zz = zz[:n_frames].copy()
    nav_siz_l = zz.shape[1]

    bins = bin_frames(zz, cfg, n_states)
    z = torch.zeros((n_states, nav_siz_l, 1, 1, 1))
    z_frames = torch.from_numpy(zz).view(n_frames, nav_siz_l, 1, 1, 1)
    for i, idx in enumerate(bins):
        z[i] = torch.mean(z_frames[idx], 0)

    siz_l = nav_siz_l + (cfg.siz_bulk_l if cfg.optimize_latent == "bulk" else 0)
    return LatentPlan(bins=bins, n_frames=n_frames, n_states=n_states, siz_l=siz_l,
                      nav_siz_l=nav_siz_l, z_init=z.float(), full_latent=zz)


def plan_full_optimisation(cfg: LatentConfig) -> LatentPlan:
    """optimize_latent 'full': one state per frame, latent learnt from zero."""
    n = cfg.n_frames
    return LatentPlan(bins=list(np.arange(n).reshape(n, 1)), n_frames=n, n_states=n,
                      siz_l=cfg.siz_l, nav_siz_l=cfg.siz_l,
                      z_init=torch.zeros((n, cfg.siz_l, 1, 1, 1)), full_latent=None)


def build_latent_plan(cfg: LatentConfig, data_dir: Path, tr: Optional[float],
                      plot_dir: Optional[Path]) -> LatentPlan:
    """Estimate the navigator (or fall back to ``lat.npy``) and bin it."""
    if cfg.optimize_latent == "full":
        return plan_full_optimisation(cfg)

    from dataloader.navigator import estimate_latent  # heavy imports (statsmodels)

    try:
        zz = trim_navigator(
            estimate_latent(data_dir, cfg.latent_type, cfg.invert_resp, cfg.lf_width,
                            cfg.hf_width, tr=tr, plot_dir=plot_dir), cfg)
    except Exception as exc:  # the legacy code fell back to lat.npy on any failure
        lat_path = Path(data_dir) / "lat.npy"
        if not lat_path.exists():
            raise
        logger.warning("Navigator extraction failed (%s); using %s", exc, lat_path)
        zz = np.load(lat_path)

    return plan_from_navigator(zz, cfg)
