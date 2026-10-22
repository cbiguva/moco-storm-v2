"""Render motion-state volumes from a trained generator and template.

Command line::

    python -m evaluation.render RESULTS.h5 --run Run0 --states 0 10 20 --out-dir renders/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from model.generator import MotionGenerator
from model.spatial_transformer import SpatialTransformer, warp_template

logger = logging.getLogger(__name__)


def render_volume(G: MotionGenerator, template: torch.Tensor, z: torch.Tensor) -> np.ndarray:
    """|warp(template, G(z))| for one latent ``(1, siz_l, 1, 1, 1)``; returns ``(N, N, N)``."""
    device = next(G.parameters()).device
    template = template.to(device)
    transformer = SpatialTransformer(template.shape[:3]).to(device)
    out = warp_template(template, G(z.to(device)), transformer)
    return torch.complex(out[0], out[1]).abs().detach().cpu().numpy()


def representative_state(latent: torch.Tensor | np.ndarray) -> int:
    """Index of the state closest to the per-dimension median latent.

    Distances are measured after scaling each dimension by its standard
    deviation. State 0 is a poor default: with sorted binning it collects the
    start-up transient frames, an out-of-distribution latent.
    """
    z = latent.detach().cpu().numpy() if isinstance(latent, torch.Tensor) else np.asarray(latent)
    z = z.reshape(z.shape[0], -1)
    scale = z.std(axis=0)
    scale[scale == 0] = 1.0
    return int(np.argmin((((z - np.median(z, axis=0)) / scale) ** 2).sum(axis=1)))


def main(argv: Optional[Sequence[str]] = None) -> None:
    from evaluation.results import load_run
    from executor.results import nifti_affine, save_nifti
    from utils.log import configure_logging

    parser = argparse.ArgumentParser(description="Render MoCo-SToRM motion states to NIfTI.")
    parser.add_argument("results_file", type=Path)
    parser.add_argument("--run", default="Run0")
    parser.add_argument("--states", nargs="+", default=["median"],
                        help="state indices to render; 'median' picks the state nearest the median latent")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    configure_logging()

    run = load_run(args.results_file, args.run)
    G = run.build_generator(args.device)
    template = torch.from_numpy(run.template)
    latent = run.latent_tensor()
    affine = nifti_affine(run.params.get("im_size", [run.size] * 3), float(run.params.get("fov", 320)))

    states = [representative_state(latent) if s == "median" else int(s) for s in args.states]
    logger.info("Rendering states %s", states)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for s in states:
            path = args.out_dir / f"{args.run}_state{s:04d}.nii.gz"
            save_nifti(render_volume(G, template, latent[s:s + 1]), affine, path)
            logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
