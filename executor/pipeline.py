"""End-to-end multi-resolution reconstruction pipeline.

Stage 1 (64^3) fits the template, the generator and, depending on
``latent.optimize_latent``, the latent. Each further stage (128, 256, 320, up to
``data.im_size``) transfers the generator to the finer grid, optionally aligns
it to the upsampled coarse flow, upsamples the template, and refits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader.binning import LatentPlan, build_latent_plan
from dataloader.operators import DataAndOperators
from executor.config import ReconConfig
from executor.results import save_results
from executor.trainer import align_upsampled_generator, train_generator, train_generator_and_latent
from model.generator import MotionGenerator
from model.spatial_transformer import SpatialTransformer
from utils.device import select_gpu
from utils.plotting import plot_latent_matrix, plot_latent_zoom
from utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)

# Optimiser settings inherited from the original driver.
_BETAS = (0.4, 0.999)
_LR_TEMPLATE_STAGE1_LATENT = 5e-2
_LR_LATENT = 5e-2
_LR_TEMPLATE = 4e-2
_LR_GENERATOR = 1e-4
_MIN_LR = 1e-8


def _scheduler(optimizer) -> ReduceLROnPlateau:
    n_groups = len(optimizer.param_groups)
    return ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=20,
                             min_lr=[_MIN_LR] * n_groups if n_groups > 1 else _MIN_LR)


@dataclass
class ReconResult:
    output_dir: Path
    generator: MotionGenerator
    template: torch.Tensor          # (N, N, N, 2) real/imag
    latent: torch.Tensor            # (n_states, siz_l, 1, 1, 1)
    plan: LatentPlan


def run_reconstruction(cfg: ReconConfig, output_dir: Path) -> ReconResult:
    """Run every stage and write results into ``output_dir``."""
    start = datetime.now()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = output_dir / "snapshots"
    snapshots.mkdir(exist_ok=True)

    rng = seed_everything(cfg.runtime.seed)
    torch.backends.cudnn.benchmark = True
    device = select_gpu(cfg.runtime.gpu, cfg.runtime.min_free_gb)
    torch.cuda.set_device(device)
    logger.info("Using %s, output in %s", device, output_dir)

    # Latent
    plan = build_latent_plan(cfg.latent, Path(cfg.data.data_dir), cfg.data.tr, snapshots)
    logger.info("Motion states: %d frames -> %d states, generator latent width %d",
                plan.n_frames, plan.n_states, plan.siz_l)
    if cfg.latent.optimize_latent != "full":
        state_latent = plan.z_init.squeeze(-1).squeeze(-1).squeeze(-1).numpy()
        plot_latent_matrix(state_latent, snapshots / "latent_binned.png")
        plot_latent_zoom(plan.full_latent, plan.bins, state_latent, snapshots / "latent_zoom.png")

    # Stage 1
    dop = DataAndOperators(cfg, device, rng=rng, plot_dir=snapshots)
    dop.change_scale(64, plan.bins, plan.n_frames)

    G = MotionGenerator(64, plan.siz_l, cfg.model.gen_base, cfg.model.out_channels).to(device)
    template = _initial_template(dop, 64, device)
    latent = _stage1(cfg, plan, dop, G, template, snapshots)
    logger.info("Stage 1 done")

    # Stage 2 .. n
    prev_size = 64
    for size in cfg.stage_sizes[1:]:
        dop.change_scale(size, plan.bins, plan.n_frames)
        G, template = _refine_stage(cfg, size, prev_size, dop, G, template, latent, snapshots)
        prev_size = size
        logger.info("Stage %d (%d^3) done", cfg.stage_sizes.index(size) + 1, size)

    save_results(cfg, output_dir, G, template, latent, plan, start, datetime.now())
    return ReconResult(output_dir, G, template, latent, plan)


def _initial_template(dop: DataAndOperators, size: int, device: torch.device) -> torch.Tensor:
    """Coil-combined image (sum_c conj(S_c) x_c) at ``size``^3, as a real/imag leaf tensor."""
    combined = torch.zeros((size,) * 3, dtype=torch.complex64, device=device)
    for c in range(dop.n_coils):
        factors = [size / s for s in dop.coil_images.shape[1:]]
        img = zoom(dop.coil_images[c].real, factors, order=3) + 1j * zoom(dop.coil_images[c].imag, factors, order=3)
        combined += torch.from_numpy(img.astype(np.complex64)).to(device) * torch.conj(dop.csm_currscale[c])
    return torch.view_as_real(combined).detach().clone().requires_grad_()


def _stage1(cfg: ReconConfig, plan: LatentPlan, dop, G: MotionGenerator, template: torch.Tensor,
            snapshots: Path) -> torch.Tensor:
    device = template.device
    mode = cfg.latent.optimize_latent
    transformer = SpatialTransformer((64,) * 3).to(device)
    gen_params = [p for p in G.parameters() if p.requires_grad]

    if mode == "none":
        latent = plan.z_init.to(device)
        optimizer = Adam([{"params": [template], "lr": _LR_TEMPLATE},
                          {"params": gen_params, "lr": _LR_GENERATOR}], betas=_BETAS)
        train_generator(cfg, 64, optimizer, _scheduler(optimizer), transformer, latent, template, G, dop,
                        epochs=cfg.training.epochs[64], snapshot_dir=snapshots)
        return latent

    if mode == "full":
        n = plan.n_frames
        alpha = torch.tensor([5 / n, 20 / n, 50 / n, 100 / n][:plan.siz_l], device=device)
        trainable = plan.z_init.clone().to(device).requires_grad_()
        z_fixed = None
    else:  # bulk
        n = plan.n_states
        alpha = torch.tensor([5 / n], device=device)
        trainable = torch.zeros((n, cfg.latent.siz_bulk_l, 1, 1, 1), device=device, requires_grad=True)
        z_fixed = plan.z_init.to(device)

    optimizer = Adam([{"params": [template], "lr": _LR_TEMPLATE_STAGE1_LATENT},
                      {"params": [trainable], "lr": _LR_LATENT},
                      {"params": gen_params, "lr": _LR_GENERATOR}], betas=_BETAS)
    latent = train_generator_and_latent(cfg, optimizer, _scheduler(optimizer), transformer, trainable,
                                        template, G, dop, alpha, z_fixed=z_fixed, snapshot_dir=snapshots)
    plan.full_latent = latent.squeeze().cpu().numpy().reshape(plan.n_states, -1)
    if mode == "bulk":
        plan.trained_bulk = trainable.detach().squeeze().cpu().numpy()
    return latent


def _refine_stage(cfg: ReconConfig, size: int, prev_size: int, dop, G_low: MotionGenerator,
                  template_low: torch.Tensor, latent: torch.Tensor, snapshots: Path):
    t = cfg.training
    device = template_low.device
    scale = size / prev_size
    freeze = t.freeze_generator_above64

    G = G_low.transfer_to(size)
    if t.align_generators or freeze:
        G = align_upsampled_generator(G_low, G, latent, scale, epochs=t.alignment_epochs,
                                      lr=t.alignment_lr, unfreeze_after=not freeze)
    del G_low
    torch.cuda.empty_cache()

    template = torch.nn.functional.interpolate(
        template_low.detach().permute(3, 0, 1, 2).unsqueeze(0), scale_factor=scale, mode="trilinear"
    ).squeeze(0).permute(1, 2, 3, 0).contiguous().requires_grad_()
    del template_low
    torch.cuda.empty_cache()

    transformer = SpatialTransformer((size,) * 3).to(device)

    if not t.align_generators:
        # Warm up only the two convs next to the output before training everything.
        top = [p for block in G.trainable_top_blocks() for p in block.parameters() if p.requires_grad]
        optimizer = Adam([{"params": [template], "lr": _LR_TEMPLATE},
                          {"params": top, "lr": _LR_GENERATOR}], betas=_BETAS)
        train_generator(cfg, size, optimizer, _scheduler(optimizer), transformer, latent, template, G, dop,
                        epochs=t.top_layer_epochs, log=False, snapshot_dir=snapshots)
        del optimizer

    if freeze:
        for p in G.parameters():
            p.requires_grad = False
        G.eval()
        optimizer = Adam([{"params": [template], "lr": _LR_TEMPLATE}], betas=_BETAS)
    else:
        optimizer = Adam([{"params": [template], "lr": _LR_TEMPLATE},
                          {"params": [p for p in G.parameters() if p.requires_grad], "lr": _LR_GENERATOR}],
                         betas=_BETAS)
    train_generator(cfg, size, optimizer, _scheduler(optimizer), transformer, latent, template, G, dop,
                    epochs=t.epochs[size], frozen_generator=freeze, snapshot_dir=snapshots)
    del optimizer
    torch.cuda.empty_cache()
    return G, template
