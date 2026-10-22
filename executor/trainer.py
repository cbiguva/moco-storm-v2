"""Optimisation loops: joint template/generator(/latent) fitting and generator alignment.

The loss for one motion state is

    sum_c || W_s (F_s (S_c * warp(template, G(z_s))) - y_{s,c}) ||_2  + regularizers

summed over a random subset of coils ``c`` each step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from model.generator import MotionGenerator
from model.regularizers import latent_smoothness, tv_smooth
from model.spatial_transformer import SpatialTransformer, warp_template
from utils.plotting import plot_latent_matrix, save_slice

logger = logging.getLogger(__name__)

STAGE_NAMES = {64: "Stage1", 128: "Stage2", 256: "Stage3", 320: "Stage4"}


def snapshot_slice(size: int) -> int:
    """Slice index used for progress snapshots (36 at 64^3, 144 at 256^3, ...)."""
    return int(size * 36 / 64)


@dataclass
class ResidualStats:
    """Accumulates ||W(Fx-y)||^2 and ||Wy||^2 for the scale-free relative residual."""

    device: torch.device
    resid_sq: torch.Tensor = field(init=False)
    data_sq: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.resid_sq = torch.zeros((), device=self.device)
        self.data_sq = torch.zeros((), device=self.device)

    def relative(self) -> float:
        return torch.sqrt(self.resid_sq / (self.data_sq + 1e-20)).item()


def _coil_subset(n_coils: int, subset: Optional[int]) -> torch.Tensor:
    if subset is not None and subset > 0:
        return torch.randperm(n_coils)[:min(subset, n_coils)]
    return torch.arange(n_coils)


def data_consistency(dop, template: torch.Tensor, flow: torch.Tensor, transformer: SpatialTransformer,
                     state: int, coil_subset: Optional[int], stats: ResidualStats):
    """Weighted k-space residual norm for one motion state, summed over coils.

    Returns ``(loss, debug)`` where ``debug`` holds intermediates for the NaN report.
    """
    out_mc = warp_template(template, flow, transformer)
    coil_images = torch.view_as_real(dop.csm_currscale * torch.complex(out_mc[0], out_mc[1]))
    fwd, weight = dop.fwd_ops[state], dop.weight_ops[state]

    loss = 0
    kdatasim = None
    for ch in _coil_subset(dop.n_coils, coil_subset).tolist():
        y = dop.ksp_states[state][ch:ch + 1].to(stats.device)
        kdatasim = fwd.apply(coil_images[ch:ch + 1])
        resid = weight.apply(kdatasim - y)
        loss = loss + torch.norm(resid)
        with torch.no_grad():
            stats.resid_sq += (resid.detach() ** 2).sum()
            stats.data_sq += (weight.apply(y) ** 2).sum()
    return loss, {"out_mc": out_mc, "coil_images": coil_images, "kdatasim": kdatasim}


def _log_epoch(size: int, epoch: int, epoch_err: float, stats: ResidualStats, template: torch.Tensor,
               snapshot_dir: Optional[Path], max_grad_norm: Optional[float] = None) -> None:
    logger.info("%s epoch %d: loss %.6g, relative residual %.4f",
                STAGE_NAMES[size], epoch, epoch_err, stats.relative())
    if max_grad_norm is not None:
        logger.info("%s epoch %d: max grad norm %.4e", STAGE_NAMES[size], epoch, max_grad_norm)
    if snapshot_dir is not None:
        mag = torch.complex(template[..., 0], template[..., 1]).detach().abs().cpu().numpy()
        save_slice(mag, snapshot_slice(size), snapshot_dir / f"{STAGE_NAMES[size]}_epoch{epoch}.png")


# Stage 1 (latent)
def train_generator_and_latent(cfg, optimizer, scheduler, transformer: SpatialTransformer,
                               latent: torch.Tensor, template: torch.Tensor, G: MotionGenerator, dop,
                               alpha: torch.Tensor, z_fixed: Optional[torch.Tensor] = None,
                               snapshot_dir: Optional[Path] = None) -> torch.Tensor:
    """Stage-1 fit of template, generator and (part of) the latent.

    ``latent`` is the trainable leaf: the whole latent in 'full' mode, or only
    the extra bulk columns in 'bulk' mode, where ``z_fixed`` holds the
    navigator columns they are concatenated to. Returns the final latent.
    """
    device = template.device
    t = cfg.training
    n_states = latent.shape[0]

    def combined():
        return latent if z_fixed is None else torch.cat([z_fixed, latent], dim=1)

    for epoch in range(t.epochs[64]):
        stats = ResidualStats(device)
        epoch_err = torch.zeros((), device=device)
        for f in np.random.permutation(n_states):
            optimizer.zero_grad()
            flow = G(combined()[f].unsqueeze(0))
            loss, _ = data_consistency(dop, template, flow, transformer, int(f),
                                       cfg.coils.random_coil_subset, stats)
            loss = loss + 0.1 * t.tv_reg * tv_smooth(template.permute(3, 0, 1, 2))
            loss = loss + latent_smoothness(latent, alpha)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            epoch_err += loss.detach()
        scheduler.step(epoch_err)
        if epoch % t.log_every == 0:
            _log_epoch(64, epoch, epoch_err.item(), stats, template, snapshot_dir)

    final = combined().detach()
    if snapshot_dir is not None:
        plot_latent_matrix(final.squeeze().cpu().numpy().reshape(n_states, -1),
                           snapshot_dir / "latent_trained.png")
    logger.info("Motion estimation done")
    return final


# Stage (fixed latent z and optimize the template and generator)
def train_generator(cfg, size: int, optimizer, scheduler, transformer: SpatialTransformer,
                    latent: torch.Tensor, template: torch.Tensor, G: MotionGenerator, dop,
                    epochs: int, frozen_generator: bool = False, log: bool = True,
                    snapshot_dir: Optional[Path] = None) -> None:
    """Fit template and generator at one resolution with the latent held fixed.

    Gradients are clipped to ``training.grad_clip`` (off when null; the norm is
    still logged). A step whose loss or gradient is non-finite is rejected and
    the first one is reported with intermediate magnitudes.
    """
    device = template.device
    t = cfg.training
    max_norm = t.grad_clip if t.grad_clip is not None else float("inf")
    nan_reported = False

    for epoch in range(epochs):
        stats = ResidualStats(device)
        epoch_err = torch.zeros((), device=device)
        max_grad_norm = 0.0
        for f in np.random.permutation(latent.shape[0]):
            optimizer.zero_grad()
            z_in = latent[f].unsqueeze(0)
            if frozen_generator:
                with torch.no_grad():
                    flow = G(z_in)
            else:
                flow = G(z_in)

            loss, debug = data_consistency(dop, template, flow, transformer, int(f),
                                           cfg.coils.random_coil_subset, stats)
            loss = loss + t.tv_reg * tv_smooth(template.permute(3, 0, 1, 2))
            loss.backward()

            params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)

            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                if not nan_reported:
                    nan_reported = True
                    _report_non_finite(size, epoch, int(f), loss, grad_norm, template, flow, debug)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            max_grad_norm = max(max_grad_norm, float(grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            epoch_err += loss.detach()

        scheduler.step(epoch_err)
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
        if log and epoch % t.log_every == 0:
            _log_epoch(size, epoch, epoch_err.item(), stats, template, snapshot_dir, max_grad_norm)


def _abs_minmax(x: Optional[torch.Tensor]):
    if x is None:
        return None
    x = x.detach()
    x = x.abs() if x.is_complex() else x
    return float(x.min()), float(x.max())


def _report_non_finite(size, epoch, frame, loss, grad_norm, template, flow, debug) -> None:
    logger.error("[NaN sentinel] non-finite step at %s, epoch %d, state %d: loss=%s, grad_norm=%s",
                 STAGE_NAMES[size], epoch, frame, loss.item(), float(grad_norm))
    for name, value in [("template", template), ("flow", flow), ("out_mc", debug["out_mc"]),
                        ("coil_images", debug["coil_images"]), ("kdatasim", debug["kdatasim"])]:
        logger.error("[NaN sentinel]   |%s| min/max = %s", name, _abs_minmax(value))


# Alignment
def align_upsampled_generator(G_low: MotionGenerator, G_high: MotionGenerator, latent: torch.Tensor,
                              scale_factor: float, epochs: int = 50, lr: float = 1e-4,
                              unfreeze_after: bool = True) -> MotionGenerator:
    """Fit the top two convs of ``G_high`` so its flow matches the upsampled ``G_low`` flow."""
    
    logger.info("Aligning %d -> %d generator for %d epochs", G_low.out_size, G_high.out_size, epochs)
    G_low.eval()
    for p in G_low.parameters():
        p.requires_grad = False
    G_high.train()
    for p in G_high.parameters():
        p.requires_grad = False
    for block in G_high.trainable_top_blocks():
        for p in block.parameters():
            p.requires_grad = True

    optimizer = torch.optim.Adam([p for p in G_high.parameters() if p.requires_grad],
                                 lr=lr, betas=(0.5, 0.999))
    n_states = latent.shape[0]
    avg = float("nan")
    for epoch in range(epochs):
        total = 0.0
        for s in np.random.permutation(n_states):
            optimizer.zero_grad()
            z_in = latent[s:s + 1]
            with torch.no_grad():
                target = F.interpolate(G_low(z_in), scale_factor=scale_factor,
                                       mode="trilinear", align_corners=False)
            loss = F.mse_loss(G_high(z_in), target)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / n_states
        if epoch % 10 == 0:
            logger.info("Alignment epoch %d/%d, average loss %.6f", epoch, epochs, avg)
    
    logger.info("Alignment complete, final loss %.6f", avg)

    del optimizer
    torch.cuda.empty_cache()
    for p in G_high.parameters():
        p.requires_grad = unfreeze_after
    G_high.eval()
    
    return G_high
