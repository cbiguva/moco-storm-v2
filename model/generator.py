"""Latent-to-deformation generators for the multi-resolution reconstruction pipeline.

Layout (d = gen_base)::

    Conv(siz_l->30, k1) Tanh
    Conv(30->8d) LReLU Up2          1  -> 2
    Conv(8d->4d) LReLU Up2          2  -> 4
    Conv(4d->4d) LReLU Up2          4  -> 8
    Conv(4d->2d) LReLU Up2          8  -> 16
    Conv(2d->d)  LReLU Up2          16 -> 32
    [Conv(d->d)  LReLU Up2] x n_extra
    Conv(d->out) Tanh  Up(final)

The output is a displacement field in voxels of the output grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Spec:
    n_extra: int          # d->d blocks after the 32^3 level
    final_scale: float    # upsampling after the output conv
    flow_scale: float     # multiplier on the Tanh output


_SPECS: Dict[int, _Spec] = {
    64: _Spec(n_extra=0, final_scale=2.0, flow_scale=1.0),
    96: _Spec(n_extra=1, final_scale=3 / 2, flow_scale=1.0),
    128: _Spec(n_extra=1, final_scale=2.0, flow_scale=2.0),
    256: _Spec(n_extra=2, final_scale=2.0, flow_scale=4.0),
    320: _Spec(n_extra=3, final_scale=320 / 256, flow_scale=4.0),
}

SUPPORTED_SIZES = tuple(sorted(_SPECS))


class MotionGenerator(nn.Module):
    """Maps a latent ``(B, siz_l, 1, 1, 1)`` to a flow ``(B, out_channels, N, N, N)``."""

    def __init__(self, out_size: int, siz_latent: int, gen_base: int = 52, out_channels: int = 3):
        super().__init__()
        if out_size not in _SPECS:
            raise ValueError(f"Unsupported generator size {out_size}; choose from {SUPPORTED_SIZES}")
        spec = _SPECS[out_size]
        d = gen_base
        self.out_size = out_size
        self.siz_latent = siz_latent
        self.gen_base = gen_base
        self.out_channels = out_channels
        self.flow_scale = spec.flow_scale

        layers = [nn.Conv3d(siz_latent, 30, 1, 1, 0), nn.Tanh()]
        widths = [30, d * 8, d * 4, d * 4, d * 2, d] + [d] * spec.n_extra
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            layers += [
                nn.Conv3d(c_in, c_out, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode="trilinear"),
            ]
        layers += [
            nn.Conv3d(d, out_channels, 3, 1, 1),
            nn.Tanh(),
            nn.Upsample(scale_factor=spec.final_scale, mode="trilinear"),
        ]
        self.conv_blocks = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(z)
        if self.flow_scale != 1.0:
            x = x * self.flow_scale
        return x

    def transfer_to(self, out_size: int) -> "MotionGenerator":
        """Build a generator for ``out_size`` initialised from this one.

        Layers whose shape differs (the output conv, whose index moves when a 
        stage inserts a block) keep PyTorch's default initialisation and are 
        reported in the log.
        """
        target = MotionGenerator(out_size, self.siz_latent, self.gen_base, self.out_channels)
        target_sd = target.state_dict()
        skipped = []
        for name, p in self.state_dict().items():
            if name not in target_sd:
                continue
            if target_sd[name].shape != p.shape:
                skipped.append(f"{name} {tuple(p.shape)}->{tuple(target_sd[name].shape)}")
                continue
            target_sd[name].copy_(p)
        target.load_state_dict(target_sd)
        if skipped:
            logger.warning("transfer %d->%d: shape mismatch, left at init: %s",
                           self.out_size, out_size, "; ".join(skipped))
        return target.to(next(self.parameters()).device)

    def trainable_top_blocks(self):
        """The two convs adjacent to the output."""
        children = list(self.conv_blocks.children())
        return [children[-3], children[-6]]
