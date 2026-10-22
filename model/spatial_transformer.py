"""Warp a volume by a dense displacement field given in voxels."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """``out = transformer(volume, flow)``.

    volume: ``(B, C, *shape)``; flow: ``(B, ndim, *shape)`` in voxels.
    The identity grid is a buffer, so ``.to(device)`` moves it with the module.
    """

    def __init__(self, shape: Sequence[int], mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in shape]
        grid = torch.stack(torch.meshgrid(*vectors, indexing="ij")).unsqueeze(0).float()
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, inp: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return F.grid_sample(inp, new_locs, align_corners=True, mode=self.mode)


def warp_template(template: torch.Tensor, flow: torch.Tensor, transformer: SpatialTransformer) -> torch.Tensor:
    """Warp a ``(N, N, N, 2)`` real/imag template; returns ``(2, N, N, N)``."""
    return transformer(template.permute(3, 0, 1, 2).unsqueeze(0), flow).squeeze(0)
