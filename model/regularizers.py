"""Regularisation terms used in the loss function."""

import torch


def latent_smoothness(z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Weighted squared first difference of a latent ``(T, L, 1, 1, 1)`` along T."""
    diff = z[1:, ...] - z[:-1, ...]
    per_dim = torch.sum(diff * diff, axis=0).squeeze()
    return torch.sum(alpha * per_dim, axis=0)


def tv_smooth(x: torch.Tensor) -> torch.Tensor:
    """Squared-gradient (Tikhonov) smoothness of a ``(C, X, Y, Z)`` volume."""
    total = 0
    for axis in (1, 2, 3):
        diff = torch.diff(x, dim=axis)
        total = total + torch.sum(diff * diff)
    return total
