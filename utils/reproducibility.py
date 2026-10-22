"""Seeding to produce reproducible results."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int]) -> np.random.Generator:
    """Seed python, numpy and torch; return a numpy Generator for explicit use.

    GPU kernels (grid_sample backward, cuFFT) are not bit-deterministic, so a
    fixed seed makes runs repeatable in their random choices (state order,
    coil subsets, PCA subsampling) but not bit-identical.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
