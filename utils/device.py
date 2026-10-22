"""GPU selection."""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_GB = 1024 ** 3


def select_gpu(index: Optional[int] = None, min_free_gb: float = 40.0) -> torch.device:
    """Return the requested GPU, or the first with ``min_free_gb`` free memory.

    Indices are relative to ``CUDA_VISIBLE_DEVICES``, which SGE sets to the
    GPUs granted to the job.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("MoCo-SToRM needs a CUDA GPU; none is visible (submit to a GPU node).")
    if index is not None:
        return torch.device(f"cuda:{index}")

    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        logger.info("GPU %d: %.1f / %.1f GB free", i, free / _GB, total / _GB)
        if free > min_free_gb * _GB:
            return torch.device(f"cuda:{i}")
    raise RuntimeError(f"No GPU found with more than {min_free_gb} GB free memory")
