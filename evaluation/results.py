"""Reads the ``Results.h5`` written after successful training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import torch

from model.generator import MotionGenerator


@dataclass
class RunData:
    name: str
    params: Dict[str, Any]          
    template: np.ndarray            # (N, N, N, 2)
    training_latent: np.ndarray     # (n_states, siz_l)
    state_dict: Dict[str, torch.Tensor]
    attrs: Dict[str, Any]

    @property
    def size(self) -> int:
        return int(self.template.shape[0])

    def build_generator(self, device: torch.device | str = "cpu") -> MotionGenerator:
        """Rebuild the generator; the latent width is read from the saved weights."""
        siz_l = int(self.state_dict["conv_blocks.0.weight"].shape[1])
        G = MotionGenerator(self.size, siz_l, int(self.params["gen_base"]), int(self.params["out_channels"]))
        G.load_state_dict(self.state_dict)
        return G.to(device).eval()

    def latent_tensor(self) -> torch.Tensor:
        z = self.training_latent.reshape(self.training_latent.shape[0], -1)
        return torch.from_numpy(z.astype(np.float32)).view(z.shape[0], z.shape[1], 1, 1, 1)


def _run_index(name: str) -> int:
    return int(name[3:]) if name[3:].isdigit() else -1


def list_runs(results_file: str | Path) -> List[Dict[str, str]]:
    with h5py.File(results_file, "r") as f:
        names = sorted((k for k in f.keys() if k.startswith("Run")), key=_run_index)
        return [{"name": n, **{k: str(v) for k, v in f[n].attrs.items()}} for n in names]


def load_run(results_file: str | Path, run: str = "Run0") -> RunData:
    with h5py.File(results_file, "r") as f:
        if run not in f:
            raise KeyError(f"{run} not in {results_file}; available: {[r['name'] for r in list_runs(results_file)]}")
        g = f[run]
        params_json = g["ParametersJSON"][()]
        if isinstance(params_json, bytes):
            params_json = params_json.decode("utf-8")
        latent = g["TrainingLatent"][:]
        if latent.ndim == 1:
            latent = latent.reshape(-1, 1)
        state = {k: torch.from_numpy(g["GeneratorState"][k][:].astype(np.float32))
                 for k in g["GeneratorState"].keys()}
        return RunData(name=run, params=json.loads(params_json), template=g["Template"][:],
                       training_latent=latent, state_dict=state, attrs=dict(g.attrs))
