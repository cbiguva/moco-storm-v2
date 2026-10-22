"""Typed, validated configuration for running the reconstruction.

A run is fully described by one YAML file (see ``configs/default.yaml``). The
config is split into sections that mirror the pipeline stages.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

STAGE_SIZES = (64, 128, 256, 320)

_LATENT_TYPES = ("in_exp", "resp_cardiac")
_OPTIMIZE_LATENT = ("none", "full", "bulk")
_COIL_TYPES = ("lowres", "jsense")
_COIL_SELECTION = ("outside_fov", "center_of_mass")
_SENSE_PROCESS = ("smooth", "polynomial", "none")


class ConfigError(ValueError):
    """Raised when a configuration is invalid."""


@dataclass
class DataConfig:
    data_dir: str = ""                      # holds MRI_Raw.h5 (or ksp.npy/dcf.npy/coord.npy)
    output_dir: Optional[str] = None        # default: <data_dir>/mocostorm_runs/<run_name>
    im_size: List[int] = field(default_factory=lambda: [256, 256, 256])
    fov: float = 320.0                      # mm, used for the NIfTI header
    zoom_factor: float = 1.0
    tr: Optional[float] = None              # s; only needed when reading ksp.npy


@dataclass
class CoilConfig:
    n_virtual_coils: int = 8
    sensitivity_method: str = "lowres"      # lowres | jsense
    selection: str = "outside_fov"          # outside_fov | center_of_mass
    outside_fov_threshold: float = 0.1
    sense_process: str = "smooth"           # smooth | polynomial | none (lowres only)
    random_coil_subset: Optional[int] = 4   # coils per step; null = all coils


@dataclass
class LatentConfig:
    latent_type: str = "resp_cardiac"       # in_exp | resp_cardiac
    resp_latent: bool = True
    bulk_latent: bool = False
    invert_resp: bool = True
    optimize_latent: str = "none"           # none | full | bulk
    n_frames: int = 300                     # only used when optimize_latent == full
    n_states: int = 300
    siz_l: int = 2
    siz_bulk_l: int = 1
    bulk_states: int = 1
    cardiac_states: int = 8
    lf_width: float = 0.5                   # s, respiratory smoother width
    hf_width: float = 0.1                   # s, cardiac smoother width


@dataclass
class ModelConfig:
    gen_base: int = 52
    out_channels: int = 3


@dataclass
class TrainingConfig:
    epochs: Dict[int, int] = field(default_factory=lambda: {64: 200, 128: 100, 256: 100, 320: 100})
    tv_reg: float = 1e-9
    grad_clip: Optional[float] = None       # null = no clipping (norm still logged)
    align_generators: bool = True
    alignment_epochs: int = 50
    alignment_lr: float = 1e-4
    freeze_generator_above64: bool = False
    top_layer_epochs: int = 10              # used only when align_generators is false
    log_every: int = 25


@dataclass
class OperatorConfig:
    dcf_epsilon: float = 0.005
    maxeig_iter: int = 30
    nufft_oversamp: float = 2.0
    probe_operators: bool = True
    probe_value: float = 1000.0


@dataclass
class RuntimeConfig:
    run_name: Optional[str] = None          # default: timestamp
    seed: Optional[int] = 0                 # null = unseeded
    gpu: Optional[int] = None               # null = first GPU with min_free_gb free
    min_free_gb: float = 40.0
    verbose: bool = True


@dataclass
class ReconConfig:
    data: DataConfig = field(default_factory=DataConfig)
    coils: CoilConfig = field(default_factory=CoilConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    operators: OperatorConfig = field(default_factory=OperatorConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # Build
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReconConfig":
        d = copy.deepcopy(d or {})
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ConfigError(f"Unknown config section(s): {sorted(unknown)}")
        kwargs = {}
        for f in dataclasses.fields(cls):
            section_cls = type(getattr(cls(), f.name))
            kwargs[f.name] = _build_section(section_cls, d.get(f.name) or {}, f.name)
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    # Validate
    def validate(self) -> None:
        d, c, l, t = self.data, self.coils, self.latent, self.training
        if not d.data_dir:
            raise ConfigError("data.data_dir is required")
        if len(d.im_size) != 3 or len(set(d.im_size)) != 1:
            raise ConfigError(f"data.im_size must be a cubic [N, N, N], got {d.im_size}")
        if d.im_size[0] not in STAGE_SIZES:
            raise ConfigError(f"data.im_size must be one of {STAGE_SIZES}, got {d.im_size[0]}")
        _check_choice("latent.latent_type", l.latent_type, _LATENT_TYPES)
        _check_choice("latent.optimize_latent", l.optimize_latent, _OPTIMIZE_LATENT)
        _check_choice("coils.sensitivity_method", c.sensitivity_method, _COIL_TYPES)
        _check_choice("coils.selection", c.selection, _COIL_SELECTION)
        _check_choice("coils.sense_process", c.sense_process, _SENSE_PROCESS)
        for size in self.stage_sizes:
            if size not in t.epochs:
                raise ConfigError(f"training.epochs is missing an entry for stage {size}")
        if l.n_states < 1 or c.n_virtual_coils < 1:
            raise ConfigError("latent.n_states and coils.n_virtual_coils must be >= 1")

        if l.optimize_latent != "full":
            if l.resp_latent and not l.bulk_latent and l.latent_type == "in_exp":
                raise ConfigError(
                    "latent_type 'in_exp' with resp_latent=true and bulk_latent=false is not "
                    "implemented (the legacy code called a missing compute_respiration_bins). "
                    "Use latent_type 'resp_cardiac', or set resp_latent=false.")
            if l.bulk_latent and not l.resp_latent and l.siz_l < 3:
                raise ConfigError("bulk_latent binning sorts on the 3rd latent column; set latent.siz_l >= 3")

    # Helpers
    @property
    def stage_sizes(self) -> List[int]:
        """Resolutions trained in order, e.g. [64, 128, 256] for a 256 matrix."""
        return [s for s in STAGE_SIZES if s == 64 or s <= self.data.im_size[0]]

    def resolve_output_dir(self, timestamp: str) -> Path:
        name = self.runtime.run_name or timestamp
        if self.data.output_dir:
            return Path(self.data.output_dir).expanduser() / name
        return Path(self.data.data_dir).expanduser() / "mocostorm_runs" / name

    def to_legacy_dict(self, siz_l: int) -> Dict[str, Any]:
        """Flat parameter dict in the format of the legacy JSON files.

        Written to Results.h5 as ``ParametersJSON`` so the original viewer can
        still rebuild the generator. ``siz_l`` is the resolved latent width the
        saved weights were trained with (it grows in bulk mode).
        """
        d, c, l, m, t = self.data, self.coils, self.latent, self.model, self.training
        legacy = {
            "name": self.runtime.run_name or "parameters",
            "im_size": list(d.im_size), "directory": str(d.data_dir), "fov": d.fov,
            "zoom_factor": d.zoom_factor, "gen_base": m.gen_base, "out_channels": m.out_channels,
            "nVcoils": c.n_virtual_coils, "coilType": c.sensitivity_method,
            "select_coils": c.selection, "outside_fov_threshold": c.outside_fov_threshold,
            "sense_process": c.sense_process, "random_coil_subset": c.random_coil_subset,
            "latent_type": l.latent_type, "resp_latent": l.resp_latent, "bulk_latent": l.bulk_latent,
            "invert_resp": l.invert_resp, "optimize_latent": l.optimize_latent,
            "nStates": l.n_states, "siz_l": siz_l, "siz_bulk_l": l.siz_bulk_l,
            "BulkStates": l.bulk_states, "CardiacStates": l.cardiac_states,
            "lf_width": l.lf_width, "hf_width": l.hf_width, "tv_reg": t.tv_reg,
        }
        for size, n in t.epochs.items():
            legacy[f"epoch{size}"] = n
        return legacy


# Loading
def load_config(path: str | Path, overrides: Optional[List[str]] = None) -> ReconConfig:
    """Load a YAML config, or a legacy flat JSON parameter file.

    ``overrides`` are ``section.key=value`` strings applied on top, with the
    value parsed as YAML (``training.epochs.64=10``, ``runtime.seed=null``).
    """
    path = Path(path)
    with open(path, "r") as fh:
        raw = json.load(fh) if path.suffix == ".json" else yaml.safe_load(fh)
    raw = raw or {}
    if path.suffix == ".json" and not _looks_sectioned(raw):
        raw = convert_legacy_params(raw)
    for item in overrides or []:
        _apply_override(raw, item)
    return ReconConfig.from_dict(raw)


# legacy flat key -> (section, key) if json file is used
_LEGACY_MAP = {
    "directory": ("data", "data_dir"), "im_size": ("data", "im_size"), "fov": ("data", "fov"),
    "zoom_factor": ("data", "zoom_factor"), "TR": ("data", "tr"),
    "nVcoils": ("coils", "n_virtual_coils"), "coilType": ("coils", "sensitivity_method"),
    "select_coils": ("coils", "selection"), "selectCoils": ("coils", "selection"),
    "outside_fov_threshold": ("coils", "outside_fov_threshold"),
    "sense_process": ("coils", "sense_process"), "random_coil_subset": ("coils", "random_coil_subset"),
    "latent_type": ("latent", "latent_type"), "resp_latent": ("latent", "resp_latent"),
    "bulk_latent": ("latent", "bulk_latent"), "invert_resp": ("latent", "invert_resp"),
    "optimize_latent": ("latent", "optimize_latent"), "nFrames": ("latent", "n_frames"),
    "nStates": ("latent", "n_states"), "siz_l": ("latent", "siz_l"),
    "siz_bulk_l": ("latent", "siz_bulk_l"), "BulkStates": ("latent", "bulk_states"),
    "CardiacStates": ("latent", "cardiac_states"), "lf_width": ("latent", "lf_width"),
    "hf_width": ("latent", "hf_width"),
    "gen_base": ("model", "gen_base"), "out_channels": ("model", "out_channels"),
    "tv_reg": ("training", "tv_reg"), "grad_clip": ("training", "grad_clip"),
    "align_generators": ("training", "align_generators"),
    "alignment_epochs": ("training", "alignment_epochs"),
    "alignment_lr": ("training", "alignment_lr"),
    "freeze_generator_above64": ("training", "freeze_generator_above64"),
    "maxeig_iter": ("operators", "maxeig_iter"), "nufft_oversamp": ("operators", "nufft_oversamp"),
    "probe_operators": ("operators", "probe_operators"), "probe_value": ("operators", "probe_value"),
    "verbose": ("runtime", "verbose"), "name": ("runtime", "run_name"),
}

_LEGACY_DROPPED = {"t1_reg", "contrast_reg", "channels", "dtype", "device", "max_disp_mm", "resp_band"}


def convert_legacy_params(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Map a legacy flat parameter dict onto the sectioned layout."""
    out: Dict[str, Dict[str, Any]] = {}
    dropped = []
    for key, value in flat.items():
        if key in _LEGACY_MAP:
            section, new_key = _LEGACY_MAP[key]
            out.setdefault(section, {})[new_key] = value
        elif key.startswith("epoch") and key[5:].isdigit():
            out.setdefault("training", {}).setdefault("epochs", {})[int(key[5:])] = value
        elif key in _LEGACY_DROPPED:
            dropped.append(key)
        else:
            raise ConfigError(f"Unrecognised legacy parameter '{key}'")
    if dropped:
        logger.warning("Ignoring legacy parameters not used by this version: %s", sorted(dropped))
    # The legacy name was a label for the JSON file, not a run directory.
    out.get("runtime", {}).pop("run_name", None)
    # 'none' and false both meant "no latent optimisation" in old files.
    lat = out.get("latent", {})
    if lat.get("optimize_latent") is False:
        lat["optimize_latent"] = "none"
    # The legacy grad_clip default was +inf, i.e. off.
    if out.get("training", {}).get("grad_clip") == float("inf"):
        out["training"]["grad_clip"] = None
    return out


# Internals
def _build_section(section_cls, values: Dict[str, Any], name: str):
    known = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{name}': {sorted(unknown)}")
    obj = section_cls(**values)
    if section_cls is TrainingConfig:
        defaults = TrainingConfig().epochs
        defaults.update({int(k): int(v) for k, v in obj.epochs.items()})
        obj.epochs = defaults
    return obj


def _check_choice(name: str, value: Any, choices) -> None:
    if value not in choices:
        raise ConfigError(f"{name} must be one of {choices}, got {value!r}")


def _looks_sectioned(raw: Dict[str, Any]) -> bool:
    sections = {f.name for f in dataclasses.fields(ReconConfig)}
    return bool(raw) and set(raw) <= sections


def _apply_override(raw: Dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ConfigError(f"Override must look like section.key=value, got '{item}'")
    dotted, value = item.split("=", 1)
    keys = dotted.strip().split(".")
    node = raw
    for k in keys[:-1]:
        k = int(k) if k.isdigit() else k
        node = node.setdefault(k, {})
    last = int(keys[-1]) if keys[-1].isdigit() else keys[-1]
    node[last] = yaml.safe_load(value)
