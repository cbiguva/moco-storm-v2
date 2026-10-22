"""
Self-navigation: respiratory, cardiac and bulk-motion latents from the k-space DC signal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import convolve, find_peaks, firls, savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

from dataloader.raw_io import RAW_H5_NAME
from utils.plotting import plot_traces

logger = logging.getLogger(__name__)


# DC signal
def extract_dc_from_h5(h5_path: str | Path) -> Tuple[np.ndarray, float]:
    """Per-coil DC samples (zero mean, unit std), time-sorted; and TR in seconds."""
    with h5py.File(h5_path, "r") as hf:
        n_coils = int(np.squeeze(hf["Kdata"].attrs["Num_Coils"]))
        n_enc = int(np.squeeze(hf["Kdata"].attrs["Num_Encodings"]))
        per_enc = []
        for enc in range(n_enc):
            time = np.squeeze(hf["Gating"][f"TIME_E{enc}"])
            order = np.argsort(time)
            coils = []
            for c in range(n_coils):
                ds = hf["Gating"][f"K0_E{enc}_C{c}"]
                dc = ds["real"][0][order] + 1j * ds["imag"][0][order]
                dc = dc - np.mean(dc)
                std = np.std(dc)
                coils.append(dc / std if std > 0 else dc)
            per_enc.append(np.stack(coils))
    k_dc = np.concatenate(per_enc, axis=1)

    sorted_time = time[order]
    tr = float(sorted_time[1] - sorted_time[0])
    if tr > 1:            # stored in microseconds
        tr *= 1e-6
    logger.info("Navigator TR: %.6g s", tr)
    return k_dc, tr


def extract_dc_from_ksp(ksp_path: str | Path) -> np.ndarray:
    ksp = np.load(ksp_path)
    k_dc = ksp[:, :, ksp.shape[2] // 2].copy()
    for c in range(k_dc.shape[0]):
        k_dc[c] -= np.mean(k_dc[c])
        std = np.std(k_dc[c])
        if std > 0:
            k_dc[c] /= std
    return k_dc


# Extraction
def _robust_normalise(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    return (x - med) / (1.4826 * np.median(np.abs(x - med)))


def _odd(n: int) -> int:
    return n + ((n - 1) % 2)


def select_coil(k_dc: np.ndarray, tr: float) -> int:
    """Coil with the largest robust spread after a 0.02-0.4 Hz FIR band-pass."""
    n, fl, fh, fw, fs = 9999, 0.02, 0.4, 0.01, 1 / tr
    taps = firls(n, [0, fl - fw, fl, fh, fh + fw, fs / 2], [0, 0, 1, 1, 0, 0], fs=fs)
    best, c_selected = 0.0, 0
    for c in range(len(k_dc)):
        padded = np.pad(k_dc[c], [n // 2, n // 2], mode="reflect")
        resp = convolve(padded, taps, mode="valid")
        sigma = 1.4826 * np.median(np.abs(resp - np.median(resp)))
        logger.info("  coil %2d: sigma=%.4g", c, abs(sigma))
        if sigma > best:
            best, c_selected = sigma, c
    logger.info("Navigator coil: %d", c_selected)
    return c_selected


def extract_resp_from_dc(k_dc: np.ndarray, coil: int, tr: float, invert: bool = True) -> np.ndarray:
    resp = savgol_filter(_robust_normalise(np.abs(k_dc[coil])), _odd(int(2 / tr)), 2)
    return -resp if invert else resp


def extract_resp_and_cardiac_from_dc(k_dc: np.ndarray, coil: int, tr: float, invert: bool = True,
                                     lf_width: float = 0.5, hf_width: float = 0.1
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Split the navigator into a respiratory (low) and cardiac (high) band."""
    dc = _robust_normalise(np.abs(k_dc[coil]))
    w_lf = _odd(int(lf_width / tr))
    t = np.arange(len(dc))
    dc_hf = np.zeros_like(dc)
    for _ in range(3):
        dc_lf = savgol_filter(dc - dc_hf, w_lf, 2)
        dc_lw = lowess(dc - dc_lf, t, frac=1.0 / tr / len(dc), xvals=t)
        dc_hf = dc - (dc_lf + dc_lw)
    dc_hf = savgol_filter(dc - (dc_lf + dc_lw), int(2 * hf_width / tr), 2)
    if invert:
        dc_lf = -dc_lf
    return dc_lf, dc_hf


def find_extrema(resp: np.ndarray, tr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Alternating local extrema of the respiratory trace and a max/min flag for each."""
    extrema, _ = find_peaks(resp, distance=int(2 / tr))
    if len(extrema) == 0 or extrema[0] != 0:
        extrema = np.insert(extrema, 0, 0)

    def is_max(p):
        return int(np.mean(resp[p + 1:p + 20]) < resp[p])

    flags = np.zeros(len(extrema))
    missed, missed_flags = [], []
    last = -1
    for e, p in enumerate(extrema):
        flags[e] = is_max(p)
        if flags[e] == last:
            # Two extrema of the same kind in a row: insert the opposite one between them.
            seg = resp[extrema[e - 1]:p]
            if flags[e]:
                missed.append(np.argmin(seg) + extrema[e - 1]); missed_flags.append(0)
            else:
                missed.append(np.argmax(seg) + extrema[e - 1]); missed_flags.append(1)
        last = flags[e]

    extrema = np.append(extrema, missed)
    flags = np.append(flags, missed_flags)
    order = np.argsort(extrema)
    return extrema[order].astype(int), flags[order].astype(int)


def extract_latent(resp: np.ndarray, extrema: np.ndarray, is_max: np.ndarray, tr: float
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bulk, respiration, insp/exp) latents from the respiratory trace."""
    t = np.arange(len(resp))
    env_min = CubicSpline(extrema[is_max == 0], resp[extrema[is_max == 0]])(t)
    env_max = CubicSpline(extrema[is_max == 1], resp[extrema[is_max == 1]])(t)
    bulk = savgol_filter((env_max + env_min) / 2, int(25 / tr), 2)

    respiration = resp - bulk
    respiration[respiration < -2] = -2          # clip start-up transients
    respiration -= respiration.min()

    bulk = bulk - bulk.min()
    bulk = 2 * (bulk / bulk.max() - 0.5)        # scale to [-1, 1]

    in_exp = np.zeros(len(respiration))
    for e in range(len(extrema) - 1):
        in_exp[extrema[e]:extrema[e + 1]] = 0 if is_max[e] else 10
    in_exp[extrema[-1]:] = 0 if is_max[-1] else 10
    return bulk, respiration, in_exp


# Pipeline
def estimate_latent(data_dir: str | Path, latent_type: str, invert_resp: bool = True,
                    lf_width: float = 0.5, hf_width: float = 0.1, tr: Optional[float] = None,
                    plot_dir: Optional[Path] = None) -> np.ndarray:
    """Navigator latents ``(n_frames, 3)``: respiration, insp/exp or cardiac, bulk."""
    data_dir = Path(data_dir)
    if (data_dir / RAW_H5_NAME).exists():
        k_dc, tr = extract_dc_from_h5(data_dir / RAW_H5_NAME)
    elif (data_dir / "ksp.npy").exists():
        if tr is None:
            raise ValueError("data.tr must be set when reading the navigator from ksp.npy")
        k_dc = extract_dc_from_ksp(data_dir / "ksp.npy")
    else:
        raise FileNotFoundError(f"Neither {RAW_H5_NAME} nor ksp.npy found in {data_dir}")

    coil = select_coil(k_dc, tr)
    if latent_type == "in_exp":
        resp = extract_resp_from_dc(k_dc, coil, tr, invert=invert_resp)
        second, second_name = None, "Insp/Exp Latent"
    elif latent_type == "resp_cardiac":
        resp, second = extract_resp_and_cardiac_from_dc(k_dc, coil, tr, invert=invert_resp,
                                                        lf_width=lf_width, hf_width=hf_width)
        second_name = "Cardiac Latent"
    else:
        raise ValueError(f"Unknown latent_type '{latent_type}'")

    extrema, is_max = find_extrema(resp, tr)
    bulk, respiration, in_exp = extract_latent(resp, extrema, is_max, tr)
    if second is None:
        second = in_exp

    latent = np.stack([respiration, second, bulk], axis=1)
    if plot_dir is not None:
        plot_traces({"Respiration Latent": respiration, second_name: second, "Bulk Latent": bulk},
                    plot_dir / "latent.png")
    return latent
