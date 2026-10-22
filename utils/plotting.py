"""Diagnostic figures written during a run."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def visualize_coils(coil_images: np.ndarray, center_of_mass: np.ndarray, filename: Path) -> None:
    """Three orthogonal slices through each coil image's centre of mass."""
    n = coil_images.shape[0]
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n), squeeze=False)
    fig.suptitle("Center of Mass Location for All Coils")
    for i in range(n):
        com = center_of_mass[:, i].astype(int)
        views = [
            (coil_images[i, :, :, com[2]], (com[1], com[0]), f"Coil {i}: Sagittal Slice (X={com[0]})"),
            (coil_images[i, com[0], :, :], (com[2], com[1]), f"Coil {i}: Axial Slice (Y={com[1]})"),
            (coil_images[i, :, com[1], :], (com[2], com[0]), f"Coil {i}: Coronal Slice (Z={com[2]})"),
        ]
        for ax, (img, (px, py), title) in zip(axes[i], views):
            ax.imshow(np.abs(img), cmap="gray")
            ax.plot(px, py, "r+", markersize=10)
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_traces(traces: Dict[str, np.ndarray], filename: Path) -> None:
    """One subplot per named 1-D trace."""
    fig, axes = plt.subplots(len(traces), 1, figsize=(10, 2.5 * len(traces)), squeeze=False)
    for ax, (title, values) in zip(axes[:, 0], traces.items()):
        ax.plot(values)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_latent_matrix(latent: np.ndarray, filename: Path, prefix: str = "Latent") -> None:
    latent = latent.reshape(latent.shape[0], -1)
    plot_traces({f"{prefix} {i + 1}": latent[:, i] for i in range(latent.shape[1])}, filename)


def plot_latent_zoom(frame_latent: np.ndarray, bins, state_latent: np.ndarray, filename: Path,
                     n_states_shown: int = 50) -> None:
    """Per-frame navigator over the frames of the first states, with each frame's state latent.

    ``frame_latent`` is ``(n_frames, L)``, ``bins[i]`` the frames of state i and
    ``state_latent`` ``(n_states, L)``. The window covers frames 0 up to the last
    frame of state ``n_states_shown - 1`` (the whole window for time binning).
    """
    frame_latent = frame_latent.reshape(frame_latent.shape[0], -1)
    state_latent = state_latent.reshape(state_latent.shape[0], -1)
    n_shown = min(n_states_shown, len(bins))
    end = int(max(np.max(b) for b in bins[:n_shown])) + 1
    state_of_frame = np.empty(frame_latent.shape[0], dtype=int)
    for i, b in enumerate(bins):
        state_of_frame[b] = i
    frames = np.arange(end)
    fig, axes = plt.subplots(frame_latent.shape[1], 1, figsize=(12, 3 * frame_latent.shape[1]), squeeze=False)
    for col, ax in enumerate(axes[:, 0]):
        ax.plot(frames, frame_latent[:end, col], lw=0.6, label="per frame (navigator)")
        ax.step(frames, state_latent[state_of_frame[:end], col], where="post", lw=1.2,
                label="state latent (bin mean)")
        ax.set_title(f"Latent {col + 1}: frames 0-{end - 1} (first {n_shown} states)")
        ax.set_xlabel("frame")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def save_slice(volume: np.ndarray, index: int, filename: Path, cmap: str = "gray",
               vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    fig, ax = plt.subplots()
    ax.imshow(volume[index], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.savefig(filename)
    plt.close(fig)
