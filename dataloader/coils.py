"""Coil selection, PCA coil compression and coil sensitivity estimation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import sigpy as sp
import sigpy.mri as mr
import torch
from scipy.ndimage import gaussian_filter

from utils.plotting import visualize_coils

logger = logging.getLogger(__name__)


def reconstruct_coil_images(ksp: np.ndarray, coord: np.ndarray, dcf: np.ndarray,
                            matrix: Optional[Sequence[int]] = None, oversamp: float = 2.1,
                            width: int = 5, device: int = 0) -> np.ndarray:
    """Density-compensated adjoint NUFFT of each coil, ``(n_coils, *matrix)``."""

    if matrix is None:
        matrix = sp.estimate_shape(coord)
    n_coils = ksp.shape[0]
    images = np.zeros((n_coils, *matrix), dtype=np.complex64)
    dev = sp.Device(device)
    
    with dev:
        dcf_d = sp.to_device(dcf, dev)
        coord_d = sp.to_device(coord, dev)
        for c in range(n_coils):
            logger.info("Reconstructing coil %d of %d", c, n_coils)
            k = sp.to_device(ksp[c], dev) * dcf_d
            images[c] = sp.to_device(
                sp.nufft_adjoint(k, coord_d, oshape=tuple(matrix), oversamp=oversamp, width=width), -1)
    
    return images


def center_of_mass_3d(images: np.ndarray) -> np.ndarray:
    """Per-coil centre of mass ``(3, n_coils)`` of the thresholded squared magnitude."""

    img = np.abs(images) if np.iscomplexobj(images) else images.astype(np.float64)
    lo = img.min(axis=(1, 2, 3), keepdims=True)
    hi = img.max(axis=(1, 2, 3), keepdims=True)
    img = np.square((img - lo) / (hi - lo))
    img[img < 0.1 * img.max(axis=(1, 2, 3), keepdims=True)] = 0

    grids = np.meshgrid(*[np.arange(s) for s in img.shape[1:]], indexing="ij")
    com = np.zeros((3, img.shape[0]))
    for c in range(img.shape[0]):
        mass = img[c].sum()
        if mass > 0:
            for axis in range(3):
                com[axis, c] = np.sum(grids[axis] * img[c]) / mass
    return com



def select_coils(ksp: np.ndarray, coord: np.ndarray, dcf: np.ndarray, method: str,
                 outside_fov_threshold: float, plot_dir: Optional[Path] = None,
                 device: int = 0) -> Tuple[np.ndarray, List[int]]:
    """Drop coils that see mostly signal from outside the FOV or sit at its edge.

    Works on a 64^3 reconstruction of the k-space centre.

    ``outside_fov``: reconstruct at 2x FOV and keep coils whose outside/inside
    energy ratio is below ``outside_fov_threshold``.
    ``center_of_mass``: keep coils whose centre of mass lies within 20-80 % of
    the matrix along x (S/I) and 15-85 % along z (L/R).
    """
    native, double = [64, 64, 64], [128, 128, 128]
    n_coils = ksp.shape[0]
    radius = np.sqrt(np.sum(coord[1] ** 2, axis=1))
    n_read = int(np.sum(radius < native[0] / 2))
    ksp_sm, coord_sm, dcf_sm = ksp[:, :, :n_read], coord[:, :n_read, :], dcf[:, :n_read]

    if method == "outside_fov":
        images = reconstruct_coil_images(ksp_sm, coord_sm * 2, dcf_sm, double, device=device)
        com = center_of_mass_3d(images)
        idx = np.arange(native[0]) + native[0] // 2
        xi, yi, zi = np.meshgrid(idx, idx, idx, indexing="ij")
        in_fov = np.zeros([n_coils] + double)
        in_fov[:, xi, yi, zi] = np.abs(images[:, xi, yi, zi])
        out_fov = np.abs(images) - in_fov
        in_e = np.sum(in_fov ** 2, axis=(1, 2, 3))
        out_e = np.sum(out_fov ** 2, axis=(1, 2, 3))
        selected = [i for i in range(n_coils) if out_e[i] / in_e[i] < outside_fov_threshold]

    elif method == "center_of_mass":
        images = reconstruct_coil_images(ksp_sm, coord_sm, dcf_sm, native, device=device)
        com = center_of_mass_3d(images)
        x_ok = (com[0] >= 0.2 * native[0]) & (com[0] <= 0.8 * native[0])
        z_ok = (com[2] >= 0.15 * native[2]) & (com[2] <= 0.85 * native[2])
        selected = list(np.where(x_ok & z_ok)[0])
    else:
        raise ValueError(f"Invalid coil selection method '{method}'")

    logger.info("Coil selection (%s): kept %d of %d coils: %s", method, len(selected), n_coils, selected)
    if not selected:
        raise RuntimeError(f"Coil selection '{method}' rejected every coil; relax the threshold")
    if plot_dir is not None:
        visualize_coils(images[selected], com[:, selected], plot_dir / "selected_coil_images.png")
    
    return ksp[selected], selected



def pca_coil_compression(ksp: np.ndarray, target_channels: int,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Compress ``(n_coils, ...)`` k-space to ``target_channels`` virtual coils.

    The SVD is computed on a random 5 % subsample of the samples.
    """
    rng = rng or np.random.default_rng()
    n_coils = ksp.shape[0]
    if n_coils == target_channels:
        logger.info("Coil compression skipped: already %d channels", n_coils)
        return ksp

    mask = rng.random(ksp.shape[1:]) < 0.05
    calib = np.stack([ksp[c][mask] for c in range(n_coils)])
    u, s, _ = np.linalg.svd(calib, full_matrices=False)
    logger.info("Coil compression %d -> %d, singular values: %s", n_coils, target_channels, s)

    moved = np.moveaxis(ksp, 0, -1)[..., None]           # (..., n_coils, 1)
    compressed = np.squeeze(np.matmul(u, moved), axis=-1)[..., :target_channels]
    
    return np.moveaxis(compressed, -1, 0)



def jsense_sensitivities(ksp: np.ndarray, coord: np.ndarray, dcf: np.ndarray,
                         im_size: Sequence[int], device: int = 0) -> torch.Tensor:
    maps = mr.app.JsenseRecon(ksp, coord=coord, weights=dcf, img_shape=list(im_size),
                              mps_ker_width=12, ksp_calib_width=32, max_iter=10,
                              max_inner_iter=10, lamda=0.0, show_pbar=False,
                              device=sp.Device(device)).run()
    
    return torch.tensor(sp.to_device(maps, -1))


def lowres_sensitivities(ksp: np.ndarray, coord: np.ndarray, dcf: np.ndarray,
                         im_size: Sequence[int], sense_process: str = "smooth",
                         device: int = 0) -> torch.Tensor:
    """Coil images from the central 16^3 of k-space divided by their root-sum-of-squares."""
    dev = sp.Device(device)
    xp = dev.xp
    im_size = tuple(im_size)
    radius = np.sum(np.abs(coord) ** 2, axis=2) ** 0.5
    n_read = int(np.max(np.sum(radius <= 8, axis=1)))
    k_lr = ksp[:, :, :n_read] * dcf[:, :n_read]
    n_coils = k_lr.shape[0]

    with dev:
        coord_lr = sp.to_device(coord[:, :n_read, :], dev)
        rss = xp.zeros(im_size, dtype=np.complex64)
        coil_lr = xp.zeros((n_coils, *im_size), dtype=np.complex64)
        for c in range(n_coils):
            coil_lr[c] = sp.nufft_adjoint(sp.to_device(k_lr[c], dev), coord_lr, oshape=im_size,
                                          oversamp=2.1, width=4.0)
            rss += coil_lr[c] * xp.conj(coil_lr[c])
        rss = xp.abs(xp.sqrt(rss))

        if sense_process == "polynomial":
            csm = fit_weighted_polynomial(sp.to_device(coil_lr / rss, -1),
                                          sp.to_device(xp.abs(coil_lr), -1))
        elif sense_process == "smooth":
            csm = np.zeros((n_coils, *im_size), dtype=np.complex64)
            for c in range(n_coils):
                csm[c] = gaussian_filter(sp.to_device(coil_lr[c] / rss, -1), [5.0, 5.0, 5.0])
        else:
            csm = sp.to_device(coil_lr / rss, -1)
    return torch.tensor(csm)


def fit_weighted_polynomial(maps: np.ndarray, weights: np.ndarray, degree: int = 3) -> np.ndarray:
    """Locally weighted 3-D polynomial fit of each coil map.

    Each voxel solves its own weighted least-squares problem with weights
    ``|coil| / distance^2`` over the whole grid, so the cost is O(N^6) per coil:
    only usable for small matrices.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_coils, nx, ny, nz = maps.shape
    axes = [torch.linspace(-1, 1, n, device=device) for n in (nx, ny, nz)]
    X, Y, Z = (g.flatten() for g in torch.meshgrid(*axes, indexing="ij"))
    pts = torch.stack([X, Y, Z], dim=1)
    A = torch.stack([X ** i * Y ** j * Z ** k
                     for i in range(degree + 1)
                     for j in range(degree + 1 - i)
                     for k in range(degree + 1 - i - j)], dim=1)

    fitted = np.zeros_like(maps, dtype=np.complex64)
    n_points, batch = nx * ny * nz, 100
    for coil in range(n_coils):
        m = torch.tensor(maps[coil].flatten(), dtype=torch.cfloat, device=device)
        w_coil = torch.tensor(np.abs(weights[coil]).flatten(), dtype=torch.float32, device=device)
        out = torch.zeros(n_points, dtype=torch.cfloat, device=device)
        for start in range(0, n_points, batch):
            stop = min(start + batch, n_points)
            w = w_coil.unsqueeze(0) / (torch.cdist(pts[start:stop], pts) + 1e-6) ** 2
            w = w / w.sum(dim=1, keepdim=True)
            for i in range(start, stop):
                WA = w[i - start, :, None] * A
                coef_r = torch.linalg.lstsq(WA, w[i - start] * m.real).solution
                coef_i = torch.linalg.lstsq(WA, w[i - start] * m.imag).solution
                out[i] = torch.complex((A[i] * coef_r).sum(), (A[i] * coef_i).sum())
        fitted[coil] = out.cpu().numpy().reshape(nx, ny, nz)
    
    return fitted
