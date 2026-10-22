"""Reading raw radial k-space from MRI_Raw.h5 or ksp / dcf / coord.npy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Default format works with all of our code
RAW_H5_NAME = "MRI_Raw.h5"


def find_h5_file(data_dir: str | Path) -> Path:
    path = Path(data_dir) / RAW_H5_NAME
    if not path.exists():
        raise FileNotFoundError(f"{RAW_H5_NAME} not found in {data_dir}")
    return path


def extract_data_from_h5(h5_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read k-space, DCF and coordinates, with views sorted by acquisition time.

    Returns
    -------
    ksp : (n_coils, n_views, n_samples) complex64
    dcf : (n_views, n_samples) float32
    coord : (n_views, n_samples, 3) float32, in the file's KX, KY, KZ order
    Multiple encodings are concatenated along the view axis.
    """
    with h5py.File(h5_path, "r") as hf:
        kdata = hf["Kdata"]
        gating = hf["Gating"]
        n_enc = int(np.squeeze(kdata.attrs["Num_Encodings"]))
        n_coils = int(np.squeeze(kdata.attrs["Num_Coils"]))
        n_views, n_samples = kdata["KData_E0_C0"].shape[1:]

        ksp = np.zeros((n_coils, n_views, n_samples, n_enc), dtype=np.complex64)
        dcf = np.zeros((n_views, n_samples, n_enc), dtype=np.float32)
        coord = np.zeros((n_views, n_samples, 3, n_enc), dtype=np.float32)

        for enc in range(n_enc):
            order = np.argsort(np.squeeze(gating[f"TIME_E{enc}"]))
            for c in range(n_coils):
                ds = kdata[f"KData_E{enc}_C{c}"]
                ksp[c, :, :, enc] = ds["real"][0][order] + 1j * ds["imag"][0][order]
            dcf[:, :, enc] = kdata[f"KW_E{enc}"][0][order]
            for dim, axis in enumerate(("KX", "KY", "KZ")):
                coord[:, :, dim, enc] = kdata[f"{axis}_E{enc}"][0][order]

    if n_enc == 1:
        ksp, dcf, coord = ksp[..., 0], dcf[..., 0], coord[..., 0]
    else:
        # Encodings are stacked along the view axis
        ksp = np.concatenate([ksp[..., e] for e in range(n_enc)], axis=1)
        dcf = np.concatenate([dcf[..., e] for e in range(n_enc)], axis=0)
        coord = np.concatenate([coord[..., e] for e in range(n_enc)], axis=0)

    logger.info("Raw data: ksp %s, dcf %s, coord %s", ksp.shape, dcf.shape, coord.shape)
    return ksp, dcf, coord


def load_raw_data(data_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``(ksp, dcf, coord)`` from ``MRI_Raw.h5``, falling back to ``.npy`` files.

    Coordinates read from HDF5 have their axis order reversed (KZ, KY, KX) to
    match the image axis convention; ``coord.npy`` is assumed to be stored in
    that convention already.
    """
    data_dir = Path(data_dir)
    try:
        h5_path = find_h5_file(data_dir)
    except FileNotFoundError:
        logger.info("%s not found, loading ksp.npy / dcf.npy / coord.npy", RAW_H5_NAME)
        return (np.load(data_dir / "ksp.npy"), np.load(data_dir / "dcf.npy"),
                np.load(data_dir / "coord.npy"))

    logger.info("Reading %s", h5_path)
    ksp, dcf, coord = extract_data_from_h5(h5_path)
    return ksp, dcf, np.ascontiguousarray(coord[:, :, ::-1])


def crop_to_matrix(ksp: np.ndarray, coord: np.ndarray, dcf: np.ndarray,
                   matrix: int, zoom_factor: float = 1.0):
    """Scale coordinates by ``zoom_factor`` and drop samples outside ``matrix/2``.

    Radius is taken from view 1, as in the original ``utils.resolution.zoom``;
    radial readouts share the same radius profile.
    """
    coord = coord * zoom_factor
    radius = np.sqrt(np.sum(coord[1] ** 2, axis=1))
    mask = radius < matrix / 2
    return ksp[:, :, mask], coord[:, mask, :], dcf[:, mask]
