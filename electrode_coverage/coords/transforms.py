"""Coordinate space transforms (MNI, ACPC, Talairach)."""

from __future__ import annotations

import logging

import numpy as np
from nibabel.affines import apply_affine

from electrode_coverage.coords.spaces import (
    CoordinateSpace,
    classify_space,
)

logger = logging.getLogger(__name__)

# Lancaster icbm2tal (Talairach -> MNI152) affine matrix.
# Reference: Lancaster JL et al. (2007) NeuroImage 35(1):13-20.
# doi:10.1016/j.neuroimage.2007.01.025
LANCASTER_TAL2MNI = np.array([
    [1.0164, 0.0000, 0.0000, 0.0],
    [0.0000, 1.0236, 0.0000, 0.0],
    [-0.0045, 0.0000, 1.0215, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def flag_suspicious_units(coords: np.ndarray) -> bool:
    """Check whether coordinates have suspicious units.

    Returns True if coordinates appear to be in meters (max < 1.0 mm) or
    unrealistically large (max > 300 mm) for MNI space. Does NOT modify
    the coordinates — just flags them.
    """
    abs_max = np.abs(coords).max()
    if abs_max < 1.0:
        logger.warning(
            "Coordinates may be in meters rather than mm (max abs value=%.4f). "
            "Flagging as suspicious — no automatic conversion applied.",
            abs_max,
        )
        return True
    if abs_max > 300.0:
        logger.warning(
            "Coordinates appear unrealistically large for MNI space "
            "(max abs value=%.1f mm). Flagging as suspicious.",
            abs_max,
        )
        return True
    return False


def transform_to_mni(
    coords: np.ndarray, coordinate_system: str,
) -> tuple[np.ndarray, str, bool]:
    """Transform coordinates to MNI space.

    Returns ``(coords, status, suspicious_units)`` where status is one of:

    - ``"mni_native"`` — already in an MNI152 variant
    - ``"acpc_approximate"`` — ACPC treated as approximate MNI (no real transform)
    - ``"mni_from_talairach"`` — Talairach converted via Lancaster et al. 2007
    - ``"surface_space"`` — fsaverage / surface coords, cannot screen volumetrically
    - ``"native_space"`` — scanner / subject-native, no transform available
    - ``"no_transform"`` — unrecognised coordinate system

    The ``suspicious_units`` flag is True when coordinates may be in the
    wrong units (e.g. meters instead of mm, or unrealistically large).
    """
    space = classify_space(coordinate_system)

    if space == CoordinateSpace.MNI152:
        suspicious = flag_suspicious_units(coords)
        return coords, "mni_native", suspicious

    if space == CoordinateSpace.ACPC:
        suspicious = flag_suspicious_units(coords)
        logger.warning(
            "ACPC coordinates treated as approximate MNI. No subject-specific "
            "registration available. Results for ACPC datasets should be "
            "interpreted with caution."
        )
        return coords, "acpc_approximate", suspicious

    if space == CoordinateSpace.TALAIRACH:
        logger.info("Transforming Talairach -> MNI via Lancaster et al. 2007.")
        mni_coords = apply_affine(LANCASTER_TAL2MNI, coords)
        return mni_coords, "mni_from_talairach", False

    if space == CoordinateSpace.SURFACE:
        logger.info(
            "Coordinate system '%s' is surface-based (not volumetric). "
            "Cannot screen against a volumetric mask.",
            coordinate_system,
        )
        return coords, "surface_space", False

    if space == CoordinateSpace.NATIVE:
        logger.info(
            "Coordinate system '%s' is scanner/subject-native. "
            "Requires subject-specific registration (not available).",
            coordinate_system,
        )
        return coords, "native_space", False

    logger.info("Coordinate system '%s' is unrecognised — no transform available.", coordinate_system)
    return coords, "no_transform", False
