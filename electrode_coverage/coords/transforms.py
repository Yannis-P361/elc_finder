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


def flag_suspicious_units(coords: np.ndarray) -> tuple[np.ndarray, bool]:
    """Check whether coordinates have suspicious units, auto-correcting if possible.

    Returns ``(coords, suspicious)`` where coords may be converted from
    meters to mm if all absolute values are < 1.0.
    """
    abs_max = np.abs(coords).max()
    if abs_max < 1.0:
        logger.warning(
            "Coordinates appear to be in meters (max abs=%.4f). "
            "Auto-converting to mm (*1000).",
            abs_max,
        )
        return coords * 1000, True
    if abs_max > 300.0:
        logger.warning(
            "Coordinates appear unrealistically large for MNI space "
            "(max abs=%.1f mm). Flagging as suspicious.",
            abs_max,
        )
        return coords, True
    return coords, False


def _detect_mislabeled_surface(
    coords: np.ndarray, coordinate_system: str, space: CoordinateSpace,
) -> CoordinateSpace:
    """Detect volumetric coordinates mislabeled as surface space.

    Real fsaverage surface coordinates are integer vertex indices (0–160k).
    If coordinates are labeled as surface but contain float values in the
    typical MNI volumetric range, they are likely mislabeled MNI coordinates.
    """
    if space != CoordinateSpace.SURFACE:
        return space

    abs_max = np.abs(coords).max()
    # After potential m->mm conversion, MNI coords are in ~1-100 range.
    # fsaverage vertex indices are integers in the 0-160k range.
    has_decimals = not np.allclose(coords, np.round(coords), atol=0.01)
    in_mni_range = 1.0 < abs_max < 200.0

    if has_decimals and in_mni_range:
        logger.warning(
            "Coordinates labeled '%s' (surface) appear to be volumetric MNI "
            "coordinates (float values in %.1f–%.1f range). "
            "Reclassifying as MNI152 for screening.",
            coordinate_system, coords.min(), coords.max(),
        )
        return CoordinateSpace.MNI152

    return space


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

    # Detect mislabeled surface coordinates (e.g. fsaverage label on MNI data).
    space = _detect_mislabeled_surface(coords, coordinate_system, space)

    if space == CoordinateSpace.MNI152:
        coords, suspicious = flag_suspicious_units(coords)
        return coords, "mni_native", suspicious

    if space == CoordinateSpace.ACPC:
        coords, suspicious = flag_suspicious_units(coords)
        logger.warning(
            "ACPC coordinates treated as approximate MNI. No subject-specific "
            "registration available. Results for ACPC datasets should be "
            "interpreted with caution."
        )
        return coords, "acpc_approximate", suspicious

    if space == CoordinateSpace.TALAIRACH:
        coords, suspicious = flag_suspicious_units(coords)
        logger.info("Transforming Talairach -> MNI via Lancaster et al. 2007.")
        mni_coords = apply_affine(LANCASTER_TAL2MNI, coords)
        return mni_coords, "mni_from_talairach", suspicious

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
