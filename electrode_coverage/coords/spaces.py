"""Coordinate space classification and grouping."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd

# ── Known coordinate system strings ──────────────────────────────────────────

MNI_COORDINATE_SYSTEMS = frozenset({
    # BIDS standard names (case-sensitive as they appear in coordsystem.json)
    "MNI152Lin",
    "MNI152NLin2009aSym", "MNI152NLin2009aAsym",
    "MNI152NLin2009bSym", "MNI152NLin2009bAsym",
    "MNI152NLin2009cSym", "MNI152NLin2009cAsym",
    "MNI152NLin6Sym", "MNI152NLin6Asym",
    # Common non-standard variants found in real datasets
    "MNI152NLin6ASym",   # typo in several OpenNeuro datasets (capital S)
    "MNI305",
    "mni",               # RAM-style shorthand
})

ACPC_COORDINATE_SYSTEMS = frozenset({"ACPC", "acpc"})

TALAIRACH_COORDINATE_SYSTEMS = frozenset({"Talairach", "TALAIRACH", "TAL", "tal"})

# Surface-based spaces — electrodes are in surface vertex coordinates,
# NOT volumetric mm. Cannot be screened against a volumetric mask.
SURFACE_COORDINATE_SYSTEMS = frozenset({
    "fsaverage", "fsaveragesym", "fsaverage5", "fsaverage6",
    "individual",  # subject-native FreeSurfer surface
    "MNI305 (fsaverage)",
})

# Scanner/image-native spaces — require subject-specific transforms
# that are not available in the electrode metadata alone.
NATIVE_COORDINATE_SYSTEMS = frozenset({
    "ScanRAS", "scanRAS", "SCANRAS",
    "T1w",                 # subject T1 native space
    "orig",                # FreeSurfer orig space
    "vox", "ct_voxel",     # voxel coordinates (RAM)
    "fs",                  # FreeSurfer native coords (RAM)
    "ind", "ind.dural",    # individual surface (RAM)
    "avg", "avg.dural",    # average surface (RAM)
})

# Explicit labels for coordinate systems that cannot be transformed
UNTRANSFORMABLE_SYSTEMS = frozenset({"Other", "other", "unknown", "n/a"})


# ── Space family classification ──────────────────────────────────────────────

class CoordinateSpace(Enum):
    """Coordinate space family for electrode positions."""
    MNI152 = "mni152"           # All MNI152 template variants
    TALAIRACH = "talairach"     # Talairach atlas space
    ACPC = "acpc"               # AC-PC aligned (treated as approximate MNI)
    SURFACE = "surface"         # FreeSurfer surface-based (fsaverage etc.)
    NATIVE = "native"           # Scanner/subject-native (ScanRAS, voxel, etc.)
    UNKNOWN = "unknown"         # Unrecognised or missing

    @property
    def can_transform_to_mni(self) -> bool:
        """Whether this space family can be transformed to volumetric MNI."""
        return self in (CoordinateSpace.MNI152, CoordinateSpace.TALAIRACH, CoordinateSpace.ACPC)

    @property
    def description(self) -> str:
        _DESC = {
            "mni152": "MNI152 volumetric template",
            "talairach": "Talairach atlas (transformed via Lancaster)",
            "acpc": "AC-PC aligned (treated as approximate MNI)",
            "surface": "FreeSurfer surface coordinates (not volumetric — cannot screen against mask)",
            "native": "Scanner/subject-native (requires subject-specific registration — not available)",
            "unknown": "Unrecognised or missing coordinate system",
        }
        return _DESC.get(self.value, "")


def classify_space(coordinate_system: Optional[str]) -> CoordinateSpace:
    """Classify a coordinate system string into a space family.

    Handles all known BIDS, FreeSurfer, and non-standard variants found
    in OpenNeuro iEEG datasets.
    """
    if coordinate_system is None:
        return CoordinateSpace.UNKNOWN

    # Exact match first (case-sensitive)
    if coordinate_system in MNI_COORDINATE_SYSTEMS:
        return CoordinateSpace.MNI152
    if coordinate_system in ACPC_COORDINATE_SYSTEMS:
        return CoordinateSpace.ACPC
    if coordinate_system in TALAIRACH_COORDINATE_SYSTEMS:
        return CoordinateSpace.TALAIRACH
    if coordinate_system in SURFACE_COORDINATE_SYSTEMS:
        return CoordinateSpace.SURFACE
    if coordinate_system in NATIVE_COORDINATE_SYSTEMS:
        return CoordinateSpace.NATIVE
    if coordinate_system in UNTRANSFORMABLE_SYSTEMS:
        return CoordinateSpace.UNKNOWN

    # Prefix-based fallback (case-insensitive)
    cs_upper = coordinate_system.upper()
    if cs_upper.startswith("MNI"):
        return CoordinateSpace.MNI152
    if "FSAVERAGE" in cs_upper:
        return CoordinateSpace.SURFACE
    if "TALAIRACH" in cs_upper:
        return CoordinateSpace.TALAIRACH

    return CoordinateSpace.UNKNOWN


def is_mni_space(coordinate_system: Optional[str]) -> bool:
    """Check whether a coordinate system string indicates MNI space."""
    return classify_space(coordinate_system) == CoordinateSpace.MNI152


def group_electrodes_by_space(
    df: pd.DataFrame, space_col: str = "coordinate_system",
) -> dict[CoordinateSpace, pd.DataFrame]:
    """Group electrodes DataFrame by coordinate space family.

    Returns a dict mapping each present :class:`CoordinateSpace` to its
    subset DataFrame.
    """
    families = df[space_col].map(classify_space)
    groups: dict[CoordinateSpace, pd.DataFrame] = {}
    for space in CoordinateSpace:
        mask = families == space
        if mask.any():
            groups[space] = df[mask].copy()
    return groups
