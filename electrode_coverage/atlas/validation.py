"""Validate that a NIfTI atlas is in MNI space and usable as an ROI mask."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationMessage:
    severity: Severity
    message: str


def validate_mni_atlas(img: nib.Nifti1Image) -> list[ValidationMessage]:
    """Validate that a NIfTI image is a usable MNI-space atlas.

    Returns a list of validation messages. Messages with severity ERROR
    indicate the atlas cannot be used; WARNING messages indicate potential
    issues that should be reviewed.
    """
    messages: list[ValidationMessage] = []
    data = np.asarray(img.dataobj)

    # 1. Dimensionality check
    if data.ndim == 4 and data.shape[3] == 1:
        messages.append(ValidationMessage(
            Severity.INFO,
            f"4D image with single volume (shape {data.shape}), will squeeze to 3D.",
        ))
    elif data.ndim != 3:
        messages.append(ValidationMessage(
            Severity.ERROR,
            f"Expected 3D volume, got {data.ndim}D image (shape {data.shape}).",
        ))
        return messages  # Can't continue with wrong dimensionality

    # 2. Non-zero content
    nonzero_count = np.count_nonzero(data)
    if nonzero_count == 0:
        messages.append(ValidationMessage(
            Severity.ERROR,
            "Atlas contains no non-zero voxels — mask would be empty.",
        ))
        return messages

    messages.append(ValidationMessage(
        Severity.INFO, f"Non-zero voxels: {nonzero_count:,}",
    ))

    # 3. Voxel size check
    voxel_sizes = np.abs(np.diag(img.affine)[:3])
    for i, (axis, vs) in enumerate(zip("XYZ", voxel_sizes)):
        if vs < 0.5 or vs > 4.0:
            messages.append(ValidationMessage(
                Severity.WARNING,
                f"Unusual voxel size along {axis}: {vs:.2f} mm "
                f"(typical MNI atlases use 0.5–4.0 mm).",
            ))

    # 4. MNI bounding box heuristic
    shape = data.shape[:3]
    corners = np.array([
        [0, 0, 0],
        [shape[0] - 1, 0, 0],
        [0, shape[1] - 1, 0],
        [0, 0, shape[2] - 1],
        [shape[0] - 1, shape[1] - 1, shape[2] - 1],
    ], dtype=float)
    from nibabel.affines import apply_affine
    world_corners = apply_affine(img.affine, corners)
    bbox_min = world_corners.min(axis=0)
    bbox_max = world_corners.max(axis=0)

    # Typical MNI152 bounding box (with generous margin)
    mni_ranges = {
        "X": (-100, 100),
        "Y": (-140, 100),
        "Z": (-80, 120),
    }
    for i, axis in enumerate("XYZ"):
        lo, hi = mni_ranges[axis]
        if bbox_min[i] < lo - 30 or bbox_max[i] > hi + 30:
            messages.append(ValidationMessage(
                Severity.WARNING,
                f"Bounding box along {axis} [{bbox_min[i]:.0f}, {bbox_max[i]:.0f}] mm "
                f"extends beyond typical MNI range [{lo}, {hi}] mm.",
            ))

    # 5. sform/qform code check
    header = img.header
    sform_code = int(header.get("sform_code", 0))
    qform_code = int(header.get("qform_code", 0))
    if sform_code == 4 or qform_code == 4:
        messages.append(ValidationMessage(
            Severity.INFO, "NIfTI header indicates MNI space (code=4).",
        ))
    elif sform_code == 1 or qform_code == 1:
        messages.append(ValidationMessage(
            Severity.WARNING,
            "NIfTI header indicates scanner-native space (code=1), not MNI (code=4). "
            "Verify this atlas is actually in MNI coordinates.",
        ))
    elif sform_code == 0 and qform_code == 0:
        messages.append(ValidationMessage(
            Severity.WARNING,
            "NIfTI header has no sform/qform code set. "
            "Cannot verify coordinate space from header alone.",
        ))

    # 6. Orientation check
    orientation = nib.aff2axcodes(img.affine)
    if orientation != ("R", "A", "S"):
        messages.append(ValidationMessage(
            Severity.WARNING,
            f"Atlas orientation is {''.join(orientation)}, not RAS. "
            f"MNI atlases are typically in RAS orientation.",
        ))

    # 7. Unique labels (for labeled atlases)
    unique_vals = np.unique(data)
    if len(unique_vals) > 2:
        messages.append(ValidationMessage(
            Severity.INFO,
            f"Labeled atlas with {len(unique_vals)} unique values "
            f"(range: {unique_vals.min():.1f} to {unique_vals.max():.1f}).",
        ))
    elif len(unique_vals) == 2:
        messages.append(ValidationMessage(
            Severity.INFO, "Binary mask (2 unique values).",
        ))

    return messages


def format_atlas_info(img: nib.Nifti1Image, source: str) -> str:
    """Format atlas metadata and validation results as a human-readable string."""
    messages = validate_mni_atlas(img)
    data = np.asarray(img.dataobj)
    shape = data.shape[:3]
    voxel_sizes = np.abs(np.diag(img.affine)[:3])

    # Bounding box
    corners = np.array([
        [0, 0, 0],
        [shape[0] - 1, shape[1] - 1, shape[2] - 1],
    ], dtype=float)
    from nibabel.affines import apply_affine
    world_corners = apply_affine(img.affine, corners)
    bbox_min = world_corners.min(axis=0)
    bbox_max = world_corners.max(axis=0)

    orientation = "".join(nib.aff2axcodes(img.affine))
    sform_code = int(img.header.get("sform_code", 0))
    sform_labels = {0: "unknown", 1: "scanner", 2: "aligned", 3: "Talairach", 4: "MNI"}

    lines = [
        f"Atlas: {source}",
        f"Shape: {shape}, Voxel: {voxel_sizes[0]:.1f} x {voxel_sizes[1]:.1f} x {voxel_sizes[2]:.1f} mm",
        f"Orientation: {orientation}, sform_code: {sform_code} ({sform_labels.get(sform_code, '?')})",
        f"MNI bounding box: X[{bbox_min[0]:.0f}, {bbox_max[0]:.0f}] "
        f"Y[{bbox_min[1]:.0f}, {bbox_max[1]:.0f}] Z[{bbox_min[2]:.0f}, {bbox_max[2]:.0f}]",
    ]

    # Unique values summary
    unique_vals = np.unique(data)
    if len(unique_vals) <= 50:
        nonzero_labels = unique_vals[unique_vals != 0]
        lines.append(f"Unique labels: {len(unique_vals)} total, {len(nonzero_labels)} non-zero")
        if len(nonzero_labels) <= 20:
            lines.append(f"  Labels: {', '.join(str(int(v)) if v == int(v) else str(v) for v in nonzero_labels)}")
    else:
        lines.append(
            f"Unique values: {len(unique_vals)} (range: {unique_vals.min():.2f} to {unique_vals.max():.2f})"
        )

    lines.append(f"Non-zero voxels: {np.count_nonzero(data):,}")
    lines.append("")

    # Validation results
    has_errors = any(m.severity == Severity.ERROR for m in messages)
    has_warnings = any(m.severity == Severity.WARNING for m in messages)

    if has_errors:
        lines.append("Status: FAIL")
    elif has_warnings:
        lines.append("Status: PASS with warnings")
    else:
        lines.append("Status: PASS — appears to be a valid MNI-space atlas")

    for m in messages:
        if m.severity == Severity.ERROR:
            lines.append(f"  ERROR: {m.message}")
        elif m.severity == Severity.WARNING:
            lines.append(f"  WARNING: {m.message}")

    return "\n".join(lines)
