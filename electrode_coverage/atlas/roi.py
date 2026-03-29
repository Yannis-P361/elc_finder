"""Create a binary ROI mask from an atlas or pre-made mask."""

from __future__ import annotations

import logging
from math import ceil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from electrode_coverage.atlas.loaders import load_atlas
from electrode_coverage.atlas.validation import Severity, validate_mni_atlas
from electrode_coverage.config import ROIConfig

logger = logging.getLogger(__name__)


def _run_validation(img: nib.Nifti1Image, source: str) -> None:
    """Run MNI atlas validation and log results. Raises on hard errors."""
    messages = validate_mni_atlas(img)
    for m in messages:
        if m.severity == Severity.ERROR:
            raise ValueError(f"Atlas validation failed ({source}): {m.message}")
        elif m.severity == Severity.WARNING:
            logger.warning("Atlas validation (%s): %s", source, m.message)


def create_roi_mask(config: ROIConfig, cache_dir: Path) -> nib.Nifti1Image:
    """Build a binary NIfTI mask for the configured ROI.

    Handles three cases:
    1. Pre-made mask file (``config.mask_path``)
    2. Atlas + label extraction (``config.atlas_source`` + ``config.atlas_labels``)
    3. Probabilistic atlas thresholding (``config.atlas_source`` + ``config.threshold``)
    """
    if config.mask_path is not None:
        logger.info("Loading ROI mask from %s", config.mask_path)
        img = nib.load(str(config.mask_path))
        _run_validation(img, str(config.mask_path))
        data = np.asarray(img.dataobj)
        # Auto-squeeze single-volume 4D
        if data.ndim == 4 and data.shape[3] == 1:
            data = data[:, :, :, 0]
            img = nib.Nifti1Image(data, img.affine, img.header)
        mask = (data > config.threshold).astype(np.uint8)
        ref_img = img
    elif config.atlas_source is not None:
        atlas_img = load_atlas(config.atlas_source, cache_dir / "atlas")
        _run_validation(atlas_img, config.atlas_source)
        atlas_data = np.asarray(atlas_img.dataobj)
        # Auto-squeeze single-volume 4D
        if atlas_data.ndim == 4 and atlas_data.shape[3] == 1:
            atlas_data = atlas_data[:, :, :, 0]
            atlas_img = nib.Nifti1Image(atlas_data, atlas_img.affine, atlas_img.header)

        if config.atlas_labels is not None:
            mask = np.isin(atlas_data, config.atlas_labels).astype(np.uint8)
            logger.info(
                "%s mask: %d voxels from labels %s",
                config.name, mask.sum(), config.atlas_labels,
            )
        else:
            # Probabilistic: threshold
            mask = (atlas_data > config.threshold).astype(np.uint8)
            logger.info(
                "%s mask: %d voxels (threshold=%.2f)",
                config.name, mask.sum(), config.threshold,
            )
        ref_img = atlas_img
    else:
        raise ValueError("ROIConfig has neither mask_path nor atlas_source.")

    # Dilation
    if config.dilate_mm > 0:
        voxel_sizes = np.abs(np.diag(ref_img.affine)[:3])
        iterations = ceil(config.dilate_mm / voxel_sizes.min())
        struct = generate_binary_structure(3, 1)
        mask = binary_dilation(mask, structure=struct, iterations=iterations).astype(np.uint8)
        logger.info(
            "After %.1fmm dilation (%d iter): %d voxels",
            config.dilate_mm, iterations, mask.sum(),
        )

    return nib.Nifti1Image(mask, ref_img.affine, ref_img.header)
