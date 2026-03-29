"""Voxel-based spatial intersection of electrodes against a mask."""

from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


def check_coords_in_mask(coords_mni: np.ndarray, mask: nib.Nifti1Image) -> np.ndarray:
    """Check which MNI coordinates fall within a binary mask.

    Parameters
    ----------
    coords_mni : (N, 3) array of MNI coordinates in mm.
    mask : NIfTI image with binary (or thresholded) mask data.

    Returns
    -------
    (N,) boolean array — True where the electrode is inside the mask.
    """
    mask_data = np.asarray(mask.dataobj)
    inv_affine = np.linalg.inv(mask.affine)
    mask_shape = np.array(mask_data.shape)

    ones = np.ones((coords_mni.shape[0], 1))
    coords_vox = (inv_affine @ np.hstack([coords_mni, ones]).T).T[:, :3]
    coords_vox_int = np.round(coords_vox).astype(int)
    in_bounds = np.all((coords_vox_int >= 0) & (coords_vox_int < mask_shape), axis=1)

    in_mask = np.zeros(len(coords_mni), dtype=bool)
    if in_bounds.any():
        ijk = coords_vox_int[in_bounds]
        in_mask[in_bounds] = mask_data[ijk[:, 0], ijk[:, 1], ijk[:, 2]] > 0
    return in_mask


def compute_distance_to_roi(
    coords_mni: np.ndarray, mask: nib.Nifti1Image,
) -> np.ndarray:
    """Compute the Euclidean distance (in mm) from each electrode to the
    nearest ROI boundary voxel.

    Electrodes inside the ROI get distance 0.0.
    Electrodes outside get a positive distance in mm.
    Out-of-bounds electrodes get NaN.

    Parameters
    ----------
    coords_mni : (N, 3) array of MNI coordinates in mm.
    mask : NIfTI image with binary mask data.

    Returns
    -------
    (N,) float array — distance in mm to nearest ROI surface.
    """
    mask_data = np.asarray(mask.dataobj) > 0
    inv_affine = np.linalg.inv(mask.affine)
    mask_shape = np.array(mask_data.shape)

    # Voxel sizes for converting voxel distance → mm
    voxel_sizes = np.abs(np.diag(mask.affine)[:3])

    # Euclidean distance transform: distance from each non-mask voxel to
    # the nearest mask voxel, in mm (using voxel spacing).
    dist_map = distance_transform_edt(~mask_data, sampling=voxel_sizes)

    # Map MNI coords → voxel indices
    ones = np.ones((coords_mni.shape[0], 1))
    coords_vox = (inv_affine @ np.hstack([coords_mni, ones]).T).T[:, :3]
    coords_vox_int = np.round(coords_vox).astype(int)
    in_bounds = np.all((coords_vox_int >= 0) & (coords_vox_int < mask_shape), axis=1)

    distances = np.full(len(coords_mni), np.nan)
    if in_bounds.any():
        ijk = coords_vox_int[in_bounds]
        distances[in_bounds] = dist_map[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    return distances
