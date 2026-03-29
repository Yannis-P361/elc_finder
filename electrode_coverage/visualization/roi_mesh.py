"""Convert a binary NIfTI mask into a smooth 3D mesh for Plotly."""

from __future__ import annotations

from math import ceil

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


def mask_to_smooth_mesh(
    mask_data: np.ndarray,
    affine: np.ndarray,
    sigma: float = 1.2,
    level: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a binary mask to a smoothed mesh in MNI coordinates.

    Uses Gaussian smoothing before marching cubes to produce a continuous
    surface rather than blocky voxel edges.

    Returns (vertices, faces) both as numpy arrays.
    """
    smoothed = gaussian_filter(mask_data.astype(np.float32), sigma=sigma)
    vertices, faces, _, _ = marching_cubes(smoothed, level=level)
    # Transform voxel → MNI mm
    ones = np.ones((vertices.shape[0], 1))
    mni = (affine @ np.hstack([vertices, ones]).T).T[:, :3]
    return mni, faces


def create_roi_mesh(
    mask: nib.Nifti1Image,
    name: str = "ROI",
    color: str = "crimson",
    opacity: float = 0.45,
    sigma: float = 1.5,
    level: float = 0.25,
) -> list[go.Mesh3d]:
    """Create Plotly Mesh3d trace(s) for a NIfTI mask.

    Parameters
    ----------
    mask : NIfTI image with binary mask data.
    name : Display name for the legend.
    color : Mesh colour.
    opacity : Mesh opacity (0–1).
    sigma : Gaussian smoothing sigma for marching cubes.
    level : Iso-surface level for marching cubes.
    """
    mask_data = np.asarray(mask.dataobj)
    if mask_data.max() == 0:
        return []

    verts, faces = mask_to_smooth_mesh(mask_data, mask.affine, sigma=sigma, level=level)

    trace = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity,
        name=name,
        hoverinfo="skip",
        lighting=dict(ambient=0.5, diffuse=0.5, specular=0.2),
        showlegend=True,
    )
    return [trace]
