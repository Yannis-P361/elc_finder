"""Brain surface mesh loading for 3D visualization."""

from __future__ import annotations

import nibabel as nib
import plotly.graph_objects as go


def load_brain_surface(mesh: str = "fsaverage5") -> list[go.Mesh3d]:
    """Load fsaverage pial surface meshes for both hemispheres.

    Parameters
    ----------
    mesh : str
        Surface resolution. Default ``"fsaverage5"`` (~10k vertices) is a
        good balance between detail and rendering speed.
    """
    from nilearn.datasets import fetch_surf_fsaverage

    fsaverage = fetch_surf_fsaverage(mesh=mesh)
    traces = []
    for hemi, key in [("Left", "pial_left"), ("Right", "pial_right")]:
        surf = nib.load(fsaverage[key])
        vertices = surf.darrays[0].data
        faces = surf.darrays[1].data
        traces.append(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="lightgrey", opacity=0.08,
            name=f"Brain ({hemi})",
            hoverinfo="skip",
            lighting=dict(ambient=0.6, diffuse=0.4, specular=0.1),
            visible=True,
            showlegend=True,
        ))
    return traces
