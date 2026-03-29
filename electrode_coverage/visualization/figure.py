"""Assemble the full 3D interactive figure for electrode coverage."""

from __future__ import annotations

import logging
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from electrode_coverage.visualization.brain import load_brain_surface
from electrode_coverage.visualization.roi_mesh import create_roi_mesh

logger = logging.getLogger(__name__)

# 20-colour palette for dataset distinction
DATASET_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def _get_mni_coords(df: pd.DataFrame) -> tuple:
    """Get MNI coordinates, preferring mni_x/y/z columns, falling back to x/y/z."""
    if "mni_x" in df.columns:
        return df["mni_x"], df["mni_y"], df["mni_z"]
    return df["x"], df["y"], df["z"]


def create_electrode_traces(
    df: pd.DataFrame,
    roi_name: str = "ROI",
    show_all: bool = False,
    color_by: str = "dataset",
) -> list[go.Scatter3d]:
    """Create scatter traces for electrodes, colour-coded by dataset.

    Parameters
    ----------
    df : DataFrame with columns: dataset, subject, channel_name, x, y, z,
         mni_x, mni_y, mni_z, in_roi, coord_status.
    roi_name : Name of the ROI for hover labels.
    show_all : If True, include non-ROI electrodes (dimmed).
    color_by : Column to colour-code by (default: ``"dataset"``).
    """
    traces = []

    # ROI electrodes
    roi_df = df[df["in_roi"].astype(str) == "True"].copy()
    mx, my, mz = _get_mni_coords(roi_df)
    roi_df = roi_df.assign(px=mx, py=my, pz=mz).dropna(subset=["px", "py", "pz"])

    if roi_df.empty:
        logger.warning("No electrodes in ROI to plot.")
    else:
        datasets = sorted(roi_df[color_by].unique())
        color_map = {ds: DATASET_COLORS[i % len(DATASET_COLORS)] for i, ds in enumerate(datasets)}

        for ds in datasets:
            subset = roi_df[roi_df[color_by] == ds]
            traces.append(go.Scatter3d(
                x=subset["px"], y=subset["py"], z=subset["pz"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=color_map[ds],
                    symbol="circle",
                    line=dict(width=0.5, color="white"),
                ),
                name=f"{ds} ({len(subset)})",
                text=[
                    f"{row['dataset']}<br>"
                    f"sub-{row.get('subject', '?')}<br>"
                    f"{row.get('channel_name', '?')}<br>"
                    f"MNI: ({row['px']:.1f}, {row['py']:.1f}, {row['pz']:.1f})"
                    + (f"<br>Type: {row['channel_type']}" if row.get('channel_type') else "")
                    + (f"<br>Annotation: {row['clinical_annotation']}" if row.get('clinical_annotation') else "")
                    for _, row in subset.iterrows()
                ],
                hoverinfo="text",
                showlegend=True,
            ))

    # SOZ electrode highlighting (distinct markers)
    if "is_soz" in df.columns:
        soz_df = roi_df[roi_df["is_soz"] == True].copy()  # noqa: E712
        if not soz_df.empty:
            traces.append(go.Scatter3d(
                x=soz_df["px"], y=soz_df["py"], z=soz_df["pz"],
                mode="markers",
                marker=dict(
                    size=8,
                    color="red",
                    symbol="diamond",
                    line=dict(width=1, color="yellow"),
                ),
                name=f"SOZ in {roi_name} ({len(soz_df)})",
                text=[
                    f"SOZ<br>"
                    f"{row['dataset']}<br>"
                    f"sub-{row.get('subject', '?')}<br>"
                    f"{row.get('channel_name', '?')}<br>"
                    f"MNI: ({row['px']:.1f}, {row['py']:.1f}, {row['pz']:.1f})<br>"
                    f"Annotation: {row.get('clinical_annotation', '')}"
                    for _, row in soz_df.iterrows()
                ],
                hoverinfo="text",
                showlegend=True,
            ))

    # Irritative zone highlighting
    if "is_irritative_zone" in df.columns:
        irz_df = roi_df[roi_df["is_irritative_zone"] == True].copy()  # noqa: E712
        if not irz_df.empty:
            traces.append(go.Scatter3d(
                x=irz_df["px"], y=irz_df["py"], z=irz_df["pz"],
                mode="markers",
                marker=dict(
                    size=7,
                    color="orange",
                    symbol="diamond",
                    line=dict(width=1, color="white"),
                ),
                name=f"Irritative zone in {roi_name} ({len(irz_df)})",
                text=[
                    f"Irritative Zone<br>"
                    f"{row['dataset']}<br>"
                    f"sub-{row.get('subject', '?')}<br>"
                    f"{row.get('channel_name', '?')}<br>"
                    f"MNI: ({row['px']:.1f}, {row['py']:.1f}, {row['pz']:.1f})"
                    for _, row in irz_df.iterrows()
                ],
                hoverinfo="text",
                showlegend=True,
            ))

    # Non-ROI electrodes (dimmed, toggle via legend)
    if show_all:
        non_roi = df[df["in_roi"].astype(str) != "True"].copy()
        nmx, nmy, nmz = _get_mni_coords(non_roi)
        non_roi = non_roi.assign(px=nmx, py=nmy, pz=nmz).dropna(subset=["px", "py", "pz"])
        non_roi = non_roi[non_roi["coord_status"] != "no_transform"]
        if len(non_roi) > 0:
            traces.append(go.Scatter3d(
                x=non_roi["px"], y=non_roi["py"], z=non_roi["pz"],
                mode="markers",
                marker=dict(size=1.5, color="grey", opacity=0.15),
                name=f"Other electrodes ({len(non_roi)})",
                hoverinfo="skip",
                showlegend=True,
                visible="legendonly",
            ))

    return traces


def build_figure(
    result,
    roi_mask: Optional[nib.Nifti1Image] = None,
    show_all: bool = False,
    markers: Optional[list[dict]] = None,
) -> go.Figure:
    """Assemble the full 3D figure.

    Parameters
    ----------
    result : :class:`CoverageResult` with ``all_electrodes`` DataFrame.
    roi_mask : NIfTI mask to render as a 3D mesh. If None, only electrodes shown.
    show_all : Show non-ROI electrodes (dimmed).
    markers : Optional list of annotation markers, each a dict with keys:
              ``x``, ``y``, ``z``, ``name``, ``color`` (and optional ``text``).
    """
    fig = go.Figure()

    roi_name = getattr(result, "roi_name", "ROI")
    roi_config = getattr(result, "roi_config", None)
    roi_color = getattr(roi_config, "color", "crimson") if roi_config else "crimson"

    # Brain surface
    logger.info("Loading brain surface mesh...")
    for trace in load_brain_surface():
        fig.add_trace(trace)

    # ROI mesh
    if roi_mask is not None:
        logger.info("Creating %s 3D mesh...", roi_name)
        for trace in create_roi_mesh(roi_mask, name=roi_name, color=roi_color):
            fig.add_trace(trace)

    # Optional annotation markers (e.g. DBS sweet spots)
    if markers:
        for m in markers:
            fig.add_trace(go.Scatter3d(
                x=[m["x"]], y=[m["y"]], z=[m["z"]],
                mode="markers+text",
                marker=dict(
                    size=m.get("size", 8),
                    color=m.get("color", "yellow"),
                    symbol=m.get("symbol", "diamond"),
                    line=dict(width=1, color="black"),
                ),
                text=[m.get("text", m.get("name", ""))],
                textposition="top center",
                textfont=dict(color=m.get("color", "yellow"), size=10),
                name=m.get("name", "Marker"),
                hovertext=[m.get("hovertext", m.get("name", ""))],
                hoverinfo="text",
                showlegend=True,
            ))

    # Electrodes
    logger.info("Adding electrode markers...")
    df = result.all_electrodes if hasattr(result, "all_electrodes") else result
    for trace in create_electrode_traces(df, roi_name=roi_name, show_all=show_all):
        fig.add_trace(trace)

    fig.update_layout(
        title=dict(
            text=f"{roi_name} — Electrode Coverage",
            font=dict(size=18, color="white"),
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgb(20, 20, 30)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.8, y=0.3, z=0.3),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="rgb(20, 20, 30)",
        legend=dict(
            font=dict(color="white", size=11),
            bgcolor="rgba(30, 30, 40, 0.8)",
            bordercolor="rgba(100, 100, 100, 0.5)",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig
