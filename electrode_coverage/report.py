"""Summary statistics and text report generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from electrode_coverage.pipeline import CoverageResult


def generate_summary(result: CoverageResult) -> str:
    """Generate a text summary of pipeline results."""
    from electrode_coverage.coords.spaces import CoordinateSpace, classify_space

    df = result.all_electrodes
    lines = []

    lines.append(f"{'=' * 70}")
    lines.append(f"  Electrode Coverage Report: {result.roi_name}")
    lines.append(f"{'=' * 70}")
    lines.append("")

    # Overall counts
    n_total = len(df)
    n_checked = df["in_roi"].notna().sum()
    n_hits = (df["in_roi"] == True).sum()  # noqa: E712
    n_unchecked = df["in_roi"].isna().sum()
    n_datasets = df["dataset"].nunique()

    lines.append(f"  Total electrodes:   {n_total:,}")
    lines.append(f"  Datasets:           {n_datasets}")
    lines.append(f"  Screened:           {n_checked:,}  (had MNI-transformable coordinates)")
    lines.append(f"  In {result.roi_name}:  {n_hits:,}")
    lines.append(f"  Not screened:       {n_unchecked:,}  (surface/native/unknown space)")
    lines.append("")

    # Coordinate space breakdown with descriptions
    if "coord_space_family" in df.columns:
        lines.append("  Coordinate space breakdown:")
        lines.append(f"    {'Space':<12s} {'Count':>8s}  {'Screened':>8s}  Description")
        lines.append(f"    {'-' * 64}")
        for space_val, count in df["coord_space_family"].value_counts().items():
            try:
                space_enum = CoordinateSpace(space_val)
                desc = space_enum.description
                can_screen = space_enum.can_transform_to_mni
            except ValueError:
                desc = ""
                can_screen = False
            screened = df.loc[
                (df["coord_space_family"] == space_val) & df["in_roi"].notna()
            ].shape[0]
            flag = "yes" if can_screen else "no"
            lines.append(f"    {space_val:<12s} {count:>8,}  {flag:>8s}  {desc}")
        lines.append("")

    # Specific coordinate system strings
    if "coordinate_system" in df.columns:
        lines.append("  Coordinate system detail (original strings from metadata):")
        for cs, count in df["coordinate_system"].value_counts().head(15).items():
            space = classify_space(cs)
            lines.append(f"    {str(cs):<30s} {count:>8,}  -> {space.value}")
        remaining = df["coordinate_system"].nunique() - 15
        if remaining > 0:
            lines.append(f"    ... and {remaining} more")
        lines.append("")

    # Channel annotation summary
    if "is_soz" in df.columns:
        has_ch = df["channel_type"].notna().sum()
        ds_with_ch = df.loc[df["channel_type"].notna(), "dataset"].nunique() if has_ch > 0 else 0
        n_soz = (df["is_soz"] == True).sum()  # noqa: E712
        n_irz = (df["is_irritative_zone"] == True).sum()  # noqa: E712
        n_resected = (df["is_resected"] == True).sum()  # noqa: E712
        n_bad = (df["is_bad"] == True).sum()  # noqa: E712

        soz_in_roi = 0
        if n_soz > 0 and "in_roi" in df.columns:
            soz_in_roi = ((df["is_soz"] == True) & (df["in_roi"] == True)).sum()  # noqa: E712

        lines.append("  Channel annotations:")
        lines.append(f"    Datasets with channels.tsv: {ds_with_ch} / {n_datasets}")
        lines.append(f"    Channels with annotations:  {has_ch:,}")
        lines.append(f"    SOZ channels:               {n_soz:,}")
        lines.append(f"    SOZ channels in {result.roi_name}:     {soz_in_roi:,}")
        lines.append(f"    Irritative zone channels:   {n_irz:,}")
        lines.append(f"    Resected channels:          {n_resected:,}")
        lines.append(f"    Bad channels:               {n_bad:,}")
        lines.append("")

    # Suspicious units flag
    if "suspicious_units" in df.columns:
        n_suspicious = (df["suspicious_units"] == True).sum()  # noqa: E712
        if n_suspicious > 0:
            lines.append(f"  WARNING: {n_suspicious:,} electrodes flagged with suspicious coordinate units.")
            lines.append("  These may be in meters instead of mm, or unrealistically large.")
            lines.append("")

    # Coordinate status breakdown
    if "coord_status" in df.columns:
        lines.append("  Transform status:")
        for status, count in df["coord_status"].value_counts().items():
            lines.append(f"    {status:<25s} {count:>8,}")
        lines.append("")

    # Per-dataset summary
    lines.append("  Per-dataset breakdown:")
    lines.append(f"    {'Dataset':<30s} {'Total':>8s} {'Screened':>10s} {'Hits':>8s}")
    lines.append(f"    {'-' * 58}")
    for ds, stats in result.dataset_summary.items():
        lines.append(
            f"    {ds:<30s} {stats['total']:>8,} {stats['screened']:>10,} {stats['hits']:>8,}"
        )
    lines.append("")

    # ROI-positive electrode list (if not too many)
    roi_df = result.in_roi
    if len(roi_df) > 0 and len(roi_df) <= 200:
        lines.append(f"  Electrodes in {result.roi_name} ({len(roi_df)}):")
        cols = ["dataset", "subject", "channel_name", "mni_x", "mni_y", "mni_z", "coord_status"]
        cols = [c for c in cols if c in roi_df.columns]
        lines.append(roi_df[cols].to_string(index=False, max_rows=50))
    elif len(roi_df) > 200:
        lines.append(f"  {len(roi_df)} electrodes in {result.roi_name} (too many to list here).")

    lines.append("")
    lines.append(f"{'=' * 70}")
    return "\n".join(lines)
