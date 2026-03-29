"""Main pipeline orchestration: ElectrodeCoveragePipeline and CoverageResult."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from electrode_coverage.atlas.roi import create_roi_mask
from electrode_coverage.config import PipelineConfig
from electrode_coverage.coords.spaces import CoordinateSpace, classify_space
from electrode_coverage.coords.transforms import transform_to_mni
from electrode_coverage.io.bids import ElectrodeFileEntry, find_electrode_files, read_electrode_tsv
from electrode_coverage.io.channels import extract_annotations, read_channels_tsv
from electrode_coverage.io.local import load_electrodes
from electrode_coverage.io.openneuro import download_electrode_metadata
from electrode_coverage.io.prescreening import filter_datasets
from electrode_coverage.screening import check_coords_in_mask, compute_distance_to_roi

logger = logging.getLogger(__name__)


@dataclass
class CoverageResult:
    """Structured output from the pipeline."""

    all_electrodes: pd.DataFrame
    roi_name: str = "ROI"
    roi_config: Optional[object] = None

    @property
    def in_roi(self) -> pd.DataFrame:
        """Electrodes that fall inside the ROI."""
        return self.all_electrodes[self.all_electrodes["in_roi"] == True].copy()  # noqa: E712

    @property
    def space_summary(self) -> dict[str, int]:
        """Electrode counts per coordinate space family."""
        col = "coord_space_family"
        if col in self.all_electrodes.columns:
            return self.all_electrodes[col].value_counts().to_dict()
        return {}

    @property
    def dataset_summary(self) -> dict[str, dict]:
        """Per-dataset statistics: total, screened, hits."""
        summary = {}
        for ds, grp in self.all_electrodes.groupby("dataset"):
            total = len(grp)
            screened = grp["in_roi"].notna().sum()
            hits = (grp["in_roi"] == True).sum()  # noqa: E712
            summary[ds] = {"total": total, "screened": int(screened), "hits": int(hits)}
        return summary

    def save_csv(self, path: Path | str) -> None:
        """Save full results to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.all_electrodes.to_csv(path, index=False)
        logger.info("Results saved to %s", path)

    def print_summary(self) -> None:
        """Print a text summary to stdout."""
        from electrode_coverage.report import generate_summary
        print(generate_summary(self))

    def visualize(self, output_path: Path | str, show_all: bool = False, markers: list[dict] | None = None) -> None:
        """Generate an interactive 3D HTML visualization."""
        from electrode_coverage.visualization.figure import build_figure
        from electrode_coverage.atlas.roi import create_roi_mask as _build_mask

        # Rebuild mask for mesh visualization
        if self.roi_config is not None:
            from electrode_coverage.config import PipelineConfig
            cfg = self.roi_config
            cache_dir = Path.home() / ".electrode_coverage" / "cache"
            mask = _build_mask(cfg, cache_dir)
        else:
            mask = None

        fig = build_figure(self, roi_mask=mask, show_all=show_all, markers=markers)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), include_plotlyjs=True)
        logger.info("Visualization saved to %s", output_path)


class ElectrodeCoveragePipeline:
    """Generalisable electrode coverage pipeline.

    Example::

        from electrode_coverage import ElectrodeCoveragePipeline, PipelineConfig, ROIConfig

        config = PipelineConfig(
            roi=ROIConfig(name="Hippocampus", mask_path="hippo_mask.nii.gz")
        )
        pipe = ElectrodeCoveragePipeline(config)
        pipe.add_openneuro_datasets(["ds003708"])
        pipe.add_local_files(["my_electrodes.csv"], coordinate_system="MNI152NLin2009cAsym")
        result = pipe.run()
        result.print_summary()
        result.visualize("output.html")
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._bids_datasets: list[tuple[Path, str]] = []
        self._openneuro_ids: list[str] = []
        self._local_frames: list[pd.DataFrame] = []

    def add_bids_dataset(self, path: Path | str, name: Optional[str] = None) -> None:
        """Register a local BIDS dataset for screening."""
        path = Path(path)
        self._bids_datasets.append((path, name or path.name))

    def add_openneuro_datasets(self, dataset_ids: list[str]) -> None:
        """Register OpenNeuro dataset IDs for download and screening."""
        self._openneuro_ids.extend(dataset_ids)

    def add_all_openneuro(self) -> None:
        """Discover and register ALL iEEG datasets on OpenNeuro.

        Queries the OpenNeuro GraphQL API for every dataset with modality
        ``ieeg``. Falls back to a curated list if the API is unreachable.
        """
        from electrode_coverage.io.openneuro import discover_ieeg_datasets
        ids = discover_ieeg_datasets()
        logger.info("Adding %d OpenNeuro iEEG datasets.", len(ids))
        self._openneuro_ids.extend(ids)

    def add_local_files(
        self,
        paths: list[Path | str],
        coordinate_system: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> None:
        """Register local electrode files (CSV, TSV, JSON)."""
        for p in paths:
            df = load_electrodes(p, coordinate_system=coordinate_system, dataset_name=dataset_name)
            self._local_frames.append(df)

    def run(self) -> CoverageResult:
        """Execute the pipeline and return results."""
        cache_dir = self.config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Build ROI mask
        logger.info("Building ROI mask: %s", self.config.roi.name)
        roi_mask = create_roi_mask(self.config.roi, cache_dir)

        # Collect all datasets to screen
        datasets: list[tuple[Path, str]] = list(self._bids_datasets)
        for ds_id in self._openneuro_ids:
            try:
                ds_path = download_electrode_metadata(
                    ds_id, cache_dir / "datasets",
                    cache_ttl=self.config.cache_ttl,
                    refresh=self.config.refresh_cache,
                )
                datasets.append((ds_path, ds_id))
            except RuntimeError as exc:
                logger.error("Skipping %s: %s", ds_id, exc)

        # Pre-screen datasets if filtering is enabled
        if self.config.require_channels or self.config.require_annotations or self.config.require_soz:
            datasets = filter_datasets(
                datasets,
                require_channels=self.config.require_channels,
                require_annotations=self.config.require_annotations,
                require_soz=self.config.require_soz,
            )

        all_rows: list[dict] = []

        # Screen BIDS datasets
        for ds_path, ds_name in datasets:
            rows = self._screen_bids_dataset(ds_path, ds_name, roi_mask)
            all_rows.extend(rows)

        # Screen local files
        for df in self._local_frames:
            rows = self._screen_local_electrodes(df, roi_mask)
            all_rows.extend(rows)

        result_df = pd.DataFrame(all_rows)
        if result_df.empty:
            logger.warning("No electrodes found across all data sources.")
            result_df = pd.DataFrame(columns=[
                "dataset", "subject", "session", "space", "channel_name",
                "x", "y", "z", "mni_x", "mni_y", "mni_z",
                "coordinate_system", "coord_space_family", "coord_status",
                "suspicious_units", "distance_mm", "in_roi", "source",
                "channel_type", "channel_status", "is_soz",
                "is_irritative_zone", "is_resected", "is_bad",
                "clinical_annotation",
            ])

        # Log summary
        if not result_df.empty:
            n_checked = result_df["in_roi"].notna().sum()
            n_hits = (result_df["in_roi"] == True).sum()  # noqa: E712
            n_unchecked = result_df["in_roi"].isna().sum()
            logger.info(
                "Summary: %d dataset(s), %d electrodes, %d checked (%d in %s), %d unchecked.",
                result_df["dataset"].nunique(), len(result_df),
                n_checked, n_hits, self.config.roi.name, n_unchecked,
            )

        # Save CSV if configured
        if self.config.output_csv is not None:
            self.config.output_csv.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(self.config.output_csv, index=False)
            logger.info("Results saved to %s", self.config.output_csv)

        return CoverageResult(
            all_electrodes=result_df,
            roi_name=self.config.roi.name,
            roi_config=self.config.roi,
        )

    def _screen_bids_dataset(
        self, bids_root: Path, dataset_name: str, roi_mask,
    ) -> list[dict]:
        """Screen a single BIDS dataset."""
        electrode_files = find_electrode_files(bids_root)
        if not electrode_files:
            logger.warning("No electrodes.tsv files found in %s", bids_root)
            return []

        all_rows: list[dict] = []
        for entry in electrode_files:
            try:
                df = read_electrode_tsv(entry.electrodes_tsv)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", entry.electrodes_tsv, exc)
                continue

            # Merge channel annotations if available
            annotations_df = None
            if self.config.include_channels and entry.channels_tsv is not None:
                try:
                    ch_df = read_channels_tsv(entry.channels_tsv)
                    if not ch_df.empty:
                        annotations_df = extract_annotations(ch_df)
                except Exception as exc:
                    logger.warning("Failed to read %s: %s", entry.channels_tsv, exc)

            rows = self._process_electrode_df(
                df,
                coord_sys=entry.coordinate_system,
                dataset=dataset_name,
                subject=entry.subject,
                session=entry.session,
                space=entry.space,
                source="openneuro" if dataset_name in self._openneuro_ids else "bids",
                roi_mask=roi_mask,
                annotations_df=annotations_df,
            )
            all_rows.extend(rows)
        return all_rows

    def _screen_local_electrodes(self, df: pd.DataFrame, roi_mask) -> list[dict]:
        """Screen a locally-loaded electrode DataFrame.

        If the DataFrame contains multiple coordinate systems (e.g. RAM JSON
        with mni, tal, vox, etc.), splits by system and processes each group.
        """
        dataset = df["dataset"].iloc[0] if "dataset" in df.columns else "local"
        all_rows: list[dict] = []

        if "coordinate_system" in df.columns and df["coordinate_system"].nunique() > 1:
            # Multi-space data: process each coordinate system separately
            for coord_sys, group_df in df.groupby("coordinate_system"):
                subject = group_df["subject"].iloc[0] if "subject" in group_df.columns else None
                rows = self._process_electrode_df(
                    group_df.reset_index(drop=True),
                    coord_sys=coord_sys,
                    dataset=dataset,
                    subject=subject,
                    session=None,
                    space=None,
                    source="local",
                    roi_mask=roi_mask,
                )
                all_rows.extend(rows)
        else:
            coord_sys = df["coordinate_system"].iloc[0] if "coordinate_system" in df.columns else None
            subject = df["subject"].iloc[0] if "subject" in df.columns else None
            all_rows = self._process_electrode_df(
                df,
                coord_sys=coord_sys,
                dataset=dataset,
                subject=subject,
                session=None,
                space=None,
                source="local",
                roi_mask=roi_mask,
            )
        return all_rows

    def _process_electrode_df(
        self, df: pd.DataFrame, *, coord_sys: Optional[str],
        dataset: str, subject: Optional[str], session: Optional[str],
        space: Optional[str], source: str, roi_mask,
        annotations_df: Optional[pd.DataFrame] = None,
    ) -> list[dict]:
        """Process a single electrode DataFrame: transform, screen, produce rows."""
        has_coords = (
            all(c in df.columns for c in ("x", "y", "z"))
            and df[["x", "y", "z"]].notna().any(axis=None)
        )

        space_family = classify_space(coord_sys).value
        coord_status = "no_transform"
        suspicious_units = False
        n = len(df)

        in_roi_values = pd.Series([None] * n, dtype=object)
        distance_values = pd.Series([np.nan] * n)
        mni_x = pd.Series([np.nan] * n)
        mni_y = pd.Series([np.nan] * n)
        mni_z = pd.Series([np.nan] * n)

        if has_coords and coord_sys is not None:
            valid_mask = df[["x", "y", "z"]].notna().all(axis=1)
            at_origin = (df["x"] == 0) & (df["y"] == 0) & (df["z"] == 0)
            valid_mask = valid_mask & ~at_origin
            coords = df.loc[valid_mask, ["x", "y", "z"]].values

            if len(coords) > 0:
                mni_coords, coord_status, suspicious_units = transform_to_mni(coords, coord_sys)

                # Only screen if we actually have MNI coordinates
                _MNI_STATUSES = ("mni_native", "acpc_approximate", "mni_from_talairach")
                if coord_status in _MNI_STATUSES:
                    valid_idx = valid_mask.values.nonzero()[0]
                    mni_x.iloc[valid_idx] = mni_coords[:, 0]
                    mni_y.iloc[valid_idx] = mni_coords[:, 1]
                    mni_z.iloc[valid_idx] = mni_coords[:, 2]

                    # Distance to ROI surface (mm)
                    distances = compute_distance_to_roi(mni_coords, roi_mask)
                    distance_values.iloc[valid_idx] = distances

                    # In-mask check (exact voxel overlap)
                    in_mask = check_coords_in_mask(mni_coords, roi_mask)

                    # Apply proximity threshold: also include electrodes
                    # within proximity_mm of the ROI
                    proximity_mm = self.config.proximity_mm
                    if proximity_mm > 0:
                        in_proximity = distances <= proximity_mm
                        combined = in_mask | in_proximity
                    else:
                        combined = in_mask

                    in_roi_full = pd.Series([False] * n)
                    in_roi_full.iloc[valid_idx] = combined
                    in_roi_values = in_roi_full
                    n_hits = combined.sum()
                    if n_hits > 0:
                        n_exact = in_mask.sum()
                        n_nearby = n_hits - n_exact
                        if n_nearby > 0:
                            logger.info(
                                "  %d/%d electrodes in %s (%d in mask, %d within %.1fmm) (%s)",
                                n_hits, len(df), self.config.roi.name,
                                n_exact, n_nearby, proximity_mm, coord_status,
                            )
                        else:
                            logger.info(
                                "  %d/%d electrodes in %s (%s)",
                                n_hits, len(df), self.config.roi.name, coord_status,
                            )

        # Build annotation lookup by channel name
        annot_lookup: dict[str, dict] = {}
        if annotations_df is not None and not annotations_df.empty:
            for _, arow in annotations_df.iterrows():
                annot_lookup[arow["name"]] = {
                    "channel_type": arow.get("channel_type"),
                    "channel_status": arow.get("channel_status"),
                    "is_soz": arow.get("is_soz", False),
                    "is_irritative_zone": arow.get("is_irritative_zone", False),
                    "is_resected": arow.get("is_resected", False),
                    "is_bad": arow.get("is_bad", False),
                    "clinical_annotation": arow.get("clinical_annotation"),
                }
        _EMPTY_ANNOT = {
            "channel_type": None, "channel_status": None,
            "is_soz": None, "is_irritative_zone": None,
            "is_resected": None, "is_bad": None,
            "clinical_annotation": None,
        }

        rows = []
        for idx, row in df.iterrows():
            row_subject = row.get("subject", subject) if "subject" in df.columns else subject
            ch_name = row.get("name")
            annot = annot_lookup.get(ch_name, _EMPTY_ANNOT) if ch_name else _EMPTY_ANNOT
            rows.append({
                "dataset": dataset,
                "subject": row_subject,
                "session": session,
                "space": space,
                "channel_name": ch_name,
                "x": row.get("x"), "y": row.get("y"), "z": row.get("z"),
                "mni_x": mni_x.iloc[idx], "mni_y": mni_y.iloc[idx], "mni_z": mni_z.iloc[idx],
                "coordinate_system": coord_sys,
                "coord_space_family": space_family,
                "coord_status": coord_status,
                "suspicious_units": suspicious_units,
                "distance_mm": distance_values.iloc[idx],
                "in_roi": in_roi_values.iloc[idx],
                "source": source,
                **annot,
            })
        return rows
