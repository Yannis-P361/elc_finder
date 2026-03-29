"""Pre-screen datasets for channel annotations before full processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from electrode_coverage.io.bids import find_electrode_files
from electrode_coverage.io.channels import (
    has_annotations,
    has_soz_annotations,
    read_channels_tsv,
)

logger = logging.getLogger(__name__)


@dataclass
class PreScreenResult:
    """Pre-screening result for a single dataset."""
    dataset_id: str
    path: Path
    has_channels: bool = False
    has_annotations: bool = False
    has_soz: bool = False


def prescreen_dataset(bids_root: Path, dataset_id: str) -> PreScreenResult:
    """Lightweight check for channel annotations in a BIDS dataset.

    Checks whether the dataset has channels.tsv files, whether those
    files contain annotation columns with non-empty values, and whether
    any channels are annotated as seizure onset zone (SOZ).
    """
    result = PreScreenResult(dataset_id=dataset_id, path=bids_root)

    entries = find_electrode_files(bids_root)
    for entry in entries:
        if entry.channels_tsv is not None:
            result.has_channels = True
            try:
                ch_df = read_channels_tsv(entry.channels_tsv)
                if not ch_df.empty:
                    if has_annotations(ch_df):
                        result.has_annotations = True
                    if has_soz_annotations(ch_df):
                        result.has_soz = True
            except Exception as exc:
                logger.debug("Could not read %s: %s", entry.channels_tsv, exc)

        # Early exit if we already found SOZ (no need to check more files)
        if result.has_soz:
            break

    return result


def filter_datasets(
    datasets: list[tuple[Path, str]],
    *,
    require_channels: bool = False,
    require_annotations: bool = False,
    require_soz: bool = False,
) -> list[tuple[Path, str]]:
    """Filter datasets based on channel annotation pre-screening.

    Returns only datasets that pass the specified filter criteria.
    Logs a summary of the filtering funnel.
    """
    if not (require_channels or require_annotations or require_soz):
        return datasets

    total = len(datasets)
    results = []
    n_has_channels = 0
    n_has_annotations = 0
    n_has_soz = 0

    for ds_path, ds_name in datasets:
        ps = prescreen_dataset(ds_path, ds_name)
        if ps.has_channels:
            n_has_channels += 1
        if ps.has_annotations:
            n_has_annotations += 1
        if ps.has_soz:
            n_has_soz += 1

        # Apply filter chain (most restrictive first)
        if require_soz and not ps.has_soz:
            continue
        if require_annotations and not ps.has_annotations:
            continue
        if require_channels and not ps.has_channels:
            continue
        results.append((ds_path, ds_name))

    # Log filtering funnel
    active_filter = "require_soz" if require_soz else (
        "require_annotations" if require_annotations else "require_channels"
    )
    logger.info(
        "Pre-screening: %d datasets checked, %d have channels.tsv, "
        "%d have annotations, %d have SOZ labels. "
        "Processing %d datasets (filtered by %s).",
        total, n_has_channels, n_has_annotations, n_has_soz,
        len(results), active_filter,
    )
    return results
