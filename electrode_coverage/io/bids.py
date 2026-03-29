"""BIDS dataset traversal and electrode file parsing."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ElectrodeFileEntry:
    """A single BIDS electrode file with its metadata."""

    electrodes_tsv: Path
    coordsystem_json: Optional[Path]
    coordinate_system: Optional[str]
    channels_tsv: Optional[Path]
    subject: Optional[str]
    session: Optional[str]
    space: Optional[str]


def parse_bids_entities(filename: str) -> dict[str, Optional[str]]:
    """Extract BIDS entities (sub, ses, space, task, run, acq) from a filename."""
    entities: dict[str, Optional[str]] = {}
    for m in re.finditer(r"(sub|ses|space|task|run|acq)-([a-zA-Z0-9]+)", filename):
        entities[m.group(1)] = m.group(2)
    return entities


def find_electrode_files(bids_root: Path) -> list[ElectrodeFileEntry]:
    """Discover all iEEG ``*_electrodes.tsv`` files in a BIDS dataset."""
    patterns = [
        "sub-*/ieeg/*_electrodes.tsv",
        "sub-*/ses-*/ieeg/*_electrodes.tsv",
        "derivatives/**/sub-*/ieeg/*_electrodes.tsv",
        "derivatives/**/sub-*/ses-*/ieeg/*_electrodes.tsv",
    ]
    tsv_files: set[Path] = set()
    for pattern in patterns:
        tsv_files.update(bids_root.glob(pattern))

    results: list[ElectrodeFileEntry] = []
    seen_keys: dict[tuple, int] = {}
    for tsv_path in sorted(tsv_files):
        entities = parse_bids_entities(tsv_path.name)
        cs_name = tsv_path.name.replace("_electrodes.tsv", "_coordsystem.json")
        cs_path = tsv_path.parent / cs_name
        coordinate_system = None
        if cs_path.exists():
            try:
                with open(cs_path) as f:
                    coordinate_system = json.load(f).get("iEEGCoordinateSystem")
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Could not read %s: %s", cs_path, exc)
        else:
            logger.warning("No coordsystem.json found for %s", tsv_path)

        # Look for matching channels.tsv (same entities, _channels.tsv suffix)
        ch_name = tsv_path.name.replace("_electrodes.tsv", "_channels.tsv")
        ch_path = tsv_path.parent / ch_name
        channels_tsv = ch_path if ch_path.exists() else None

        key = (entities.get("sub"), entities.get("ses"), entities.get("space"))
        is_derivative = "derivatives" in tsv_path.relative_to(bids_root).parts
        if key in seen_keys:
            prev_idx = seen_keys[key]
            prev_is_deriv = "derivatives" in results[prev_idx].electrodes_tsv.relative_to(bids_root).parts
            if prev_is_deriv and not is_derivative:
                logger.debug("Replacing derivatives duplicate with raw: %s", tsv_path.name)
                results[prev_idx] = None  # type: ignore[assignment]
            else:
                logger.debug("Skipping duplicate (derivatives): %s", tsv_path.name)
                continue

        seen_keys[key] = len(results)
        results.append(ElectrodeFileEntry(
            electrodes_tsv=tsv_path,
            coordsystem_json=cs_path if cs_path.exists() else None,
            coordinate_system=coordinate_system,
            channels_tsv=channels_tsv,
            subject=entities.get("sub"),
            session=entities.get("ses"),
            space=entities.get("space"),
        ))
    results = [r for r in results if r is not None]
    logger.info("Found %d electrodes.tsv file(s) in %s", len(results), bids_root)
    return results


def read_electrode_tsv(path: Path) -> pd.DataFrame:
    """Read a BIDS ``*_electrodes.tsv`` and coerce coordinate columns to float."""
    df = pd.read_csv(path, sep="\t")
    df = df.replace("n/a", np.nan)
    for col in ("x", "y", "z"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
