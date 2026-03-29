"""Load electrode coordinates from local files (CSV, TSV, JSON)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Common column name variants → canonical names
_COLUMN_MAP = {
    # coordinate columns
    "X": "x", "Y": "y", "Z": "z",
    "mni_x": "x", "mni_y": "y", "mni_z": "z",
    "MNI_X": "x", "MNI_Y": "y", "MNI_Z": "z",
    "coord_x": "x", "coord_y": "y", "coord_z": "z",
    # name columns
    "contact_name": "name", "channel_name": "name",
    "contact": "name", "channel": "name", "label": "name",
    "electrode": "name",
    # subject columns
    "patient": "subject", "pt": "subject", "patient_id": "subject",
    "sub": "subject", "subject_id": "subject",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common column variants to canonical names (x, y, z, name, subject)."""
    rename = {}
    for col in df.columns:
        canonical = _COLUMN_MAP.get(col)
        if canonical is not None and canonical not in df.columns:
            rename[col] = canonical
    if rename:
        logger.debug("Normalizing columns: %s", rename)
        df = df.rename(columns=rename)
    return df


def load_electrodes(
    path: Path | str,
    coordinate_system: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load electrode data from a local file (CSV, TSV, or JSON).

    Auto-detects format from file extension and normalizes column names.
    If *coordinate_system* is given it is added as a column.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(path)
    elif ext == ".json":
        df = _load_json(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    df = df.replace("n/a", np.nan)
    df = normalize_columns(df)

    # Coerce coordinate columns
    for col in ("x", "y", "z"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if coordinate_system is not None:
        df["coordinate_system"] = coordinate_system

    if dataset_name is not None:
        df["dataset"] = dataset_name
    elif "dataset" not in df.columns:
        df["dataset"] = path.stem

    if "source" not in df.columns:
        df["source"] = "local"

    return df


def load_from_directory(
    dir_path: Path | str,
    pattern: str = "*.csv",
    coordinate_system: Optional[str] = None,
) -> list[pd.DataFrame]:
    """Load all electrode files matching *pattern* from a directory."""
    dir_path = Path(dir_path)
    frames = []
    for p in sorted(dir_path.glob(pattern)):
        try:
            frames.append(load_electrodes(p, coordinate_system=coordinate_system))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", p, exc)
    return frames


def _load_json(path: Path) -> pd.DataFrame:
    """Load electrodes from a JSON file, handling nested formats (e.g. RAM)."""
    with open(path) as f:
        data = json.load(f)

    # Flat list of dicts: [{"name": ..., "x": ..., "y": ..., "z": ...}, ...]
    if isinstance(data, list):
        return pd.DataFrame(data)

    # RAM-style nested: {subject: {contacts: {name: {atlases: {space: {x, y, z}}}}}}
    # or: {leads: {name: {contacts: {name: {coordinate_spaces: {space: {raw: [x,y,z]}}}}}}}
    rows = []
    if _is_ram_subject_contacts(data):
        rows = _parse_ram_subject_contacts(data)
    elif _is_ram_localization(data):
        rows = _parse_ram_localization(data)
    elif _is_ram_contacts(data):
        rows = _parse_ram_contacts(data)
    else:
        # Try to flatten a generic dict
        return pd.json_normalize(data)

    return pd.DataFrame(rows)


def _is_ram_subject_contacts(data: dict) -> bool:
    """Check for {subject: {contacts: {name: {atlases: ...}}}} format."""
    if not isinstance(data, dict):
        return False
    for val in data.values():
        if isinstance(val, dict) and "contacts" in val:
            contacts = val["contacts"]
            if isinstance(contacts, dict):
                for cv in contacts.values():
                    if isinstance(cv, dict) and "atlases" in cv:
                        return True
    return False


def _parse_ram_subject_contacts(data: dict) -> list[dict]:
    """Parse {subject: {contacts: {name: {atlases: {atlas: {x, y, z, region}}}}}}."""
    rows = []
    for subject_id, subject_data in data.items():
        contacts = subject_data.get("contacts", {})
        if not isinstance(contacts, dict):
            continue
        for contact_name, contact_data in contacts.items():
            atlases = contact_data.get("atlases", {})
            if not isinstance(atlases, dict):
                continue
            for atlas_name, atlas_data in atlases.items():
                if not isinstance(atlas_data, dict):
                    continue
                x = atlas_data.get("x")
                y = atlas_data.get("y")
                z = atlas_data.get("z")
                if x is not None and y is not None and z is not None:
                    row = {
                        "name": contact_name,
                        "subject": subject_id,
                        "x": x, "y": y, "z": z,
                        "coordinate_system": atlas_name,
                    }
                    region = atlas_data.get("region")
                    if region is not None:
                        row["region"] = region
                    rows.append(row)
    return rows


def _is_ram_localization(data: dict) -> bool:
    """Check if this looks like a RAM localization.json with leads→contacts→coordinate_spaces."""
    if not isinstance(data, dict):
        return False
    for val in data.values():
        if isinstance(val, dict) and "contacts" in val:
            return True
    return False


def _parse_ram_localization(data: dict) -> list[dict]:
    """Parse RAM localization JSON: {lead: {contacts: {name: {coordinate_spaces: {space: ...}}}}}."""
    rows = []
    for lead_name, lead_data in data.items():
        contacts = lead_data.get("contacts", {})
        for contact_name, contact_data in contacts.items():
            spaces = contact_data.get("coordinate_spaces", {})
            for space_name, space_data in spaces.items():
                coords = space_data.get("raw", space_data.get("corrected"))
                if coords is None or not isinstance(coords, (list, tuple)):
                    continue
                if len(coords) >= 3:
                    rows.append({
                        "name": contact_name,
                        "lead": lead_name,
                        "x": coords[0],
                        "y": coords[1],
                        "z": coords[2],
                        "coordinate_system": space_name,
                    })
    return rows


def _is_ram_contacts(data: dict) -> bool:
    """Check if this looks like RAM contacts_*.json with atlases."""
    if not isinstance(data, dict):
        return False
    for val in data.values():
        if isinstance(val, dict) and "atlases" in val:
            return True
    return False


def _parse_ram_contacts(data: dict) -> list[dict]:
    """Parse RAM contacts JSON: {contact: {atlases: {atlas: {x, y, z, region}}}}."""
    rows = []
    for contact_name, contact_data in data.items():
        atlases = contact_data.get("atlases", {})
        for atlas_name, atlas_data in atlases.items():
            if not isinstance(atlas_data, dict):
                continue
            x = atlas_data.get("x")
            y = atlas_data.get("y")
            z = atlas_data.get("z")
            if x is not None and y is not None and z is not None:
                row = {
                    "name": contact_name,
                    "x": x, "y": y, "z": z,
                    "coordinate_system": atlas_name,
                }
                region = atlas_data.get("region")
                if region is not None:
                    row["region"] = region
                rows.append(row)
    return rows
