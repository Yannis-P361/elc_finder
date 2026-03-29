"""Parse BIDS channels.tsv files and extract clinical annotations."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Regex patterns for clinical annotation detection (case-insensitive).
_SOZ_PATTERNS = re.compile(
    r"\bsoz\b|seizure[\s\-_]?onset|onset[\s\-_]?zone", re.IGNORECASE
)
_IRRITATIVE_PATTERNS = re.compile(
    r"\birritative\b|\birz\b|\binterictal\b", re.IGNORECASE
)
_RESECTED_PATTERNS = re.compile(
    r"\bresect(?:ed|ion)?\b", re.IGNORECASE
)

# Columns in channels.tsv that may carry clinical annotations.
_ANNOTATION_COLUMNS = ("status_description", "description")


def read_channels_tsv(path: Path) -> pd.DataFrame:
    """Read a BIDS ``*_channels.tsv`` file.

    Returns a DataFrame with at least the ``name`` column. Other standard
    BIDS columns (``type``, ``status``, ``status_description``,
    ``description``) are included if present.
    """
    df = pd.read_csv(path, sep="\t")
    df = df.replace("n/a", np.nan)
    if "name" not in df.columns:
        logger.warning("channels.tsv at %s has no 'name' column — skipping.", path)
        return pd.DataFrame(columns=["name"])
    return df


def extract_annotations(channels_df: pd.DataFrame) -> pd.DataFrame:
    """Extract clinical annotations from a channels DataFrame.

    Scans ``status``, ``status_description``, and ``description`` columns
    for known clinical keywords (SOZ, irritative zone, resected, bad).

    Returns a DataFrame with columns:
    - ``name`` — channel name (for joining with electrodes)
    - ``channel_type`` — from BIDS ``type`` column (ECOG, SEEG, EKG, etc.)
    - ``channel_status`` — from BIDS ``status`` column (good/bad)
    - ``is_soz`` — True if channel matches seizure onset zone patterns
    - ``is_irritative_zone`` — True if channel matches irritative zone patterns
    - ``is_resected`` — True if channel matches resected patterns
    - ``is_bad`` — True if channel status is "bad"
    - ``clinical_annotation`` — raw annotation text (combined from description fields)
    """
    n = len(channels_df)
    result = pd.DataFrame({
        "name": channels_df["name"],
        "channel_type": channels_df.get("type"),
        "channel_status": channels_df.get("status"),
        "is_soz": [False] * n,
        "is_irritative_zone": [False] * n,
        "is_resected": [False] * n,
        "is_bad": [False] * n,
        "clinical_annotation": [None] * n,
    })

    # Flag bad channels
    if "status" in channels_df.columns:
        result["is_bad"] = channels_df["status"].str.lower().eq("bad").fillna(False)

    # Scan text columns for clinical annotations
    annotation_texts = []
    for col in _ANNOTATION_COLUMNS:
        if col in channels_df.columns:
            annotation_texts.append(channels_df[col].fillna(""))

    if annotation_texts:
        combined = annotation_texts[0]
        for extra in annotation_texts[1:]:
            combined = combined + " " + extra
        combined = combined.str.strip()

        result["clinical_annotation"] = combined.replace("", None)
        result["is_soz"] = combined.apply(lambda s: bool(_SOZ_PATTERNS.search(s)) if s else False)
        result["is_irritative_zone"] = combined.apply(
            lambda s: bool(_IRRITATIVE_PATTERNS.search(s)) if s else False
        )
        result["is_resected"] = combined.apply(
            lambda s: bool(_RESECTED_PATTERNS.search(s)) if s else False
        )

    return result


def has_annotations(channels_df: pd.DataFrame) -> bool:
    """Check if a channels DataFrame contains any non-empty clinical annotations."""
    for col in _ANNOTATION_COLUMNS:
        if col in channels_df.columns:
            non_empty = channels_df[col].replace("n/a", np.nan).dropna()
            if len(non_empty) > 0 and any(non_empty.str.strip() != ""):
                return True
    return False


def has_soz_annotations(channels_df: pd.DataFrame) -> bool:
    """Check if any channel is annotated as seizure onset zone."""
    for col in _ANNOTATION_COLUMNS:
        if col in channels_df.columns:
            values = channels_df[col].fillna("")
            if values.apply(lambda s: bool(_SOZ_PATTERNS.search(s)) if s else False).any():
                return True
    return False
