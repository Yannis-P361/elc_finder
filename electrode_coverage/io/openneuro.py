"""Download electrode metadata from OpenNeuro via GitHub API."""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _make_ssl_context() -> ssl.SSLContext:
    """Create an SSL context, using certifi CA bundle as fallback.

    On macOS with python.org Python, the default certificate store is
    often empty.  This function tries certifi first, then falls back to
    the system default.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _urlopen(req, **kwargs):
    """urllib.request.urlopen wrapper with SSL fallback."""
    try:
        return urllib.request.urlopen(req, **kwargs)
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            ctx = _make_ssl_context()
            try:
                return urllib.request.urlopen(req, context=ctx, **kwargs)
            except urllib.error.URLError:
                raise RuntimeError(
                    "SSL certificate verification failed. Fix with one of:\n"
                    "  pip install certifi\n"
                    "  # or on macOS:\n"
                    "  /Applications/Python\\ 3.13/Install\\ Certificates.command"
                ) from exc
        raise

_OPENNEURO_GRAPHQL_URL = "https://openneuro.org/crn/graphql"

# Default cache time-to-live: 7 days (in seconds).
_DEFAULT_CACHE_TTL = 7 * 24 * 60 * 60


def discover_ieeg_datasets(timeout: int = 30) -> list[str]:
    """Query the OpenNeuro API for all publicly available iEEG datasets.

    Uses the OpenNeuro GraphQL endpoint with ``modality: "ieeg"`` to
    retrieve every dataset tagged with the iEEG modality.  This query
    runs **live** every time — new datasets are found automatically.

    Returns a sorted list of dataset IDs (e.g. ``["ds002799", "ds003029", ...]``).
    Falls back to a curated list ONLY if the API is unreachable.
    """
    query = '{ datasets(first: 1000, modality: "ieeg") { edges { node { id } } } }'
    data = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        _OPENNEURO_GRAPHQL_URL, data=data,
        headers={"Content-Type": "application/json", "User-Agent": "electrode-coverage-pipeline"},
    )
    try:
        with _urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning(
            "OpenNeuro API query failed (%s). "
            "Using curated fallback list (%d datasets). "
            "This list may be outdated — check your internet connection.",
            exc, len(_CURATED_IEEG_DATASET_IDS),
        )
        return list(_CURATED_IEEG_DATASET_IDS)

    edges = result.get("data", {}).get("datasets", {}).get("edges", [])
    ids = sorted(e["node"]["id"] for e in edges if e.get("node", {}).get("id"))
    logger.info("Discovered %d iEEG datasets on OpenNeuro (live query).", len(ids))
    return ids


# Curated fallback list — known iEEG datasets as of March 2026.
# ONLY used when the OpenNeuro GraphQL API is unreachable.
# The live query (above) is always the primary discovery mechanism.
_CURATED_IEEG_DATASET_IDS = sorted([
    "ds002799", "ds003029", "ds003078", "ds003374", "ds003498", "ds003688",
    "ds003708", "ds003844", "ds003848", "ds003876", "ds004100", "ds004127",
    "ds004194", "ds004080", "ds004370", "ds004457", "ds004473", "ds004551",
    "ds004624", "ds004642", "ds004696", "ds004703", "ds004752", "ds004770",
    "ds004774", "ds004789", "ds004809", "ds004819", "ds004859", "ds004865",
    "ds004944", "ds004977", "ds004993", "ds005007", "ds005059", "ds005083",
    "ds005169", "ds005398", "ds005411", "ds005415", "ds005448", "ds005489",
    "ds005491", "ds005494", "ds005522", "ds005523", "ds005545", "ds005557",
    "ds005558", "ds005574", "ds005592", "ds005624", "ds005670", "ds005691",
    "ds005931", "ds005953", "ds006065", "ds006107", "ds006136", "ds006233",
    "ds006234", "ds006253", "ds006392", "ds006519", "ds006890", "ds006910",
    "ds006914", "ds007095", "ds007118", "ds007119", "ds007120",
])

_GITHUB_API_TREE_URL = (
    "https://api.github.com/repos/OpenNeuroDatasets/{dataset_id}/git/trees/main?recursive=1"
)
_GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/OpenNeuroDatasets/{dataset_id}/main/{filepath}"
)

_TARGET_SUFFIXES = ("_electrodes.tsv", "_coordsystem.json", "_channels.tsv")


def _get_github_token() -> Optional[str]:
    """Get GitHub token from environment for higher API rate limits.

    Without a token: 60 requests/hour.
    With a token (free): 5,000 requests/hour.

    Set via: export GITHUB_TOKEN=ghp_your_token_here
    """
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _read_download_meta(meta_path: Path) -> dict:
    """Read the download metadata JSON, or return empty dict."""
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_download_meta(meta_path: Path, meta: dict) -> None:
    """Write download metadata JSON."""
    meta_path.write_text(json.dumps(meta, indent=2))


def _cache_is_fresh(meta: dict, cache_ttl: int) -> bool:
    """Check if cached data is still within its TTL."""
    ts = meta.get("downloaded_at", 0)
    suffixes = set(meta.get("suffixes", []))
    has_all = set(_TARGET_SUFFIXES).issubset(suffixes)
    return has_all and (time.time() - ts) < cache_ttl


def download_electrode_metadata(
    dataset_id: str,
    cache_dir: Optional[Path] = None,
    cache_ttl: int = _DEFAULT_CACHE_TTL,
    refresh: bool = False,
) -> Path:
    """Download electrode and channel metadata from an OpenNeuro dataset.

    Downloads ``*_electrodes.tsv``, ``*_coordsystem.json``, and
    ``*_channels.tsv`` files via the GitHub mirror, caching them locally.

    Parameters
    ----------
    dataset_id : str
        OpenNeuro dataset ID (e.g. ``"ds003708"``).
    cache_dir : Path, optional
        Parent directory for dataset caches.
    cache_ttl : int
        Cache time-to-live in seconds. Datasets older than this are
        re-checked for updates. Default: 7 days. Use 0 for always fresh.
    refresh : bool
        If True, force re-download regardless of cache state.

    Returns the local dataset root directory.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".electrode_coverage" / "cache" / "datasets"
    dataset_dir = cache_dir / dataset_id
    meta_path = dataset_dir / ".download_meta.json"

    # Migrate from old marker-based cache
    old_marker = dataset_dir / ".download_complete"
    if old_marker.exists() and not meta_path.exists():
        # Old-style cache: lacks channels.tsv download and timestamp
        meta = {"downloaded_at": old_marker.stat().st_mtime, "suffixes": ["_electrodes.tsv", "_coordsystem.json"]}
        _write_download_meta(meta_path, meta)
        old_marker.unlink()

    meta = _read_download_meta(meta_path)

    if not refresh and _cache_is_fresh(meta, cache_ttl):
        age_days = (time.time() - meta.get("downloaded_at", 0)) / 86400
        logger.info(
            "Using cached dataset %s (%.1f days old, TTL=%d days).",
            dataset_id, age_days, cache_ttl // 86400,
        )
        return dataset_dir

    reason = "forced refresh" if refresh else "cache stale or missing"
    logger.info("Fetching %s from GitHub (%s)...", dataset_id, reason)
    tree_url = _GITHUB_API_TREE_URL.format(dataset_id=dataset_id)
    req = urllib.request.Request(tree_url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "electrode-coverage-pipeline")
    token = _get_github_token()
    if token:
        req.add_header("Authorization", f"token {token}")
    try:
        with _urlopen(req) as resp:
            tree_data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(f"Dataset {dataset_id} not found on GitHub.") from exc
        raise RuntimeError(f"GitHub API error for {dataset_id}: {exc}") from exc

    target_files = [
        e["path"] for e in tree_data.get("tree", [])
        if e["type"] == "blob" and any(e["path"].endswith(s) for s in _TARGET_SUFFIXES)
    ]
    if not target_files:
        logger.warning("No electrode/channel files found in %s.", dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_download_meta(meta_path, {
            "downloaded_at": time.time(),
            "suffixes": list(_TARGET_SUFFIXES),
            "file_count": 0,
        })
        return dataset_dir

    logger.info("Found %d electrode/channel metadata files in %s", len(target_files), dataset_id)
    downloaded = 0
    for filepath in target_files:
        local_path = dataset_dir / filepath
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and not refresh:
            continue
        raw_url = _GITHUB_RAW_URL.format(dataset_id=dataset_id, filepath=filepath)
        logger.debug("Downloading %s", filepath)
        try:
            with _urlopen(raw_url) as resp:
                local_path.write_bytes(resp.read())
            downloaded += 1
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            logger.warning("Failed to download %s: %s", filepath, exc)

    _write_download_meta(meta_path, {
        "downloaded_at": time.time(),
        "suffixes": list(_TARGET_SUFFIXES),
        "file_count": len(target_files),
        "tree_sha": tree_data.get("sha"),
    })
    logger.info(
        "Downloaded %d new files for %s (%d total in dataset) to %s",
        downloaded, dataset_id, len(target_files), dataset_dir,
    )
    return dataset_dir
