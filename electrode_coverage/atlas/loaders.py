"""Load brain atlases from various sources (local, NeuroVault, FSL, URL)."""

from __future__ import annotations

import json
import logging
import ssl
import urllib.request
from pathlib import Path
from typing import Optional

import nibabel as nib

logger = logging.getLogger(__name__)


def _make_ssl_context() -> ssl.SSLContext:
    """Create an SSL context, using certifi CA bundle as fallback."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _urlopen(url, **kwargs):
    """urllib.request.urlopen wrapper with SSL fallback."""
    try:
        return urllib.request.urlopen(url, **kwargs)
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            ctx = _make_ssl_context()
            try:
                return urllib.request.urlopen(url, context=ctx, **kwargs)
            except urllib.error.URLError:
                raise RuntimeError(
                    "SSL certificate verification failed. Fix with one of:\n"
                    "  pip install certifi\n"
                    "  # or on macOS:\n"
                    "  /Applications/Python\\ 3.13/Install\\ Certificates.command"
                ) from exc
        raise

_NEUROVAULT_URL_TEMPLATE = "https://neurovault.org/media/images/{collection}/{{filename}}"
_FSL_STANDARD_RAW = "https://git.fmrib.ox.ac.uk/fsl/data_standard/-/raw/master/{filename}"
_FSL_ATLASES_RAW = "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/master/{subdir}/{filename}"

# Mapping of known atlas prefixes to their subdirectory in data_atlases.
_FSL_ATLAS_SUBDIRS: dict[str, str] = {
    "Cerebellum": "Cerebellum",
    "HarvardOxford": "HarvardOxford",
    "JHU": "JHU",
    "Juelich": "Juelich",
    "MNI": "MNI",
    "MarsParietal": "MarsParietalParcellation",
    "MarsTPJ": "MarsTPJParcellation",
    "Neubert": "NeubertVentralFrontalParcellation",
    "SMATT": "SMATT",
    "STN": "STN",
    "Sallet": "SalletDorsalFrontalParcellation",
    "Striatum": "Striatum",
    "Talairach": "Talairach",
    "Thalamus": "Thalamus",
}

_FSL_GITLAB_API = "https://git.fmrib.ox.ac.uk/api/v4/projects/{project}/repository/tree"


def load_atlas(source: str, cache_dir: Optional[Path] = None) -> nib.Nifti1Image:
    """Load a brain atlas from *source*.

    Supported source formats:

    - Local file path: ``"/path/to/atlas.nii.gz"``
    - NeuroVault image ID: ``"neurovault:1401"``
    - FSL standard template: ``"fsl:Fornix_FMRIB_FA1mm"``
    - HTTP(S) URL: ``"https://example.com/atlas.nii.gz"``
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".electrode_coverage" / "cache" / "atlas"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Local file
    p = Path(source)
    if p.exists():
        logger.info("Loading atlas from local path: %s", p)
        return nib.load(str(p))

    # neurovault:<id>
    if source.lower().startswith("neurovault:"):
        image_id = int(source.split(":", 1)[1])
        return fetch_from_neurovault(image_id, cache_dir)

    # fsl:<template_name>
    if source.lower().startswith("fsl:"):
        template_name = source.split(":", 1)[1]
        return fetch_from_fsl(template_name, cache_dir)

    # HTTP(S) URL
    if source.startswith("http://") or source.startswith("https://"):
        return _download_and_cache(source, cache_dir)

    raise ValueError(
        f"Cannot resolve atlas source: {source!r}. "
        "Use a local path, 'neurovault:<id>', 'fsl:<name>', or an HTTP URL."
    )


def fetch_from_neurovault(image_id: int, cache_dir: Path) -> nib.Nifti1Image:
    """Fetch a NIfTI image from NeuroVault by image ID."""
    # Try nilearn first (handles its own caching)
    try:
        from nilearn.datasets import fetch_neurovault_ids
        logger.info("Fetching NeuroVault image %d via nilearn...", image_id)
        result = fetch_neurovault_ids(image_ids=[image_id], verbose=0)
        logger.info("Atlas loaded from nilearn cache: %s", result.images[0])
        return nib.load(result.images[0])
    except Exception as exc:
        logger.debug("nilearn fetch failed (%s), trying direct download...", exc)

    # Direct download fallback — known images
    _KNOWN = {
        1401: "https://neurovault.org/media/images/264/JHU-ICBM-labels-1mm.nii.gz",
    }
    url = _KNOWN.get(image_id)
    if url is None:
        raise RuntimeError(
            f"NeuroVault image {image_id} not in known list and nilearn fetch failed."
        )
    return _download_and_cache(url, cache_dir)


def fetch_from_fsl(template_name: str, cache_dir: Path) -> nib.Nifti1Image:
    """Fetch an FSL standard-space template or atlas from FSL GitLab.

    Tries ``data_standard`` first (for templates like Fornix_FMRIB_FA1mm),
    then ``data_atlases/<subdir>`` for known atlas families (HarvardOxford,
    JHU, Juelich, etc.).
    """
    filename = f"{template_name}.nii.gz"
    cached = cache_dir / filename
    if cached.exists():
        logger.info("Using cached FSL template: %s", cached)
        return nib.load(str(cached))

    # Build candidate URLs: data_standard first, then matching atlas subdir.
    urls = [_FSL_STANDARD_RAW.format(filename=filename)]
    for prefix, subdir in _FSL_ATLAS_SUBDIRS.items():
        if template_name.startswith(prefix):
            urls.append(_FSL_ATLASES_RAW.format(subdir=subdir, filename=filename))
            break

    for url in urls:
        logger.info("Trying FSL download: %s", url)
        try:
            with _urlopen(url) as resp:
                cached.write_bytes(resp.read())
            logger.info("Downloaded FSL template %s from %s", template_name, url)
            return nib.load(str(cached))
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue

    raise RuntimeError(
        f"Failed to download FSL template '{template_name}': not found in "
        "data_standard or data_atlases. Run "
        "'python -m electrode_coverage --list-fsl-atlases' to see available "
        "names, or use a direct URL with --roi-atlas."
    )


def list_fsl_atlases() -> list[str]:
    """Query FSL GitLab and return all available atlas/template names.

    Searches both ``data_standard`` (templates) and ``data_atlases`` (atlas
    families) and returns names usable with ``fsl:<name>``.
    """
    names: list[str] = []

    # data_standard — top-level .nii.gz files
    try:
        url = _FSL_GITLAB_API.format(project="fsl%2Fdata_standard") + "?per_page=100"
        with _urlopen(url) as resp:
            entries = json.loads(resp.read().decode())
        for e in entries:
            if e["name"].endswith(".nii.gz"):
                names.append(e["name"].removesuffix(".nii.gz"))
    except Exception as exc:
        logger.warning("Could not query FSL data_standard: %s", exc)

    # data_atlases — each subdirectory contains .nii.gz files
    for subdir in _FSL_ATLAS_SUBDIRS.values():
        try:
            url = (
                _FSL_GITLAB_API.format(project="fsl%2Fdata_atlases")
                + f"?path={subdir}&per_page=100"
            )
            with _urlopen(url) as resp:
                entries = json.loads(resp.read().decode())
            for e in entries:
                if e["name"].endswith(".nii.gz"):
                    names.append(e["name"].removesuffix(".nii.gz"))
        except Exception as exc:
            logger.warning("Could not query FSL data_atlases/%s: %s", subdir, exc)

    names.sort()
    return names


def _download_and_cache(url: str, cache_dir: Path) -> nib.Nifti1Image:
    """Download a NIfTI file from *url*, caching locally."""
    filename = url.rsplit("/", 1)[-1]
    cached = cache_dir / filename
    if cached.exists():
        logger.info("Using cached file: %s", cached)
        return nib.load(str(cached))
    logger.info("Downloading %s...", url)
    try:
        with _urlopen(url) as resp:
            cached.write_bytes(resp.read())
    except Exception as exc:
        raise RuntimeError(f"Download failed ({url}): {exc}") from exc
    return nib.load(str(cached))
