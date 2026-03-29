"""Configuration dataclasses for the electrode coverage pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ROIConfig:
    """Defines the region of interest for electrode screening.

    Provide ONE of:
      - ``mask_path``: a pre-made binary NIfTI mask
      - ``atlas_source`` + ``atlas_labels``: an atlas image and label indices
    """

    name: str = "ROI"

    # Option A: pre-made mask
    mask_path: Optional[Path] = None

    # Option B: atlas + labels
    atlas_source: Optional[str] = None  # "neurovault:1401", "fsl:Fornix_FMRIB", URL, or local path
    atlas_labels: Optional[tuple[int, ...]] = None

    # Mask refinement
    dilate_mm: float = 2.0
    threshold: float = 0.0  # for probabilistic maps: voxels > threshold

    # Visualisation
    color: str = "crimson"

    # --- factory methods ---

    @classmethod
    def from_mask(cls, path, name: str = "ROI", **kwargs) -> ROIConfig:
        return cls(name=name, mask_path=Path(path), **kwargs)

    @classmethod
    def from_atlas(cls, source: str, labels: tuple[int, ...], name: str = "ROI", **kwargs) -> ROIConfig:
        return cls(name=name, atlas_source=source, atlas_labels=labels, **kwargs)

    def __post_init__(self):
        if self.mask_path is not None:
            self.mask_path = Path(self.mask_path)
        if self.mask_path is None and self.atlas_source is None:
            raise ValueError("Provide either mask_path or atlas_source (+atlas_labels).")


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    roi: ROIConfig = field(default_factory=lambda: ROIConfig(name="ROI", mask_path=Path(".")))
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".electrode_coverage" / "cache")
    output_dir: Optional[Path] = None
    output_csv: Optional[Path] = None

    # Proximity: include electrodes within this distance (mm) of the ROI.
    # 0 = exact overlap only. E.g. 5.0 = include electrodes within 5mm.
    proximity_mm: float = 0.0

    # Channel annotation settings
    include_channels: bool = True
    require_channels: bool = False
    require_annotations: bool = False
    require_soz: bool = False

    # Cache settings
    cache_ttl: int = 7 * 24 * 60 * 60  # 7 days in seconds
    refresh_cache: bool = False
