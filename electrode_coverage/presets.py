"""Built-in presets that reproduce known pipeline configurations."""

from __future__ import annotations

from pathlib import Path

from electrode_coverage.config import PipelineConfig, ROIConfig


def fornix_config(
    dilate_mm: float = 2.0,
    cache_dir: Path | None = None,
    output_csv: Path | None = None,
) -> PipelineConfig:
    """Preset that reproduces the original fornix electrode coverage pipeline.

    Uses the Fornix_FMRIB_FA1mm probabilistic tractography template
    (Brown et al. 2017, NeuroImage: Clinical).
    """
    roi = ROIConfig(
        name="Fornix",
        atlas_source="fsl:Fornix_FMRIB_FA1mm",
        dilate_mm=dilate_mm,
        threshold=0.0,
        color="crimson",
    )
    return PipelineConfig(
        roi=roi,
        cache_dir=cache_dir or Path.home() / ".electrode_coverage" / "cache",
        output_csv=output_csv,
    )


def fornix_jhu_config(
    dilate_mm: float = 2.0,
    output_csv: Path | None = None,
) -> PipelineConfig:
    """Fornix preset using the JHU ICBM-DTI-81 atlas labels (body + cres/stria).

    Falls back to the JHU white-matter atlas instead of the FMRIB template.
    Labels: 1 (column body), 5 (cres/stria terminalis).
    """
    roi = ROIConfig(
        name="Fornix (JHU)",
        atlas_source="neurovault:1401",
        atlas_labels=(1, 5),
        dilate_mm=dilate_mm,
        color="crimson",
    )
    return PipelineConfig(roi=roi, output_csv=output_csv)


# DBS sweet spot markers (Rios et al. 2022, Nature Comms)
FORNIX_DBS_MARKERS = [
    {
        "x": -3.6, "y": -1.5, "z": -3.6,
        "name": "AD DBS sweet spot (Rios 2022)",
        "text": "DBS target (L)",
        "hovertext": "AD DBS sweet spot (Left)\nMNI: (-3.6, -1.5, -3.6)\nRios et al. 2022",
        "color": "yellow", "symbol": "diamond", "size": 8,
    },
    {
        "x": 3.6, "y": -1.5, "z": -3.6,
        "name": "AD DBS sweet spot (Rios 2022)",
        "text": "DBS target (R)",
        "hovertext": "AD DBS sweet spot (Right)\nMNI: (3.6, -1.5, -3.6)\nRios et al. 2022",
        "color": "yellow", "symbol": "diamond", "size": 8,
    },
]
