"""Generalisable electrode coverage pipeline for intracranial EEG data.

Screen intracranial electrodes against any brain atlas ROI, with support
for OpenNeuro, local BIDS datasets, and arbitrary coordinate files.
"""

__version__ = "0.1.0"

from electrode_coverage.config import PipelineConfig, ROIConfig
from electrode_coverage.pipeline import CoverageResult, ElectrodeCoveragePipeline

__all__ = [
    "ElectrodeCoveragePipeline",
    "CoverageResult",
    "PipelineConfig",
    "ROIConfig",
]
