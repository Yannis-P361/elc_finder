"""Command-line interface for the electrode coverage pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from electrode_coverage.config import PipelineConfig, ROIConfig
from electrode_coverage.pipeline import ElectrodeCoveragePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen intracranial electrodes against a brain atlas ROI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Fornix preset with OpenNeuro data:\n"
            "  %(prog)s --preset fornix --openneuro ds003708 -o results.csv --visualize fornix.html\n\n"
            "  # Custom ROI mask:\n"
            "  %(prog)s --roi-mask hippo.nii.gz --roi-name Hippocampus "
            "--openneuro ds003708 -o results.csv\n\n"
            "  # Local electrode files:\n"
            "  %(prog)s --roi-mask mask.nii.gz --local-csv electrodes.csv "
            "--local-space MNI152NLin2009cAsym -o results.csv\n"
        ),
    )

    # ROI definition (mutually exclusive groups)
    roi_group = parser.add_argument_group("ROI definition")
    roi_group.add_argument("--preset", choices=["fornix", "fornix-jhu"],
                           help="Use a built-in ROI preset.")
    roi_group.add_argument("--roi-mask", type=Path, metavar="PATH",
                           help="Path to a pre-made binary NIfTI mask.")
    roi_group.add_argument("--roi-atlas", type=str, metavar="SOURCE",
                           help="Atlas source: local path, 'neurovault:<id>', 'fsl:<name>', or URL.")
    roi_group.add_argument("--roi-labels", type=int, nargs="+", metavar="LABEL",
                           help="Label indices to extract from the atlas.")
    roi_group.add_argument("--roi-name", type=str, default="ROI",
                           help="Human-readable ROI name (default: ROI).")
    roi_group.add_argument("--roi-color", type=str, default="crimson",
                           help="ROI mesh colour for visualization.")
    roi_group.add_argument("--dilate-mm", type=float, default=2.0,
                           help="Mask dilation in mm (default: 2.0).")
    roi_group.add_argument("--threshold", type=float, default=0.0,
                           help="Threshold for probabilistic masks (default: 0.0).")
    roi_group.add_argument("--proximity-mm", type=float, default=0.0,
                           help="Include electrodes within this distance (mm) of the ROI "
                                "(default: 0 = exact overlap only). E.g. --proximity-mm 5.")
    roi_group.add_argument("--atlas-info", action="store_true",
                           help="Print atlas metadata and validation, then exit.")
    roi_group.add_argument("--list-fsl-atlases", action="store_true",
                           help="List all available FSL atlas names for use with 'fsl:<name>', then exit.")

    # Data sources
    data_group = parser.add_argument_group("Data sources")
    data_group.add_argument("bids_roots", type=Path, nargs="*",
                            help="Local BIDS dataset paths.")
    data_group.add_argument("--openneuro", nargs="+", metavar="ID", default=None,
                            help="OpenNeuro dataset IDs to download and scan.")
    data_group.add_argument("--all-openneuro", action="store_true",
                            help="Discover and scan ALL iEEG datasets on OpenNeuro.")
    data_group.add_argument("--local-csv", nargs="+", type=Path, metavar="PATH",
                            help="Local electrode coordinate files (CSV, TSV, JSON).")
    data_group.add_argument("--local-space", type=str, default=None,
                            help="Coordinate system for --local-csv files.")

    # Channel annotations
    annot_group = parser.add_argument_group("Channel annotations")
    annot_group.add_argument("--include-channels", action="store_true", default=True,
                             help="Parse channels.tsv and merge annotations (default: on).")
    annot_group.add_argument("--no-channels", dest="include_channels", action="store_false",
                             help="Skip channel annotation parsing.")
    annot_group.add_argument("--require-channels", action="store_true",
                             help="Only process datasets that have channels.tsv files.")
    annot_group.add_argument("--require-annotations", action="store_true",
                             help="Only process datasets with clinical annotations in channels.tsv.")
    annot_group.add_argument("--require-soz", action="store_true",
                             help="Only process datasets with SOZ-annotated channels.")

    # Cache
    cache_group = parser.add_argument_group("Cache")
    cache_group.add_argument("--refresh-cache", action="store_true",
                             help="Force re-download of cached datasets.")
    cache_group.add_argument("--cache-ttl", type=int, default=7, metavar="DAYS",
                             help="Cache time-to-live in days (default: 7). Use 0 for always fresh.")

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output", "-o", type=Path, default=None,
                           help="Output CSV path.")
    out_group.add_argument("--visualize", type=Path, default=None, metavar="PATH",
                           help="Generate interactive 3D HTML visualization.")
    out_group.add_argument("--search-report", action="store_true",
                           help="Print a search/filtering funnel summary.")
    out_group.add_argument("--verbose", "-v", action="store_true",
                           help="Verbose logging.")

    return parser


def _print_search_report(result) -> None:
    """Print a search/filtering funnel summary."""
    df = result.all_electrodes
    n_datasets = df["dataset"].nunique()
    n_with_electrodes = df.loc[df[["x", "y", "z"]].notna().any(axis=1), "dataset"].nunique()
    n_screenable = df.loc[df["in_roi"].notna(), "dataset"].nunique()
    n_with_hits = df.loc[df["in_roi"] == True, "dataset"].nunique()  # noqa: E712

    lines = [
        "",
        "Search pipeline funnel:",
        f"  1. Datasets processed:         {n_datasets}",
        f"  2. With electrode coordinates:  {n_with_electrodes} / {n_datasets}",
        f"  3. With MNI-screenable coords:  {n_screenable} / {n_datasets}",
        f"  4. With hits in {result.roi_name}:        {n_with_hits} / {n_datasets}",
    ]
    if "channel_type" in df.columns:
        n_with_channels = df.loc[df["channel_type"].notna(), "dataset"].nunique()
        lines.insert(3, f"  2b. With channels.tsv:          {n_with_channels} / {n_datasets}")
    if "is_soz" in df.columns:
        n_with_soz = df.loc[df["is_soz"] == True, "dataset"].nunique()  # noqa: E712
        if n_with_soz > 0:
            lines.append(f"  5. With SOZ annotations:        {n_with_soz} / {n_datasets}")

    print("\n".join(lines))


def _handle_atlas_info(args: argparse.Namespace) -> None:
    """Load and validate an atlas, print info, then exit."""
    import nibabel as nib
    from electrode_coverage.atlas.loaders import load_atlas
    from electrode_coverage.atlas.validation import format_atlas_info

    cache_dir = Path.home() / ".electrode_coverage" / "cache"
    if args.roi_mask:
        img = nib.load(str(args.roi_mask))
        source = str(args.roi_mask)
    elif args.roi_atlas:
        img = load_atlas(args.roi_atlas, cache_dir / "atlas")
        source = args.roi_atlas
    elif args.preset:
        from electrode_coverage.presets import fornix_config, fornix_jhu_config
        cfg = {"fornix": fornix_config, "fornix-jhu": fornix_jhu_config}[args.preset]()
        from electrode_coverage.atlas.roi import create_roi_mask
        # Just load the atlas for info, don't build mask
        if cfg.roi.atlas_source:
            img = load_atlas(cfg.roi.atlas_source, cache_dir / "atlas")
            source = cfg.roi.atlas_source
        elif cfg.roi.mask_path:
            img = nib.load(str(cfg.roi.mask_path))
            source = str(cfg.roi.mask_path)
        else:
            print("ERROR: Preset has no atlas source or mask path.")
            return
    else:
        print("ERROR: Provide --roi-mask, --roi-atlas, or --preset to inspect.")
        return

    print(format_atlas_info(img, source))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # List FSL atlases and exit
    if args.list_fsl_atlases:
        from electrode_coverage.atlas.loaders import list_fsl_atlases
        print("Querying FSL GitLab for available atlases...")
        names = list_fsl_atlases()
        if not names:
            print("No atlases found (check your internet connection).")
            return
        print(f"\n{len(names)} available FSL atlases (use as 'fsl:<name>'):\n")
        for name in names:
            print(f"  fsl:{name}")
        return

    # Atlas info mode — inspect atlas and exit
    if args.atlas_info:
        _handle_atlas_info(args)
        return

    # Build configuration
    if args.preset:
        from electrode_coverage.presets import fornix_config, fornix_jhu_config
        config = {"fornix": fornix_config, "fornix-jhu": fornix_jhu_config}[args.preset](
            dilate_mm=args.dilate_mm, output_csv=args.output,
        )
    elif args.roi_mask:
        config = PipelineConfig(
            roi=ROIConfig(
                name=args.roi_name, mask_path=args.roi_mask,
                dilate_mm=args.dilate_mm, threshold=args.threshold,
                color=args.roi_color,
            ),
            output_csv=args.output,
        )
    elif args.roi_atlas:
        labels = tuple(args.roi_labels) if args.roi_labels else None
        config = PipelineConfig(
            roi=ROIConfig(
                name=args.roi_name, atlas_source=args.roi_atlas,
                atlas_labels=labels, dilate_mm=args.dilate_mm,
                threshold=args.threshold, color=args.roi_color,
            ),
            output_csv=args.output,
        )
    else:
        parser.error("Provide --preset, --roi-mask, or --roi-atlas.")

    if not args.bids_roots and not args.openneuro and not args.local_csv and not args.all_openneuro:
        parser.error("Provide at least one data source: BIDS paths, --openneuro, --all-openneuro, or --local-csv.")

    # Apply proximity and annotation settings
    config.proximity_mm = args.proximity_mm
    config.include_channels = args.include_channels
    config.require_channels = args.require_channels
    config.require_annotations = args.require_annotations
    config.require_soz = args.require_soz
    config.refresh_cache = args.refresh_cache
    config.cache_ttl = args.cache_ttl * 24 * 60 * 60  # days to seconds

    # Build and run pipeline
    pipe = ElectrodeCoveragePipeline(config)
    for root in (args.bids_roots or []):
        pipe.add_bids_dataset(root)
    if args.all_openneuro:
        pipe.add_all_openneuro()
    if args.openneuro:
        pipe.add_openneuro_datasets(args.openneuro)
    if args.local_csv:
        pipe.add_local_files(args.local_csv, coordinate_system=args.local_space)

    result = pipe.run()

    if result.all_electrodes.empty:
        print("No electrodes found.")
        return

    result.print_summary()

    # Search/filtering funnel report
    if args.search_report:
        _print_search_report(result)

    if args.visualize:
        result.visualize(args.visualize)
        print(f"Visualization saved to {args.visualize}")
