"""Command-line entry point for batch colony morphometrics.

Usage
-----
    python -m src.cli --input data/plates --output results --dish-diameter 90
"""

from __future__ import annotations

import argparse
import logging

from src.batch import run_batch


def main() -> None:
    """Parse CLI arguments and run a batch over the given input folder."""
    parser = argparse.ArgumentParser(
        description="Batch-process a folder of colony biofilm plate photographs "
        "into shape/texture morphometrics."
    )
    parser.add_argument("--input", required=True, help="Folder containing plate photographs.")
    parser.add_argument(
        "--output", required=True, help="Folder to write measurements.csv and overlays/ into."
    )
    parser.add_argument(
        "--dish-diameter",
        type=float,
        default=90.0,
        help="Known Petri dish diameter in mm (default: 90).",
    )
    parser.add_argument(
        "--threshold-offset",
        type=float,
        default=0.0,
        help="Offset added to the automatic Otsu threshold (default: 0.0).",
    )
    parser.add_argument(
        "--min-object-size",
        type=int,
        default=500,
        help="Minimum object area in pixels to keep as colony (default: 500).",
    )
    parser.add_argument(
        "--morph-kernel-size",
        type=int,
        default=5,
        help="Radius of the morphological cleanup structuring element (default: 5).",
    )
    parser.add_argument(
        "--colony-brighter-than-background",
        action="store_true",
        help="Set if the colony appears brighter than the surrounding agar/background "
        "(e.g. a top-lit opaque biofilm imaged against a dark background), instead of "
        "the default assumption that the colony is darker than its surroundings.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Log per-image progress and warnings."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    segment_kwargs = {
        "colony_darker_than_background": not args.colony_brighter_than_background,
        "threshold_offset": args.threshold_offset,
        "min_object_size": args.min_object_size,
        "morph_kernel_size": args.morph_kernel_size,
    }

    df = run_batch(
        args.input,
        args.output,
        dish_diameter_mm=args.dish_diameter,
        segment_kwargs=segment_kwargs,
    )

    n_failed = df["warning"].fillna("").str.contains("failed|Could not read", case=False).sum()
    print(f"Processed {len(df)} image(s); {n_failed} failed. Results written to {args.output}/measurements.csv")


if __name__ == "__main__":
    main()
