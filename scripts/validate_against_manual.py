"""Validate the automated pipeline against manual ImageJ colony area measurements.

Reprocesses a folder of real plate photographs through the pipeline and
compares the resulting colony area against a CSV of areas measured by hand
in ImageJ (the workflow this tool automates -- see the README's "Validation"
section). Reports Pearson correlation and Bland-Altman agreement, and saves
scatter + Bland-Altman plots.

Usage
-----
    python -m scripts.validate_against_manual \\
        --images-dir "path/to/real/plates" \\
        --manual-csv "path/to/Results_Quant_Biofilms.csv"

The images folder and CSV are expected to follow this project's MSc dataset
naming convention: each image is named `<prefix>.lif_-_<colony>_QBf.tif` and
the CSV has a `Label` column reading `<prefix>.lif - <colony>` plus a
`colonyArea_micron2` column. Adjust `filename_to_label` if matching against
a differently-named manual dataset.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.batch import run_batch

logger = logging.getLogger(__name__)


def filename_to_label(filename: str) -> str:
    """Convert an image filename to the manual CSV's `Label` value.

    `21_04_09.lif_-_DepsA_1_QBf.tif` -> `21_04_09.lif - DepsA_1`
    """
    stem = Path(filename).stem
    if stem.endswith("_QBf"):
        stem = stem[: -len("_QBf")]
    return stem.replace(".lif_-_", ".lif - ", 1)


def load_manual_measurements(manual_csv: Path) -> pd.DataFrame:
    """Load the manual ImageJ CSV and convert area to mm^2 for comparison."""
    manual = pd.read_csv(manual_csv)
    manual = manual.rename(columns={"colonyArea_micron2": "manual_area_mm2"})
    manual["manual_area_mm2"] = manual["manual_area_mm2"] / 1e6
    return manual[["Label", "manual_area_mm2"]]


def merge_with_manual(automated: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Join automated results (with a `Label` column already added) to manual measurements."""
    merged = automated.merge(manual, on="Label", how="inner")
    merged = merged.rename(columns={"area": "automated_area_mm2"})
    return merged


def summarize_agreement(merged: pd.DataFrame) -> dict:
    """Compute agreement statistics between manual and automated area."""
    manual = merged["manual_area_mm2"].to_numpy()
    automated = merged["automated_area_mm2"].to_numpy()
    diff = automated - manual
    mean = (automated + manual) / 2

    pearson_r = float(np.corrcoef(manual, automated)[0, 1])
    slope, intercept = np.polyfit(manual, automated, 1)
    mean_bias = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    loa_low = mean_bias - 1.96 * sd_diff
    loa_high = mean_bias + 1.96 * sd_diff
    rmse = float(np.sqrt(np.mean(diff**2)))
    # Percent difference is undefined where the manual measurement is 0 (a
    # plate manually scored as having no growth) -- exclude those rows
    # rather than let a division by zero blow up the mean.
    nonzero_manual = manual > 0
    mean_pct_diff = float(np.mean(diff[nonzero_manual] / manual[nonzero_manual]) * 100)

    return {
        "n": len(merged),
        "pearson_r": pearson_r,
        "regression_slope": float(slope),
        "regression_intercept": float(intercept),
        "mean_bias_mm2": mean_bias,
        "mean_pct_diff": mean_pct_diff,
        "loa_low_mm2": loa_low,
        "loa_high_mm2": loa_high,
        "rmse_mm2": rmse,
        "mean_x": mean,
        "diff": diff,
    }


def plot_scatter(merged: pd.DataFrame, stats: dict, output_path: Path) -> None:
    manual = merged["manual_area_mm2"].to_numpy()
    automated = merged["automated_area_mm2"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(manual, automated, alpha=0.6, edgecolor="none")

    lims = [0, max(manual.max(), automated.max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x")

    slope, intercept = stats["regression_slope"], stats["regression_intercept"]
    fit_x = np.array(lims)
    ax.plot(fit_x, slope * fit_x + intercept, color="tab:red", linewidth=1.5, label="Fit")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Manual (ImageJ) area, mm$^2$")
    ax.set_ylabel("Automated (pipeline) area, mm$^2$")
    ax.set_title(f"Manual vs. automated colony area (n={stats['n']}, r={stats['pearson_r']:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_bland_altman(stats: dict, output_path: Path) -> None:
    mean_x, diff = stats["mean_x"], stats["diff"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mean_x, diff, alpha=0.6, edgecolor="none")
    ax.axhline(stats["mean_bias_mm2"], color="tab:red", linewidth=1.5, label="Mean bias")
    ax.axhline(stats["loa_low_mm2"], color="gray", linestyle="--", linewidth=1, label="±1.96 SD")
    ax.axhline(stats["loa_high_mm2"], color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Mean of manual and automated area, mm$^2$")
    ax.set_ylabel("Automated − manual area, mm$^2$")
    ax.set_title("Bland-Altman agreement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, help="Folder of real plate photographs.")
    parser.add_argument("--manual-csv", required=True, help="CSV of manual ImageJ measurements.")
    parser.add_argument(
        "--output",
        default="results/validation",
        help="Folder for batch output + validation CSV (default: results/validation).",
    )
    parser.add_argument(
        "--plots-output",
        default="docs/images",
        help="Folder to save the scatter/Bland-Altman PNGs into (default: docs/images).",
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

    output_dir = Path(args.output)
    plots_dir = Path(args.plots_output)
    plots_dir.mkdir(parents=True, exist_ok=True)

    segment_kwargs = {
        "colony_darker_than_background": not args.colony_brighter_than_background,
    }
    automated = run_batch(args.images_dir, output_dir, segment_kwargs=segment_kwargs)

    metadata_calibrated = (automated["calibration_source"] == "metadata").sum()
    print(
        f"Processed {len(automated)} image(s); "
        f"{metadata_calibrated} calibrated from embedded TIFF metadata."
    )
    uncalibrated = automated[automated["calibration_source"] != "metadata"]
    if not uncalibrated.empty:
        print(f"WARNING: {len(uncalibrated)} image(s) did not calibrate from metadata:")
        for _, row in uncalibrated.iterrows():
            print(f"  {row['filename']}: {row['warning']}")

    automated["Label"] = automated["filename"].apply(filename_to_label)
    manual = load_manual_measurements(Path(args.manual_csv))
    merged = merge_with_manual(automated, manual)
    unmatched = set(automated["Label"]) - set(merged["Label"])
    if unmatched:
        print(f"WARNING: {len(unmatched)} image(s) had no matching manual measurement:")
        for label in sorted(unmatched):
            print(f"  {label}")

    merged_path = output_dir / "validation_measurements.csv"
    merged.to_csv(merged_path, index=False)

    no_colony = merged[merged["warning"].astype(str).str.contains("No colony", na=False)]
    if not no_colony.empty:
        print(f"\n{len(no_colony)} image(s) had no colony detected (area recorded as 0):")
        for _, row in no_colony.iterrows():
            print(f"  {row['filename']} (manual: {row['manual_area_mm2']:.1f} mm^2)")

    stats = summarize_agreement(merged)
    print()
    print(f"n = {stats['n']}")
    print(f"Pearson r = {stats['pearson_r']:.4f}")
    print(f"Regression: automated = {stats['regression_slope']:.4f} * manual + {stats['regression_intercept']:.2f}")
    print(f"Mean bias = {stats['mean_bias_mm2']:.2f} mm^2 ({stats['mean_pct_diff']:.1f}%)")
    print(f"95% limits of agreement = [{stats['loa_low_mm2']:.2f}, {stats['loa_high_mm2']:.2f}] mm^2")
    print(f"RMSE = {stats['rmse_mm2']:.2f} mm^2")

    plot_scatter(merged, stats, plots_dir / "validation_scatter.png")
    plot_bland_altman(stats, plots_dir / "validation_bland_altman.png")
    print(f"\nSaved merged measurements to {merged_path}")
    print(f"Saved plots to {plots_dir}/validation_scatter.png and {plots_dir}/validation_bland_altman.png")


if __name__ == "__main__":
    main()
