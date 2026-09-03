"""Batch processing: run segmentation, calibration, and metrics over a folder
of plate photographs and assemble results/measurements.csv plus per-image
overlay images.

Every image is processed independently and failures are caught per-image --
one corrupt or unreadable file logs a warning and is recorded with an error
message in the output CSV, but never aborts the rest of the batch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: batch runs must not require a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibrate import CalibrationResult, calibrate
from src.metrics import compute_all_metrics
from src.segment import SegmentationResult, load_image, segment_colony

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

METRIC_COLUMNS = [
    "area",
    "perimeter",
    "circularity",
    "solidity",
    "equivalent_diameter",
    "texture_contrast",
    "texture_entropy",
]
CSV_COLUMNS = [
    "filename",
    *METRIC_COLUMNS,
    "calibrated",
    "mm_per_pixel",
    "calibration_source",
    "warning",
]


def find_images(input_dir: Path) -> list[Path]:
    """Recursively find plate photographs in a folder.

    Parameters
    ----------
    input_dir : Path
        Folder to search, including subfolders.

    Returns
    -------
    list[Path]
        Paths with a .jpg/.jpeg/.png/.tif/.tiff extension (case-insensitive),
        sorted for reproducible processing order.
    """
    input_dir = Path(input_dir)
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Rescale an image to floats in [0, 1] so `imshow` renders it correctly.

    `load_image` preserves the source dtype: an 8-bit photograph is fine as
    given, but scientific TIFFs are frequently 16-bit containers holding a
    much narrower true range (e.g. a 12-bit sensor's 0-4095), which
    matplotlib would otherwise clip to solid white. A 1st-99.5th percentile
    stretch avoids a few hot or dead pixels compressing the real contrast
    into a narrow band, at the cost of losing that same tiny fraction of
    outlier pixels to clipping in the *displayed* overlay -- segmentation
    itself is unaffected, since it runs on `segment.to_grayscale`'s own
    conversion, not this display copy.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), as returned by `segment.load_image`.

    Returns
    -------
    np.ndarray
        Image rescaled to float64 in [0, 1], same shape as `image`.
    """
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float64)
    low, high = np.percentile(image, (1, 99.5))
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def save_overlay(
    image,
    colony_mask,
    calibration: CalibrationResult,
    output_path: Path,
) -> None:
    """Save an annotated overlay showing the detected colony boundary and dish rim.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3).
    colony_mask : np.ndarray
        Boolean colony mask.
    calibration : CalibrationResult
        Output of `calibrate.calibrate`; the dish rim circle is only drawn
        if calibration succeeded via dish rim detection (`source == "dish_rim"`) --
        metadata-sourced calibration carries no dish geometry to draw.
    output_path : Path
        Where to write the overlay PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_overlay_on_axes(ax, image, colony_mask, calibration)
    ax.axis("off")
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def draw_overlay_on_axes(ax, image, colony_mask, calibration: CalibrationResult) -> None:
    """Draw the plate image with colony boundary and dish rim onto an existing Axes.

    Factored out of `save_overlay` so the GUI's live preview (an embedded
    Axes it redraws on every parameter change) shows exactly the same
    annotation as the batch-saved overlay PNGs, with one implementation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into. Not cleared first -- call `ax.clear()` beforehand
        if reusing an Axes across frames.
    image : np.ndarray
        RGB image, shape (H, W, 3).
    colony_mask : np.ndarray
        Boolean colony mask.
    calibration : CalibrationResult
        Output of `calibrate.calibrate`; the dish rim circle is only drawn
        if calibration succeeded via dish rim detection (`source == "dish_rim"`) --
        metadata-sourced calibration carries no dish geometry to draw.
    """
    ax.imshow(_normalize_for_display(image))
    if colony_mask.any():
        ax.contour(colony_mask, colors="lime", linewidths=2)
    if calibration.source == "dish_rim":
        rim = plt.Circle(
            calibration.dish_center,
            calibration.dish_radius_px,
            color="cyan",
            fill=False,
            linewidth=2,
        )
        ax.add_patch(rim)


def process_image(
    path: Path,
    dish_diameter_mm: float = 90.0,
    segment_kwargs: dict | None = None,
    overlay_path: Path | None = None,
) -> dict:
    """Run the full pipeline on one image and return a results row.

    Never raises: any failure (unreadable file, segmentation error, etc.) is
    caught and reported as a warning in the returned row instead.

    Parameters
    ----------
    path : Path
        Path to a plate photograph.
    dish_diameter_mm : float, default 90.0
        Known Petri dish diameter, passed to `calibrate.calibrate`.
    segment_kwargs : dict or None, default None
        Extra keyword arguments forwarded to `segment.segment_colony`
        (e.g. threshold_offset, min_object_size).
    overlay_path : Path or None, default None
        If given, an annotated overlay is saved here.

    Returns
    -------
    dict
        One row matching `CSV_COLUMNS`. Metric values are NaN (via missing
        keys) if processing failed before metrics could be computed.
    """
    row: dict = {"filename": path.name}
    try:
        image = load_image(str(path))
    except Exception as exc:
        message = " ".join(str(exc).split())
        logger.warning("Could not read %s: %s", path, message)
        row["warning"] = f"Could not read image: {message}"
        row["calibrated"] = False
        return row

    try:
        seg: SegmentationResult = segment_colony(image, **(segment_kwargs or {}))
        cal: CalibrationResult = calibrate(image, dish_diameter_mm=dish_diameter_mm, path=path)
        metrics = compute_all_metrics(seg.colony_mask, seg.gray, mm_per_pixel=cal.mm_per_pixel)
        row.update(metrics)
        row["calibrated"] = cal.calibrated
        row["mm_per_pixel"] = cal.mm_per_pixel
        row["calibration_source"] = cal.source

        warning = cal.warning or ""
        if not seg.colony_mask.any():
            warning = (warning + "; " if warning else "") + "No colony detected"
        row["warning"] = warning

        if overlay_path is not None:
            save_overlay(image, seg.colony_mask, cal, overlay_path)
    except Exception as exc:
        message = " ".join(str(exc).split())
        logger.warning("Failed to process %s: %s", path, message)
        row["warning"] = f"Processing failed: {message}"
        row["calibrated"] = False

    return row


def run_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    dish_diameter_mm: float = 90.0,
    segment_kwargs: dict | None = None,
    overrides: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Process every plate photograph in a folder and write results/measurements.csv.

    Parameters
    ----------
    input_dir : str or Path
        Folder to recursively search for plate photographs.
    output_dir : str or Path
        Folder to write measurements.csv into; overlays go in
        `output_dir/overlays/`. Created if it doesn't exist.
    dish_diameter_mm : float, default 90.0
        Known Petri dish diameter in mm, used for calibration.
    segment_kwargs : dict or None, default None
        Extra keyword arguments forwarded to `segment.segment_colony` for
        every image (e.g. threshold_offset, min_object_size).
    overrides : dict[str, dict] or None, default None
        Per-image segmentation parameter overrides keyed by filename
        (e.g. from the GUI's "accept and save parameters for this image").
        Where present, these are merged over `segment_kwargs` for that one
        image; every other image still uses `segment_kwargs` unchanged.

    Returns
    -------
    pd.DataFrame
        One row per image found, columns matching `CSV_COLUMNS`. Also
        written to `output_dir/measurements.csv`.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    overrides = overrides or {}

    image_paths = find_images(input_dir)
    if not image_paths:
        logger.warning("No images found in %s", input_dir)

    rows = []
    for path in image_paths:
        overlay_path = overlays_dir / f"{path.stem}_overlay.png"
        image_kwargs = {**(segment_kwargs or {}), **overrides.get(path.name, {})}
        row = process_image(
            path,
            dish_diameter_mm=dish_diameter_mm,
            segment_kwargs=image_kwargs,
            overlay_path=overlay_path,
        )
        rows.append(row)

    df = pd.DataFrame(rows).reindex(columns=CSV_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "measurements.csv", index=False)
    return df
