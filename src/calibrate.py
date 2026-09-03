"""Pixel-to-millimetre calibration, via embedded TIFF metadata or dish rim detection.

Two independent calibration sources are supported:

- **Metadata** (`mm_per_pixel_from_tiff_metadata`): reads the pixel size an
  acquisition/analysis tool (e.g. ImageJ) already wrote into the TIFF's
  ImageJ-format resolution tags. This is the only option that works for
  photographs cropped to a single colony, where the dish rim isn't in frame.
- **Dish rim detection** (`detect_dish_circle`): OpenCV's Hough circle
  transform locates the Petri dish rim; given the known physical dish
  diameter, the detected rim radius in pixels yields a mm-per-pixel scale
  factor. Requires the whole dish to be visible in the photograph.

`calibrate()` tries metadata first (when a file path is given) and falls
back to dish rim detection, so `metrics.py` can report real-world units
instead of raw pixel counts either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
from skimage import color

# Millimetres per unit, for units ImageJ's "unit=" ImageDescription field
# commonly records. Units with no fixed physical length (e.g. "pixel",
# meaning the image was never calibrated) are deliberately omitted so they
# fall through to `None` (no calibration) rather than a wrong scale factor.
_MM_PER_UNIT = {
    "micron": 0.001,
    "microns": 0.001,
    "um": 0.001,
    "µm": 0.001,
    "mm": 1.0,
    "millimeter": 1.0,
    "millimetre": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimetre": 10.0,
    "inch": 25.4,
    "in": 25.4,
}


@dataclass
class CalibrationResult:
    """Outcome of attempting to calibrate a plate photograph.

    Attributes
    ----------
    mm_per_pixel : float or None
        Millimetres represented by one pixel. None if calibration failed.
    dish_center : tuple[float, float] or None
        (x, y) pixel coordinates of the detected dish center. None unless
        `source` is `"dish_rim"`.
    dish_radius_px : float or None
        Detected dish rim radius in pixels. None unless `source` is
        `"dish_rim"`.
    calibrated : bool
        Whether a calibration was obtained, by either source. When False,
        downstream metrics should be reported in pixel units and flagged as
        uncalibrated.
    warning : str or None
        Human-readable explanation when `calibrated` is False.
    source : str or None
        Which calibration method succeeded: `"metadata"` (embedded TIFF
        resolution tags) or `"dish_rim"` (Hough circle transform). None if
        `calibrated` is False.
    """

    mm_per_pixel: float | None
    dish_center: tuple[float, float] | None
    dish_radius_px: float | None
    calibrated: bool
    warning: str | None
    source: str | None = None


def mm_per_pixel_from_tiff_metadata(path: str | Path) -> float | None:
    """Read a pixel-to-mm calibration from ImageJ-format TIFF resolution tags.

    Tools like ImageJ/Fiji write the pixel size a microscope or camera
    reports at acquisition time into the `XResolution`/`YResolution` TIFF
    tags (as pixels-per-unit) plus a matching `unit=...` line in
    `ImageDescription`. This is the only reliable calibration source for a
    photograph cropped to a single colony, since `detect_dish_circle` needs
    the whole dish rim in frame.

    Only TIFFs whose `ImageDescription` starts with `"ImageJ="` are trusted:
    an arbitrary TIFF's `ResolutionUnit`/`XResolution` tags are frequently
    left at meaningless defaults (e.g. 72 dpi) by tools that never measured
    a real pixel size, and there is no way to tell that apart from a
    genuine calibration without the ImageJ marker confirming intent.

    Parameters
    ----------
    path : str or Path
        Path to a TIFF file.

    Returns
    -------
    float or None
        Millimetres per pixel, or None if the file isn't TIFF, wasn't
        written by ImageJ, has no recognised unit (including an explicitly
        uncalibrated `unit=pixel`), or is missing the resolution tags.
    """
    try:
        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            description_tag = page.tags.get("ImageDescription")
            description = description_tag.value if description_tag is not None else ""
            if not description.startswith("ImageJ="):
                return None

            unit = None
            for line in description.splitlines():
                if line.startswith("unit="):
                    unit = line[len("unit=") :].strip().lower()
                    break
            mm_per_unit = _MM_PER_UNIT.get(unit)
            if mm_per_unit is None:
                return None

            x_resolution_tag = page.tags.get("XResolution")
            if x_resolution_tag is None:
                return None
            numerator, denominator = x_resolution_tag.value
            if numerator == 0 or denominator == 0:
                return None
            pixels_per_unit = numerator / denominator
            if pixels_per_unit <= 0:
                return None
            return mm_per_unit / pixels_per_unit
    except (OSError, ValueError, KeyError, tifffile.TiffFileError):
        return None


def mm_per_pixel_from_radius(dish_radius_px: float, dish_diameter_mm: float = 90.0) -> float:
    """Convert a dish rim radius in pixels to a mm-per-pixel scale factor.

    Parameters
    ----------
    dish_radius_px : float
        Detected dish rim radius, in pixels.
    dish_diameter_mm : float, default 90.0
        The Petri dish's true physical diameter (standard dish sizes are
        60, 90, or 100 mm; 90 mm is the most common for colony biofilm
        assays).

    Returns
    -------
    float
        Millimetres per pixel.
    """
    return dish_diameter_mm / (2.0 * dish_radius_px)


def detect_dish_circle(
    image: np.ndarray,
    min_radius_frac: float = 0.3,
    max_radius_frac: float = 0.48,
    param1: float = 100.0,
    param2: float = 30.0,
) -> tuple[float, float, float] | None:
    """Locate the Petri dish rim with a Hough circle transform.

    Searches only for circles whose radius is a plausible fraction of the
    image's shorter side, since the dish is assumed to fill most of the
    frame (as in a standard top-down plate photograph) but not touch the
    edges. If several circles are detected, the one closest to the image
    center is returned, since off-center detections are usually spurious
    (reflections, bench edges, lettering on the dish lid).

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), as returned by `segment.load_image`.
    min_radius_frac, max_radius_frac : float
        Bounds on candidate radius as a fraction of min(H, W).
    param1 : float, default 100.0
        Upper Canny edge-detection threshold passed to `cv2.HoughCircles`.
    param2 : float, default 30.0
        Accumulator threshold for circle centers; lower values detect more
        (and weaker/spurious) circles.

    Returns
    -------
    tuple[float, float, float] or None
        (center_x, center_y, radius) in pixels, or None if no circle was
        found within the expected size range.
    """
    gray = color.rgb2gray(image)
    gray_uint8 = (gray * 255).astype(np.uint8)
    blurred = cv2.medianBlur(gray_uint8, 5)

    height, width = gray_uint8.shape
    min_dim = min(height, width)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min_dim * 0.5,
        param1=param1,
        param2=param2,
        minRadius=int(min_dim * min_radius_frac),
        maxRadius=int(min_dim * max_radius_frac),
    )
    if circles is None:
        return None

    candidates = circles[0, :]
    center = np.array([width / 2, height / 2])
    distances = np.hypot(candidates[:, 0] - center[0], candidates[:, 1] - center[1])
    best = candidates[np.argmin(distances)]
    return float(best[0]), float(best[1]), float(best[2])


def calibrate(
    image: np.ndarray,
    dish_diameter_mm: float = 90.0,
    path: str | Path | None = None,
    **hough_kwargs,
) -> CalibrationResult:
    """Compute a pixel-to-mm calibration, preferring embedded metadata over dish rim detection.

    Tries `mm_per_pixel_from_tiff_metadata` first when `path` is given,
    since it works even when the dish rim isn't in frame (e.g. a photo
    cropped to one colony among several on the same plate). Falls back to
    `detect_dish_circle` -- which needs the whole dish visible -- when no
    path is given or the file carries no usable metadata calibration.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), as returned by `segment.load_image`.
    dish_diameter_mm : float, default 90.0
        The Petri dish's true physical diameter, used only by the dish rim
        fallback. Override for non-standard dishes (e.g. 60 mm or 100 mm).
    path : str, Path, or None, default None
        Path to the source image file, so its TIFF metadata can be checked
        for an existing calibration before falling back to dish rim
        detection. Pass this whenever it's available.
    **hough_kwargs
        Forwarded to `detect_dish_circle` (e.g. `param2` to loosen/tighten
        detection sensitivity).

    Returns
    -------
    CalibrationResult
        Falls back gracefully with `calibrated=False` and a clear warning
        if neither calibration source succeeds -- metrics should still be
        computed, just reported in pixel units.
    """
    if path is not None:
        mm_per_pixel = mm_per_pixel_from_tiff_metadata(path)
        if mm_per_pixel is not None:
            return CalibrationResult(
                mm_per_pixel=mm_per_pixel,
                dish_center=None,
                dish_radius_px=None,
                calibrated=True,
                warning=None,
                source="metadata",
            )

    circle = detect_dish_circle(image, **hough_kwargs)
    if circle is None:
        return CalibrationResult(
            mm_per_pixel=None,
            dish_center=None,
            dish_radius_px=None,
            calibrated=False,
            warning=(
                "No embedded calibration metadata and dish rim not detected "
                "by Hough circle transform; falling back to pixel units. "
                "Check lighting/framing or pass a manual "
                "dish_diameter_mm-consistent crop."
            ),
            source=None,
        )

    cx, cy, radius_px = circle
    mm_per_pixel = mm_per_pixel_from_radius(radius_px, dish_diameter_mm)
    return CalibrationResult(
        mm_per_pixel=mm_per_pixel,
        dish_center=(cx, cy),
        dish_radius_px=radius_px,
        calibrated=True,
        warning=None,
        source="dish_rim",
    )
