"""Pixel-to-millimetre calibration via Petri dish rim detection.

The dish rim is located with OpenCV's Hough circle transform. Given the
known physical dish diameter, the detected rim radius in pixels yields a
mm-per-pixel scale factor that `metrics.py` can use to report real-world
units instead of raw pixel counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage import color


@dataclass
class CalibrationResult:
    """Outcome of attempting to calibrate a plate photograph.

    Attributes
    ----------
    mm_per_pixel : float or None
        Millimetres represented by one pixel. None if calibration failed.
    dish_center : tuple[float, float] or None
        (x, y) pixel coordinates of the detected dish center. None if
        calibration failed.
    dish_radius_px : float or None
        Detected dish rim radius in pixels. None if calibration failed.
    calibrated : bool
        Whether dish detection succeeded. When False, downstream metrics
        should be reported in pixel units and flagged as uncalibrated.
    warning : str or None
        Human-readable explanation when `calibrated` is False.
    """

    mm_per_pixel: float | None
    dish_center: tuple[float, float] | None
    dish_radius_px: float | None
    calibrated: bool
    warning: str | None


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


def calibrate(image: np.ndarray, dish_diameter_mm: float = 90.0, **hough_kwargs) -> CalibrationResult:
    """Detect the dish rim and compute a pixel-to-mm calibration.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), as returned by `segment.load_image`.
    dish_diameter_mm : float, default 90.0
        The Petri dish's true physical diameter. Override for non-standard
        dishes (e.g. 60 mm or 100 mm).
    **hough_kwargs
        Forwarded to `detect_dish_circle` (e.g. `param2` to loosen/tighten
        detection sensitivity).

    Returns
    -------
    CalibrationResult
        Falls back gracefully with `calibrated=False` and a clear warning
        if the dish rim cannot be found -- metrics should still be
        computed, just reported in pixel units.
    """
    circle = detect_dish_circle(image, **hough_kwargs)
    if circle is None:
        return CalibrationResult(
            mm_per_pixel=None,
            dish_center=None,
            dish_radius_px=None,
            calibrated=False,
            warning=(
                "Dish rim not detected by Hough circle transform; "
                "falling back to pixel units. Check lighting/framing or "
                "pass a manual dish_diameter_mm-consistent crop."
            ),
        )

    cx, cy, radius_px = circle
    mm_per_pixel = mm_per_pixel_from_radius(radius_px, dish_diameter_mm)
    return CalibrationResult(
        mm_per_pixel=mm_per_pixel,
        dish_center=(cx, cy),
        dish_radius_px=radius_px,
        calibrated=True,
        warning=None,
    )
