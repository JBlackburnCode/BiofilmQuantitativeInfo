"""Tests for src/calibrate.py using a synthetic plate photograph."""

import numpy as np
import pytest
from skimage import draw, filters

from src.calibrate import calibrate, detect_dish_circle, mm_per_pixel_from_radius


@pytest.fixture
def synthetic_dish_image() -> np.ndarray:
    """A bright disk (agar) of known radius on a dark background (bench).

    A little Gaussian blur is applied so the rim has the soft edge a real
    photograph would have -- a razor-sharp CG edge aliases badly under
    Canny/Hough and isn't representative of what this is actually detecting
    in practice. No colony is drawn -- these tests only exercise dish rim
    detection and the pixel-to-mm maths, not segmentation.
    """
    shape = (500, 500)
    image = np.full((*shape, 3), 30, dtype=np.uint8)
    rr, cc = draw.disk((250, 250), 200, shape=shape)
    image[rr, cc] = 200
    blurred = filters.gaussian(image.astype(float), sigma=2, channel_axis=2)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def test_mm_per_pixel_from_radius_known_values():
    # A 100px radius dish that is truly 90mm across: each pixel is 90/200 mm.
    assert mm_per_pixel_from_radius(100.0, dish_diameter_mm=90.0) == pytest.approx(0.45)


def test_mm_per_pixel_from_radius_scales_with_diameter():
    # Doubling the stated dish diameter should double the mm-per-pixel factor.
    small = mm_per_pixel_from_radius(100.0, dish_diameter_mm=60.0)
    large = mm_per_pixel_from_radius(100.0, dish_diameter_mm=120.0)
    assert large == pytest.approx(2 * small)


def test_detect_dish_circle_finds_known_radius(synthetic_dish_image):
    circle = detect_dish_circle(synthetic_dish_image)
    assert circle is not None
    _cx, _cy, radius = circle
    assert radius == pytest.approx(200.0, rel=0.05)


def test_calibrate_converts_pixels_to_mm_correctly(synthetic_dish_image):
    result = calibrate(synthetic_dish_image, dish_diameter_mm=90.0)
    assert result.calibrated
    assert result.warning is None
    # True mm-per-pixel for a 200px-radius dish that is 90mm across.
    expected = 90.0 / (2 * 200.0)
    assert result.mm_per_pixel == pytest.approx(expected, rel=0.05)


def test_calibrate_falls_back_gracefully_when_no_dish_present():
    blank = np.full((200, 200, 3), 128, dtype=np.uint8)
    result = calibrate(blank)
    assert not result.calibrated
    assert result.mm_per_pixel is None
    assert result.warning is not None
