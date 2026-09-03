"""Tests for src/calibrate.py using a synthetic plate photograph."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from skimage import draw, filters

from src.calibrate import (
    calibrate,
    detect_dish_circle,
    mm_per_pixel_from_radius,
    mm_per_pixel_from_tiff_metadata,
)


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
    assert result.source is None


def test_calibrate_records_dish_rim_as_the_source(synthetic_dish_image):
    result = calibrate(synthetic_dish_image, dish_diameter_mm=90.0)
    assert result.source == "dish_rim"


def _write_imagej_tiff(path: Path, x_resolution: tuple[int, int], unit: str) -> None:
    """Write a minimal ImageJ-format TIFF carrying a pixel-size calibration."""
    array = np.zeros((10, 10), dtype=np.uint16)
    tifffile.imwrite(
        path,
        array,
        imagej=True,
        resolution=(x_resolution, x_resolution),
        metadata={"unit": unit},
    )


def test_mm_per_pixel_from_tiff_metadata_reads_micron_calibration(tmp_path):
    # 2 pixels per micron -> each pixel is 0.5 micron -> 0.0005 mm.
    path = tmp_path / "calibrated.tif"
    _write_imagej_tiff(path, (2, 1), "um")
    assert mm_per_pixel_from_tiff_metadata(path) == pytest.approx(0.0005)


def test_mm_per_pixel_from_tiff_metadata_converts_other_units(tmp_path):
    # 10 pixels per mm -> each pixel is 0.1 mm.
    path = tmp_path / "calibrated_mm.tif"
    _write_imagej_tiff(path, (10, 1), "mm")
    assert mm_per_pixel_from_tiff_metadata(path) == pytest.approx(0.1)


def test_mm_per_pixel_from_tiff_metadata_returns_none_for_uncalibrated_unit(tmp_path):
    # ImageJ writes unit="pixel" when a file was never calibrated to a real
    # physical unit -- that's not a usable scale factor.
    path = tmp_path / "uncalibrated.tif"
    _write_imagej_tiff(path, (1, 1), "pixel")
    assert mm_per_pixel_from_tiff_metadata(path) is None


def test_mm_per_pixel_from_tiff_metadata_returns_none_for_non_imagej_tiff(tmp_path):
    # A plain TIFF's XResolution tag (e.g. a default 72 dpi some tool wrote)
    # can't be distinguished from a genuine calibration without ImageJ's
    # marker confirming a real pixel size was recorded.
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((10, 10), dtype=np.uint8), resolution=(72, 72))
    assert mm_per_pixel_from_tiff_metadata(path) is None


def test_mm_per_pixel_from_tiff_metadata_returns_none_for_missing_file(tmp_path):
    assert mm_per_pixel_from_tiff_metadata(tmp_path / "does_not_exist.tif") is None


def test_calibrate_prefers_metadata_over_dish_rim_when_path_given(
    tmp_path, synthetic_dish_image
):
    path = tmp_path / "calibrated.tif"
    _write_imagej_tiff(path, (2, 1), "um")

    result = calibrate(synthetic_dish_image, dish_diameter_mm=90.0, path=path)

    assert result.calibrated
    assert result.source == "metadata"
    assert result.mm_per_pixel == pytest.approx(0.0005)
    # Metadata calibration carries no dish-rim geometry.
    assert result.dish_center is None
    assert result.dish_radius_px is None


def test_calibrate_falls_back_to_dish_rim_when_metadata_missing(tmp_path, synthetic_dish_image):
    path = tmp_path / "uncalibrated.tif"
    _write_imagej_tiff(path, (1, 1), "pixel")

    result = calibrate(synthetic_dish_image, dish_diameter_mm=90.0, path=path)

    assert result.calibrated
    assert result.source == "dish_rim"
