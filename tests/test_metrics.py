"""Tests for src/metrics.py using synthetic masks with known geometry."""

import numpy as np
import pytest
from skimage import draw

from src.metrics import (
    circularity,
    equivalent_diameter,
    solidity,
    texture_contrast,
    texture_entropy,
)


@pytest.fixture
def circle_mask() -> np.ndarray:
    """A filled disk, radius 80px, on a 300x300 canvas -- the convex reference shape."""
    shape = (300, 300)
    mask = np.zeros(shape, dtype=bool)
    rr, cc = draw.disk((150, 150), 80, shape=shape)
    mask[rr, cc] = True
    return mask


@pytest.fixture
def cross_mask() -> np.ndarray:
    """A plus-sign shape -- deeply concave, four reentrant corners."""
    shape = (300, 300)
    mask = np.zeros(shape, dtype=bool)
    mask[120:180, 60:240] = True
    mask[60:240, 120:180] = True
    return mask


def test_circularity_near_one_for_circle(circle_mask):
    assert circularity(circle_mask) == pytest.approx(1.0, abs=0.02)


def test_circularity_lower_for_wrinkled_margin(cross_mask):
    # A plus-sign encloses much less area per unit perimeter than a circle,
    # mimicking a wrinkled/furrowed colony margin.
    assert circularity(cross_mask) < 0.7


def test_solidity_near_one_for_convex_shape(circle_mask):
    assert solidity(circle_mask) == pytest.approx(1.0, abs=0.02)


def test_solidity_clearly_separates_convex_from_concave(circle_mask, cross_mask):
    assert solidity(cross_mask) < 0.8
    assert solidity(circle_mask) - solidity(cross_mask) > 0.15


def test_equivalent_diameter_matches_known_radius(circle_mask):
    # Area of a radius-80 disk is pi*80^2; equivalent diameter should recover ~160px.
    assert equivalent_diameter(circle_mask) == pytest.approx(160.0, rel=0.02)


def test_equivalent_diameter_respects_calibration(circle_mask):
    mm_per_pixel = 0.05
    px_diameter = equivalent_diameter(circle_mask)
    mm_diameter = equivalent_diameter(circle_mask, mm_per_pixel=mm_per_pixel)
    assert mm_diameter == pytest.approx(px_diameter * mm_per_pixel)


def test_empty_mask_metrics_are_zero():
    empty = np.zeros((50, 50), dtype=bool)
    assert circularity(empty) == 0.0
    assert solidity(empty) == 0.0


def test_texture_contrast_higher_for_noisy_colony(circle_mask):
    gray_smooth = np.full(circle_mask.shape, 0.5)

    rng = np.random.default_rng(0)
    gray_noisy = np.full(circle_mask.shape, 0.5)
    gray_noisy[circle_mask] = np.clip(
        0.5 + 0.3 * rng.standard_normal(circle_mask.sum()), 0, 1
    )

    smooth_contrast = texture_contrast(gray_smooth, circle_mask)
    noisy_contrast = texture_contrast(gray_noisy, circle_mask)

    assert smooth_contrast == pytest.approx(0.0, abs=1e-9)
    assert noisy_contrast > smooth_contrast


def test_texture_entropy_is_nonnegative_and_higher_for_disordered_colony(circle_mask):
    gray_smooth = np.full(circle_mask.shape, 0.5)

    rng = np.random.default_rng(0)
    gray_noisy = np.full(circle_mask.shape, 0.5)
    gray_noisy[circle_mask] = np.clip(
        0.5 + 0.3 * rng.standard_normal(circle_mask.sum()), 0, 1
    )

    smooth_entropy = texture_entropy(gray_smooth, circle_mask)
    noisy_entropy = texture_entropy(gray_noisy, circle_mask)

    assert smooth_entropy >= 0.0
    assert noisy_entropy >= 0.0
    assert noisy_entropy > smooth_entropy
