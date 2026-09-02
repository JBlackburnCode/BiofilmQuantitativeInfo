"""Tests for src/segment.py using a synthetic plate photograph."""

import numpy as np
import pytest
from skimage import draw, measure

from src.segment import segment_colony


@pytest.fixture
def synthetic_plate() -> np.ndarray:
    """A synthetic plate photograph: dark bench, bright agar disk, dark colony blob.

    Deliberately includes noise speckle and a second small dark blob outside
    the dish rim, so the test also exercises small-object removal and the
    largest-connected-component selection, not just a trivial single-blob case.
    """
    shape = (400, 400)
    image = np.full((*shape, 3), 40, dtype=np.uint8)  # dark bench

    rr, cc = draw.disk((200, 200), 170, shape=shape)
    image[rr, cc] = 190  # agar

    rr, cc = draw.disk((200, 200), 70, shape=shape)
    image[rr, cc] = 100  # colony, darker than agar

    # Small unconnected debris just outside the dish -- should be discarded.
    rr, cc = draw.disk((30, 30), 5, shape=shape)
    image[rr, cc] = 90

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 3, size=shape)
    image = np.clip(image.astype(float) + noise[..., None], 0, 255).astype(np.uint8)

    return image


def test_segmentation_returns_single_connected_component(synthetic_plate):
    result = segment_colony(synthetic_plate, illumination_kernel_radius=150)
    labels = measure.label(result.colony_mask)
    assert labels.max() == 1


def test_segmentation_finds_colony_of_expected_size(synthetic_plate):
    result = segment_colony(synthetic_plate, illumination_kernel_radius=150)
    expected_area = np.pi * 70**2
    assert result.colony_mask.sum() == pytest.approx(expected_area, rel=0.1)


def test_segmentation_excludes_debris_outside_dish(synthetic_plate):
    result = segment_colony(synthetic_plate, illumination_kernel_radius=150)
    assert not result.colony_mask[25:35, 25:35].any()


def test_segmentation_handles_blank_image_without_crashing():
    blank = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = segment_colony(blank)
    assert result.colony_mask.sum() == 0
