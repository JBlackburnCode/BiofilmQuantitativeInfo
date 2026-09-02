"""Tests for src/batch.py, covering the folder-recursion and never-crash contract."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage import draw, io

from src.batch import CSV_COLUMNS, find_images, run_batch


def _make_plate_image() -> np.ndarray:
    """A minimal synthetic plate photograph: dish + colony, no real data needed."""
    shape = (300, 300)
    image = np.full((*shape, 3), 40, dtype=np.uint8)
    rr, cc = draw.disk((150, 150), 130, shape=shape)
    image[rr, cc] = 190
    rr, cc = draw.disk((150, 150), 50, shape=shape)
    image[rr, cc] = 100
    return image


@pytest.fixture
def plates_dir(tmp_path: Path) -> Path:
    """A folder with two valid synthetic plate photos and one corrupt file."""
    plates = tmp_path / "plates"
    plates.mkdir()

    io.imsave(plates / "plate01.png", _make_plate_image())
    io.imsave(plates / "plate02.png", _make_plate_image())
    (plates / "plate03_corrupt.png").write_bytes(b"not a real image")
    (plates / "notes.txt").write_text("this should be ignored, wrong extension")

    return plates


def test_find_images_only_matches_known_extensions(plates_dir):
    found = find_images(plates_dir)
    names = {p.name for p in found}
    assert names == {"plate01.png", "plate02.png", "plate03_corrupt.png"}


def test_batch_continues_past_corrupt_file(plates_dir, tmp_path):
    output_dir = tmp_path / "results"
    df = run_batch(plates_dir, output_dir, dish_diameter_mm=90.0)

    assert len(df) == 3
    assert set(df["filename"]) == {"plate01.png", "plate02.png", "plate03_corrupt.png"}

    corrupt_row = df[df["filename"] == "plate03_corrupt.png"].iloc[0]
    assert corrupt_row["calibrated"] == False  # noqa: E712
    assert pd.isna(corrupt_row["area"])
    assert "could not read" in corrupt_row["warning"].lower()


def test_batch_computes_metrics_for_valid_images(plates_dir, tmp_path):
    output_dir = tmp_path / "results"
    df = run_batch(plates_dir, output_dir, dish_diameter_mm=90.0)

    good_rows = df[df["filename"].isin(["plate01.png", "plate02.png"])]
    assert (good_rows["area"] > 0).all()
    assert (good_rows["circularity"] > 0).all()


def test_batch_writes_csv_with_expected_columns(plates_dir, tmp_path):
    output_dir = tmp_path / "results"
    run_batch(plates_dir, output_dir, dish_diameter_mm=90.0)

    csv_path = output_dir / "measurements.csv"
    assert csv_path.exists()
    on_disk = pd.read_csv(csv_path)
    assert list(on_disk.columns) == CSV_COLUMNS


def test_batch_saves_overlays_only_for_processed_images(plates_dir, tmp_path):
    output_dir = tmp_path / "results"
    run_batch(plates_dir, output_dir, dish_diameter_mm=90.0)

    overlay_files = {p.name for p in (output_dir / "overlays").iterdir()}
    assert overlay_files == {"plate01_overlay.png", "plate02_overlay.png"}
