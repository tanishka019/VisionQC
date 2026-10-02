"""Tests for the ML module. The end-to-end test downloads backbone weights and
takes ~30s on CPU, so it only runs when VISIONQC_RUN_SLOW=1."""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ml import model as ml

CALIB = {"image_min": 10.0, "image_max": 20.0}


def test_normalise_score_maps_calibration_range():
    assert ml.normalise_score(10.0, CALIB) == 0.0
    assert ml.normalise_score(20.0, CALIB) == 0.5
    assert ml.normalise_score(30.0, CALIB) == 1.0
    assert ml.normalise_score(99.0, CALIB) == 1.0


def test_normalise_score_single_calibration_image():
    score = ml.normalise_score(22.0, {"image_min": 20.0, "image_max": 20.0})
    assert 0.5 < score <= 1.0


def test_make_folds_cover_all_images_once_and_are_deterministic():
    paths = [Path(f"{i}.png") for i in range(12)]
    folds = ml.make_folds(paths, k=5)
    assert len(folds) == 5 and all(folds)
    assert sorted(p for f in folds for p in f) == sorted(paths)
    assert ml.make_folds(list(reversed(paths)), k=5) == folds


def test_make_folds_small_sets():
    folds = ml.make_folds([Path(f"{i}.png") for i in range(5)], k=5)
    assert [len(f) for f in folds] == [1] * 5


def test_train_rejects_too_few_images(tmp_path):
    for i in range(3):
        Image.new("RGB", (32, 32)).save(tmp_path / f"{i}.png")
    with pytest.raises(ValueError):
        ml.train(tmp_path)


def _part(seed: int, defect: bool = False) -> Image.Image:
    """A synthetic 'part': textured grey disc; the defect is a dark scratch."""
    rng = np.random.default_rng(seed)
    base = (rng.normal(160, 6, (256, 256, 3))).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    d.ellipse([48, 48, 208, 208], fill=(110, 110, 120))
    if defect:
        d.line([(90, 100), (170, 150)], fill=(10, 10, 10), width=8)
    return img


@pytest.mark.skipif(os.getenv("VISIONQC_RUN_SLOW") != "1", reason="set VISIONQC_RUN_SLOW=1")
def test_train_and_inspect_end_to_end(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    for i in range(12):
        _part(i).save(train_dir / f"good_{i}.png")
    result = ml.train(train_dir, "disc")
    assert result["image_count"] == 12
    assert ml.CHECKPOINT.exists()

    ml._model = None  # force reload from checkpoint
    goods = []
    for seed in range(100, 106):
        path = tmp_path / f"good_{seed}.png"
        _part(seed).save(path)
        goods.append(ml.inspect(path, 0.5))
    bad_path = tmp_path / "bad.png"
    _part(101, defect=True).save(bad_path)
    bad = ml.inspect(bad_path, 0.5)

    # Cross-validated calibration keeps false rejects rare, and a clear defect must stand out
    assert sum(g["result"] == "PASS" for g in goods) >= 5, [g["score"] for g in goods]
    assert bad["result"] == "FAIL"
    assert bad["score"] > max(g["score"] for g in goods)
    assert Path(bad["heatmap_path"]).exists()
