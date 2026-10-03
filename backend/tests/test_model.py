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


HEAT_CAL = {"image_min": 10.0, "image_max": 20.0, "pixel_typical": 10.0, "pixel_max": 20.0}


def test_heat_is_transparent_for_normal_texture():
    # at/below the hottest pixel seen on good photos nothing may be drawn
    assert ml.heat_intensity(np.full((8, 8), 10.0), HEAT_CAL).max() == 0.0
    assert ml.heat_intensity(np.full((8, 8), 10.0 + 10.0 * 0.5), HEAT_CAL).max() == 0.0
    assert ml.heat_intensity(np.full((8, 8), 20.0), HEAT_CAL).max() < 0.6


def test_heat_grows_with_how_unusual_a_region_is():
    mild = ml.heat_intensity(np.full((4, 4), 10.0 + 10.0 * 0.9), HEAT_CAL).mean()
    strong = ml.heat_intensity(np.full((4, 4), 10.0 + 10.0 * 1.4), HEAT_CAL).mean()
    assert 0.0 < mild < strong <= 1.0


def test_outline_is_drawn_only_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "RESULTS_DIR", tmp_path)
    img = Image.new("RGB", (600, 450), (200, 170, 100))
    amap = np.full((45, 60), 10.0)
    amap[15:30, 20:40] = 10.0 + 10.0 * 1.4          # one clearly hot region
    with_outline = np.asarray(Image.open(ml._save_heatmap(tmp_path / "a.png", img, amap, HEAT_CAL, outline=True))).astype(int)
    no_outline   = np.asarray(Image.open(ml._save_heatmap(tmp_path / "b.png", img, amap, HEAT_CAL, outline=False))).astype(int)
    base = np.asarray(img).astype(int)

    assert with_outline.shape == no_outline.shape == (450, 600, 3)
    assert (np.abs(with_outline - no_outline).max(axis=2) > 30).sum() > 100     # the outline itself
    assert (np.abs(no_outline - base).max(axis=2) > 30)[200:290, 200:400].any()  # the heat fill, on the hot region
    assert np.abs(no_outline - base)[:100].max() <= 6                            # ...and nowhere else


def test_clean_photo_is_left_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "RESULTS_DIR", tmp_path)
    img = Image.new("RGB", (600, 450), (200, 170, 100))
    out = np.asarray(Image.open(ml._save_heatmap(tmp_path / "c.png", img, np.full((45, 60), 10.0), HEAT_CAL, outline=True))).astype(int)
    assert np.abs(out - np.asarray(img).astype(int)).max() <= 6                  # JPEG noise only


def test_input_size_keeps_the_photos_aspect_ratio(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "IMAGE_LONG_SIDE", 384)

    def photos(size):
        paths = []
        for i in range(3):
            p = tmp_path / f"{size[0]}x{size[1]}_{i}.png"
            Image.new("RGB", size).save(p)
            paths.append(p)
        return paths

    monkeypatch.setattr(ml, "FIT_ASPECT", True)
    assert ml.choose_input_size(photos((1920, 1440))) == (384, 288)    # landscape 4:3
    assert ml.choose_input_size(photos((1000, 1500))) == (256, 384)    # portrait 2:3
    w, h = ml.choose_input_size(photos((640, 480)))
    assert w % 16 == 0 and h % 16 == 0                                  # matches the backbone strides

    monkeypatch.setattr(ml, "FIT_ASPECT", False)
    assert ml.choose_input_size(photos((1920, 1440))) == (384, 384)    # opt-out: square


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
