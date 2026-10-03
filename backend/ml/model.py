"""
model.py — PatchCore anomaly detection via Anomalib (v3)

Training builds PatchCore's memory bank directly from the uploaded "good"
images. The score scale is calibrated with k-fold cross-validation: every
image is scored against a bank built without it, giving one unbiased
"unseen good part" score per image:

    score = 0.5 + 0.5 * (raw - calib_max) / (calib_max - calib_min)   (clamped to [0, 1])

so the most typical good image scores ~0, the most unusual one
scores 0.5, and the default threshold of 0.5 is a sensible start.
"""
import os
import json
import random
import shutil
import logging
import threading
import numpy as np
from pathlib import Path
from PIL import Image

logger = logging.getLogger("visionqc.model")

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = Path(os.getenv("VISIONQC_DATA_DIR", BASE_DIR))
RESULTS_DIR = DATA_DIR / "results"
MODEL_DIR   = DATA_DIR / "model_artifacts"
CHECKPOINT  = MODEL_DIR / "patchcore.pt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_LONG_SIDE = int(os.getenv("VISIONQC_IMAGE_SIZE", "256"))   # long side of the model input, in pixels
FIT_ASPECT      = os.getenv("VISIONQC_FIT_ASPECT", "0") == "1"   # keep the photos' aspect ratio instead of squashing to a square
IMAGE_SIZE     = (IMAGE_LONG_SIDE, IMAGE_LONG_SIDE)              # (w, h); recomputed from the training photos when FIT_ASPECT
BACKBONE       = os.getenv("VISIONQC_BACKBONE", "resnet18")
LAYERS         = tuple(os.getenv("VISIONQC_LAYERS", "layer2,layer3").split(","))
CORESET_RATIO  = float(os.getenv("VISIONQC_CORESET_RATIO", "0.02"))
BATCH_SIZE     = int(os.getenv("VISIONQC_BATCH_SIZE", "8"))
CALIB_FOLDS    = 5
# Coreset sampling is O(candidates x bank size); pre-sampling candidate patches (adjacent
# patches are highly redundant) makes it much faster. 0 = use every patch.
MAX_CANDIDATES = int(os.getenv("VISIONQC_MAX_CANDIDATES", "3000"))
BANK_PATCHES   = float(os.getenv("VISIONQC_BANK_PATCHES", "250"))   # target number of patches kept in the memory bank
SEED           = 0
MIN_IMAGES     = 5
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)

_model       = None   # anomalib PatchcoreModel (torch module) with memory bank
_calibration = None   # {"image_min", "image_max", "pixel_typical", "pixel_max": float}
_lock        = threading.Lock()
_img_size    = IMAGE_SIZE   # (w, h) used by _to_tensor; set from the training photos / checkpoint


def _build_model(backbone: str = None, layers: tuple = None):
    from anomalib.models import Patchcore
    bb = backbone or BACKBONE
    ly = layers or LAYERS
    module = Patchcore(
        backbone=bb,
        layers=ly,
        coreset_sampling_ratio=CORESET_RATIO,
    )
    return module.model


def choose_input_size(paths: list[Path]) -> tuple[int, int]:
    """(w, h) of the model input.

    A square by default. With FIT_ASPECT the training photos' median aspect ratio is kept, so
    parts aren't squashed (a squashed part looks different from a real one and blurs the map);
    both sides are snapped to a multiple of 16 to match the backbone's strides.
    """
    if not FIT_ASPECT:
        return (IMAGE_LONG_SIDE, IMAGE_LONG_SIDE)
    ratios = []
    for p in paths[:50]:
        with Image.open(p) as im:
            ratios.append(im.width / im.height)
    r = float(np.median(ratios))
    w, h = (IMAGE_LONG_SIDE, IMAGE_LONG_SIDE / r) if r >= 1 else (IMAGE_LONG_SIDE * r, IMAGE_LONG_SIDE)
    snap = lambda v: max(32, int(round(v / 16)) * 16)
    return (snap(w), snap(h))


def _to_tensor(img: Image.Image, size: tuple[int, int] | None = None):
    """PIL image → normalised [3, H, W] tensor, matching the backbone's ImageNet stats."""
    import torchvision.transforms.functional as TF
    tensor = TF.to_tensor(img.convert("RGB").resize(size or _img_size, Image.BILINEAR))
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def _batches(paths: list[Path], size: tuple[int, int] | None = None):
    import torch
    for i in range(0, len(paths), BATCH_SIZE):
        chunk = paths[i:i + BATCH_SIZE]
        yield torch.stack([_to_tensor(Image.open(p), size) for p in chunk])


def _score_raw(model, batch):
    """Run inference; returns (pred_scores [B], anomaly_maps [B, H, W]) as numpy."""
    import torch
    model.eval()
    with torch.no_grad():
        out = model(batch)
    return (
        out["pred_score"].reshape(-1).cpu().numpy(),
        out["anomaly_map"].reshape(batch.shape[0], *batch.shape[-2:]).cpu().numpy(),
    )


def is_trained() -> bool:
    return CHECKPOINT.exists()


def reset_model():
    """Delete all model artifacts."""
    global _model, _calibration
    with _lock:
        _model = None
        _calibration = None
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)


def make_folds(paths: list[Path], k: int = CALIB_FOLDS, seed: int = SEED) -> list[list[Path]]:
    """Split images into `k` disjoint, deterministic folds (each at least 1 image)."""
    paths = sorted(paths)
    random.Random(seed).shuffle(paths)
    k = max(2, min(k, len(paths)))
    return [paths[i::k] for i in range(k)]


def _fit_bank(model, embeddings: list) -> None:
    """Build the PatchCore memory bank (coreset) from per-image patch embeddings."""
    import torch
    stacked = torch.vstack(embeddings)
    total = stacked.shape[0]
    if MAX_CANDIDATES and total > MAX_CANDIDATES:
        generator = torch.Generator().manual_seed(SEED)
        stacked = stacked[torch.randperm(total, generator=generator)[:MAX_CANDIDATES]]
        ratio = min(0.5, max(0.04, BANK_PATCHES / stacked.shape[0]))
    else:
        ratio = max(CORESET_RATIO, 0.05)
    model.subsample_embedding(stacked, ratio)


def train(training_images_dir: Path, product_name: str = "product", progress=None) -> dict:
    """
    Train PatchCore on a folder of 'good' images.

    The score scale is calibrated with k-fold cross-validation: every image is scored
    against a memory bank built WITHOUT it, giving one unbiased "unseen good part" score
    per image. (Calibrating on images the bank was built from, or on only a few images,
    makes the scale far too tight and good parts get rejected.)

    `progress(percent, message)` is called as training advances (optional).
    """
    import torch
    global _model, _calibration, _img_size

    def report(pct: int, msg: str):
        logger.info(msg)
        if progress:
            progress(pct, msg)

    image_files = sorted(f for f in Path(training_images_dir).glob("*") if f.suffix.lower() in IMAGE_EXTS)
    if len(image_files) < MIN_IMAGES:
        raise ValueError(f"Need at least {MIN_IMAGES} images, got {len(image_files)}")

    torch.manual_seed(SEED)  # keep runs reproducible
    try:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    except Exception:
        pass
    folds = make_folds(image_files)
    size = choose_input_size(image_files)
    report(15, f"Loading {BACKBONE} backbone…")
    model = _build_model()

    # 1. Extract patch embeddings once per image (the backbone is frozen)
    model.train()  # PatchcoreModel returns embeddings in train mode
    emb: dict = {}
    with torch.no_grad():
        for i in range(0, len(image_files), BATCH_SIZE):
            chunk = image_files[i:i + BATCH_SIZE]
            batch = torch.stack([_to_tensor(Image.open(p), size) for p in chunk])
            for path, e in zip(chunk, torch.chunk(model(batch), len(chunk))):
                emb[path] = e
            done = min(i + BATCH_SIZE, len(image_files))
            report(15 + int(35 * done / len(image_files)), f"Extracting features ({done}/{len(image_files)})…")

    # 2. Calibrate with cross-validation
    scores, map_max, map_median = [], [], []
    for f, held_out in enumerate(folds):
        report(50 + int(35 * f / len(folds)), f"Calibrating (fold {f + 1}/{len(folds)})…")
        held = set(held_out)
        _fit_bank(model, [e for p, e in emb.items() if p not in held])
        for batch in _batches(held_out, size):
            s, m = _score_raw(model, batch)
            scores.extend(s.tolist())
            map_max.extend(m.max(axis=(1, 2)).tolist())
            map_median.extend(np.median(m, axis=(1, 2)).tolist())
    calibration = {
        "image_min": float(min(scores)),
        "image_max": float(max(scores)),
        "pixel_typical": float(np.median(map_median)),
        "pixel_max": float(max(map_max)),
    }

    # 3. Final memory bank from ALL images
    report(88, "Building final memory bank…")
    _fit_bank(model, list(emb.values()))
    model.eval()

    report(95, "Saving optimized model checkpoint…")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": model.memory_bank.cpu(),
            "calibration": calibration,
            "config": {
                "backbone": BACKBONE, "layers": list(LAYERS),
                "image_size": list(size), "product_name": product_name,
                "image_count": len(image_files),
            },
        },
        CHECKPOINT,
    )

    with _lock:
        _model, _calibration, _img_size = model, calibration, size

    logger.info(f"Training complete: {json.dumps(calibration)}")
    return {"status": "trained", "image_count": len(image_files), "calibration": calibration}


def _load_model_if_needed():
    global _model, _calibration, _img_size
    if _model is not None:
        return
    import torch
    if not CHECKPOINT.exists():
        raise RuntimeError("No trained model found. Please train first.")
    logger.info(f"Loading checkpoint: {CHECKPOINT}")
    ckpt  = torch.load(CHECKPOINT, map_location="cpu")
    cfg = ckpt.get("config", {})
    saved_bb = cfg.get("backbone", BACKBONE)
    saved_layers = tuple(cfg.get("layers", LAYERS)) if "layers" in cfg else LAYERS
    model = _build_model(backbone=saved_bb, layers=saved_layers)
    model.memory_bank = ckpt["memory_bank"]
    model.eval()
    _model, _calibration = model, ckpt["calibration"]
    _img_size = tuple(cfg.get("image_size") or IMAGE_SIZE)   # old checkpoints stay at the size they were trained at


def normalise_score(raw: float, calibration: dict) -> float:
    hi, lo = calibration["image_max"], calibration["image_min"]
    spread = max(hi - lo, 0.1 * abs(hi), 1e-6)  # guard: a single calibration image
    return float(min(max(0.5 + 0.5 * (raw - hi) / spread, 0.0), 1.0))


def inspect(image_path: Path, threshold: float) -> dict:
    """
    Run anomaly detection on a single image.
    Returns score, confidence, result, heatmap_path.
    """
    with _lock:
        _load_model_if_needed()
        img = Image.open(image_path).convert("RGB")
        raw_scores, maps = _score_raw(_model, _to_tensor(img).unsqueeze(0))
        calibration = _calibration

    raw_score = float(raw_scores[0])
    score     = normalise_score(raw_score, calibration)
    result    = "FAIL" if score >= threshold else "PASS"
    confidence = round((score if result == "FAIL" else 1 - score) * 100, 1)

    heatmap_path = _save_heatmap(image_path, img, maps[0], calibration)

    return {
        "score":        round(score, 4),
        "raw_score":    round(raw_score, 4),
        "confidence":   confidence,
        "result":       result,
        "heatmap_path": str(heatmap_path),
    }


def normalised_map(amap: np.ndarray, calibration: dict) -> np.ndarray:
    """Anomaly map scaled so a typical good pixel is 0, the hottest good pixel 0.5, twice as far 1."""
    typical, peak = calibration["pixel_typical"], calibration["pixel_max"]
    return np.clip(0.5 * (amap - typical) / max(peak - typical, 1e-6), 0.0, 1.0)


def _save_heatmap(orig_path: Path, orig_img: Image.Image, amap: np.ndarray, calibration: dict) -> Path:
    """Overlay an inferno heatmap on the original image and save.

    The map is scaled against calibrated values from good images (not
    per-image min/max) and used as the overlay's opacity, so normal regions show the
    original image and only true anomalies glow.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps

    w, h = orig_img.size
    norm_map = normalised_map(amap, calibration)

    heatmap_rgb = (colormaps["inferno"](norm_map)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb).resize((w, h))
    alpha_img   = Image.fromarray((norm_map * 0.8 * 255).astype(np.uint8)).resize((w, h))

    blended = Image.composite(heatmap_img, orig_img.convert("RGB"), alpha_img)

    out_path = RESULTS_DIR / f"heatmap_{orig_path.stem}.jpg"
    blended.save(out_path, "JPEG", quality=92)
    return out_path
