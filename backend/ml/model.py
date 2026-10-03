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

try:
    from . import shape
except ImportError:   # run as a plain script / tests that put ml/ on the path
    import shape

logger = logging.getLogger("visionqc.model")

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = Path(os.getenv("VISIONQC_DATA_DIR", BASE_DIR))
RESULTS_DIR = DATA_DIR / "results"
MODEL_DIR   = DATA_DIR / "model_artifacts"
CHECKPOINT  = MODEL_DIR / "patchcore.pt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Defaults picked by benchmarking on photos with defects drawn at known positions, judging BOTH the heatmap
# and the PASS/FAIL verdict (vs. the previous 256x256 square / layer2+layer3: pixel AUROC 0.90-0.92 -> 0.98,
# heatmap peak inside the defect 46-51% -> 69-78%, large defects still flagged FAIL ~100%).
# Dropping layer3 gave the sharpest maps but wrecked the verdict (large defects flagged 0-6%), so it stays.
IMAGE_LONG_SIDE = int(os.getenv("VISIONQC_IMAGE_SIZE", "384"))   # long side of the model input, in pixels
FIT_ASPECT      = os.getenv("VISIONQC_FIT_ASPECT", "1") == "1"   # keep the photos' aspect ratio instead of squashing to a square
IMAGE_SIZE     = (IMAGE_LONG_SIDE, IMAGE_LONG_SIDE)              # (w, h); recomputed from the training photos when FIT_ASPECT
BACKBONE       = os.getenv("VISIONQC_BACKBONE", "wide_resnet50_2")
# layer1 (stride 4) localises small defects; layer2/layer3 carry the context the verdict needs.
LAYERS         = tuple(os.getenv("VISIONQC_LAYERS", "layer2,layer3").split(","))
CORESET_RATIO  = float(os.getenv("VISIONQC_CORESET_RATIO", "0.02"))
BATCH_SIZE     = int(os.getenv("VISIONQC_BATCH_SIZE", "8"))
CALIB_FOLDS    = 5
SCORE_MARGIN   = 0.2
ALIGN          = os.getenv("VISIONQC_ALIGN", "1") == "1"                       # rotate/crop single long parts to a canonical pose
ALIGN_CANVAS   = tuple(int(v) for v in os.getenv("VISIONQC_ALIGN_SIZE", "448x256").split("x"))
# Coreset sampling is O(candidates x bank size); pre-sampling candidate patches (adjacent
# patches are highly redundant) makes it much faster. 0 = use every patch.
MAX_CANDIDATES = int(os.getenv("VISIONQC_MAX_CANDIDATES", "1"))      # 0 = use every patch (slow)
CAND_PER_IMAGE, MIN_CAND, MAX_CAND_CAP = 480, 8000, 20000             # patches the coreset picks from: 480 per photo
BANK_PER_IMAGE, MIN_BANK, BANK_CAP = 60, 1200, 2400                   # patches kept: 60 per photo
SEED           = 0
MIN_IMAGES     = 5
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)

_model       = None   # anomalib PatchcoreModel (torch module) with memory bank
_calibration = None   # {"image_min", "image_max", "pixel_typical", "pixel_max": float}
_lock        = threading.Lock()
_align_size  = None   # (w, h) canvas when the model was trained on aligned parts, else None
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


def _prepare(img: Image.Image):
    """Apply the part alignment the model was trained with. Returns (image, M); M is None when not aligned."""
    if _align_size:
        got = shape.align(img, _align_size)
        if got is not None:
            return got
    return img, None


def _to_tensor(img: Image.Image, size: tuple[int, int] | None = None, prepared: bool = False):
    """PIL image → normalised [3, H, W] tensor, matching the backbone's ImageNet stats."""
    import torchvision.transforms.functional as TF
    if not prepared:
        img = _prepare(img)[0]
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
    """Build the PatchCore memory bank (coreset) from per-image patch embeddings.

    The bank grows with the number of photos (more photos = more kinds of normal to remember); a fixed
    size made scores collapse when 50 photos were used instead of 25.
    """
    import torch
    n = len(embeddings)
    cand = int(min(MAX_CAND_CAP, max(MIN_CAND, CAND_PER_IMAGE * n))) if MAX_CANDIDATES else 0
    bank = min(BANK_CAP, max(MIN_BANK, BANK_PER_IMAGE * n))
    stacked = torch.vstack(embeddings)
    total = stacked.shape[0]
    if cand and total > cand:
        generator = torch.Generator().manual_seed(SEED)
        stacked = stacked[torch.randperm(total, generator=generator)[:cand]]
        ratio = min(0.5, max(0.04, bank / stacked.shape[0]))
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
    global _model, _calibration, _img_size, _align_size

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
    # Align single long parts (screws, bolts…) to a canonical pose when every photo allows it
    _align_size = None
    if ALIGN:
        ok = sum(shape.align(Image.open(p), ALIGN_CANVAS) is not None for p in image_files)
        if ok >= 0.8 * len(image_files):
            _align_size = ALIGN_CANVAS
    size = _align_size or choose_input_size(image_files)
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
        "image_anchor": float(np.percentile(scores, 95)),
        "pixel_typical": float(np.median(map_median)),
        "pixel_max": float(max(map_max)),
    }

    # Straightness of the part (only when every photo shows one long object on a plain background)
    devs = [m["dev"] for m in (shape.measure(Image.open(p)) for p in image_files) if m is not None]
    if len(devs) >= 0.8 * len(image_files):
        calibration["shape_min"], calibration["shape_max"] = float(min(devs)), float(max(devs))

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
                "image_size": list(size), "align": list(_align_size) if _align_size else None, "product_name": product_name,
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
    global _model, _calibration, _img_size, _align_size
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
    _align_size = tuple(cfg["align"]) if cfg.get("align") else None
    _img_size = tuple(cfg.get("image_size") or IMAGE_SIZE)   # old checkpoints stay at the size they were trained at


def normalise_score(raw: float, calibration: dict) -> float:
    hi, lo = calibration.get("image_anchor", calibration["image_max"]), calibration["image_min"]   # p95 of unseen-good scores
    spread = max(hi - lo, 0.1 * abs(hi), 1e-6)  # guard: a single calibration image
    # The hottest of n good photos is beaten by a new good photo about 1 time in n+1, so the 0.5 mark sits a
    # little above it (SCORE_MARGIN of the spread) to keep false rejects rare on small training sets.
    hi = hi + SCORE_MARGIN * spread
    return float(min(max(0.5 + 0.5 * (raw - hi) / spread, 0.0), 1.0))


def shape_limit(calibration: dict) -> float:
    """Straightness deviation that maps to score 0.5: the worst normal part plus a margin."""
    lo, hi = calibration["shape_min"], calibration["shape_max"]
    return hi + 0.5 * (hi - lo) + 0.004


def shape_score(dev: float, calibration: dict) -> float:
    limit = shape_limit(calibration)
    return float(min(max(0.5 + 0.5 * (dev - limit) / (0.5 * limit), 0.0), 1.0))


def inspect(image_path: Path, threshold: float) -> dict:
    """
    Run anomaly detection on a single image.
    Returns score, confidence, result, heatmap_path.
    """
    with _lock:
        _load_model_if_needed()
        img = Image.open(image_path).convert("RGB")
        prepared, M = _prepare(img)
        raw_scores, maps = _score_raw(_model, _to_tensor(prepared, prepared=True).unsqueeze(0))
        calibration = _calibration
    amap = maps[0]
    if M is not None:   # heat was computed on the aligned part: map it back onto the original photo
        import cv2
        typical = calibration["pixel_typical"]
        amap = cv2.warpAffine(amap.astype(np.float32), M, img.size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=float(typical))
        keep = shape.part_mask(img)   # dust and shadows on the plain background are not defects
        if keep is not None:
            amap = np.where(keep, amap, typical)

    raw_score = float(raw_scores[0])
    score     = normalise_score(raw_score, calibration)
    shape_map = None
    if "shape_max" in calibration:
        m = shape.measure(img)
        if m is not None:
            s_score = shape_score(m["dev"], calibration)
            if s_score > score:
                score = s_score
            if s_score >= 0.45:
                shape_map = m["resid"] / shape_limit(calibration)   # 1.0 = at the limit
    result    = "FAIL" if score >= threshold else "PASS"
    confidence = round((score if result == "FAIL" else 1 - score) * 100, 1)

    heatmap_path = _save_heatmap(image_path, img, amap, calibration, outline=(result == "FAIL"), shape_map=shape_map, relative=True)

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


# Heat drawing. R = hottest normal pixel - typical pixel (both measured on the training photos).
# Nothing is drawn below HEAT_FLOOR * R above typical, so ordinary texture stays clear; the scale tops out at
# HEAT_TOP * R. Tuned on photos with defects at known positions: the share of drawn heat that sits on the
# defect rose from ~12% to ~59% and haze on good photos fell ~50x versus the old inferno-opacity drawing.
HEAT_FLOOR = 0.65
HEAT_TOP   = 1.30
RELATIVE_FLOOR = 0.30   # the always-visible heat view starts here (the calibrated defect heat starts at HEAT_FLOOR)
_HEAT_STOPS = np.array([[232, 71, 43], [255, 125, 30], [255, 205, 50]], dtype=float)   # vermilion → orange → amber


def heat_intensity(amap: np.ndarray, calibration: dict) -> np.ndarray:
    """0..1 heat to draw for each pixel of the anomaly map (0 = transparent)."""
    typical, peak = calibration["pixel_typical"], calibration["pixel_max"]
    x = (amap - typical) / max(peak - typical, 1e-6)
    return np.clip((x - HEAT_FLOOR) / (HEAT_TOP - HEAT_FLOOR), 0.0, 1.0)


def _heat_colours(heat: np.ndarray) -> np.ndarray:
    """Warm ramp that never darkens the photo (unlike inferno's near-black low end)."""
    pos = np.linspace(0.0, 1.0, len(_HEAT_STOPS))
    return np.stack([np.interp(heat, pos, _HEAT_STOPS[:, c]) for c in range(3)], axis=-1)


def _save_heatmap(orig_path: Path, orig_img: Image.Image, amap: np.ndarray, calibration: dict, outline: bool = True, shape_map=None, relative: bool = False) -> Path:
    """Draw the heat over the original photo and save it.

    Heat is calibrated against what normal photos look like (see HEAT_FLOOR) and drawn with an opacity that
    follows its intensity. When `outline` is set (a FAIL verdict) the hot region also gets a thin outline so
    the location is unambiguous; a PASS never outlines anything, so the picture can't contradict the verdict.
    """
    from PIL import ImageFilter

    w, h = orig_img.size
    heat_small = Image.fromarray((heat_intensity(amap, calibration) * 255).astype(np.uint8))
    heat = np.asarray(heat_small.resize((w, h), Image.BICUBIC), dtype=float) / 255.0

    hard = heat    # the calibrated heat alone decides where the FAIL outline goes
    if relative:   # always-visible view: how unusual each spot is, even when the part passes (faint = normal)
        typical, peak = calibration["pixel_typical"], calibration["pixel_max"]
        x = (amap - typical) / max(peak - typical, 1e-6)
        x_img = np.asarray(Image.fromarray(x.astype(np.float32)).resize((w, h), Image.BICUBIC))
        heat = np.maximum(heat, 0.8 * np.clip((x_img - RELATIVE_FLOOR) / (HEAT_TOP - RELATIVE_FLOOR), 0.0, 1.0))

    if shape_map is not None:   # bent-part evidence: heat on the part, strongest where the centre line strays
        sm = np.asarray(Image.fromarray(shape_map.astype(np.float32)).resize((w, h), Image.BILINEAR))
        heat = np.maximum(heat, np.clip((sm - HEAT_FLOOR) / (HEAT_TOP - HEAT_FLOOR), 0.0, 1.0))

    base  = np.asarray(orig_img.convert("RGB"), dtype=float)
    alpha = (0.72 * heat ** 0.8)[..., None]          # translucent enough to still see the defect underneath
    out   = base * (1 - alpha) + _heat_colours(heat) * alpha

    region = hard > 0.3
    if outline and region.any():
        k = max(3, (w // 300) | 1)                       # outline thickness grows with the image
        mask = Image.fromarray((region * 255).astype(np.uint8))
        edge = np.asarray(mask.filter(ImageFilter.MaxFilter(k))) > 0
        edge &= ~(np.asarray(mask.filter(ImageFilter.MinFilter(k))) > 0)
        out[edge] = 0.15 * out[edge] + 0.85 * 255

    out_path = RESULTS_DIR / f"heatmap_{orig_path.stem}.jpg"
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(out_path, "JPEG", quality=92)
    return out_path
