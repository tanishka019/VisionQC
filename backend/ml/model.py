"""
model.py — PatchCore anomaly detection via Anomalib (v2)
"""
import os
import shutil
import logging
import numpy as np
from pathlib import Path
from PIL import Image

logger = logging.getLogger("visionqc.model")

BASE_DIR    = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "good"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR   = BASE_DIR / "model_artifacts"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_engine = None
_model  = None


def _get_anomalib():
    from anomalib.models import Patchcore
    from anomalib.engine import Engine
    from anomalib.data import Folder
    return Patchcore, Engine, Folder


def is_trained() -> bool:
    checkpoints = list(MODEL_DIR.rglob("*.ckpt"))
    return len(checkpoints) > 0


def reset_model():
    """Delete all model artifacts and dataset."""
    global _engine, _model
    _engine = None
    _model  = None
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_DIR.parent.exists():
        shutil.rmtree(DATASET_DIR.parent)


def train(training_images_dir: Path, product_name: str = "product") -> dict:
    """
    Train PatchCore on a folder of 'good' images.
    Expects images in training_images_dir/*.
    """
    global _engine, _model

    Patchcore, Engine, Folder = _get_anomalib()

    # Copy images to canonical dataset folder
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in training_images_dir.glob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    if len(image_files) < 5:
        raise ValueError(f"Need at least 5 images, got {len(image_files)}")

    for f in image_files:
        shutil.copy(f, DATASET_DIR / f.name)

    logger.info(f"Training PatchCore on {len(image_files)} images for '{product_name}'…")

    datamodule = Folder(
        name=product_name,
        root=BASE_DIR / "dataset",
        normal_dir="good",
        task="segmentation",
        image_size=(256, 256),
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    model = Patchcore(backbone="wide_resnet50_2", layers_list=["layer2", "layer3"])

    engine = Engine(
        default_root_dir=str(MODEL_DIR),
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )

    engine.fit(model=model, datamodule=datamodule)

    _engine = engine
    _model  = model

    logger.info("Training complete!")
    return {"status": "trained", "image_count": len(image_files)}


def _load_model_if_needed():
    global _engine, _model
    if _model is not None:
        return

    Patchcore, Engine, _ = _get_anomalib()

    checkpoints = sorted(MODEL_DIR.rglob("*.ckpt"))
    if not checkpoints:
        raise RuntimeError("No trained model found. Please train first.")

    ckpt_path = checkpoints[-1]
    logger.info(f"Loading checkpoint: {ckpt_path}")

    _model  = Patchcore.load_from_checkpoint(str(ckpt_path))
    _engine = Engine(accelerator="auto", devices=1, logger=False, enable_progress_bar=False)


def inspect(image_path: Path, threshold: float) -> dict:
    """
    Run anomaly detection on a single image.
    Returns score, confidence, result, heatmap_path.
    """
    _load_model_if_needed()

    import torch
    import torchvision.transforms.functional as TF

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((256, 256))

    tensor = TF.to_tensor(img_resized).unsqueeze(0)  # [1, 3, 256, 256]

    _model.eval()
    with torch.no_grad():
        output = _model(tensor)

    anomaly_map = output.anomaly_map
    if anomaly_map is not None:
        amap = anomaly_map.squeeze().cpu().numpy()
    else:
        amap = np.zeros((256, 256))

    raw_score = (
        float(output.pred_score.squeeze().cpu().numpy())
        if output.pred_score is not None
        else float(amap.max())
    )

    score    = min(max(raw_score, 0.0), 1.0)
    result   = "FAIL" if score >= threshold else "PASS"
    confidence = round((score if result == "FAIL" else 1 - score) * 100, 1)

    heatmap_path = _save_heatmap(image_path, img, amap)

    return {
        "score":        round(score, 4),
        "confidence":   confidence,
        "result":       result,
        "heatmap_path": str(heatmap_path),
    }


def _save_heatmap(orig_path: Path, orig_img: Image.Image, amap: np.ndarray) -> Path:
    """Overlay a jet-colourmap heatmap on the original image and save."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    w, h = orig_img.size

    a_min, a_max = amap.min(), amap.max()
    norm_map = (amap - a_min) / (a_max - a_min) if a_max > a_min else amap

    colormap      = cm.get_cmap("inferno")   # better than jet for professional look
    heatmap_rgba  = colormap(norm_map)
    heatmap_rgb   = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_img   = Image.fromarray(heatmap_rgb).resize((w, h))

    # Blend original + heatmap
    blended = Image.blend(orig_img.convert("RGB"), heatmap_img, alpha=0.5)

    stem     = orig_path.stem
    out_path = RESULTS_DIR / f"heatmap_{stem}.jpg"
    blended.save(out_path, "JPEG", quality=92)
    return out_path
