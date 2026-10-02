"""
main.py — VisionQC FastAPI Backend
"""
import os
import shutil
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visionqc")

BASE_DIR    = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
TRAIN_DIR   = BASE_DIR / "train_temp"

for d in [UPLOADS_DIR, RESULTS_DIR, TRAIN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

db.init_db()

app = FastAPI(title="VisionQC API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve heatmap images
app.mount("/results",  StaticFiles(directory=str(RESULTS_DIR)),  name="results")
app.mount("/uploads",  StaticFiles(directory=str(UPLOADS_DIR)),  name="uploads")


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ─── Config / Threshold ───────────────────────────────────────────────────────

@app.get("/threshold")
def get_threshold():
    val = db.get_config("threshold")
    return {"threshold": float(val) if val else 0.5}


class ThresholdBody(BaseModel):
    threshold: float

@app.post("/threshold")
def set_threshold(body: ThresholdBody):
    if not (0.0 < body.threshold < 1.0):
        raise HTTPException(400, "Threshold must be between 0 and 1 (exclusive).")
    db.set_config("threshold", str(body.threshold))
    return {"threshold": body.threshold}


# ─── Model status ─────────────────────────────────────────────────────────────

@app.get("/model/status")
def model_status():
    trained = db.get_config("model_trained") == "true"
    product = db.get_config("product_name") or "Product"
    return {"trained": trained, "product_name": product}


# ─── Train ────────────────────────────────────────────────────────────────────

@app.post("/train")
async def train_model(
    files: List[UploadFile] = File(...),
    product_name: str = Form("product"),
):
    """
    Accept 20-30 good product images, train PatchCore, update DB.
    """
    if len(files) < 5:
        raise HTTPException(400, f"Upload at least 5 images (got {len(files)}).")

    # Clear and recreate temp train folder
    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        dest = TRAIN_DIR / f"{uuid.uuid4().hex}{ext}"
        content = await f.read()
        dest.write_bytes(content)
        saved += 1

    if saved < 5:
        raise HTTPException(400, "Not enough valid image files.")

    try:
        from ml.model import train
        result = train(TRAIN_DIR, product_name=product_name)
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(500, f"Training failed: {e}")

    db.set_config("model_trained", "true")
    db.set_config("product_name", product_name)

    return {"status": "success", "message": f"Model trained on {result['image_count']} images.", "image_count": result["image_count"]}


# ─── Inspect ──────────────────────────────────────────────────────────────────

@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    """
    Inspect a single image. Returns score, result, heatmap URL.
    """
    trained = db.get_config("model_trained") == "true"
    if not trained:
        raise HTTPException(400, "Model not trained yet. Go to /train first.")

    ext = Path(file.filename).suffix.lower() or ".jpg"
    uid = uuid.uuid4().hex
    image_path = UPLOADS_DIR / f"{uid}{ext}"
    content = await file.read()
    image_path.write_bytes(content)

    threshold = float(db.get_config("threshold") or 0.5)

    try:
        from ml.model import inspect
        result = inspect(image_path, threshold)
    except Exception as e:
        logger.exception("Inspection failed")
        raise HTTPException(500, f"Inspection failed: {e}")

    heatmap_url = None
    if result.get("heatmap_path"):
        heatmap_name = Path(result["heatmap_path"]).name
        heatmap_url = f"/results/{heatmap_name}"

    log_id = db.log_inspection(
        image_path=str(image_path),
        heatmap_path=result.get("heatmap_path", ""),
        score=result["score"],
        confidence=result["confidence"],
        result=result["result"],
        threshold=threshold,
    )

    return {
        "id": log_id,
        "score": result["score"],
        "confidence": result["confidence"],
        "result": result["result"],
        "threshold": threshold,
        "heatmap_url": heatmap_url,
        "image_url": f"/uploads/{uid}{ext}",
    }


# ─── History ──────────────────────────────────────────────────────────────────

@app.get("/history")
def history(limit: int = 200):
    rows = db.get_history(limit=limit)
    return {"inspections": rows}


# ─── Stats ────────────────────────────────────────────────────────────────────

@app.get("/stats")
def stats():
    today = db.get_today_stats()
    hourly = db.get_hourly_stats()
    return {"today": today, "hourly": hourly}
