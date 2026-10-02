"""
main.py — VisionQC FastAPI Backend (v2 – Professional)
"""
import os
import shutil
import uuid
import logging
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, File, UploadFile, HTTPException,
    Form, Query, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("visionqc")

BASE_DIR    = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
TRAIN_DIR   = BASE_DIR / "train_temp"

for d in [UPLOADS_DIR, RESULTS_DIR, TRAIN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

db.init_db()

# ─── Global training state (SSE) ──────────────────────────────────────────────
_train_state: dict = {"status": "idle", "progress": 0, "message": ""}
_train_subscribers: list = []

async def _broadcast_train_event(data: dict):
    for q in list(_train_subscribers):
        try:
            await q.put(data)
        except Exception:
            pass


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="VisionQC API", version="2.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    trained = db.get_config("model_trained") == "true"
    product = db.get_config("product_name") or "Product"
    total   = db.get_total_inspections()
    return {
        "status": "ok",
        "model_trained": trained,
        "product_name": product,
        "total_inspections": total,
        "version": "2.0.0",
        "uptime_ts": datetime.now().isoformat(),
    }


# ─── Config / Threshold ───────────────────────────────────────────────────────
@app.get("/threshold")
def get_threshold():
    val = db.get_config("threshold")
    return {"threshold": float(val) if val else 0.5}


class ThresholdBody(BaseModel):
    threshold: float

@app.post("/threshold")
def set_threshold(body: ThresholdBody):
    if not (0.01 <= body.threshold <= 0.99):
        raise HTTPException(400, "Threshold must be between 0.01 and 0.99.")
    db.set_config("threshold", str(body.threshold))
    return {"threshold": body.threshold, "saved": True}


# ─── Model status ─────────────────────────────────────────────────────────────
@app.get("/model/status")
def model_status():
    trained = db.get_config("model_trained") == "true"
    product = db.get_config("product_name") or "Product"
    trained_at = db.get_config("trained_at") or None
    image_count = db.get_config("train_image_count") or "0"
    return {
        "trained": trained,
        "product_name": product,
        "trained_at": trained_at,
        "image_count": int(image_count),
    }


@app.delete("/model/reset")
def reset_model():
    """Delete trained model and reset state."""
    from ml.model import reset_model as ml_reset
    try:
        ml_reset()
    except Exception as e:
        logger.warning(f"ml_reset error: {e}")
    db.set_config("model_trained", "false")
    db.set_config("product_name", "Product")
    db.set_config("trained_at", "")
    db.set_config("train_image_count", "0")
    return {"reset": True}


# ─── Train (with SSE progress) ────────────────────────────────────────────────
@app.get("/train/progress")
async def train_progress():
    """SSE stream for training progress updates."""
    queue: asyncio.Queue = asyncio.Queue()
    _train_subscribers.append(queue)

    async def event_stream():
        try:
            # Send current state immediately
            yield f"data: {json.dumps(_train_state)}\n\n"
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                    if data.get("status") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    yield "data: {\"ping\":1}\n\n"  # keepalive
        finally:
            _train_subscribers.remove(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/train")
async def train_model(
    files: List[UploadFile] = File(...),
    product_name: str = Form("product"),
):
    """Accept good product images, train PatchCore in background."""
    global _train_state

    if _train_state.get("status") == "running":
        raise HTTPException(409, "Training already in progress.")

    if len(files) < 5:
        raise HTTPException(400, f"Upload at least 5 images (got {len(files)}).")

    # Clear train temp
    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        dest = TRAIN_DIR / f"{uuid.uuid4().hex}{ext}"
        dest.write_bytes(await f.read())
        saved += 1

    if saved < 5:
        raise HTTPException(400, "Not enough valid image files.")

    # Launch background training
    asyncio.create_task(_run_training(product_name, saved))
    _train_state = {"status": "running", "progress": 5, "message": f"Saving {saved} images…"}
    await _broadcast_train_event(_train_state)

    return {"status": "started", "image_count": saved, "product_name": product_name}


async def _run_training(product_name: str, image_count: int):
    global _train_state

    async def update(progress: int, message: str):
        _train_state = {"status": "running", "progress": progress, "message": message}
        await _broadcast_train_event(_train_state)

    try:
        await update(10, "Loading model architecture…")
        await asyncio.sleep(0.5)

        loop = asyncio.get_event_loop()
        await update(20, "Extracting deep features…")

        from ml.model import train
        result = await loop.run_in_executor(
            None, lambda: train(TRAIN_DIR, product_name=product_name)
        )

        await update(90, "Saving model checkpoint…")
        await asyncio.sleep(0.3)

        db.set_config("model_trained", "true")
        db.set_config("product_name", product_name)
        db.set_config("trained_at", datetime.now().isoformat())
        db.set_config("train_image_count", str(result["image_count"]))

        _train_state = {
            "status": "done",
            "progress": 100,
            "message": f"✅ Trained on {result['image_count']} images.",
            "image_count": result["image_count"],
        }
        await _broadcast_train_event(_train_state)

    except Exception as e:
        logger.exception("Training failed")
        _train_state = {"status": "error", "progress": 0, "message": str(e)}
        await _broadcast_train_event(_train_state)


# ─── Inspect (single) ─────────────────────────────────────────────────────────
@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    """Inspect a single image. Returns score, result, heatmap URL."""
    trained = db.get_config("model_trained") == "true"
    if not trained:
        raise HTTPException(400, "Model not trained yet. Go to Train first.")

    ext = Path(file.filename).suffix.lower() or ".jpg"
    uid = uuid.uuid4().hex
    image_path = UPLOADS_DIR / f"{uid}{ext}"
    image_path.write_bytes(await file.read())

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
        filename=file.filename,
    )

    return {
        "id": log_id,
        "score": result["score"],
        "confidence": result["confidence"],
        "result": result["result"],
        "threshold": threshold,
        "heatmap_url": heatmap_url,
        "image_url": f"/uploads/{uid}{ext}",
        "filename": file.filename,
        "timestamp": datetime.now().isoformat(),
    }


# ─── Batch Inspect ────────────────────────────────────────────────────────────
@app.post("/inspect/batch")
async def batch_inspect(files: List[UploadFile] = File(...)):
    """Inspect multiple images at once."""
    trained = db.get_config("model_trained") == "true"
    if not trained:
        raise HTTPException(400, "Model not trained yet.")

    threshold = float(db.get_config("threshold") or 0.5)
    results = []

    for file in files:
        try:
            ext = Path(file.filename).suffix.lower() or ".jpg"
            uid = uuid.uuid4().hex
            image_path = UPLOADS_DIR / f"{uid}{ext}"
            image_path.write_bytes(await file.read())

            from ml.model import inspect
            r = inspect(image_path, threshold)

            heatmap_url = None
            if r.get("heatmap_path"):
                heatmap_url = f"/results/{Path(r['heatmap_path']).name}"

            log_id = db.log_inspection(
                image_path=str(image_path),
                heatmap_path=r.get("heatmap_path", ""),
                score=r["score"],
                confidence=r["confidence"],
                result=r["result"],
                threshold=threshold,
                filename=file.filename,
            )
            results.append({
                "id": log_id,
                "filename": file.filename,
                "score": r["score"],
                "confidence": r["confidence"],
                "result": r["result"],
                "threshold": threshold,
                "heatmap_url": heatmap_url,
                "image_url": f"/uploads/{uid}{ext}",
                "error": None,
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "result": "ERROR",
            })

    total   = len(results)
    passed  = sum(1 for r in results if r.get("result") == "PASS")
    failed  = sum(1 for r in results if r.get("result") == "FAIL")
    return {"results": results, "summary": {"total": total, "passed": passed, "failed": failed}}


# ─── History ──────────────────────────────────────────────────────────────────
@app.get("/history")
def history(
    limit: int = Query(200, ge=1, le=1000),
    result_filter: Optional[str] = Query(None, alias="filter"),
    page: int = Query(1, ge=1),
):
    rows = db.get_history(limit=limit, result_filter=result_filter, page=page)
    total = db.get_total_inspections()
    return {"inspections": rows, "total": total, "page": page, "limit": limit}


@app.delete("/history/{inspection_id}")
def delete_inspection(inspection_id: int):
    deleted = db.delete_inspection(inspection_id)
    if not deleted:
        raise HTTPException(404, "Inspection not found.")
    return {"deleted": True, "id": inspection_id}


@app.delete("/history")
def clear_history():
    count = db.clear_history()
    return {"cleared": count}


# ─── Stats ────────────────────────────────────────────────────────────────────
@app.get("/stats")
def stats():
    today   = db.get_today_stats()
    hourly  = db.get_hourly_stats()
    weekly  = db.get_weekly_stats()
    return {"today": today, "hourly": hourly, "weekly": weekly}


# ─── WebSocket live feed ───────────────────────────────────────────────────────
_ws_clients: list[WebSocket] = []

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """Push latest inspection result to all connected clients."""
    await ws.accept()
    _ws_clients.append(ws)
    logger.info(f"WS client connected. Total: {len(_ws_clients)}")
    try:
        while True:
            await ws.receive_text()  # keep-alive ping
    except WebSocketDisconnect:
        _ws_clients.remove(ws)
        logger.info(f"WS client disconnected. Total: {len(_ws_clients)}")


async def broadcast_inspection(data: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)
