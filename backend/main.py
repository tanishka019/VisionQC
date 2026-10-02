"""
main.py — VisionQC FastAPI Backend (v3)

Environment variables (all optional):
    VISIONQC_CORS_ORIGINS   comma-separated allowed origins
                            (default: http://localhost:5173,http://127.0.0.1:5173; "*" allows all)
    VISIONQC_API_KEY        if set, POST/DELETE requests must send it in the X-API-Key header
    VISIONQC_MAX_UPLOAD_MB  per-file upload limit in MB (default 20)
    VISIONQC_KEEP_FILES     max number of uploaded images (and heatmaps) kept on disk (default 2000)
    VISIONQC_DATA_DIR       where uploads, results, the model and the DB live (default: this folder)
"""
import io
import os
import shutil
import uuid
import logging
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import (
    FastAPI, File, UploadFile, HTTPException,
    Form, Query, Request, WebSocket, WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

import db
from ml import model as ml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("visionqc")

VERSION     = "3.0.0"
DATA_DIR    = ml.DATA_DIR
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = ml.RESULTS_DIR
TRAIN_DIR   = DATA_DIR / "train_temp"

CORS_ORIGINS  = [o.strip() for o in os.getenv(
    "VISIONQC_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",") if o.strip()]
API_KEY       = os.getenv("VISIONQC_API_KEY") or None
MAX_UPLOAD_MB = float(os.getenv("VISIONQC_MAX_UPLOAD_MB", "20"))
KEEP_FILES    = int(os.getenv("VISIONQC_KEEP_FILES", "2000"))
IMAGE_EXTS    = ml.IMAGE_EXTS

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


async def _set_train_state(state: dict):
    global _train_state
    _train_state = state
    await _broadcast_train_event(state)


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="VisionQC API", version=VERSION, docs_url="/docs")

@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """When VISIONQC_API_KEY is set, protect every state-changing request."""
    if API_KEY and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse({"detail": "Invalid or missing API key."}, status_code=401)
    return await call_next(request)


# Added last so it is the outermost layer: even 401s carry CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def model_is_trained() -> bool:
    """The DB flag and the checkpoint on disk must agree."""
    return db.get_config("model_trained") == "true" and ml.is_trained()


async def read_image_upload(file: UploadFile) -> tuple[bytes, str]:
    """Read an upload, enforcing extension, size and that it decodes as an image."""
    ext = Path(file.filename or "").suffix.lower() or ".jpg"
    if ext not in IMAGE_EXTS:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(IMAGE_EXTS))}.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File too large (max {MAX_UPLOAD_MB:g} MB).")
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValueError("File is not a valid image.")
    return data, ext


def prune_old_files(directory: Path, keep: int = KEEP_FILES):
    """Keep only the newest `keep` files in a directory."""
    files = sorted((f for f in directory.iterdir() if f.is_file()), key=lambda f: f.stat().st_mtime)
    for f in files[:-keep] if keep > 0 else files:
        f.unlink(missing_ok=True)


def remove_files(*paths: Optional[str]):
    for p in paths:
        if p:
            Path(p).unlink(missing_ok=True)


async def run_inspection(file: UploadFile) -> dict:
    """Validate, save, inspect, log and broadcast one image."""
    data, ext = await read_image_upload(file)
    uid = uuid.uuid4().hex
    image_path = UPLOADS_DIR / f"{uid}{ext}"
    image_path.write_bytes(data)

    threshold = float(db.get_config("threshold") or 0.5)
    result = await run_in_threadpool(ml.inspect, image_path, threshold)

    heatmap_url = None
    if result.get("heatmap_path"):
        heatmap_url = f"/results/{Path(result['heatmap_path']).name}"

    log_id = db.log_inspection(
        image_path=str(image_path),
        heatmap_path=result.get("heatmap_path", ""),
        score=result["score"],
        confidence=result["confidence"],
        result=result["result"],
        threshold=threshold,
        filename=file.filename,
    )

    payload = {
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
    await broadcast_inspection({"type": "inspection", **payload})
    return payload


def prune_storage():
    prune_old_files(UPLOADS_DIR)
    prune_old_files(RESULTS_DIR)


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_trained": model_is_trained(),
        "product_name": db.get_config("product_name") or "Product",
        "total_inspections": db.get_total_inspections(),
        "version": VERSION,
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
    return {
        "trained": model_is_trained(),
        "product_name": db.get_config("product_name") or "Product",
        "trained_at": db.get_config("trained_at") or None,
        "image_count": int(db.get_config("train_image_count") or "0"),
    }


@app.delete("/model/reset")
def reset_model():
    """Delete trained model and reset state."""
    if _train_state.get("status") == "running":
        raise HTTPException(409, "Training in progress.")
    try:
        ml.reset_model()
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
            if _train_state.get("status") != "running":
                return
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
    if _train_state.get("status") == "running":
        raise HTTPException(409, "Training already in progress.")

    if len(files) < ml.MIN_IMAGES:
        raise HTTPException(400, f"Upload at least {ml.MIN_IMAGES} images (got {len(files)}).")

    # Clear train temp
    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    saved, skipped = 0, []
    for f in files:
        try:
            data, ext = await read_image_upload(f)
        except ValueError as e:
            skipped.append({"filename": f.filename, "reason": str(e)})
            continue
        (TRAIN_DIR / f"{uuid.uuid4().hex}{ext}").write_bytes(data)
        saved += 1

    if saved < ml.MIN_IMAGES:
        raise HTTPException(400, f"Not enough valid image files ({saved} valid, need {ml.MIN_IMAGES}).")

    await _set_train_state({"status": "running", "progress": 5, "message": f"Saved {saved} images…"})
    asyncio.create_task(_run_training(product_name, saved))

    return {"status": "started", "image_count": saved, "skipped": skipped, "product_name": product_name}


async def _run_training(product_name: str, image_count: int):
    loop = asyncio.get_running_loop()

    def progress(pct: int, message: str):
        # called from the worker thread
        asyncio.run_coroutine_threadsafe(
            _set_train_state({"status": "running", "progress": pct, "message": message}), loop
        )

    try:
        result = await run_in_threadpool(ml.train, TRAIN_DIR, product_name, progress)

        db.set_config("model_trained", "true")
        db.set_config("product_name", product_name)
        db.set_config("trained_at", datetime.now().isoformat())
        db.set_config("train_image_count", str(result["image_count"]))

        await _set_train_state({
            "status": "done",
            "progress": 100,
            "message": f"✅ Trained on {result['image_count']} images.",
            "image_count": result["image_count"],
        })

    except Exception as e:
        logger.exception("Training failed")
        await _set_train_state({"status": "error", "progress": 0, "message": str(e)})
    finally:
        shutil.rmtree(TRAIN_DIR, ignore_errors=True)
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)


# ─── Inspect (single) ─────────────────────────────────────────────────────────
@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    """Inspect a single image. Returns score, result, heatmap URL."""
    if not model_is_trained():
        raise HTTPException(400, "Model not trained yet. Go to Train first.")

    try:
        payload = await run_inspection(file)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Inspection failed")
        raise HTTPException(500, f"Inspection failed: {e}")

    prune_storage()
    return payload


# ─── Batch Inspect ────────────────────────────────────────────────────────────
@app.post("/inspect/batch")
async def batch_inspect(files: List[UploadFile] = File(...)):
    """Inspect multiple images at once."""
    if not model_is_trained():
        raise HTTPException(400, "Model not trained yet.")

    results = []
    for file in files:
        try:
            r = await run_inspection(file)
            results.append({**r, "error": None})
        except Exception as e:
            if not isinstance(e, ValueError):
                logger.exception(f"Inspection failed for {file.filename}")
            results.append({
                "filename": file.filename,
                "error": str(e),
                "result": "ERROR",
            })

    prune_storage()
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
    total = db.get_total_inspections(result_filter=result_filter)
    return {"inspections": rows, "total": total, "page": page, "limit": limit}


@app.delete("/history/{inspection_id}")
def delete_inspection(inspection_id: int):
    paths = db.delete_inspection(inspection_id)
    if paths is None:
        raise HTTPException(404, "Inspection not found.")
    remove_files(*paths)
    return {"deleted": True, "id": inspection_id}


@app.delete("/history")
def clear_history():
    count = db.clear_history()
    for d in (UPLOADS_DIR, RESULTS_DIR):
        for f in d.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
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
    """Push every new inspection result to all connected clients."""
    await ws.accept()
    _ws_clients.append(ws)
    logger.info(f"WS client connected. Total: {len(_ws_clients)}")
    try:
        while True:
            await ws.receive_text()  # keep-alive ping
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
        logger.info(f"WS client disconnected. Total: {len(_ws_clients)}")


async def broadcast_inspection(data: dict):
    dead = []
    for ws in list(_ws_clients):
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
