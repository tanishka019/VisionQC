# VisionQC — AI Quality Inspection System
### TechForge Hackathon Submission

> **Learns "normal" from 20–30 good product photos → detects defects live from webcam or images → shows heatmap + confidence score + PASS/FAIL.**

---

## 🚀 Quick Start

Requires **Python 3.11** and **Node.js 20+**.

### Terminal 1 — Python backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1

# CPU-only torch keeps the download small; drop the extra index to use CUDA
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The first training run downloads the WideResNet-50 backbone weights (~270 MB) from Hugging Face.

### Terminal 2 — React frontend

```bash
cd frontend
cp .env.example .env              # Windows: copy .env.example .env
npm install
npm run dev
```

Open **http://localhost:5173**. API docs are at **http://localhost:8000/docs**.

### Or with Docker

```bash
docker compose up --build
```

The frontend is served on http://localhost:5173 and the API on http://localhost:8000. Uploads, the trained model and the database are kept in the `visionqc-data` volume.

### Deploying on Railway

Create **two services from this repo**, one per folder:

| Service | Root Directory | Config File Path | Variables |
|---------|----------------|------------------|-----------|
| backend | `/backend` | `/backend/railway.toml` | `VISIONQC_CORS_ORIGINS=https://<frontend-domain>` (optional: `VISIONQC_API_KEY`) |
| frontend | `/frontend` | `/frontend/railway.toml` | `VITE_API_URL=https://<backend-domain>` (and `VITE_API_KEY` if set on the backend) |

- Generate a public domain for each service (Settings → Networking), then put each domain into the other service's variable and redeploy. `VITE_API_URL` is baked in at build time, so changing it needs a rebuild.
- Attach a **volume mounted at `/data`** to the backend, or the trained model, uploads and history are lost on every redeploy.
- The backend needs roughly 2 GB of RAM (PyTorch and WideResNet-50). The first training run downloads about 270 MB of weights.

---

## 📋 How to Demo (Judge Flow)

1. **Train** → Go to `Train Model` → upload 20–30 GOOD product photos → click "Learn Normal". This takes about 20–60 s on CPU and progress streams live.
2. **Inspect** → Go to `Inspect` → start the webcam, hold the product up and click "Capture & Inspect", or upload one or more images.
3. **View the heatmap** → regions that look unlike anything in the training set glow red/yellow. Normal regions show the original image.
4. **Tune the threshold** → Go to `Dashboard` → drag the slider → Save. The dashboard updates live as inspections come in.
5. **View history** → Go to `History` → see all inspections, filter PASS/FAIL, and delete entries.

---

## 🏗️ Architecture

```
React (Vite) ←→ FastAPI ←→ PatchCore (Anomalib 1.1, WideResNet-50) ←→ SQLite
                   │
                   ├─ SSE        /train/progress   (training progress)
                   └─ WebSocket  /ws/live          (every new inspection)
```

## 📁 Structure

```
VisionQC/
├── backend/
│   ├── main.py              ← FastAPI app (routes, upload validation, SSE, WebSocket)
│   ├── db.py                ← SQLite layer
│   ├── ml/model.py          ← PatchCore training, calibration, inference, heatmaps
│   ├── tests/               ← pytest suite
│   ├── requirements.txt     ← runtime deps (pinned to a known-good set)
│   ├── requirements-dev.txt ← + pytest/httpx
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── pages/           ← Dashboard, Train, Inspect, History
│   │   └── api.js           ← API client (+ live-feed helper)
│   ├── .env.example
│   └── Dockerfile
├── images/                  ← sample screw photos (see note below)
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Service status, model state, total inspections |
| `GET`  | `/model/status` | Trained?, product name, trained-at, image count |
| `DELETE` | `/model/reset` | Delete the trained model |
| `POST` | `/train` | Upload ≥5 good images (multipart `files`, `product_name`), trains in the background |
| `GET`  | `/train/progress` | Server-Sent Events stream of training progress |
| `POST` | `/inspect` | Inspect one image (`file`) → score, confidence, PASS/FAIL, heatmap URL |
| `POST` | `/inspect/batch` | Inspect many images (`files`) → per-image results and a summary |
| `GET`  | `/history?limit=&page=&filter=PASS\|FAIL` | Inspection log |
| `DELETE` | `/history/{id}` | Delete one inspection (and its image files) |
| `DELETE` | `/history` | Clear all history (and stored images) |
| `GET`  | `/stats` | Today's totals, hourly and 7-day breakdown |
| `GET` / `POST` | `/threshold` | Get or set the decision threshold (0.01–0.99) |
| `WS`   | `/ws/live` | Pushes every new inspection result as JSON |
| `GET`  | `/uploads/*`, `/results/*` | Uploaded images and heatmaps |

## ⚙️ Configuration

Backend (environment variables, all optional):

| Variable | Default | Purpose |
|----------|---------|---------|
| `VISIONQC_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed origins (`*` allows all) |
| `VISIONQC_API_KEY` | *(unset)* | If set, every POST/DELETE must send it as `X-API-Key` |
| `VISIONQC_MAX_UPLOAD_MB` | `20` | Per-file upload limit |
| `VISIONQC_KEEP_FILES` | `2000` | Max uploaded images (and heatmaps) kept on disk; older ones are pruned |
| `VISIONQC_DATA_DIR` | `backend/` | Where uploads, heatmaps, the model and the DB are stored |

Frontend (`frontend/.env`): `VITE_API_URL` (backend URL) and `VITE_API_KEY` (only needed if the backend sets `VISIONQC_API_KEY`).

---

## 🧠 How It Works

1. **Training**: the uploaded good images are split into a memory-bank set (~80%) and a held-out calibration set (~20%, at least 1).
2. **PatchCore**: a frozen WideResNet-50 extracts patch features (layers 2 and 3) from the memory-bank images. Coreset subsampling (10%) keeps a compact memory bank of "normal" patches.
3. **Calibration**: the held-out good images are scored. Their lowest and highest raw scores define the scale, so **score 0 ≈ a typical good part and 0.5 = the most unusual good part seen**. That's why the default threshold of 0.5 is a sensible starting point.
4. **Inference**: a new image is resized to 256×256 and ImageNet-normalised. Each patch is compared with its nearest neighbour in the memory bank, giving a per-patch anomaly map and an image score.
5. **Heatmap**: the anomaly map is scaled against the calibration values and blended in, with transparency where things look normal.
6. **Decision**: `score >= threshold → FAIL`, otherwise `PASS`. Confidence is how far the score is from the opposite verdict.

The model is saved to `model_artifacts/patchcore.pt` and reloads automatically when the server restarts.

---

## ⚡ Dataset for Testing

PatchCore works best when every "good" photo shows **the same part, framed the same way** (fixed camera, similar lighting). The best public benchmark is **MVTec AD** (Screw category): https://www.mvtec.com/company/research/datasets/mvtec-ad

```
dataset/screw/train/good/*.png          → training images (320)
dataset/screw/test/scratch_head/*.png   → scratch defects
dataset/screw/test/manipulated_front/   → manipulation defects
dataset/screw/test/thread_side/         → thread defects
```

> **Note on `images/`:** the bundled `screws_*.png` photos are cluttered scenes of loose screws on wood, each laid out differently. They are fine for checking that the pipeline runs, but every photo is a different "normal", so defect separation is much weaker than on aligned single-part photos like MVTec.

---

## 🧪 Development

```bash
# Backend tests (fast; the ML layer is mocked)
cd backend && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
python -m pytest -q tests

# Include the real train → inspect test (downloads weights, ~30 s on CPU)
VISIONQC_RUN_SLOW=1 python -m pytest -q tests

# Frontend
cd frontend && npm run lint && npm run build
```

CI (`.github/workflows/ci.yml`) runs all of the above on every push and pull request.
