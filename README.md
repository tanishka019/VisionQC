# VisionQC — AI Quality Inspection System
### TechForge Hackathon Submission

> **Learns "normal" from 20–30 good product photos → detects defects live from webcam or images → shows heatmap + confidence score + PASS/FAIL.**

---

## 🚀 Quick Start (3 Terminal Commands)

### Terminal 1 — Python Backend

```powershell
# 1. Install Python 3.11 from https://python.org/downloads
# Make sure to check "Add to PATH" during install

# 2. Navigate to backend
cd C:\Users\HP\Desktop\techForge\backend

# 3. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Install dependencies (takes ~5 min first time — torch is large)
pip install -r requirements.txt

# 5. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2 — React Frontend

```powershell
cd C:\Users\HP\Desktop\techForge\frontend
npm run dev
```

### Open in Browser
```
http://localhost:5173
```

---

## 📋 How to Demo (Judge Flow)

1. **Train** → Go to `Train Model` → Upload 20–30 GOOD screw photos → Click "Learn Normal" (wait ~1-3 min)
2. **Inspect** → Go to `Inspect` → Start webcam → Hold product → Click "Capture & Inspect"
3. **View heatmap** → See red areas where defects are detected
4. **Tune threshold** → Go to `Dashboard` → Drag slider → Save
5. **View history** → Go to `History` → See all inspections with rejection rate

---

## 🏗️ Architecture

```
React (Vite) ←→ FastAPI ←→ Anomalib (PatchCore + ResNet-50) ←→ SQLite
```

## 📁 Structure

```
techForge/
├── backend/
│   ├── main.py          ← FastAPI (5 routes)
│   ├── db.py            ← SQLite
│   ├── requirements.txt
│   └── ml/
│       └── model.py     ← PatchCore pipeline
└── frontend/
    └── src/
        ├── pages/
        │   ├── Dashboard.jsx  ← Stats + threshold
        │   ├── Train.jsx      ← Upload + train
        │   ├── Inspect.jsx    ← Webcam + results
        │   └── History.jsx    ← Inspection log
        └── api.js
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/train` | Upload 20-30 images, train PatchCore |
| `POST` | `/inspect` | Inspect one image → score + heatmap |
| `GET`  | `/history` | Inspection log |
| `GET`  | `/stats` | Today's rejection rate |
| `GET/POST` | `/threshold` | Get/set decision threshold |

---

## 🧠 How It Works

1. **Training**: Uploads 20-30 good images → Anomalib feeds them to PatchCore
2. **PatchCore**: Extracts CNN features (ResNet-50 backbone) from each patch → builds a memory bank of "normal" features
3. **Inference**: New image → extract features → compare to memory bank → anomaly score per patch → aggregate to scalar score
4. **Heatmap**: Spatial anomaly map blended onto original image with jet colormap
5. **Decision**: `score >= threshold → FAIL`, else `PASS`

---

## ⚡ Dataset for Testing

Download **MVTec AD** (Screw category): https://www.mvtec.com/company/research/datasets/mvtec-ad

```
Use: dataset/screw/train/good/*.png          → as training images (320 images)
Use: dataset/screw/test/scratch_head/*.png   → to demo scratch defects
Use: dataset/screw/test/manipulated_front/   → to demo manipulation defects
Use: dataset/screw/test/thread_side/         → to demo thread defects
```
