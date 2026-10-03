# VisionQC — AI Industrial Visual Quality Inspection

> **Unsupervised Visual Quality Inspection System**  
> Learns "normal" from 20–30 good product photos → detects defects in real time from webcam or image uploads → displays localization heatmaps + confidence scores + supervisor-tunable PASS/FAIL decisions → logs production metrics and tracks today's rejection rate.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://visionqc.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌐 Live Demo

- **Live Cloud Web App**: **[https://visionqc.streamlit.app](https://visionqc.streamlit.app)**
- **GitHub Repository**: [https://github.com/tanishka019/VisionQC](https://github.com/tanishka019/VisionQC)

---

## 📌 Project Overview

In industrial manufacturing, identifying defective products on the assembly line is critical to preventing costly recalls and maintaining high quality standards. Traditional computer vision and supervised deep learning models require thousands of labeled defect examples—an impractical requirement because defects are rare, unpredictable, and diverse.

**VisionQC** addresses this challenge using an **unsupervised anomaly detection paradigm (PatchCore)**:
1. **Cold-Start Ready**: Learns what a defect-free product looks like using only **20–30 photos of normal parts**—zero defect photos or bounding box annotations required.
2. **Defect Localization**: Pinpoints unexpected surface scratches, dents, missing threads, and foreign objects with high-precision **heatmaps**.
3. **Supervisor Control**: Equips line managers with an interactive **sensitivity threshold slider** to tune tolerance limits on the fly.
4. **Quality Accountability**: Automatically logs every inspection with timestamps, scores, and images to compute **real-time rejection rates** and maintain an audit trail.

---

## ✨ Key Features

- **🧠 Few-Shot Anomaly Learning**: Memory-bank PatchCore extracts deep patch representations from a pre-trained backbone (`ResNet-18` / `WideResNet-50`) without requiring neural network retraining.
- **🔥 Deviation Heatmaps**: Draws a warm heat overlay only where the part is more unusual than anything seen on the good training photos, so normal areas stay completely clear. A FAIL result also gets a thin outline around the hottest region.
- **🎯 Real-Time PASS / FAIL Decision & Confidence**: Computes an anomaly score scaled against cross-validated calibration scores, paired with an exact confidence percentage.
- **⚙️ Supervisor-Tunable Threshold**: Provides an interactive sensitivity slider ($0.00$ to $1.00$) to dynamically control strictness without downtime.
- **📈 Live Production Rejection Rate**: Automatically tracks total inspected units, passed count, rejected count, and today's rejection percentage across production shifts.
- **📷 Multi-Modal Inspection Input**:
  - Live webcam capture with interactive snapshot.
  - Single or batch file upload (`JPG`, `PNG`, `WEBP`, `BMP`).
- **📜 Inspection History & Audit Log**: Searchable history table with PASS/FAIL filters, inspection IDs, timestamps, and one-click heatmap review.

---

## 🏗️ Architecture & Workflow

### High-Level Architecture

```
[ Industrial Camera / Webcam / File Upload ]
                     │
                     ▼
       ┌───────────────────────────┐
       │   VisionQC Core Engine    │
       │ (ResNet Backbone Extractor)│
       └─────────────┬─────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  [ Training Mode ]       [ Inspection Mode ]
         │                       │
 20-30 Normal Photos      Extract Patch Vectors
         │                       │
 Coreset Subsampling      Nearest-Neighbor Search
         │                against Memory Bank
         ▼                       │
   [Memory Bank]                 ▼
         ▲            ┌─────────────────────┐
         └────────────┤ Anomaly Score & Map ├────┐
                      └─────────────────────┘    │
                                                 ▼
                                     ┌───────────────────────┐
                                     │ Threshold Comparator  │
                                     │ (Supervisor Setpoint) │
                                     └───────────┬───────────┘
                                                 │
                               ┌─────────────────┴─────────────────┐
                               ▼                                   ▼
                       [ ✅ PASS ]                            [ ❌ FAIL ]
                               │                                   │
                               └─────────────────┬─────────────────┘
                                                 ▼
                                     ┌───────────────────────┐
                                     │ SQLite Audit Database │
                                     │ (Today's Reject Rate) │
                                     └───────────────────────┘
```

### Detailed Inspection & Scoring Workflow

1. **Feature Extraction**: A single long part on a plain background is first rotated and cropped to one pose. Images are then normalized to ImageNet statistics and passed through two convolutional layers (`layer2`, `layer3`) of a Wide ResNet-50, which gave the best accuracy on real screws (see *Measuring accuracy*).
2. **Coreset Sampling**: A greedy minimax facility location algorithm selects the most informative patch embeddings, constructing a lightweight memory bank ($10\%$ sampling ratio).
3. **Calibration (cross-validation)**: The training images are split into 5 folds, and every image is scored against a memory bank built from the *other* folds. These unbiased "unseen good part" scores set the scale so that a typical good part scores $\approx 0.0$ and the most unusual good part scores $\approx 0.5$. (Calibrating on images the bank was built from makes the scale far too tight and rejects good parts.)
4. **Heatmap Generation**: The per-pixel anomaly map is calibrated against the good training photos (nothing is drawn below the hottest normal pixel), mapped to a vermilion-to-amber ramp and blended onto the original image with opacity that follows intensity. On a benchmark with defects drawn at known positions this raised the share of drawn heat that lands on the defect from about 12% to about 59%, and cut haze on good photos about 50-fold.
5. **Persistence**: SQLite records the inspection parameters, score, verdict, threshold, and artifact paths.

---

## 💻 Technology Stack

| Layer | Technologies | Purpose |
|---|---|---|
| **AI / Machine Learning** | PyTorch 2.3+, Torchvision, Anomalib 1.1, Scikit-learn, NumPy | PatchCore anomaly detection, feature extraction, coreset subsampling |
| **Vision Backbone** | ResNet-18 (Default, optimized for cloud CPU) / WideResNet-50-2 | Feature representation pre-trained on ImageNet |
| **Interactive Web App** | Streamlit 1.35+ | Standalone cloud web application, camera integration, real-time metrics |
| **REST / WebSocket API** | FastAPI, Uvicorn, Pydantic, Starlette | High-performance backend API with SSE & WebSockets |
| **Frontend UI (Optional)** | React 18, Vite, TailwindCSS / Vanilla CSS | Full-featured dual-service web client |
| **Data & Storage** | SQLite3, Pillow (PIL), Matplotlib | Local relational logging and defect heatmap rendering |
| **DevOps & Containerization** | Docker, Docker Compose, GitHub Actions | Multi-stage container builds, CI linting, and automated testing |

---

## 🚀 Setup & Installation Instructions

### Option 1: Run the Streamlit Application (Recommended & Simplest)

Requires **Python 3.11**.

```bash
# 1. Clone the repository
git clone https://github.com/tanishka019/VisionQC.git
cd VisionQC

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Streamlit
streamlit run streamlit_app.py
```
Open **http://localhost:8501** in your browser.

---

### Option 2: Run Full-Stack (FastAPI Backend + React Frontend)

#### Terminal 1 — FastAPI Backend:
```bash
cd backend
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1 | Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
API Documentation will be available at **http://localhost:8000/docs**.

#### Terminal 2 — React Frontend:
```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```
Open **http://localhost:5173** in your browser.

---

### Option 3: Run with Docker Compose

```bash
docker compose up --build
```
- Frontend: **http://localhost:5173**
- Backend API: **http://localhost:8000**
- Data persistence: Docker volume `visionqc-data`

---

## 🔌 API Information

When running the FastAPI backend (`backend/main.py`), the following REST endpoints and WebSocket channels are exposed:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check, model status, and inspection count |
| `GET` | `/model/status` | Current training state, backbone architecture, and product label |
| `DELETE` | `/model/reset` | Clears trained model artifacts and resets memory bank |
| `POST` | `/train` | Multipart upload of $\ge 5$ good photos to train PatchCore |
| `GET` | `/train/progress` | Server-Sent Events (SSE) live progress stream |
| `POST` | `/inspect` | Upload single product image → returns score, confidence, PASS/FAIL, and heatmap URL |
| `POST` | `/inspect/batch` | Upload multiple images for batch inspection |
| `GET` | `/stats` | Aggregated statistics: total, pass/fail counts, hourly breakdown, and rejection rate |
| `GET` / `POST` | `/threshold` | Read or update the decision sensitivity threshold |
| `GET` | `/history` | Paginated inspection history with optional `?filter=PASS\|FAIL` |
| `DELETE` | `/history/{id}` | Deletes a single inspection and its associated image files |
| `WS` | `/ws/live` | WebSocket channel broadcasting inspection events in real time |

---

## 📊 Dataset Information

VisionQC is validated against the industry-standard **MVTec Anomaly Detection (MVTec AD)** benchmark dataset.

- **Dataset Source**: [MVTec AD Benchmark](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **Primary Category**: **Screws** (includes defects such as *scratch_head*, *scratch_neck*, *thread_side*, and *manipulated_front*).

---

## 📷 Screenshots & Demo Information

### Evaluation / Judge Demo Flow:

1. **Dashboard Overview**: Check the sidebar for **Today's Production KPIs** (Total Inspections, Pass Count, Reject Count, Rejection Rate %).
2. **Train Model**:
   - Navigate to **🚀 Train Model**.
   - Upload about 20 good product photos.
   - Click **🚀 Learn Normal**. The progress bar updates live as ResNet extracts patch embeddings and builds the memory bank (~20 seconds).
3. **Inspect Product**:
   - Navigate to **🔍 Inspect Product**.
   - Upload a photo or use the webcam.
4. **Analyze Results**:
   - **PASS/FAIL Banner**: Bold color-coded verdict.
   - **Metric Strip**: Anomaly Score, Supervisor Threshold, Deviation vs Limit, and Confidence %.
   - **Deviation Heatmap**: Side-by-side view highlighting anomalous regions in glowing red/yellow.
5. **Tune Supervisor Threshold**:
   - Move the **Defect Sensitivity Threshold** slider in the sidebar (e.g. from `0.50` to `0.30`).
   - Notice how sensitivity shifts the rejection boundary.
6. **Audit Trail**:
   - Navigate to **📜 History Log** to view all recorded inspections, filter by PASS/FAIL, and inspect saved defect overlays.

---

## ⚠️ Limitations & Future Scope

### Current Limitations:
- **Pose Alignment Sensitivity**: PatchCore performs best when parts are positioned with consistent orientation and lighting; heavy rotation requires data augmentation.
- **CPU Inference Overhead**: On low-tier cloud CPU instances, inference takes $\approx 150\text{–}300\text{ ms}$ per image compared to $<20\text{ ms}$ on CUDA GPUs.
- **Ephemeral Cloud Storage**: On free serverless platforms, local SQLite records and cached weights reset upon container reboot unless an external persistent volume or database is attached.

### Future Scope:
- **Edge Deployment**: Exporting backbone feature extractors to **ONNX / TensorRT / OpenVINO** for sub-10ms execution on NVIDIA Jetson and Raspberry Pi industrial gateways.
- **Defect Classification**: Adding a downstream secondary classifier to label defect types (e.g., distinguishing a "thread defect" from a "surface scratch").
- **PLC / Industrial Camera Integration**: Native support for GigE Vision and GenICam protocols to trigger frame captures from optical sensors on assembly lines.
- **MES / ERP Webhooks**: Automated alerts via Slack, Microsoft Teams, and webhook integration into SAP / FactoryTalk upon sustained rejection rate spikes.

---

## 👥 Team Members

Developed for the **TechForge Hackathon**:

1. **Tanishka Dhanudharmi**
2. **Anvesha Thakur**
3. **Lavanya Patil**
4. **Prarthna Purohit**

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Measuring accuracy

`python backend/eval/evaluate.py DATA_DIR` trains on `DATA_DIR/train_good/` and tests on `test_good/` and `test_bad/`
(defect type = file-name prefix). It prints AUROC, the catch rate and false-reject rate at several thresholds, the best
threshold, and a per-defect breakdown. On MVTec AD "screw" (25 training photos): AUROC 0.86, 81% of defects caught and
15% of good parts rejected at threshold 0.3. Parts that are one long object on a plain background are automatically
rotated and cropped to a single pose before scoring (`VISIONQC_ALIGN=0` turns this off).
