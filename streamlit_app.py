import os
import sys
import shutil
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
import streamlit as st
from PIL import Image, ImageDraw

# Ensure backend directory is in Python path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml import model
import db

# ----------------- PAGE CONFIG -----------------
def _page_icon() -> Image.Image:
    """Viewfinder mark (matches the web UI), drawn at runtime so it works on any Streamlit version."""
    size, pad, arm, w = 64, 8, 18, 6
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ink = (18, 18, 18, 255)
    for x0, x1 in ((pad, pad + arm), (size - pad - arm, size - pad)):
        for y in (pad, size - pad - w):
            d.rectangle([x0, y, x1, y + w - 1], fill=ink)
    for x in (pad, size - pad - w):
        for y0, y1 in ((pad, pad + arm), (size - pad - arm, size - pad)):
            d.rectangle([x, y0, x + w - 1, y1], fill=ink)
    c = size // 2
    d.ellipse([c - 6, c - 6, c + 6, c + 6], fill=ink)
    return img


st.set_page_config(
    page_title="VisionQC",
    page_icon=_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design tokens mirror the React UI: neutral paper + ink, with green / vermilion only for PASS / FAIL.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #f5f5f3; --surface: #ffffff; --sunken: #fafaf8;
        --ink: #121212; --ink-2: #4a4a47; --ink-3: #8a8a86;
        --line: #e4e4e0; --line-strong: #cfcfca;
        --pass: #1b7f4b; --fail: #d1381b;
        --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }

    html, body, .stApp, .stMarkdown, p, label, li, button, input, textarea, h1, h2, h3, h4,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-baseweb="tab"] {
        font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    }

    /* Chrome we don't need */
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }
    [data-testid="stHeader"] { background: transparent; }

    .block-container { max-width: 1160px; padding-top: 2.75rem; padding-bottom: 4rem; }

    /* Type */
    h1 { font-size: 1.75rem !important; font-weight: 600 !important; letter-spacing: -0.03em !important; line-height: 1.15 !important; padding: 0 0 .25rem !important; }
    h2, h3 { font-weight: 600 !important; letter-spacing: -0.015em !important; }
    h3 { font-size: 1.05rem !important; }
    .page-sub { color: var(--ink-3); margin: -0.25rem 0 1.75rem; font-size: .95rem; }
    .label { font-size: .72rem; font-weight: 500; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3); }
    .mono { font-family: var(--mono); }

    /* Sidebar */
    [data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1.5rem; }
    .brand { display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 1.05rem; letter-spacing: -0.02em; margin-bottom: 1.5rem; }
    .model-chip { display: flex; align-items: center; gap: 8px; font-size: .82rem; color: var(--ink-2); margin: .25rem 0 1.25rem; }
    .model-chip .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--line-strong); }
    .model-chip .dot.on { background: var(--pass); }

    /* Metrics as quiet figures */
    [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; }
    [data-testid="stMetricLabel"] p { font-size: .72rem; font-weight: 500; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3); }
    [data-testid="stMetricValue"] { font-size: 2.1rem; font-weight: 500; letter-spacing: -0.04em; font-variant-numeric: tabular-nums; }
    [data-testid="stMetricDelta"] { font-size: .78rem; }

    /* Controls */
    .stButton > button, .stDownloadButton > button { border-radius: 6px; font-weight: 500; height: 2.5rem; }
    [data-testid="stFileUploaderDropzone"] { border: 1px dashed var(--line-strong); border-radius: 8px; background: var(--surface); }
    [data-baseweb="tab"] { font-weight: 500; }
    [data-testid="stAlert"] { border-radius: 6px; }
    [data-testid="stImage"] img { border: 1px solid var(--line); border-radius: 0; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--ink-3) !important; opacity: 1 !important; }
    code { color: var(--ink) !important; background: var(--sunken) !important; border: 1px solid var(--line); border-radius: 4px; font-family: var(--mono) !important; font-size: .82em !important; }
    hr { border-color: var(--line) !important; }

    /* Verdict + gauge */
    .verdict { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 24px; margin-bottom: 1rem; }
    .verdict-word { font-size: 3.2rem; font-weight: 600; letter-spacing: -0.05em; line-height: 1; margin-top: 10px; }
    .verdict-word.pass { color: var(--pass); }
    .verdict-word.fail { color: var(--fail); }
    .verdict-sub { margin-top: 10px; color: var(--ink-2); font-size: .92rem; }
    .gauge { margin-top: 30px; padding-top: 14px; }
    .gauge-track { position: relative; height: 6px; background: var(--line); border-radius: 3px; }
    .gauge-fill { height: 100%; border-radius: 3px; }
    .gauge-fill.pass { background: var(--pass); }
    .gauge-fill.fail { background: var(--fail); }
    .gauge-limit { position: absolute; top: -5px; bottom: -5px; width: 2px; background: var(--ink); }
    .gauge-limit span { position: absolute; bottom: calc(100% + 4px); left: 50%; transform: translateX(-50%); font-family: var(--mono); font-size: .66rem; color: var(--ink-2); white-space: nowrap; }
    .gauge-scale { display: flex; justify-content: space-between; margin-top: 8px; font-family: var(--mono); font-size: .68rem; color: var(--ink-3); }

    /* Key / value rows */
    .rows { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 4px 18px; }
    .rows > div { display: flex; justify-content: space-between; gap: 12px; padding: 12px 0; border-bottom: 1px solid var(--line); font-size: .9rem; }
    .rows > div:last-child { border-bottom: 0; }
    .rows span:first-child { color: var(--ink-3); }
    .rows span:last-child { font-weight: 500; font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)

# Initialize database
db.init_db()

# ----------------- HELPERS -----------------
MARK_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#121212" stroke-width="2.2" '
    'stroke-linecap="square"><path d="M3 8V3h5M16 3h5v5M21 16v5h-5M8 21H3v-5"/>'
    '<circle cx="12" cy="12" r="2.4" fill="#121212" stroke="none"/></svg>'
)


def page_header(title: str, sub: str = ""):
    st.title(title)
    if sub:
        st.markdown(f"<p class='page-sub'>{sub}</p>", unsafe_allow_html=True)


def rows_html(pairs) -> str:
    """Key / value list rendered as quiet hairline rows."""
    body = "".join(f"<div><span>{k}</span><span class='mono'>{v}</span></div>" for k, v in pairs)
    return f"<div class='rows'>{body}</div>"


def verdict_html(result: dict, threshold: float) -> str:
    is_pass = result["result"] == "PASS"
    kind = "pass" if is_pass else "fail"
    score = min(max(result["score"], 0.0), 1.0)
    limit = min(max(threshold, 0.0), 1.0)
    sub = "Meets the quality standard." if is_pass else "Defect detected. Reject this part."
    return (
        "<div class='verdict'>"
        "<div class='label'>Verdict</div>"
        f"<div class='verdict-word {kind}'>{result['result']}</div>"
        f"<div class='verdict-sub'>{sub}</div>"
        "<div class='gauge'><div class='gauge-track'>"
        f"<div class='gauge-fill {kind}' style='width:{score * 100:.1f}%'></div>"
        f"<div class='gauge-limit' style='left:{limit * 100:.1f}%'><span>limit {threshold:.2f}</span></div>"
        "</div><div class='gauge-scale'><span>0</span><span>0.5</span><span>1</span></div></div>"
        "</div>"
    )


# Navigation labels (compared below)
NAV_OVERVIEW, NAV_INSPECT, NAV_TRAIN, NAV_HISTORY, NAV_SETTINGS = "Overview", "Inspect", "Train", "History", "Settings"

# ----------------- SIDEBAR -----------------
st.sidebar.markdown(f"<div class='brand'>{MARK_SVG}<span>VisionQC</span></div>", unsafe_allow_html=True)

nav_choice = st.sidebar.radio(
    "Navigation",
    [NAV_OVERVIEW, NAV_INSPECT, NAV_TRAIN, NAV_HISTORY, NAV_SETTINGS],
    index=1,
    label_visibility="collapsed",
)

st.sidebar.divider()

product_name = db.get_config("product_name") or "Product"

# Model status
is_model_ready = model.is_trained()
if is_model_ready:
    st.sidebar.markdown(
        f"<div class='model-chip'><span class='dot on'></span><span>Model ready &middot; <span class='mono'>{product_name}</span></span></div>",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown("<div class='model-chip'><span class='dot'></span><span>No model trained</span></div>", unsafe_allow_html=True)

# Threshold Slider
current_threshold = float(db.get_config("threshold") or "0.5")
threshold_val = st.sidebar.slider(
    "Detection threshold",
    min_value=0.0,
    max_value=1.0,
    value=current_threshold,
    step=0.01,
    help="Higher threshold = more lenient (fewer fails). Lower = stricter (more fails).",
)

if threshold_val != current_threshold:
    db.set_config("threshold", str(threshold_val))
    st.sidebar.caption(f"Threshold updated to {threshold_val:.2f}")


# =====================================================================
# 1. OVERVIEW
# =====================================================================
if nav_choice == NAV_OVERVIEW:
    page_header("Overview", "Today's inspections at a glance.")
    stats = db.get_today_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Inspected", stats["total"])
    with col2:
        st.metric("Passed", stats["passed"])
    with col3:
        st.metric("Failed", stats["failed"])
    with col4:
        st.metric("Rejection", f"{stats['rejection_rate']}%")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Today's summary")
        if stats["total"] > 0:
            st.markdown(
                rows_html([
                    ("Average anomaly score", f"{stats['avg_score']:.3f}"),
                    ("Average confidence", f"{stats['avg_confidence']}%"),
                    ("Passed quality check", f"{stats['passed']} of {stats['total']}"),
                ]),
                unsafe_allow_html=True,
            )
            st.progress(stats["passed"] / stats["total"])
        else:
            st.info("No inspections recorded today yet. Run one from the Inspect page.")

    with c2:
        st.subheader("Active model")
        st.markdown(
            rows_html([
                ("Backbone", model.BACKBONE),
                ("Image resolution", f"{model.IMAGE_SIZE[0]}x{model.IMAGE_SIZE[1]}"),
                ("Product", product_name),
                ("Threshold", f"{threshold_val:.2f}"),
            ]),
            unsafe_allow_html=True,
        )


# =====================================================================
# 2. INSPECT
# =====================================================================
elif nav_choice == NAV_INSPECT:
    page_header("Inspect", "Check a part against the learned normal.")

    if not is_model_ready:
        st.warning("No trained model yet. Open the Train page and add photos of good parts first.")
    else:
        sample_dir = ROOT_DIR / "images"
        sample_images = sorted(list(sample_dir.glob("*.png"))) if sample_dir.exists() else []

        tab_cam, tab_file, tab_samples = st.tabs(["Camera", "Upload", "Samples"])
        image_to_inspect = None
        source_name = "upload.jpg"

        with tab_cam:
            camera_img = st.camera_input("Point the camera at the part and capture")
            if camera_img is not None:
                image_to_inspect = Image.open(camera_img)
                source_name = "webcam_capture.jpg"

        with tab_file:
            uploaded_file = st.file_uploader("Upload a product photo", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if uploaded_file is not None:
                image_to_inspect = Image.open(uploaded_file)
                source_name = uploaded_file.name

        with tab_samples:
            if sample_images:
                st.caption(f"{len(sample_images)} bundled sample screw photos.")
                selected_sample = st.selectbox(
                    "Sample photo",
                    sample_images,
                    format_func=lambda p: p.name
                )
                if selected_sample and st.button("Inspect selected sample"):
                    image_to_inspect = Image.open(selected_sample)
                    source_name = selected_sample.name
            else:
                st.info("No sample images found in the images folder.")

        if image_to_inspect is not None:
            st.write("")
            with st.spinner("Analysing image..."):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image_to_inspect.convert("RGB").save(tmp.name, "JPEG", quality=95)
                    temp_img_path = Path(tmp.name)

                try:
                    result = model.inspect(temp_img_path, threshold=threshold_val)

                    # Save inspection record to SQLite
                    db.log_inspection(
                        image_path=str(temp_img_path),
                        heatmap_path=result["heatmap_path"],
                        score=result["score"],
                        confidence=result["confidence"],
                        result=result["result"],
                        threshold=threshold_val,
                        filename=source_name,
                    )
                except Exception as e:
                    result = None
                    st.error(f"Inference error: {e}")

            if result is not None:
                col_main, col_side = st.columns([1.7, 1], gap="large")

                with col_main:
                    st.subheader("Deviation map")
                    img_a, img_b = st.columns(2)
                    with img_a:
                        st.caption("ORIGINAL")
                        st.image(image_to_inspect, use_container_width=True)
                    with img_b:
                        st.caption("HEATMAP")
                        heatmap_file = Path(result["heatmap_path"])
                        if heatmap_file.exists():
                            st.image(str(heatmap_file), use_container_width=True)
                    st.caption("Warm areas differ from what a good part looks like.")

                with col_side:
                    st.markdown(verdict_html(result, threshold_val), unsafe_allow_html=True)
                    st.markdown(
                        rows_html([
                            ("Anomaly score", f"{result['score']:.3f}"),
                            ("Confidence", f"{result['confidence']}%"),
                            ("Threshold", f"{threshold_val:.2f}"),
                            ("Margin", f"{result['score'] - threshold_val:+.3f}"),
                        ]),
                        unsafe_allow_html=True,
                    )


# =====================================================================
# 3. TRAIN
# =====================================================================
elif nav_choice == NAV_TRAIN:
    page_header("Train", "Show VisionQC what a good part looks like. No defect photos needed.")
    st.markdown("""
    PatchCore learns what a **normal, defect-free** product looks like from **20–30 good photos**.
    Anything that deviates from it is flagged with a heatmap.
    """)

    product_input = st.text_input("Product name", value=product_name)

    sample_dir = ROOT_DIR / "images"
    has_sample_dir = sample_dir.exists() and len(list(sample_dir.glob("*.png"))) >= 5

    MODE_SERVER = "Sample folder (instant)"
    MODE_ZIP    = "Upload a .zip folder"
    MODE_DROP   = "Upload individual photos"

    train_mode = st.radio(
        "Training photos",
        [MODE_SERVER, MODE_ZIP, MODE_DROP] if has_sample_dir else [MODE_ZIP, MODE_DROP],
        index=0,
        horizontal=True,
    )

    uploaded_zip = None
    uploaded_train_files = None
    sample_count = 20

    if train_mode == MODE_SERVER:
        st.caption("Photos are read straight from the server, so there is nothing to upload.")
        sample_count = st.slider("How many sample photos to train on", 5, min(50, len(list(sample_dir.glob("*.png")))), 20)
        st.caption(f"VisionQC will learn from {sample_count} photos in `{sample_dir.name}/`.")

    elif train_mode == MODE_ZIP:
        st.caption("Zip a folder of 15–30 good photos and drop it here. A single zip uploads much faster than many files.")
        uploaded_zip = st.file_uploader(
            "Zipped folder of good product photos",
            type=["zip"],
            key="zip_uploader"
        )
        if uploaded_zip:
            st.success(f"Loaded `{uploaded_zip.name}` ({uploaded_zip.size / 1024:.1f} KB)")

    elif train_mode == MODE_DROP:
        uploaded_train_files = st.file_uploader(
            "10–25 photos of good parts (no defects)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="multi_file_uploader"
        )
        if uploaded_train_files:
            st.info(f"{len(uploaded_train_files)} photos selected.")

    if st.button("Learn normal", type="primary"):
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 1. Populate temp_dir based on chosen mode
            if train_mode == MODE_SERVER:
                samples = sorted(list(sample_dir.glob("*.png")))[:sample_count]
                for s in samples:
                    shutil.copy(s, temp_dir / s.name)

            elif train_mode == MODE_ZIP:
                if not uploaded_zip:
                    st.error("Please upload a .zip file containing your product photos first.")
                    st.stop()
                with zipfile.ZipFile(uploaded_zip, "r") as z:
                    for filename in z.namelist():
                        ext = Path(filename).suffix.lower()
                        if ext in model.IMAGE_EXTS and not Path(filename).name.startswith("."):
                            # Extract clean basename to temp_dir
                            target = temp_dir / Path(filename).name
                            with z.open(filename) as src, open(target, "wb") as dst:
                                dst.write(src.read())

            elif train_mode == MODE_DROP:
                if not uploaded_train_files or len(uploaded_train_files) < 5:
                    st.error("Please select at least 5 images (15–20 recommended) to train.")
                    st.stop()
                for uf in uploaded_train_files:
                    img = Image.open(uf)
                    img.convert("RGB").save(temp_dir / uf.name)

            progress_bar = st.progress(0)
            status_text = st.empty()

            def on_progress(pct: int, msg: str):
                progress_bar.progress(min(pct, 100))
                status_text.caption(f"{pct}%  ·  {msg}")

            with st.spinner("Learning what normal looks like..."):
                train_result = model.train(temp_dir, product_name=product_input, progress=on_progress)
                db.set_config("model_trained", "true")
                db.set_config("product_name", product_input)
                db.set_config("train_image_count", str(train_result["image_count"]))
                db.set_config("trained_at", datetime.now().isoformat())

            st.success(f"Model trained successfully on {train_result['image_count']} images using `{model.BACKBONE}`.")
            with st.expander("Calibration details"):
                st.json(train_result["calibration"])

        except Exception as err:
            st.error(f"Training failed: {err}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# 4. HISTORY
# =====================================================================
elif nav_choice == NAV_HISTORY:
    page_header("History", "Every inspection, newest first.")

    filter_choice = st.radio("Filter", ["ALL", "PASS", "FAIL"], horizontal=True, label_visibility="collapsed")
    result_filter = None if filter_choice == "ALL" else filter_choice

    history = db.get_history(limit=100, result_filter=result_filter)

    if not history:
        st.info("No inspection records found.")
    else:
        st.caption(f"Showing the last {len(history)} inspections.")

        col_left, col_right = st.columns([2, 1], gap="large")

        with col_left:
            table_data = []
            for h in history:
                table_data.append({
                    "ID": h["id"],
                    "Time": h["timestamp"][:19].replace("T", " "),
                    "Result": h["result"],
                    "Score": round(h["anomaly_score"], 3),
                    "Confidence": f"{h['confidence']}%",
                    "Threshold": round(h["threshold"], 2),
                    "File": h["filename"] or "-",
                })
            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.NumberColumn(format="%.3f"),
                    "Threshold": st.column_config.NumberColumn(format="%.2f"),
                },
            )

        with col_right:
            st.subheader("Details")
            selected_id = st.selectbox("Inspection ID", [h["id"] for h in history])
            selected_row = next((h for h in history if h["id"] == selected_id), None)
            if selected_row:
                st.markdown(
                    rows_html([
                        ("Result", selected_row["result"]),
                        ("Score", f"{selected_row['anomaly_score']:.4f}"),
                        ("Confidence", f"{selected_row['confidence']}%"),
                    ]),
                    unsafe_allow_html=True,
                )
                if selected_row["heatmap_path"] and Path(selected_row["heatmap_path"]).exists():
                    st.write("")
                    st.image(selected_row["heatmap_path"], caption=f"Heatmap #{selected_row['id']}", use_container_width=True)

        st.divider()
        if st.button("Clear history"):
            db.clear_history()
            st.success("History cleared.")
            st.rerun()


# =====================================================================
# 5. SETTINGS
# =====================================================================
elif nav_choice == NAV_SETTINGS:
    page_header("Settings", "Model configuration and reset.")

    st.subheader("Model")
    st.markdown(
        rows_html([
            ("Backbone", model.BACKBONE),
            ("Coreset subsampling ratio", model.CORESET_RATIO),
            ("Batch size", model.BATCH_SIZE),
            ("Model trained", db.get_config("model_trained")),
            ("Product label", db.get_config("product_name")),
            ("Trained at", db.get_config("trained_at") or "Never"),
        ]),
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Reset model")
    st.caption("Clearing the model removes the trained memory bank. You will need to train again.")
    if st.button("Reset trained model"):
        model.reset_model()
        db.set_config("model_trained", "false")
        st.warning("Model reset. Please train the model again.")
        st.rerun()
