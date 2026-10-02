import os
import sys
import shutil
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
import streamlit as st
from PIL import Image

# Ensure backend directory is in Python path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml import model
import db

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="VisionQC — AI Quality Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        color: white;
    }
    .status-pass {
        color: #22c55e;
        font-weight: 800;
        font-size: 2.2rem;
    }
    .status-fail {
        color: #ef4444;
        font-weight: 800;
        font-size: 2.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
db.init_db()

# ----------------- SIDEBAR -----------------
st.sidebar.title("🔍 VisionQC")
st.sidebar.caption("AI-powered visual defect inspection using PatchCore")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🔍 Inspect Product", "🚀 Train Model", "📜 History Log", "⚙️ Settings"],
    index=1
)

st.sidebar.divider()

# Model Status badge
is_model_ready = model.is_trained()
if is_model_ready:
    st.sidebar.success("● Model Status: **Trained & Ready**")
else:
    st.sidebar.warning("○ Model Status: **Untrained**")

# Threshold Slider
current_threshold = float(db.get_config("threshold") or "0.5")
st.sidebar.subheader("Sensitivity Threshold")
threshold_val = st.sidebar.slider(
    "Defect Threshold",
    min_value=0.0,
    max_value=1.0,
    value=current_threshold,
    step=0.01,
    help="Higher threshold = more lenient (fewer fails). Lower = stricter (more fails)."
)

if threshold_val != current_threshold:
    db.set_config("threshold", str(threshold_val))
    st.sidebar.info(f"Threshold updated to {threshold_val:.2f}")

product_name = db.get_config("product_name") or "Product"
st.sidebar.caption(f"Inspecting item: **{product_name}**")


# =====================================================================
# 1. DASHBOARD
# =====================================================================
if nav_choice == "📊 Dashboard":
    st.title("📊 Inspection Dashboard")
    stats = db.get_today_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Today", stats["total"])
    with col2:
        st.metric("Passed", stats["passed"])
    with col3:
        st.metric("Rejected (Defective)", stats["failed"])
    with col4:
        st.metric("Rejection Rate", f"{stats['rejection_rate']}%")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Today's Defect Summary")
        if stats["total"] > 0:
            st.write(f"- **Average Anomaly Score:** `{stats['avg_score']:.3f}`")
            st.write(f"- **Average Confidence:** `{stats['avg_confidence']}%`")
            st.progress(stats["passed"] / stats["total"])
            st.caption(f"{stats['passed']} of {stats['total']} items passed quality check.")
        else:
            st.info("No inspections recorded today yet. Run inspections on the 'Inspect Product' tab.")

    with c2:
        st.subheader("Active Model Specs")
        st.write(f"- **Backbone Architecture:** `{model.BACKBONE}`")
        st.write(f"- **Image Resolution:** `{model.IMAGE_SIZE[0]}x{model.IMAGE_SIZE[1]}`")
        st.write(f"- **Target Product:** `{product_name}`")
        st.write(f"- **Active Threshold:** `{threshold_val:.2f}`")


# =====================================================================
# 2. INSPECT PRODUCT
# =====================================================================
elif nav_choice == "🔍 Inspect Product":
    st.title("🔍 Live Product Inspection")

    if not is_model_ready:
        st.warning("⚠️ No trained model found. Please go to **Train Model** tab to train on good product photos first.")
    else:
        sample_dir = ROOT_DIR / "images"
        sample_images = sorted(list(sample_dir.glob("*.png"))) if sample_dir.exists() else []

        tab_cam, tab_file, tab_samples = st.tabs(["📷 Webcam", "📁 Upload Image", "🖼️ Sample Screws"])
        image_to_inspect = None
        source_name = "upload.jpg"

        with tab_cam:
            camera_img = st.camera_input("Point camera at product and click capture")
            if camera_img is not None:
                image_to_inspect = Image.open(camera_img)
                source_name = "webcam_capture.jpg"

        with tab_file:
            uploaded_file = st.file_uploader("Upload product photo", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if uploaded_file is not None:
                image_to_inspect = Image.open(uploaded_file)
                source_name = uploaded_file.name

        with tab_samples:
            if sample_images:
                st.caption(f"Select one of {len(sample_images)} bundled sample screw images:")
                selected_sample = st.selectbox(
                    "Choose sample",
                    sample_images,
                    format_func=lambda p: p.name
                )
                if selected_sample and st.button("Inspect Selected Sample"):
                    image_to_inspect = Image.open(selected_sample)
                    source_name = selected_sample.name
            else:
                st.info("No sample images found in /images directory.")

        if image_to_inspect is not None:
            st.divider()
            col_img, col_res = st.columns([1, 1])

            with col_img:
                st.subheader("Inspected Image")
                st.image(image_to_inspect, use_container_width=True)

            with col_res:
                st.subheader("Analysis & Diagnosis")
                with st.spinner("Running PatchCore anomaly detection..."):
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

                        # Display PASS/FAIL badge
                        is_pass = result["result"] == "PASS"
                        badge_html = f"<div class='status-pass'>✅ PASS</div>" if is_pass else f"<div class='status-fail'>❌ DEFECT DETECTED (FAIL)</div>"
                        st.markdown(badge_html, unsafe_allow_html=True)

                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric(
                                label="Anomaly Score",
                                value=f"{result['score']:.3f}",
                                delta=f"{result['score'] - threshold_val:+.3f} vs threshold",
                                delta_color="inverse"
                            )
                        with m2:
                            st.metric(label="Confidence", value=f"{result['confidence']}%")

                        st.progress(min(max(result["score"], 0.0), 1.0))
                        st.caption(f"Score: {result['score']:.3f} | Threshold: {threshold_val:.2f}")

                        heatmap_file = Path(result["heatmap_path"])
                        if heatmap_file.exists():
                            st.subheader("Defect Heatmap Overlay")
                            st.image(str(heatmap_file), caption="Glowing red/yellow areas indicate anomalous regions", use_container_width=True)

                    except Exception as e:
                        st.error(f"Inference error: {e}")


# =====================================================================
# 3. TRAIN MODEL
# =====================================================================
elif nav_choice == "🚀 Train Model":
    st.title("🚀 Train Anomaly Detection Model")
    st.markdown("""
    PatchCore learns what a **normal, defect-free** product looks like from **20–30 good photos**.
    Any subsequent product that deviates from this learned distribution will be flagged with a heatmap.
    """)

    product_input = st.text_input("Product Name / Part Label", value=product_name)

    sample_dir = ROOT_DIR / "images"
    has_sample_dir = sample_dir.exists() and len(list(sample_dir.glob("*.png"))) >= 5

    train_mode = st.radio(
        "Choose how to load training images (Fastest to Custom):",
        [
            "⚡ Server Folder (Instant — 0s Upload Time)",
            "🗜️ Upload Folder as .ZIP (Fastest for Custom Images)",
            "📁 Drag & Drop Multiple Photos"
        ] if has_sample_dir else [
            "🗜️ Upload Folder as .ZIP (Fastest for Custom Images)",
            "📁 Drag & Drop Multiple Photos"
        ],
        index=0
    )

    uploaded_zip = None
    uploaded_train_files = None
    sample_count = 20

    if train_mode == "⚡ Server Folder (Instant — 0s Upload Time)":
        st.info("💡 **Best approach**: Images are loaded directly from the server filesystem with zero browser upload delay.")
        sample_count = st.slider("How many normal photos to train on?", 5, min(50, len(list(sample_dir.glob("*.png")))), 20)
        st.caption(f"Will learn normal product distribution from {sample_count} sample screw photos in `{sample_dir.name}/`.")

    elif train_mode == "🗜️ Upload Folder as .ZIP (Fastest for Custom Images)":
        st.info("💡 **Recommended for custom images**: Zip your folder of 15–30 good photos on your PC into a `.zip` file and drop it here. It uploads in 1 single second!")
        uploaded_zip = st.file_uploader(
            "Upload zipped folder of good product photos (.zip)",
            type=["zip"],
            key="zip_uploader"
        )
        if uploaded_zip:
            st.success(f"Loaded zip archive: `{uploaded_zip.name}` ({uploaded_zip.size / 1024:.1f} KB)")

    elif train_mode == "📁 Drag & Drop Multiple Photos":
        uploaded_train_files = st.file_uploader(
            "Upload 10–25 Good Photos (No defects)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="multi_file_uploader"
        )
        if uploaded_train_files:
            st.info(f"Selected {len(uploaded_train_files)} images for training.")

    if st.button("🚀 Learn Normal (Start Training)"):
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 1. Populate temp_dir based on chosen mode
            if train_mode == "⚡ Server Folder (Instant — 0s Upload Time)":
                samples = sorted(list(sample_dir.glob("*.png")))[:sample_count]
                for s in samples:
                    shutil.copy(s, temp_dir / s.name)

            elif train_mode == "🗜️ Upload Folder as .ZIP (Fastest for Custom Images)":
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

            elif train_mode == "📁 Drag & Drop Multiple Photos":
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
                status_text.text(f"[{pct}%] {msg}")

            with st.spinner("Extracting patch embeddings and building memory bank..."):
                train_result = model.train(temp_dir, product_name=product_input, progress=on_progress)
                db.set_config("model_trained", "true")
                db.set_config("product_name", product_input)
                db.set_config("train_image_count", str(train_result["image_count"]))
                db.set_config("trained_at", datetime.now().isoformat())

            st.success(f"🎉 Model trained successfully on {train_result['image_count']} images using backbone `{model.BACKBONE}`!")
            st.json(train_result["calibration"])

        except Exception as err:
            st.error(f"Training failed: {err}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# 4. HISTORY LOG
# =====================================================================
elif nav_choice == "📜 History Log":
    st.title("📜 Inspection History")

    filter_choice = st.selectbox("Filter records", ["ALL", "PASS", "FAIL"])
    result_filter = None if filter_choice == "ALL" else filter_choice

    history = db.get_history(limit=100, result_filter=result_filter)

    if not history:
        st.info("No inspection records found.")
    else:
        st.write(f"Showing last **{len(history)}** inspections:")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            # Table formatting
            table_data = []
            for h in history:
                table_data.append({
                    "ID": h["id"],
                    "Timestamp": h["timestamp"][:19].replace("T", " "),
                    "Result": "✅ PASS" if h["result"] == "PASS" else "❌ FAIL",
                    "Score": round(h["anomaly_score"], 3),
                    "Confidence": f"{h['confidence']}%",
                    "Threshold": round(h["threshold"], 2),
                    "Filename": h["filename"] or "-",
                })
            st.dataframe(table_data, use_container_width=True)

        with col_right:
            st.subheader("Inspection Details")
            selected_id = st.selectbox("Inspect entry ID", [h["id"] for h in history])
            selected_row = next((h for h in history if h["id"] == selected_id), None)
            if selected_row:
                st.write(f"**Result:** `{selected_row['result']}`")
                st.write(f"**Score:** `{selected_row['anomaly_score']:.4f}`")
                st.write(f"**Confidence:** `{selected_row['confidence']}%`")
                if selected_row["heatmap_path"] and Path(selected_row["heatmap_path"]).exists():
                    st.image(selected_row["heatmap_path"], caption=f"Heatmap #{selected_row['id']}", use_container_width=True)

        st.divider()
        if st.button("🗑️ Clear History"):
            db.clear_history()
            st.success("History cleared.")
            st.rerun()


# =====================================================================
# 5. SETTINGS
# =====================================================================
elif nav_choice == "⚙️ Settings":
    st.title("⚙️ System & Model Configuration")

    st.write("### Model Information")
    st.write(f"- **Backbone Architecture:** `{model.BACKBONE}`")
    st.write(f"- **Coreset Subsampling Ratio:** `{model.CORESET_RATIO}`")
    st.write(f"- **Batch Size:** `{model.BATCH_SIZE}`")
    st.write(f"- **Model Trained:** `{db.get_config('model_trained')}`")
    st.write(f"- **Product Label:** `{db.get_config('product_name')}`")
    st.write(f"- **Trained At:** `{db.get_config('trained_at') or 'Never'}`")

    st.divider()

    st.write("### Reset Model")
    st.write("Clearing the model removes trained memory banks and cached weights.")
    if st.button("⚠️ Reset Trained Model", type="primary"):
        model.reset_model()
        db.set_config("model_trained", "false")
        st.warning("Model reset. Please train the model again.")
        st.rerun()
