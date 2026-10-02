// src/pages/Train.jsx
import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { trainModel } from "../api";

const MIN_IMAGES = 5;  // lower for dev; ideally 20
const IDEAL_MIN  = 20;

export default function Train({ onTrained }) {
  const [files,        setFiles]       = useState([]);
  const [previews,     setPreviews]    = useState([]);
  const [productName,  setProductName] = useState("Screw");
  const [status,       setStatus]      = useState("idle"); // idle | training | success | error
  const [message,      setMessage]     = useState("");

  const onDrop = useCallback((accepted) => {
    const imgs = accepted.filter((f) => f.type.startsWith("image/"));
    setFiles((prev) => {
      const next = [...prev, ...imgs].slice(0, 50);
      // generate previews
      next.forEach((f) => {
        if (!f.preview) f.preview = URL.createObjectURL(f);
      });
      setPreviews(next.map((f) => f.preview));
      return next;
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    multiple: true,
  });

  const removeFile = (idx) => {
    setFiles((prev) => {
      const next = [...prev];
      URL.revokeObjectURL(next[idx].preview);
      next.splice(idx, 1);
      setPreviews(next.map((f) => f.preview));
      return next;
    });
  };

  const handleTrain = async () => {
    if (files.length < MIN_IMAGES) return;
    setStatus("training");
    setMessage("");
    try {
      const res = await trainModel(files, productName || "product");
      setMessage(`✅ ${res.data.message}`);
      setStatus("success");
      onTrained && onTrained();
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Unknown error";
      setMessage(`❌ Training failed: ${msg}`);
      setStatus("error");
    }
  };

  const reset = () => {
    files.forEach((f) => URL.revokeObjectURL(f.preview));
    setFiles([]);
    setPreviews([]);
    setStatus("idle");
    setMessage("");
  };

  const readyCount = files.length;
  const isReady    = readyCount >= MIN_IMAGES && status !== "training";

  return (
    <div style={{ maxWidth: 860, margin: "0 auto" }}>

      {/* Step progress */}
      <div className="card mb-4">
        <div className="card-body" style={{ padding: "14px 22px" }}>
          <div style={{ display: "flex", gap: 0 }}>
            {["Upload Images", "Name Product", "Learn Normal"].map((s, i) => (
              <div key={s} style={{ flex: 1, display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{
                  width: 28, height: 28,
                  borderRadius: "50%",
                  background: i === 0 && status !== "success" ? "var(--primary)" :
                              i === 1 && files.length >= MIN_IMAGES && status !== "success" ? "var(--primary)" :
                              i === 2 && status === "success" ? "var(--success)" :
                              "var(--border)",
                  color: "#fff",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: ".75rem", fontWeight: 700, flexShrink: 0,
                }}>
                  {i === 2 && status === "success" ? "✓" : i + 1}
                </div>
                <span style={{ fontSize: ".8rem", color: "var(--text-muted)", fontWeight: 500 }}>{s}</span>
                {i < 2 && (
                  <div style={{ flex: 1, height: 2, background: "var(--border)", margin: "0 8px" }} />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {status === "training" && (
        <div className="card mb-4">
          <div className="card-body train-progress">
            <div className="spinner" />
            <h3 style={{ fontWeight: 700 }}>Training PatchCore…</h3>
            <p className="text-muted">This may take 1–5 minutes depending on your hardware.</p>
            <p className="text-muted" style={{ fontSize: ".78rem" }}>
              ResNet-50 feature extraction + memory bank construction
            </p>
          </div>
        </div>
      )}

      {status === "success" && (
        <div className="card mb-4" style={{ borderColor: "var(--success)" }}>
          <div className="card-body" style={{ textAlign: "center", padding: 32 }}>
            <div style={{ fontSize: "3rem", marginBottom: 10 }}>🎉</div>
            <h3 style={{ fontWeight: 800, color: "var(--success)", marginBottom: 6 }}>Model Ready!</h3>
            <p className="text-muted" style={{ marginBottom: 20 }}>{message}</p>
            <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
              <a className="btn btn-primary" href="/inspect">Start Inspecting →</a>
              <button className="btn btn-outline" onClick={reset}>Train Another</button>
            </div>
          </div>
        </div>
      )}

      {status !== "success" && (
        <>
          {/* Product name */}
          <div className="card mb-4">
            <div className="card-header"><h3>Product Name</h3></div>
            <div className="card-body">
              <input
                id="product-name-input"
                className="input"
                placeholder="e.g. Screw, Bolt, PCB Board, Gear…"
                value={productName}
                onChange={(e) => setProductName(e.target.value)}
              />
            </div>
          </div>

          {/* Dropzone */}
          <div className="card mb-4">
            <div className="card-header">
              <h3>Good Product Images</h3>
              <span className={`badge ${readyCount >= IDEAL_MIN ? "badge-pass" : readyCount >= MIN_IMAGES ? "badge-info" : "badge-fail"}`}>
                {readyCount} / {IDEAL_MIN} recommended
              </span>
            </div>
            <div className="card-body">
              <div {...getRootProps()} className={`dropzone ${isDragActive ? "active" : ""}`}>
                <input {...getInputProps()} id="image-file-input" />
                <div className="dropzone-icon">🖼️</div>
                <h3>Drop images here or click to browse</h3>
                <p>Upload 20–30 images of a <strong>good (defect-free)</strong> product</p>
                <p style={{ marginTop: 6, fontSize: ".8rem" }}>JPG, PNG, BMP, WebP accepted</p>
              </div>

              {/* Previews */}
              {files.length > 0 && (
                <div className="image-preview-grid">
                  {files.map((f, i) => (
                    <div key={i} className="image-thumb">
                      <img src={previews[i]} alt={f.name} />
                      <button
                        className="remove-btn"
                        onClick={(e) => { e.stopPropagation(); removeFile(i); }}
                        title="Remove"
                      >✕</button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Train button */}
          <div className="card">
            <div className="card-body" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
              <div>
                {readyCount < MIN_IMAGES ? (
                  <p className="text-muted">Upload at least <strong>{MIN_IMAGES}</strong> images to begin training.</p>
                ) : readyCount < IDEAL_MIN ? (
                  <p className="text-muted">You have <strong>{readyCount}</strong> images — more is better. {IDEAL_MIN - readyCount} more recommended.</p>
                ) : (
                  <p style={{ color: "var(--success)", fontWeight: 600 }}>✅ {readyCount} images ready. Looks great!</p>
                )}
                {message && status === "error" && (
                  <p style={{ color: "var(--danger)", fontSize: ".82rem", marginTop: 4 }}>{message}</p>
                )}
              </div>
              <button
                id="train-btn"
                className="btn btn-primary btn-lg"
                disabled={!isReady}
                onClick={handleTrain}
              >
                🧠 Learn Normal
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
