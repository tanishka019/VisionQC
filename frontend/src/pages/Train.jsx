// src/pages/Train.jsx — v2 with SSE progress
import { useState, useCallback, useEffect, useRef } from "react";
import { useDropzone } from "react-dropzone";
import { trainModel, resetModel, API_BASE } from "../api";

const MIN_IMAGES  = 5;
const IDEAL_MIN   = 20;

// ─── Step ───────────────────────────────────────────────────────────────────
function StepRow({ step }) {
  const steps = ["Upload Images", "Name Product", "Train Model"];
  return (
    <div className="steps-row" style={{ padding: "0 4px" }}>
      {steps.map((label, i) => {
        const done   = i < step;
        const active = i === step;
        return (
          <div key={label} className="step-item" style={{ flex: i < 2 ? 1 : "none" }}>
            <div className={`step-circle ${done ? "done" : active ? "active" : "inactive"}`}>
              {done ? "✓" : i + 1}
            </div>
            <span className="step-label" style={{ color: done || active ? "var(--text-secondary)" : "var(--text-muted)" }}>
              {label}
            </span>
            {i < 2 && <div className={`step-line ${done ? "done" : ""}`} />}
          </div>
        );
      })}
    </div>
  );
}

// ─── Progress bar ────────────────────────────────────────────────────────────
function TrainingProgress({ progress, message }) {
  return (
    <div className="train-progress-card">
      <div className="orbit-spinner" />
      <div style={{ textAlign: "center" }}>
        <h3 style={{ fontWeight: 800, fontSize: "1.1rem", marginBottom: 6 }}>Training PatchCore…</h3>
        <p className="text-muted" style={{ fontSize: ".85rem", marginBottom: 4 }}>
          {message || "Processing images…"}
        </p>
        <p className="text-muted" style={{ fontSize: ".75rem" }}>
          Wide-ResNet50 feature extraction + memory bank
        </p>
      </div>
      <div style={{ width: "100%", maxWidth: 400 }}>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
          <span className="text-xs text-muted">Progress</span>
          <span className="text-xs mono" style={{ color: "var(--primary-light)", fontWeight: 700 }}>
            {progress}%
          </span>
        </div>
      </div>
      <p className="text-muted" style={{ fontSize: ".75rem", textAlign: "center" }}>
        This may take 1–10 min depending on hardware. Keep this tab open.
      </p>
    </div>
  );
}

export default function Train({ onTrained }) {
  const [files,       setFiles]      = useState([]);
  const [previews,    setPreviews]   = useState([]);
  const [productName, setProduct]    = useState("Screw");
  const [status,      setStatus]     = useState("idle"); // idle | training | done | error
  const [progress,    setProgress]   = useState(0);
  const [message,     setMessage]    = useState("");
  const [finalMsg,    setFinalMsg]   = useState("");
  const sseRef = useRef(null);

  // SSE listener for training progress
  const listenSSE = useCallback(() => {
    if (sseRef.current) sseRef.current.close();
    const es = new EventSource(`${API_BASE}/train/progress`);
    sseRef.current = es;

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.ping) return;
        if (data.status === "running") {
          setProgress(data.progress || 0);
          setMessage(data.message || "");
        } else if (data.status === "done") {
          setProgress(100);
          setMessage(data.message || "");
          setStatus("done");
          setFinalMsg(data.message || "Model trained successfully.");
          onTrained?.();
          es.close();
        } else if (data.status === "error") {
          setStatus("error");
          setFinalMsg(data.message || "Training failed.");
          es.close();
        }
      } catch { /* noop */ }
    };

    es.onerror = () => {
      // SSE disconnects are normal after training finishes
    };
  }, [onTrained]);

  useEffect(() => {
    return () => { sseRef.current?.close(); };
  }, []);

  const onDrop = useCallback((accepted) => {
    const imgs = accepted.filter((f) => f.type.startsWith("image/"));
    setFiles((prev) => {
      const merged = [...prev, ...imgs].slice(0, 60);
      merged.forEach((f) => { if (!f.preview) f.preview = URL.createObjectURL(f); });
      setPreviews(merged.map((f) => f.preview));
      return merged;
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
    setProgress(5);
    setMessage("Uploading images…");
    try {
      await trainModel(files, productName || "product");
      // Subscribe only once the server is in the "running" state, so a stale
      // "done" from a previous run can't end this one early.
      listenSSE();
    } catch (err) {
      const msg = err.userMessage || err.message || "Unknown error";
      setStatus("error");
      setFinalMsg(`Training failed: ${msg}`);
      sseRef.current?.close();
    }
  };

  const reset = () => {
    files.forEach((f) => URL.revokeObjectURL(f.preview));
    setFiles([]); setPreviews([]);
    setStatus("idle"); setProgress(0); setMessage(""); setFinalMsg("");
  };

  const handleReset = async () => {
    if (!confirm("Reset the trained model? This cannot be undone.")) return;
    try {
      await resetModel();
      onTrained?.();
      alert("Model reset.");
    } catch { /* noop */ }
  };

  const count   = files.length;
  const isReady = count >= MIN_IMAGES && status !== "training";
  const step    = status === "done" ? 3 : count >= MIN_IMAGES ? 1 : 0;

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }} className="animate-fade-up">

      {/* Steps */}
      <div className="card mb-4">
        <div className="card-body" style={{ padding: "18px 24px" }}>
          <StepRow step={step} />
        </div>
      </div>

      {/* Training in progress */}
      {status === "training" && (
        <div className="card mb-4">
          <TrainingProgress progress={progress} message={message} />
        </div>
      )}

      {/* Success */}
      {status === "done" && (
        <div className="card mb-4" style={{ borderColor: "var(--success-border)" }}>
          <div className="card-body" style={{ textAlign: "center", padding: "40px 32px" }}>
            <div style={{ fontSize: "3.5rem", marginBottom: 14 }}>🎉</div>
            <h3 style={{ fontWeight: 900, fontSize: "1.3rem", color: "var(--success-light)", marginBottom: 8 }}>
              Model Ready!
            </h3>
            <p className="text-muted" style={{ marginBottom: 24, fontSize: ".9rem" }}>{finalMsg}</p>
            <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
              <a className="btn btn-primary btn-lg" href="/inspect">Start Inspecting →</a>
              <button className="btn btn-secondary" onClick={reset}>Train Another Model</button>
              <button className="btn btn-danger btn-sm" onClick={handleReset}>Reset Model</button>
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {status === "error" && (
        <div className="alert alert-danger mb-4">
          <span>⚠</span>
          <span>{finalMsg}</span>
          <button className="btn btn-ghost btn-sm" onClick={reset} style={{ marginLeft: "auto" }}>Retry</button>
        </div>
      )}

      {/* Form (only when not done/training) */}
      {!["done", "training"].includes(status) && (
        <>
          {/* Product name */}
          <div className="card mb-4">
            <div className="card-header">
              <span className="card-title">Product Name</span>
              <span className="text-xs text-muted">Used to label the model</span>
            </div>
            <div className="card-body">
              <input
                id="product-name-input"
                className="input"
                placeholder="e.g. Screw, PCB Board, Gear, Bolt…"
                value={productName}
                onChange={(e) => setProduct(e.target.value)}
              />
            </div>
          </div>

          {/* Dropzone */}
          <div className="card mb-4">
            <div className="card-header">
              <span className="card-title">Good Product Images</span>
              <span className={`badge ${count >= IDEAL_MIN ? "badge-pass" : count >= MIN_IMAGES ? "badge-info" : "badge-neutral"}`}>
                {count} / {IDEAL_MIN} recommended
              </span>
            </div>
            <div className="card-body">
              <div {...getRootProps()} className={`dropzone${isDragActive ? " drag-active" : ""}`}>
                <input {...getInputProps()} id="image-file-input" />
                <span className="dropzone-icon">🖼️</span>
                <h3>{isDragActive ? "Drop images here!" : "Drop images or click to browse"}</h3>
                <p>Upload <strong>20–30 defect-free</strong> product images for best accuracy</p>
                <p style={{ fontSize: ".76rem", marginTop: 6, opacity: 0.7 }}>JPG · PNG · BMP · WebP · Max 60 images</p>
              </div>

              {/* Thumbnails */}
              {files.length > 0 && (
                <div className="image-preview-grid mt-4">
                  {files.map((f, i) => (
                    <div key={i} className="image-thumb">
                      <img src={previews[i]} alt={f.name} loading="lazy" />
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

          {/* Train action bar */}
          <div className="card">
            <div className="card-body" style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 16, flexWrap: "wrap", padding: "18px 22px",
            }}>
              <div>
                {count < MIN_IMAGES ? (
                  <p className="text-muted text-sm">
                    Upload at least <strong style={{ color: "var(--text)" }}>{MIN_IMAGES}</strong> images to begin.
                    ({MIN_IMAGES - count} more needed)
                  </p>
                ) : count < IDEAL_MIN ? (
                  <p className="text-muted text-sm">
                    <strong style={{ color: "var(--warning)" }}>{count}</strong> images ready —
                    {IDEAL_MIN - count} more would improve accuracy.
                  </p>
                ) : (
                  <p style={{ color: "var(--success-light)", fontWeight: 600, fontSize: ".9rem" }}>
                    ✓ {count} images ready — optimal dataset size!
                  </p>
                )}
              </div>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                {count > 0 && (
                  <button className="btn btn-secondary" onClick={reset}>
                    Clear All
                  </button>
                )}
                <button
                  id="train-btn"
                  className="btn btn-primary btn-lg"
                  disabled={!isReady}
                  onClick={handleTrain}
                >
                  ⬢ Train Model
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
