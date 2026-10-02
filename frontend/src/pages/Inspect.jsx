// src/pages/Inspect.jsx
import { useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import { inspectImage, API_BASE } from "../api";

function ResultPanel({ result, loading }) {
  if (loading) {
    return (
      <div className="card result-panel">
        <div className="card-body loading-wrap">
          <div className="spinner" />
          <p>Analyzing image…</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card result-panel">
        <div className="empty-state">
          <div className="icon">🔬</div>
          <p>Capture a frame or upload an image<br />to see the inspection result.</p>
        </div>
      </div>
    );
  }

  const { score, confidence, result: verdict, threshold, heatmap_url, image_url } = result;
  const isPass = verdict === "PASS";

  return (
    <div className="result-panel" style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* Verdict card */}
      <div className={`card result-verdict ${isPass ? "pass" : "fail"}`}>
        <div className="verdict-icon">{isPass ? "✅" : "❌"}</div>
        <div className={`verdict-text ${isPass ? "pass" : "fail"}`}>{verdict}</div>
        <div className="verdict-sub">
          {isPass ? "Product meets quality standard" : "Defect detected — reject unit"}
        </div>
      </div>

      {/* Metrics */}
      <div className="card">
        <div className="card-body">
          <div className="score-metric">
            <span className="label">Anomaly Score</span>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div className="score-bar-track" style={{ width: 80 }}>
                <div
                  className="score-bar-fill"
                  style={{
                    width: `${score * 100}%`,
                    background: score > threshold ? "var(--danger)" : "var(--success)",
                  }}
                />
              </div>
              <span className="value">{(score * 100).toFixed(1)}%</span>
            </div>
          </div>
          <div className="score-metric">
            <span className="label">Confidence</span>
            <span className="value" style={{ color: isPass ? "var(--success)" : "var(--danger)" }}>
              {confidence.toFixed(1)}%
            </span>
          </div>
          <div className="score-metric">
            <span className="label">Threshold</span>
            <span className="value">{(threshold * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {/* Heatmap */}
      {heatmap_url && (
        <div className="card">
          <div className="card-header"><h3>Deviation Heatmap</h3></div>
          <div className="card-body">
            <p className="text-muted mb-4" style={{ fontSize: ".78rem" }}>
              Red/warm areas indicate deviation from normal.
            </p>
            <div className="heatmap-compare">
              {image_url && (
                <div>
                  <p className="section-title">Original</p>
                  <img src={`${API_BASE}${image_url}`} alt="Original" />
                </div>
              )}
              <div>
                <p className="section-title">Heatmap</p>
                <img src={`${API_BASE}${heatmap_url}`} alt="Heatmap" />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const WEBCAM_CONSTRAINTS = {
  width: { ideal: 1280 },
  height: { ideal: 720 },
  facingMode: "environment",
};

export default function Inspect({ modelTrained }) {
  const webcamRef  = useRef(null);
  const fileRef    = useRef(null);
  const [mode,     setMode]    = useState("webcam"); // webcam | upload
  const [camOn,    setCamOn]   = useState(false);
  const [result,   setResult]  = useState(null);
  const [loading,  setLoading] = useState(false);
  const [error,    setError]   = useState("");

  const runInspect = useCallback(async (file) => {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await inspectImage(file);
      setResult(res.data);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Inspection failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  const capture = useCallback(() => {
    if (!webcamRef.current) return;
    const dataUrl = webcamRef.current.getScreenshot();
    if (!dataUrl) return;
    // Convert dataUrl to Blob/File
    fetch(dataUrl)
      .then((r) => r.blob())
      .then((blob) => {
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
        runInspect(file);
      });
  }, [runInspect]);

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) runInspect(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) runInspect(file);
  };

  if (!modelTrained) {
    return (
      <div className="card" style={{ maxWidth: 520, margin: "60px auto" }}>
        <div className="card-body empty-state" style={{ padding: 48 }}>
          <div className="icon">🧠</div>
          <h3 style={{ fontWeight: 700, fontSize: "1.1rem" }}>No Model Trained</h3>
          <p>Please go to <strong>Train Model</strong> first and upload 20–30 good product images.</p>
          <a className="btn btn-primary mt-4" href="/train">Go to Train →</a>
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Mode switcher */}
      <div style={{ display: "flex", gap: 10, marginBottom: 24 }}>
        <button
          id="webcam-mode-btn"
          className={`btn ${mode === "webcam" ? "btn-primary" : "btn-outline"}`}
          onClick={() => setMode("webcam")}
        >
          📷 Webcam
        </button>
        <button
          id="upload-mode-btn"
          className={`btn ${mode === "upload" ? "btn-primary" : "btn-outline"}`}
          onClick={() => { setMode("upload"); setCamOn(false); }}
        >
          📁 Upload Image
        </button>
      </div>

      <div className="inspect-grid">

        {/* Left: camera / upload */}
        <div>
          {mode === "webcam" ? (
            <div className="card">
              <div className="card-header"><h3>Live Camera Feed</h3></div>
              <div className="card-body">
                <div className="webcam-box">
                  {camOn ? (
                    <Webcam
                      ref={webcamRef}
                      audio={false}
                      screenshotFormat="image/jpeg"
                      videoConstraints={WEBCAM_CONSTRAINTS}
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    />
                  ) : (
                    <div className="webcam-overlay">
                      <div className="icon">📷</div>
                      <p>Camera is off</p>
                    </div>
                  )}
                </div>
                <div className="webcam-controls">
                  {!camOn ? (
                    <button id="start-camera-btn" className="btn btn-primary" onClick={() => setCamOn(true)}>
                      ▶ Start Camera
                    </button>
                  ) : (
                    <>
                      <button id="capture-btn" className="btn btn-primary btn-lg" onClick={capture} disabled={loading}>
                        📸 Capture & Inspect
                      </button>
                      <button className="btn btn-outline" onClick={() => setCamOn(false)}>
                        ⏹ Stop
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="card-header"><h3>Upload Product Image</h3></div>
              <div className="card-body">
                <div
                  className="file-drop-area"
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileRef.current?.click()}
                >
                  <div style={{ fontSize: "2.5rem", marginBottom: 8 }}>🖼️</div>
                  <h3 style={{ fontWeight: 600, marginBottom: 4 }}>Drop image here</h3>
                  <p className="text-muted">Or click to browse — JPG, PNG, WebP</p>
                </div>
                <input
                  id="inspect-file-input"
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  style={{ display: "none" }}
                  onChange={handleFileUpload}
                />
              </div>
            </div>
          )}

          {error && (
            <div style={{
              marginTop: 14, padding: "12px 16px",
              background: "rgba(239,71,111,.1)",
              border: "1px solid rgba(239,71,111,.3)",
              borderRadius: "var(--radius-sm)",
              color: "var(--danger)",
              fontSize: ".85rem",
            }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Right: result */}
        <ResultPanel result={result} loading={loading} />
      </div>
    </div>
  );
}
