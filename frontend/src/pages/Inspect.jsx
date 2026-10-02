// src/pages/Inspect.jsx — v2
import { useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import { inspectImage, batchInspect, API_BASE } from "../api";

// ─── Score ring ──────────────────────────────────────────────────────────────
function ScoreRing({ score, threshold }) {
  const isPass  = score < threshold;
  const color   = isPass ? "#10b981" : "#ef4444";
  const pct     = Math.round(score * 100);
  const r       = 40;
  const circ    = 2 * Math.PI * r;
  const dash    = (score * circ).toFixed(2);

  return (
    <svg width="100" height="100" viewBox="0 0 100 100">
      <circle cx="50" cy="50" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
      <circle
        cx="50" cy="50" r={r}
        fill="none"
        stroke={color}
        strokeWidth="8"
        strokeDasharray={`${dash} ${circ}`}
        strokeLinecap="round"
        transform="rotate(-90 50 50)"
        style={{ transition: "stroke-dasharray 0.6s cubic-bezier(0.4,0,0.2,1)", filter: `drop-shadow(0 0 6px ${color}88)` }}
      />
      <text x="50" y="45" textAnchor="middle" fill={color} fontSize="16" fontWeight="800" fontFamily="monospace">
        {pct}
      </text>
      <text x="50" y="60" textAnchor="middle" fill="#4a5568" fontSize="9" fontFamily="monospace">
        SCORE
      </text>
    </svg>
  );
}

// ─── Result Panel ────────────────────────────────────────────────────────────
function ResultPanel({ result, loading }) {
  if (loading) {
    return (
      <div className="card result-panel">
        <div className="card-body loading-wrap" style={{ padding: 48 }}>
          <div className="orbit-spinner" />
          <p style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Analyzing image…</p>
          <p className="text-muted text-xs">Running PatchCore inference</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card result-panel">
        <div className="empty-state" style={{ padding: 52 }}>
          <div className="empty-icon">◎</div>
          <div className="empty-title">Awaiting Inspection</div>
          <p className="empty-sub">Capture a webcam frame or upload an image to run AI quality inspection.</p>
        </div>
      </div>
    );
  }

  const { score, confidence, result: verdict, threshold, heatmap_url, image_url, filename } = result;
  const isPass = verdict === "PASS";

  return (
    <div className="result-panel animate-fade-up">

      {/* Verdict */}
      <div className={`verdict-card ${isPass ? "pass" : "fail"}`}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 20 }}>
          <ScoreRing score={score} threshold={threshold} />
          <div>
            <div className="verdict-icon">{isPass ? "✅" : "❌"}</div>
            <div className={`verdict-label ${isPass ? "pass" : "fail"}`}>{verdict}</div>
            <div className="verdict-sub">
              {isPass ? "Meets quality standard" : "Defect detected — reject"}
            </div>
          </div>
        </div>
        {filename && (
          <div style={{ textAlign: "center", marginTop: 10 }}>
            <span className="text-xs mono" style={{ color: "var(--text-muted)", opacity: 0.7 }}>{filename}</span>
          </div>
        )}
      </div>

      {/* Metrics */}
      <div className="card metrics-card">
        <div className="card-header" style={{ padding: "14px 18px" }}>
          <span className="card-title">Inspection Metrics</span>
        </div>
        <div className="card-body" style={{ padding: 0 }}>
          <div className="metric-row" style={{ padding: "12px 18px" }}>
            <span className="metric-label">Anomaly Score</span>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ width: 80 }}>
                <div className="score-bar-track">
                  <div
                    className="score-bar-fill"
                    style={{
                      width: `${score * 100}%`,
                      background: score >= threshold ? "var(--danger)" : "var(--success)",
                    }}
                  />
                </div>
              </div>
              <span className="metric-value mono">{(score * 100).toFixed(1)}%</span>
            </div>
          </div>
          <div className="metric-row" style={{ padding: "12px 18px" }}>
            <span className="metric-label">Confidence</span>
            <span className="metric-value" style={{ color: isPass ? "var(--success-light)" : "var(--danger-light)" }}>
              {confidence.toFixed(1)}%
            </span>
          </div>
          <div className="metric-row" style={{ padding: "12px 18px" }}>
            <span className="metric-label">Threshold</span>
            <span className="metric-value mono">{(threshold * 100).toFixed(0)}%</span>
          </div>
          <div className="metric-row" style={{ padding: "12px 18px" }}>
            <span className="metric-label">Verdict</span>
            <span className={`badge ${isPass ? "badge-pass" : "badge-fail"}`}>{verdict}</span>
          </div>
        </div>
      </div>

      {/* Heatmap */}
      {heatmap_url && (
        <div className="card">
          <div className="card-header" style={{ padding: "14px 18px" }}>
            <span className="card-title">Deviation Heatmap</span>
            <span className="text-xs text-muted">Warm areas = anomaly</span>
          </div>
          <div className="card-body">
            <div className="heatmap-grid">
              {image_url && (
                <div className="heatmap-img-wrap">
                  <div className="heatmap-img-label">Original</div>
                  <img src={`${API_BASE}${image_url}`} alt="Original" loading="lazy" />
                </div>
              )}
              <div className="heatmap-img-wrap">
                <div className="heatmap-img-label">Heatmap</div>
                <img src={`${API_BASE}${heatmap_url}`} alt="Heatmap" loading="lazy" />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Webcam constraints ──────────────────────────────────────────────────────
const WEBCAM_CONSTRAINTS = {
  width: { ideal: 1280 }, height: { ideal: 720 },
  facingMode: "environment",
};

// ─── Batch result row ────────────────────────────────────────────────────────
function BatchResultRow({ item }) {
  const isPass = item.result === "PASS";
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 12, padding: "10px 0",
      borderBottom: "1px solid var(--border)",
    }}>
      <span className={`badge ${isPass ? "badge-pass" : item.result === "ERROR" ? "badge-warning" : "badge-fail"}`}>
        {item.result}
      </span>
      <span className="text-sm mono" style={{ flex: 1, color: "var(--text-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {item.filename}
      </span>
      {item.score != null && (
        <span className="text-xs mono" style={{ color: "var(--text-muted)" }}>
          {(item.score * 100).toFixed(1)}%
        </span>
      )}
      {item.error && (
        <span className="text-xs" style={{ color: "var(--danger-light)" }}>{item.error}</span>
      )}
    </div>
  );
}

export default function Inspect({ modelTrained }) {
  const webcamRef  = useRef(null);
  const fileRef    = useRef(null);
  const [mode,     setMode]    = useState("webcam"); // webcam | upload | batch
  const [camOn,    setCamOn]   = useState(false);
  const [result,   setResult]  = useState(null);
  const [loading,  setLoading] = useState(false);
  const [error,    setError]   = useState("");
  const [batchRes, setBatch]   = useState(null);
  const [inspectCount, setCount] = useState(0);

  const runInspect = useCallback(async (file) => {
    setLoading(true);
    setError("");
    setResult(null);
    setBatch(null);
    try {
      const res = await inspectImage(file);
      setResult(res.data);
      setCount((c) => c + 1);
    } catch (err) {
      setError(err.userMessage || err.message || "Inspection failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const capture = useCallback(() => {
    if (!webcamRef.current) return;
    const dataUrl = webcamRef.current.getScreenshot();
    if (!dataUrl) return;
    fetch(dataUrl)
      .then((r) => r.blob())
      .then((blob) => {
        const file = new File([blob], `capture_${Date.now()}.jpg`, { type: "image/jpeg" });
        runInspect(file);
      });
  }, [runInspect]);

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) runInspect(file);
    e.target.value = ""; // reset so same file can be re-picked
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file?.type.startsWith("image/")) runInspect(file);
  };

  const handleBatchUpload = async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setLoading(true); setError(""); setBatch(null); setResult(null);
    try {
      const res = await batchInspect(files);
      setBatch(res.data);
    } catch (err) {
      setError(err.userMessage || err.message || "Batch inspection failed");
    } finally {
      setLoading(false);
    }
    e.target.value = "";
  };

  // Not trained guard
  if (!modelTrained) {
    return (
      <div style={{ maxWidth: 520, margin: "60px auto" }} className="animate-fade-up">
        <div className="card">
          <div className="card-body empty-state" style={{ padding: 52 }}>
            <div className="empty-icon">⬢</div>
            <div className="empty-title">No Model Trained</div>
            <p className="empty-sub">
              Please go to <strong>Train Model</strong> first and upload 20–30 good product images to teach the AI what "normal" looks like.
            </p>
            <a className="btn btn-primary mt-4" href="/train">Go to Train →</a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-up">
      {/* Mode tabs + count */}
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 20, flexWrap: "wrap" }}>
        <div className="tab-group" style={{ width: "auto" }}>
          {[
            { key: "webcam", label: "📷  Webcam" },
            { key: "upload", label: "🖼️  Upload" },
            { key: "batch",  label: "📂  Batch" },
          ].map(({ key, label }) => (
            <button
              key={key}
              className={`tab-btn${mode === key ? " active" : ""}`}
              onClick={() => { setMode(key); if (key !== "webcam") setCamOn(false); }}
            >
              {label}
            </button>
          ))}
        </div>
        {inspectCount > 0 && (
          <span className="badge badge-info">
            {inspectCount} inspection{inspectCount > 1 ? "s" : ""} this session
          </span>
        )}
      </div>

      <div className="inspect-layout">

        {/* Left: input */}
        <div>
          {/* Webcam mode */}
          {mode === "webcam" && (
            <div className="card">
              <div className="card-header">
                <span className="card-title">Live Camera Feed</span>
                {camOn && (
                  <div className="live-indicator" style={{ fontSize: ".65rem" }}>
                    <span className="live-dot" />STREAMING
                  </div>
                )}
              </div>
              <div className="card-body">
                <div className="webcam-container">
                  {camOn ? (
                    <>
                      <Webcam
                        ref={webcamRef}
                        audio={false}
                        screenshotFormat="image/jpeg"
                        screenshotQuality={0.92}
                        videoConstraints={WEBCAM_CONSTRAINTS}
                        style={{ width: "100%", height: "100%", objectFit: "cover" }}
                      />
                      <div className="scan-line" />
                      <div className="webcam-overlay-badge">
                        <span className="live-dot" style={{ width: 5, height: 5 }} />
                        CAMERA ACTIVE
                      </div>
                    </>
                  ) : (
                    <div className="webcam-idle">
                      <div className="webcam-idle-icon">📷</div>
                      <p style={{ fontSize: ".85rem", color: "var(--text-muted)" }}>Camera is off</p>
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
                      <button
                        id="capture-btn"
                        className="btn btn-primary btn-lg"
                        onClick={capture}
                        disabled={loading}
                      >
                        {loading ? (
                          <><span className="spinner-sm" style={{ display:"inline-block",width:16,height:16,border:"2px solid rgba(255,255,255,.2)",borderTopColor:"#fff",borderRadius:"50%",animation:"orbit .8s linear infinite" }}/> Analyzing…</>
                        ) : "📸 Capture & Inspect"}
                      </button>
                      <button className="btn btn-secondary" onClick={() => setCamOn(false)}>⏹ Stop</button>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Upload mode */}
          {mode === "upload" && (
            <div className="card">
              <div className="card-header">
                <span className="card-title">Upload Product Image</span>
              </div>
              <div className="card-body">
                <div
                  className="file-drop-area"
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileRef.current?.click()}
                >
                  <div style={{ fontSize: "2.5rem", marginBottom: 10 }}>🖼️</div>
                  <h3 style={{ fontWeight: 700, fontSize: "1rem", marginBottom: 6 }}>
                    {loading ? "Analyzing…" : "Drop image here or click to browse"}
                  </h3>
                  <p className="text-muted text-sm">JPG · PNG · BMP · WebP</p>
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

          {/* Batch mode */}
          {mode === "batch" && (
            <div className="card">
              <div className="card-header">
                <span className="card-title">Batch Inspection</span>
                <span className="text-xs text-muted">Inspect multiple images at once</span>
              </div>
              <div className="card-body">
                <div
                  className="file-drop-area"
                  onClick={() => document.getElementById("batch-file-input").click()}
                >
                  <div style={{ fontSize: "2.5rem", marginBottom: 10 }}>📂</div>
                  <h3 style={{ fontWeight: 700, marginBottom: 6 }}>Select multiple images</h3>
                  <p className="text-muted text-sm">All images inspected in one shot</p>
                </div>
                <input
                  id="batch-file-input"
                  type="file"
                  accept="image/*"
                  multiple
                  style={{ display: "none" }}
                  onChange={handleBatchUpload}
                />

                {/* Batch results */}
                {batchRes && (
                  <div style={{ marginTop: 20 }}>
                    <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
                      <span className="badge badge-info">{batchRes.summary.total} total</span>
                      <span className="badge badge-pass">{batchRes.summary.passed} passed</span>
                      <span className="badge badge-fail">{batchRes.summary.failed} failed</span>
                    </div>
                    <div style={{ maxHeight: 300, overflowY: "auto" }}>
                      {batchRes.results.map((item, i) => (
                        <BatchResultRow key={i} item={item} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="alert alert-danger">
              <span>⚠</span>
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right: result */}
        {mode !== "batch" && (
          <ResultPanel result={result} loading={loading} />
        )}
      </div>
    </div>
  );
}
