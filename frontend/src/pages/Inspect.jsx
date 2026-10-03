// src/pages/Inspect.jsx
import { useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import { inspectImage, batchInspect, API_BASE } from "../api";
import { Icon, Frame, Status, PageHead, Segmented, ScoreGauge } from "../ui";

// ─── Result: verdict + metrics (right column) ────────────────────────────────
function ResultSummary({ result, loading }) {
  if (loading) {
    return (
      <div className="card">
        <div className="empty">
          <h3>Analysing image</h3>
          <p>Comparing against what a good part looks like.</p>
          <div className="busy" />
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card">
        <div className="empty">
          <Icon name="scan" size={40} />
          <h3>Awaiting an image</h3>
          <p>Capture a camera frame or upload a photo to get a PASS / FAIL result.</p>
        </div>
      </div>
    );
  }

  const { score, confidence, result: verdict, threshold, filename } = result;
  const isPass = verdict === "PASS";

  return (
    <div className="result-col fade">
      <div className="card verdict">
        <div className="label">Verdict</div>
        <div className={`verdict-word ${isPass ? "pass" : "fail"}`} style={{ marginTop: 10 }}>{verdict}</div>
        <p className="verdict-sub">{isPass ? "Meets the quality standard." : "Defect detected. Reject this part."}</p>
        <ScoreGauge score={score} threshold={threshold} />
        {filename && <p className="verdict-file mono">{filename}</p>}
      </div>

      <div className="card">
        <div className="card-body">
          <dl className="rows">
            <div><dt>Anomaly score</dt><dd className="mono">{(score * 100).toFixed(1)}%</dd></div>
            <div><dt>Confidence</dt><dd className="mono">{confidence.toFixed(1)}%</dd></div>
            <div><dt>Threshold</dt><dd className="mono">{(threshold * 100).toFixed(0)}%</dd></div>
          </dl>
        </div>
      </div>
    </div>
  );
}

// ─── Result: original vs deviation map (left column, under the input) ────────
function ResultImages({ result }) {
  const { heatmap_url, image_url } = result;
  return (
    <div className="card fade">
      <div className="card-head">
        <h2>Deviation map</h2>
        <span className="small muted hint-sm">Warm areas differ from normal</span>
      </div>
      <div className="card-body">
        <div className="pair">
          {image_url && (
            <figure>
              <figcaption>Original</figcaption>
              <Frame><img src={`${API_BASE}${image_url}`} alt="Original" loading="lazy" /></Frame>
            </figure>
          )}
          <figure>
            <figcaption>Heatmap</figcaption>
            <Frame><img src={`${API_BASE}${heatmap_url}`} alt="Heatmap" loading="lazy" /></Frame>
          </figure>
        </div>
      </div>
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
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 14, padding: "10px 0",
      borderBottom: "1px solid var(--line)",
    }}>
      <span style={{ width: 64 }}><Status result={item.result} /></span>
      <span className="small mono truncate" style={{ flex: 1, maxWidth: "none", color: "var(--ink-2)" }}>
        {item.filename}
      </span>
      {item.score != null && (
        <span className="small mono muted">{(item.score * 100).toFixed(1)}%</span>
      )}
      {item.error && <span className="small" style={{ color: "var(--fail)" }}>{item.error}</span>}
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
      <div style={{ maxWidth: 520, margin: "40px auto 0" }}>
        <div className="card">
          <div className="empty" style={{ padding: "56px 32px" }}>
            <Icon name="scan" size={40} />
            <h3>No model yet</h3>
            <p>
              Upload 20–30 photos of good parts on the Train page so VisionQC can learn what &ldquo;normal&rdquo; looks like.
            </p>
            <a className="btn btn-primary" href="/train">Go to Train <Icon name="arrow" size={15} /></a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <PageHead title="Inspect" sub="Check a part against the learned normal." />

      <div className="toolbar">
        <Segmented
          options={[
            { key: "webcam", label: "Camera" },
            { key: "upload", label: "Upload" },
            { key: "batch",  label: "Batch" },
          ]}
          value={mode}
          onChange={(key) => { setMode(key); if (key !== "webcam") setCamOn(false); }}
        />
        {inspectCount > 0 && (
          <span className="small muted mono">
            {inspectCount} inspection{inspectCount > 1 ? "s" : ""} this session
          </span>
        )}
      </div>

      <div className="inspect-grid">
        {/* Input */}
        <div className="area-input">
          {mode === "webcam" && (
            <div className="card">
              <div className="card-head"><h2>Camera</h2></div>
              <div className="card-body">
                <Frame>
                  <div className="viewer">
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
                        <span className="viewer-tag">LIVE</span>
                      </>
                    ) : (
                      <div className="viewer-idle">
                        <Icon name="camera" size={32} />
                        <span>Camera is off</span>
                      </div>
                    )}
                  </div>
                </Frame>
                <div className="controls">
                  {!camOn ? (
                    <button id="start-camera-btn" className="btn btn-primary" onClick={() => setCamOn(true)}>
                      <Icon name="play" size={13} /> Start camera
                    </button>
                  ) : (
                    <>
                      <button id="capture-btn" className="btn btn-primary btn-lg" onClick={capture} disabled={loading}>
                        {loading ? <><span className="spin" /> Analysing</> : "Capture & inspect"}
                      </button>
                      <button className="btn btn-secondary btn-lg" onClick={() => setCamOn(false)}>
                        <Icon name="stop" size={13} /> Stop
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {mode === "upload" && (
            <div className="card">
              <div className="card-head"><h2>Upload a photo</h2></div>
              <div className="card-body">
                <div
                  className="dropzone"
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileRef.current?.click()}
                >
                  <Icon name="upload" size={28} />
                  <h3>{loading ? "Analysing…" : "Drop an image, or click to browse"}</h3>
                  <p>JPG · PNG · BMP · WebP</p>
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

          {mode === "batch" && (
            <div className="card">
              <div className="card-head">
                <h2>Batch</h2>
                <span className="small muted hint-sm">Inspect many images at once</span>
              </div>
              <div className="card-body">
                <div
                  className="dropzone"
                  onClick={() => document.getElementById("batch-file-input").click()}
                >
                  <Icon name="folder" size={28} />
                  <h3>{loading ? "Analysing…" : "Select multiple images"}</h3>
                  <p>Every image is checked in one go</p>
                </div>
                <input
                  id="batch-file-input"
                  type="file"
                  accept="image/*"
                  multiple
                  style={{ display: "none" }}
                  onChange={handleBatchUpload}
                />

                {batchRes && (
                  <div style={{ marginTop: 24 }}>
                    <div style={{ display: "flex", gap: 20, marginBottom: 8 }} className="small">
                      <span><b className="num">{batchRes.summary.total}</b> <span className="muted">total</span></span>
                      <span style={{ color: "var(--pass)" }}><b className="num">{batchRes.summary.passed}</b> passed</span>
                      <span style={{ color: "var(--fail)" }}><b className="num">{batchRes.summary.failed}</b> failed</span>
                    </div>
                    <div style={{ maxHeight: 320, overflowY: "auto" }}>
                      {batchRes.results.map((item, i) => <BatchResultRow key={i} item={item} />)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {error && (
            <div className="alert" role="alert">
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Verdict + metrics */}
        {mode !== "batch" && (
          <div className="area-side"><ResultSummary result={result} loading={loading} /></div>
        )}

        {/* Heatmap */}
        {mode !== "batch" && !loading && result?.heatmap_url && (
          <div className="area-images"><ResultImages result={result} /></div>
        )}
      </div>
    </div>
  );
}
