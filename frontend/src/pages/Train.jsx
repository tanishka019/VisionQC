// src/pages/Train.jsx — with SSE progress
import { useState, useCallback, useEffect, useRef } from "react";
import { useDropzone } from "react-dropzone";
import { trainModel, resetModel, API_BASE } from "../api";
import { Icon, PageHead } from "../ui";

const MIN_IMAGES  = 5;
const IDEAL_MIN   = 20;

// ─── Steps ───────────────────────────────────────────────────────────────────
function Steps({ step }) {
  const steps = ["Add good photos", "Name the product", "Learn"];
  return (
    <ol className="steps" style={{ listStyle: "none" }}>
      {steps.map((label, i) => {
        const done   = i < step;
        const active = i === step;
        return (
          <li key={label} className={`step${done ? " done" : active ? " active" : ""}`}>
            <span className="n">{done ? <Icon name="check" size={13} /> : String(i + 1).padStart(2, "0")}</span>
            {label}
          </li>
        );
      })}
    </ol>
  );
}

// ─── Progress ────────────────────────────────────────────────────────────────
function TrainingProgress({ progress, message }) {
  return (
    <div className="card-body" style={{ padding: "36px 28px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 14 }}>
        <h2 style={{ fontSize: "1.05rem", fontWeight: 600, letterSpacing: "-0.01em" }}>Learning what normal looks like</h2>
        <span className="mono num" style={{ fontSize: ".9rem" }}>{progress}%</span>
      </div>
      <div className="progress"><div style={{ width: `${progress}%` }} /></div>
      <p className="small muted" style={{ marginTop: 14 }}>{message || "Processing images…"}</p>
      <p className="small muted" style={{ marginTop: 4 }}>Usually under a minute. Keep this tab open.</p>
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
  const step    = status === "done" ? 3 : status === "training" ? 2 : count >= MIN_IMAGES ? 1 : 0;

  return (
    <div style={{ maxWidth: 820 }}>
      <PageHead title="Train" sub="Show VisionQC what a good part looks like. No defect photos needed." />

      <Steps step={step} />

      {status === "training" && (
        <div className="card gap-y"><TrainingProgress progress={progress} message={message} /></div>
      )}

      {status === "done" && (
        <div className="card gap-y">
          <div className="card-body center" style={{ padding: "44px 28px" }}>
            <Icon name="check" size={28} style={{ color: "var(--pass)" }} />
            <h2 style={{ fontSize: "1.3rem", fontWeight: 600, letterSpacing: "-0.02em", margin: "12px 0 6px" }}>Model ready</h2>
            <p className="muted" style={{ marginBottom: 24 }}>{finalMsg.replace(/^\p{Extended_Pictographic}\uFE0F?\s*/u, "")}</p>
            <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
              <a className="btn btn-primary btn-lg" href="/inspect">Start inspecting <Icon name="arrow" size={15} /></a>
              <button className="btn btn-secondary btn-lg" onClick={reset}>Train another</button>
              <button className="btn btn-danger btn-lg" onClick={handleReset}>Reset model</button>
            </div>
          </div>
        </div>
      )}

      {status === "error" && (
        <div className="alert" role="alert">
          <span>{finalMsg}</span>
          <button className="btn btn-quiet btn-sm" onClick={reset}>Retry</button>
        </div>
      )}

      {!["done", "training"].includes(status) && (
        <>
          <div className="card gap-y">
            <div className="card-head">
              <h2>Product name</h2>
              <span className="small muted hint-sm">Used to label the model</span>
            </div>
            <div className="card-body">
              <input
                id="product-name-input"
                className="input"
                placeholder="e.g. Screw, circuit board, gear…"
                value={productName}
                onChange={(e) => setProduct(e.target.value)}
              />
            </div>
          </div>

          <div className="card gap-y">
            <div className="card-head">
              <h2>Good product photos</h2>
              <span className="small mono muted num">{count} / {IDEAL_MIN} recommended</span>
            </div>
            <div className="card-body">
              <div {...getRootProps()} className={`dropzone${isDragActive ? " drag-active" : ""}`}>
                <input {...getInputProps()} id="image-file-input" />
                <Icon name="upload" size={28} />
                <h3>{isDragActive ? "Drop the photos here" : "Drop photos, or click to browse"}</h3>
                <p>20–30 photos of defect-free parts work best, taken the same way each time.</p>
                <p className="small" style={{ marginTop: 6 }}>JPG · PNG · BMP · WebP · up to 60 images</p>
              </div>

              {files.length > 0 && (
                <div className="thumbs">
                  {files.map((f, i) => (
                    <div key={i} className="thumb">
                      <img src={previews[i]} alt={f.name} loading="lazy" />
                      <button
                        onClick={(e) => { e.stopPropagation(); removeFile(i); }}
                        title="Remove"
                        aria-label={`Remove ${f.name}`}
                      >
                        <Icon name="x" size={11} strokeWidth={2.4} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="card gap-y">
            <div className="card-body action-bar">
              <p className="small" style={{ color: "var(--ink-2)" }}>
                {count < MIN_IMAGES ? (
                  <>Add at least <b>{MIN_IMAGES}</b> photos to begin ({MIN_IMAGES - count} more needed).</>
                ) : count < IDEAL_MIN ? (
                  <><b>{count}</b> photos ready. {IDEAL_MIN - count} more would improve accuracy.</>
                ) : (
                  <><b>{count}</b> photos ready. That&rsquo;s a good size.</>
                )}
              </p>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                {count > 0 && <button className="btn btn-secondary" onClick={reset}>Clear all</button>}
                <button id="train-btn" className="btn btn-primary btn-lg" disabled={!isReady} onClick={handleTrain}>
                  Learn normal
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
