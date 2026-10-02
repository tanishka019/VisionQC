// src/pages/Dashboard.jsx — v2
import { useEffect, useState, useCallback } from "react";
import {
  AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";
import { getStats, getThreshold, setThreshold, subscribeLive } from "../api";

// ─── Stat Card ───────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, variant, icon, animDelay = 0 }) {
  const [displayed, setDisplayed] = useState(0);
  const target = typeof value === "number" ? value : null;

  useEffect(() => {
    if (target === null) return;
    let start = null;
    const duration = 800;
    const step = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayed(Math.round(eased * target));
      if (progress < 1) requestAnimationFrame(step);
    };
    const id = setTimeout(() => requestAnimationFrame(step), animDelay);
    return () => clearTimeout(id);
  }, [target, animDelay]);

  const displayValue = target !== null ? displayed : value;

  return (
    <div className={`stat-card ${variant || ""}`} style={{ animationDelay: `${animDelay}ms` }}>
      {icon && <div className={`stat-icon ${variant || "primary"}`}>{icon}</div>}
      <div className="stat-label">{label}</div>
      <div className="stat-value">{displayValue}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

// ─── Custom Tooltip ──────────────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "var(--bg-card)",
      border: "1px solid var(--border)",
      borderRadius: "var(--radius-sm)",
      padding: "10px 14px",
      boxShadow: "var(--shadow-md)",
    }}>
      <p style={{ fontSize: ".78rem", color: "var(--text-muted)", marginBottom: 6 }}>{label}</p>
      {payload.map((p) => (
        <div key={p.name} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: ".82rem", marginBottom: 3 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: p.color, display: "inline-block" }} />
          <span style={{ color: "var(--text-secondary)" }}>{p.name}:</span>
          <span style={{ color: "var(--text)", fontWeight: 700 }}>{p.value}</span>
        </div>
      ))}
    </div>
  );
};

// ─── Hours array ─────────────────────────────────────────────────────────────
const HOURS = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, "0"));

export default function Dashboard() {
  const [stats,     setStats]     = useState({ today: {}, hourly: [], weekly: [] });
  const [threshold, setThreshV]   = useState(0.5);
  const [saving,    setSaving]    = useState(false);
  const [saved,     setSaved]     = useState(false);
  const [loading,   setLoading]   = useState(true);
  const [chartTab,  setChartTab]  = useState("hourly"); // hourly | weekly

  const refresh = useCallback(() => {
    Promise.all([getStats(), getThreshold()])
      .then(([sRes, tRes]) => {
        setStats(sRes.data);
        setThreshV(tRes.data.threshold);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 15000);
    // Refresh stats (not the threshold being edited) as soon as an inspection lands
    const unsubscribe = subscribeLive(() => {
      getStats().then((r) => setStats(r.data)).catch(() => {});
    });
    return () => { clearInterval(t); unsubscribe(); };
  }, [refresh]);

  // Build hourly chart data (fill up to current hour)
  const curHour = new Date().getHours();
  const hourlyData = HOURS
    .filter((h) => Number(h) <= curHour)
    .map((h) => {
      const row = (stats.hourly || []).find((r) => r.hour === h);
      return { hour: `${h}:00`, passed: row?.passed || 0, failed: row?.failed || 0 };
    });

  // Weekly chart data
  const weeklyData = (stats.weekly || []).map((r) => ({
    day: new Date(r.day).toLocaleDateString("en-IN", { weekday: "short", day: "numeric" }),
    passed: r.passed,
    failed: r.failed,
    total:  r.total,
  }));

  const chartData = chartTab === "hourly" ? hourlyData : weeklyData;
  const chartKey  = chartTab === "hourly" ? "hour"     : "day";

  const handleSave = async () => {
    setSaving(true);
    try {
      await setThreshold(threshold);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch { /* noop */ }
    finally { setSaving(false); }
  };

  const { total = 0, passed = 0, failed = 0, rejection_rate = 0, avg_confidence = 0 } = stats.today || {};
  const passRate = total > 0 ? ((passed / total) * 100).toFixed(1) : "—";

  // Threshold sentiment
  const thresholdInfo =
    threshold <= 0.35 ? { label: "High Sensitivity", color: "var(--danger-light)",  hint: "More items will be flagged as FAIL." } :
    threshold >= 0.65 ? { label: "Lenient",           color: "var(--warning)",       hint: "Only major defects will trigger FAIL." } :
                        { label: "Balanced",           color: "var(--success-light)", hint: "Recommended starting point." };

  return (
    <div className="animate-fade-up">
      {/* Stat cards */}
      <div className="stat-grid">
        <StatCard label="Inspected Today" value={total}          sub="Units processed"   variant="primary" icon="◉"  animDelay={0}   />
        <StatCard label="Passed"          value={passed}         sub={`${passRate}% pass rate`} variant="success" icon="✓"  animDelay={80}  />
        <StatCard label="Failed"          value={failed}         sub="Defects detected"  variant="danger"  icon="✕"  animDelay={160} />
        <StatCard label="Rejection Rate"  value={`${rejection_rate}%`} sub="Today's average"   variant={rejection_rate > 20 ? "danger" : "warning"} icon="⚡" animDelay={240} />
      </div>

      {/* Charts + Threshold */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 20 }}>

        {/* Chart card */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              {chartTab === "hourly" ? "Hourly Throughput" : "7-Day Overview"}
            </span>
            <div className="tab-group" style={{ width: "auto" }}>
              {["hourly", "weekly"].map((t) => (
                <button
                  key={t}
                  className={`tab-btn${chartTab === t ? " active" : ""}`}
                  style={{ padding: "5px 12px", fontSize: ".75rem" }}
                  onClick={() => setChartTab(t)}
                >
                  {t === "hourly" ? "Today" : "Week"}
                </button>
              ))}
            </div>
          </div>
          <div className="card-body">
            {loading ? (
              <div className="loading-wrap" style={{ padding: 40 }}>
                <div className="spinner" />
                <span>Loading data…</span>
              </div>
            ) : chartData.length === 0 ? (
              <div className="empty-state" style={{ padding: 40 }}>
                <div className="empty-icon">📭</div>
                <div className="empty-title">No data yet</div>
                <p className="empty-sub">Head to <strong>Inspect</strong> to begin quality checks.</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                {chartTab === "hourly" ? (
                  <AreaChart data={chartData} margin={{ top: 8, right: 4, left: -24, bottom: 0 }}>
                    <defs>
                      <linearGradient id="gPass" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#10b981" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="gFail" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey={chartKey} tick={{ fontSize: 10, fill: "#4a5568" }} tickLine={false} axisLine={false} />
                    <YAxis tick={{ fontSize: 10, fill: "#4a5568" }} tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 12, color: "#a0aec0", paddingTop: 12 }} />
                    <Area type="monotone" dataKey="passed" name="Passed" stroke="#10b981" fill="url(#gPass)" strokeWidth={2} dot={false} />
                    <Area type="monotone" dataKey="failed" name="Failed" stroke="#ef4444" fill="url(#gFail)" strokeWidth={2} dot={false} />
                  </AreaChart>
                ) : (
                  <BarChart data={chartData} margin={{ top: 8, right: 4, left: -24, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey={chartKey} tick={{ fontSize: 10, fill: "#4a5568" }} tickLine={false} axisLine={false} />
                    <YAxis tick={{ fontSize: 10, fill: "#4a5568" }} tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 12, color: "#a0aec0", paddingTop: 12 }} />
                    <Bar dataKey="passed" name="Passed" fill="#10b981" radius={[3,3,0,0]} />
                    <Bar dataKey="failed" name="Failed" fill="#ef4444" radius={[3,3,0,0]} />
                  </BarChart>
                )}
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Threshold card */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Detection Threshold</span>
            <span
              className="badge"
              style={{
                background: threshold <= 0.35 ? "var(--danger-bg)" : threshold >= 0.65 ? "var(--warning-bg)" : "var(--success-bg)",
                color: thresholdInfo.color,
                borderColor: "transparent",
                fontSize: ".62rem",
              }}
            >
              {thresholdInfo.label}
            </span>
          </div>
          <div className="card-body slider-section">

            <p className="text-muted" style={{ fontSize: ".8rem", lineHeight: 1.7 }}>
              Anomaly score above this threshold → <strong style={{ color: "var(--danger-light)" }}>FAIL</strong>.
              Lower = stricter.
            </p>

            {/* Big number */}
            <div className="slider-value-row">
              <div style={{ display: "flex", flexDirection: "column" }}>
                <span className="section-label">Current Threshold</span>
                <div>
                  <span className="threshold-num">{threshold.toFixed(2)}</span>
                  <span className="threshold-unit">/ 1.00</span>
                </div>
              </div>

              {/* Mini donut indicator */}
              <svg width="56" height="56" viewBox="0 0 56 56">
                <circle cx="28" cy="28" r="22" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="5" />
                <circle
                  cx="28" cy="28" r="22"
                  fill="none"
                  stroke={threshold <= 0.35 ? "#ef4444" : threshold >= 0.65 ? "#f59e0b" : "#10b981"}
                  strokeWidth="5"
                  strokeDasharray={`${threshold * 138.2} 138.2`}
                  strokeLinecap="round"
                  transform="rotate(-90 28 28)"
                  style={{ transition: "stroke-dasharray 0.4s ease, stroke 0.3s ease" }}
                />
                <text x="28" y="33" textAnchor="middle" fill="#a0aec0" fontSize="9" fontFamily="monospace" fontWeight="600">
                  {Math.round(threshold * 100)}%
                </text>
              </svg>
            </div>

            <div className="slider-wrap">
              <input
                id="threshold-slider"
                type="range"
                className="threshold-slider"
                min={0.1} max={0.9} step={0.01}
                value={threshold}
                onChange={(e) => setThreshV(parseFloat(e.target.value))}
              />
              <div className="slider-labels">
                <span>0.10 · Strict</span>
                <span>0.90 · Lenient</span>
              </div>
            </div>

            {/* Hint box */}
            <div
              className="threshold-hint"
              style={{
                background: threshold <= 0.35 ? "var(--danger-bg)"  : threshold >= 0.65 ? "var(--warning-bg)"  : "var(--success-bg)",
                border: `1px solid ${threshold <= 0.35 ? "var(--danger-border)" : threshold >= 0.65 ? "var(--warning-border)" : "var(--success-border)"}`,
                color:  thresholdInfo.color,
              }}
            >
              <span style={{ fontSize: "1rem" }}>
                {threshold <= 0.35 ? "⚠" : threshold >= 0.65 ? "⚙" : "✓"}
              </span>
              <span style={{ fontSize: ".78rem" }}>{thresholdInfo.hint}</span>
            </div>

            <button
              id="save-threshold-btn"
              className={`btn btn-lg w-full ${saved ? "btn-success" : "btn-primary"}`}
              onClick={handleSave}
              disabled={saving}
            >
              {saving ? (
                <><span className="spinner-sm" style={{ borderTopColor: "#fff", borderColor: "rgba(255,255,255,0.2)", display: "inline-block", borderRadius: "50%", width: 16, height: 16, borderWidth: 2, animation: "orbit .8s linear infinite" }} /> Saving…</>
              ) : saved ? "✓ Saved!" : "Save Threshold"}
            </button>
          </div>
        </div>
      </div>

      {/* Second row – avg confidence strip */}
      {!loading && total > 0 && (
        <div className="card mt-4">
          <div className="card-body" style={{ display: "flex", gap: 32, flexWrap: "wrap", padding: "16px 22px" }}>
            <div>
              <div className="section-label">Avg Confidence</div>
              <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--primary-light)", fontVariantNumeric: "tabular-nums" }}>
                {avg_confidence.toFixed(1)}%
              </div>
            </div>
            <div style={{ width: 1, background: "var(--border)" }} />
            <div>
              <div className="section-label">Pass Rate</div>
              <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--success-light)", fontVariantNumeric: "tabular-nums" }}>
                {passRate}%
              </div>
            </div>
            <div style={{ width: 1, background: "var(--border)" }} />
            <div>
              <div className="section-label">Threshold</div>
              <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text)", fontVariantNumeric: "tabular-nums", fontFamily: "monospace" }}>
                {threshold.toFixed(2)}
              </div>
            </div>
            <div style={{ flex: 1 }} />
            <div style={{ display: "flex", alignItems: "center" }}>
              <span className="text-muted text-xs">Auto-refreshes every 15s</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
