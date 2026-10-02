// src/pages/Dashboard.jsx
import { useEffect, useState, useCallback } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from "recharts";
import { getStats, getThreshold, setThreshold } from "../api";

function StatCard({ label, value, sub, className }) {
  return (
    <div className={`stat-card ${className || ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

const HOURS = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, "0"));

export default function Dashboard() {
  const [stats,      setStats]      = useState({ today: {}, hourly: [] });
  const [threshold,  setThresholdV] = useState(0.5);
  const [saved,      setSaved]      = useState(false);
  const [loading,    setLoading]    = useState(true);

  const refresh = useCallback(() => {
    Promise.all([getStats(), getThreshold()]).then(([sRes, tRes]) => {
      setStats(sRes.data);
      setThresholdV(tRes.data.threshold);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 15000);
    return () => clearInterval(t);
  }, [refresh]);

  // Build chart data (fill missing hours with 0)
  const chartData = HOURS.map((h) => {
    const row = (stats.hourly || []).find((r) => r.hour === h);
    return { hour: `${h}:00`, passed: row?.passed || 0, failed: row?.failed || 0 };
  }).filter((d) => d.passed > 0 || d.failed > 0 || Number(d.hour.split(":")[0]) <= new Date().getHours());

  const handleSave = async () => {
    try {
      await setThreshold(threshold);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {/* ignore */}
  };

  const { total = 0, passed = 0, failed = 0, rejection_rate = 0 } = stats.today || {};

  return (
    <div>
      {/* Stat Cards */}
      <div className="stat-grid">
        <StatCard label="Total Inspected" value={total} sub="Today" className="primary" />
        <StatCard label="Passed"          value={passed} sub="Today" className="success" />
        <StatCard label="Failed"          value={failed} sub="Today" className="danger" />
        <StatCard label="Rejection Rate"  value={`${rejection_rate}%`} sub="Today" className={rejection_rate > 20 ? "danger" : ""} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 24 }}>
        {/* Chart */}
        <div className="card">
          <div className="card-header">
            <h3>Hourly Inspections</h3>
            <span className="text-muted">Today</span>
          </div>
          <div className="card-body">
            {loading ? (
              <div className="loading-wrap"><div className="spinner" /></div>
            ) : chartData.length === 0 ? (
              <div className="empty-state">
                <div className="icon">📭</div>
                <p>No inspections yet today.<br />Head to <strong>Inspect</strong> to begin.</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="gPass" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#06d6a0" stopOpacity={.25} />
                      <stop offset="95%" stopColor="#06d6a0" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="gFail" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#ef476f" stopOpacity={.25} />
                      <stop offset="95%" stopColor="#ef476f" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e5ef" />
                  <XAxis dataKey="hour" tick={{ fontSize: 11, fill: "#7b8099" }} />
                  <YAxis tick={{ fontSize: 11, fill: "#7b8099" }} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{ borderRadius: 8, border: "1px solid #e2e5ef", fontSize: 13 }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Area type="monotone" dataKey="passed" name="Passed" stroke="#06d6a0" fill="url(#gPass)" strokeWidth={2} dot={false} />
                  <Area type="monotone" dataKey="failed" name="Failed" stroke="#ef476f" fill="url(#gFail)" strokeWidth={2} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Threshold */}
        <div className="card">
          <div className="card-header">
            <h3>Detection Threshold</h3>
          </div>
          <div className="card-body threshold-section">
            <p className="text-muted" style={{ fontSize: ".82rem", lineHeight: 1.6 }}>
              Scores above the threshold are marked <strong style={{ color: "var(--danger)" }}>FAIL</strong>.
              Lower = stricter. Higher = more lenient.
            </p>

            <div>
              <div className="threshold-value-display mb-4">
                <span className="section-title">Current Threshold</span>
                <span className="threshold-num">{threshold.toFixed(2)}</span>
              </div>
              <div className="slider-wrap">
                <div className="slider-labels">
                  <span>0.10 (Strict)</span>
                  <span>0.90 (Lenient)</span>
                </div>
                <input
                  id="threshold-slider"
                  type="range"
                  className="threshold-slider"
                  min={0.1} max={0.9} step={0.01}
                  value={threshold}
                  onChange={(e) => setThresholdV(parseFloat(e.target.value))}
                />
              </div>
            </div>

            {/* Live interpretation */}
            <div style={{
              padding: "12px 14px",
              background: threshold <= 0.35 ? "rgba(239,71,111,.07)" :
                          threshold >= 0.65 ? "rgba(6,214,160,.07)" :
                          "rgba(67,97,238,.07)",
              borderRadius: "var(--radius-sm)",
              fontSize: ".8rem",
              color: "var(--text-muted)",
            }}>
              {threshold <= 0.35 && "⚠️ High sensitivity — more items will FAIL."}
              {threshold > 0.35 && threshold < 0.65 && "✅ Balanced — recommended starting point."}
              {threshold >= 0.65 && "⚙️ Lenient — only extreme defects will FAIL."}
            </div>

            <button
              id="save-threshold-btn"
              className={`btn ${saved ? "btn-success" : "btn-primary"} btn-lg w-full`}
              onClick={handleSave}
            >
              {saved ? "✓ Saved!" : "Save Threshold"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
