// src/pages/Dashboard.jsx
import { useEffect, useState, useCallback } from "react";
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";
import { getStats, getThreshold, setThreshold, subscribeLive } from "../api";
import { COLORS } from "../theme";
import { Icon, PageHead, Segmented, Spinner } from "../ui";

// ─── KPI cell ────────────────────────────────────────────────────────────────
function Kpi({ label, value, sub, tone, animDelay = 0 }) {
  const [displayed, setDisplayed] = useState(0);
  const target = typeof value === "number" ? value : null;

  useEffect(() => {
    if (target === null) return;
    let start = null;
    const duration = 700;
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

  return (
    <div className="kpi">
      <div className="label">{label}</div>
      <div className={`kpi-value ${tone || ""}`}>{target !== null ? displayed : value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

// ─── Chart tooltip ───────────────────────────────────────────────────────────
const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#fff", border: `1px solid ${COLORS.line}`, borderRadius: 6,
      padding: "8px 12px", fontSize: ".8rem",
    }}>
      <div style={{ color: COLORS.ink3, marginBottom: 4 }}>{label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: p.color }} />
          <span style={{ color: COLORS.ink2 }}>{p.name}</span>
          <span style={{ marginLeft: "auto", paddingLeft: 12, fontWeight: 600 }}>{p.value}</span>
        </div>
      ))}
    </div>
  );
};

const HOURS = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, "0"));
const AXIS = { fontSize: 11, fill: COLORS.ink3 };

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

  // Hourly chart data (fill up to current hour)
  const curHour = new Date().getHours();
  const hourlyData = HOURS
    .filter((h) => Number(h) <= curHour)
    .map((h) => {
      const row = (stats.hourly || []).find((r) => r.hour === h);
      return { hour: `${h}:00`, passed: row?.passed || 0, failed: row?.failed || 0 };
    });

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

  const mode =
    threshold <= 0.35 ? { label: "Strict",   hint: "More items will be flagged as FAIL." } :
    threshold >= 0.65 ? { label: "Lenient",  hint: "Only major defects will trigger FAIL." } :
                        { label: "Balanced", hint: "Recommended starting point." };

  const pct = ((threshold - 0.1) / 0.8) * 100;

  return (
    <div>
      <PageHead title="Overview" sub="Today's inspections at a glance." />

      <div className="kpis" style={{ "--cols": 4 }}>
        <Kpi label="Inspected" value={total}  sub="units today"                animDelay={0} />
        <Kpi label="Passed"    value={passed} sub={`${passRate}% pass rate`}   tone="pass" animDelay={60} />
        <Kpi label="Failed"    value={failed} sub="defects detected"           tone={failed > 0 ? "fail" : ""} animDelay={120} />
        <Kpi label="Rejection" value={`${rejection_rate}%`} sub="of today's units" tone={rejection_rate > 20 ? "fail" : ""} animDelay={180} />
      </div>

      <div className="split">
        {/* Throughput */}
        <div className="card">
          <div className="card-head">
            <h2>{chartTab === "hourly" ? "Throughput by hour" : "Last 7 days"}</h2>
            <Segmented
              options={[{ key: "hourly", label: "Today" }, { key: "weekly", label: "Week" }]}
              value={chartTab}
              onChange={setChartTab}
            />
          </div>
          <div className="card-body">
            {loading ? (
              <div className="loading"><Spinner /></div>
            ) : chartData.length === 0 ? (
              <div className="empty">
                <h3>No data yet</h3>
                <p>Run a few checks on the Inspect page and they will show up here.</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                {chartTab === "hourly" ? (
                  <LineChart data={chartData} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke={COLORS.line} />
                    <XAxis dataKey={chartKey} tick={AXIS} tickLine={false} axisLine={false} />
                    <YAxis tick={AXIS} tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip content={<ChartTooltip />} cursor={{ stroke: COLORS.line }} />
                    <Legend iconType="plainline" wrapperStyle={{ fontSize: 12, color: COLORS.ink2, paddingTop: 12 }} />
                    <Line type="monotone" dataKey="passed" name="Passed" stroke={COLORS.pass} strokeWidth={1.75} dot={false} />
                    <Line type="monotone" dataKey="failed" name="Failed" stroke={COLORS.fail} strokeWidth={1.75} dot={false} />
                  </LineChart>
                ) : (
                  <BarChart data={chartData} margin={{ top: 8, right: 8, left: -24, bottom: 0 }} barGap={2}>
                    <CartesianGrid vertical={false} stroke={COLORS.line} />
                    <XAxis dataKey={chartKey} tick={AXIS} tickLine={false} axisLine={false} />
                    <YAxis tick={AXIS} tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(18,18,18,0.04)" }} />
                    <Legend iconType="square" iconSize={8} wrapperStyle={{ fontSize: 12, color: COLORS.ink2, paddingTop: 12 }} />
                    <Bar dataKey="passed" name="Passed" fill={COLORS.pass} maxBarSize={18} />
                    <Bar dataKey="failed" name="Failed" fill={COLORS.fail} maxBarSize={18} />
                  </BarChart>
                )}
              </ResponsiveContainer>
            )}
          </div>

          {!loading && total > 0 && (
            <div className="statline">
              <div><span className="label">Avg confidence</span><b>{avg_confidence.toFixed(1)}%</b></div>
              <div><span className="label">Pass rate</span><b>{passRate}%</b></div>
              <div><span className="label">Threshold</span><b className="mono">{threshold.toFixed(2)}</b></div>
            </div>
          )}
        </div>

        {/* Threshold */}
        <div className="card">
          <div className="card-head">
            <h2>Detection threshold</h2>
            <span className="small muted">{mode.label}</span>
          </div>
          <div className="card-body">
            <p className="small muted" style={{ marginBottom: 20 }}>
              An anomaly score at or above this value is marked <strong style={{ color: "var(--fail)", fontWeight: 600 }}>FAIL</strong>.
              Lower is stricter.
            </p>

            <div className="threshold-num num">{threshold.toFixed(2)}<small>/ 1.00</small></div>

            <input
              id="threshold-slider"
              type="range"
              className="range"
              style={{ "--pct": `${pct}%` }}
              min={0.1} max={0.9} step={0.01}
              value={threshold}
              onChange={(e) => setThreshV(parseFloat(e.target.value))}
            />
            <div className="range-ends"><span>0.10 · strict</span><span>0.90 · lenient</span></div>

            <p className="hint" style={{ margin: "18px 0 20px" }}>{mode.hint}</p>

            <button
              id="save-threshold-btn"
              className="btn btn-primary btn-block"
              onClick={handleSave}
              disabled={saving}
            >
              {saving ? <><span className="spin" /> Saving</> : saved ? <><Icon name="check" size={15} /> Saved</> : "Save threshold"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
