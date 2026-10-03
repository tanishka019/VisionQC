// src/pages/Dashboard.jsx
import { useEffect, useState, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";
import { getStats, getThreshold, setThreshold, subscribeLive, downloadTodayReport } from "../api";
import { COLORS } from "../theme";
import { Icon, PageHead, HourStrip, Spinner } from "../ui";

// Counts up to `target` once on mount / when the target changes.
function useCountUp(target, delay = 0, duration = 800) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (typeof target !== "number") return;
    let start = null;
    let raf;
    const step = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      setShown(Math.round((1 - Math.pow(1 - progress, 3)) * target));
      if (progress < 1) raf = requestAnimationFrame(step);
    };
    const id = setTimeout(() => { raf = requestAnimationFrame(step); }, delay);
    return () => { clearTimeout(id); cancelAnimationFrame(raf); };
  }, [target, delay, duration]);
  return typeof target === "number" ? shown : target;
}

function HeroStat({ label, value, sub, tone, delay }) {
  const shown = useCountUp(typeof value === "number" ? value : null, delay);
  return (
    <div className="hero-stat">
      <div className="label">{label}</div>
      <div className={`figure ${tone || ""}`}>{typeof value === "number" ? shown : value}</div>
      <small>{sub}</small>
    </div>
  );
}

// ─── Chart tooltip ───────────────────────────────────────────────────────────
const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#141414", color: "#f3f2ee", borderRadius: 10,
      padding: "9px 13px", fontSize: ".8rem",
    }}>
      <div style={{ color: "#a2a19b", marginBottom: 4, fontFamily: "var(--mono)", fontSize: ".7rem" }}>{label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: p.color === COLORS.ink ? "#cbf55c" : p.color }} />
          <span style={{ color: "#a2a19b" }}>{p.name}</span>
          <span style={{ marginLeft: "auto", paddingLeft: 14, fontWeight: 600 }}>{p.value}</span>
        </div>
      ))}
    </div>
  );
};

const AXIS = { fontSize: 11, fill: COLORS.ink3 };

export default function Dashboard() {
  const [stats,     setStats]     = useState({ today: {}, hourly: [], weekly: [] });
  const [threshold, setThreshV]   = useState(0.25);
  const [saving,    setSaving]    = useState(false);
  const [saved,     setSaved]     = useState(false);
  const [loading,   setLoading]   = useState(true);

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

  const now = new Date();
  const dateStr = now.toLocaleDateString("en-IN", { weekday: "short", day: "numeric", month: "short" }).toUpperCase();

  const weeklyData = (stats.weekly || []).map((r) => ({
    day: new Date(r.day).toLocaleDateString("en-IN", { weekday: "short", day: "numeric" }),
    passed: r.passed,
    failed: r.failed,
    total:  r.total,
  }));

  const [reporting, setReporting] = useState(false);
  const handleReport = async () => {
    setReporting(true);
    try {
      const res = await downloadTodayReport();
      const url = URL.createObjectURL(res.data);
      const a = document.createElement("a");
      a.href = url;
      a.download = `visionqc-report-${new Date().toISOString().slice(0, 10)}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch { /* noop */ }
    finally { setReporting(false); }
  };

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
  const bigTotal = useCountUp(total, 0, 900);

  const mode =
    threshold <= 0.35 ? { label: "Strict",   hint: "More items will be flagged as FAIL." } :
    threshold >= 0.65 ? { label: "Lenient",  hint: "Only major defects will trigger FAIL." } :
                        { label: "Balanced", hint: "Recommended starting point." };

  const pct = ((threshold - 0.1) / 0.8) * 100;

  return (
    <div>
      <PageHead title="Overview" sub="Today's inspections at a glance." />
      <div style={{ display: "flex", justifyContent: "flex-end", margin: "-8px 0 16px" }}>
        <button id="download-report-btn" className="btn btn-primary" onClick={handleReport} disabled={reporting}>
          {reporting ? <><span className="spin" /> Preparing</> : "Download today's report"}
        </button>
      </div>

      <section className="panel hero">
        <div className="hero-top">
          <span className="label">Today</span>
          <span className="label">{dateStr}</span>
        </div>

        <div className="hero-grid">
          <div className="hero-big">
            <div className="label">Inspected</div>
            <div className="big" style={{ marginTop: 14 }}>{bigTotal}</div>
            <p>units checked so far today</p>
          </div>
          <div className="hero-side">
            <HeroStat label="Passed"    value={passed} sub={`${passRate}% pass rate`} tone="pass" delay={80} />
            <HeroStat label="Failed"    value={failed} sub="defects found"            tone={failed > 0 ? "fail" : ""} delay={160} />
            <HeroStat label="Rejection" value={`${rejection_rate}%`} sub="of today's units" delay={240} />
          </div>
        </div>

        <div className="strip">
          <div className="label" style={{ marginBottom: 14 }}>Activity by hour</div>
          <HourStrip rows={stats.hourly} currentHour={now.getHours()} />
        </div>
      </section>

      <div className="split">
        {/* Last 7 days */}
        <div className="card">
          <div className="card-head">
            <h2>Last 7 days</h2>
            <span className="small muted hint-sm">Passed vs failed per day</span>
          </div>
          <div className="card-body">
            {loading ? (
              <div className="loading"><Spinner /></div>
            ) : weeklyData.length === 0 ? (
              <div className="empty">
                <h3>No data yet</h3>
                <p>Run a few checks on the Inspect page and they will show up here.</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={weeklyData} margin={{ top: 8, right: 8, left: -24, bottom: 0 }} barGap={3}>
                  <CartesianGrid vertical={false} stroke={COLORS.line} />
                  <XAxis dataKey="day" tick={AXIS} tickLine={false} axisLine={false} />
                  <YAxis tick={AXIS} tickLine={false} axisLine={false} allowDecimals={false} />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(14,14,14,0.05)" }} />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, color: COLORS.ink2, paddingTop: 14 }} />
                  <Bar dataKey="passed" name="Passed" fill={COLORS.ink}  radius={[4, 4, 0, 0]} maxBarSize={16} />
                  <Bar dataKey="failed" name="Failed" fill={COLORS.fail} radius={[4, 4, 0, 0]} maxBarSize={16} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          {!loading && total > 0 && (
            <div style={{ display: "flex", gap: 40, flexWrap: "wrap", padding: "18px 22px", borderTop: "1px solid var(--line)" }}>
              <div><span className="label">Avg confidence</span><div className="mono num" style={{ fontSize: "1.05rem", marginTop: 4 }}>{avg_confidence.toFixed(1)}%</div></div>
              <div><span className="label">Pass rate</span><div className="mono num" style={{ fontSize: "1.05rem", marginTop: 4 }}>{passRate}%</div></div>
              <div><span className="label">Threshold</span><div className="mono num" style={{ fontSize: "1.05rem", marginTop: 4 }}>{threshold.toFixed(2)}</div></div>
            </div>
          )}
        </div>

        {/* Threshold */}
        <div className="card">
          <div className="card-head">
            <h2>Detection threshold</h2>
            <span className="pill neutral">{mode.label}</span>
          </div>
          <div className="card-body">
            <p className="small muted" style={{ marginBottom: 22 }}>
              An anomaly score at or above this value is marked <strong style={{ color: "var(--fail)", fontWeight: 600 }}>FAIL</strong>.
              Lower is stricter.
            </p>

            <div className="threshold-num num">{threshold.toFixed(2)}<small>/ 1.00</small></div>

            <input
              id="threshold-slider"
              type="range"
              className="range"
              style={{ "--pct": `${pct}%`, marginTop: 22 }}
              min={0.1} max={0.9} step={0.01}
              value={threshold}
              onChange={(e) => setThreshV(parseFloat(e.target.value))}
            />
            <div className="range-ends"><span>0.10 · STRICT</span><span>0.90 · LENIENT</span></div>

            <p className="hint" style={{ margin: "20px 0 22px" }}>{mode.hint}</p>

            <button
              id="save-threshold-btn"
              className="btn btn-primary btn-lg btn-block"
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
