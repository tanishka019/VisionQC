// src/pages/History.jsx
import { useEffect, useState, useCallback } from "react";
import { getHistory } from "../api";

function fmt(ts) {
  try {
    const d = new Date(ts);
    return d.toLocaleString("en-IN", {
      day: "2-digit", month: "short",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  } catch {
    return ts;
  }
}

function ScoreBar({ value, threshold }) {
  const pct = Math.min(Math.max(value * 100, 0), 100);
  const color = value >= threshold ? "var(--danger)" : "var(--success)";
  return (
    <div className="score-bar-wrap">
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span style={{ fontSize: ".82rem", fontWeight: 600, minWidth: 36 }}>
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

export default function History() {
  const [rows,    setRows]    = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter,  setFilter]  = useState("all"); // all | pass | fail
  const [search,  setSearch]  = useState("");

  const refresh = useCallback(() => {
    getHistory(500).then((r) => {
      setRows(r.data.inspections || []);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 15000);
    return () => clearInterval(t);
  }, [refresh]);

  const filtered = rows.filter((r) => {
    if (filter === "pass" && r.result !== "PASS") return false;
    if (filter === "fail" && r.result !== "FAIL") return false;
    if (search && !r.timestamp.includes(search)) return false;
    return true;
  });

  const totalShown = filtered.length;
  const failCount  = filtered.filter((r) => r.result === "FAIL").length;
  const rateShown  = totalShown > 0 ? ((failCount / totalShown) * 100).toFixed(1) : "0.0";

  return (
    <div>
      {/* Controls */}
      <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 6 }}>
          {["all", "pass", "fail"].map((f) => (
            <button
              key={f}
              id={`filter-${f}-btn`}
              className={`btn ${filter === f ? "btn-primary" : "btn-outline"}`}
              style={{ padding: "7px 14px", fontSize: ".8rem", textTransform: "capitalize" }}
              onClick={() => setFilter(f)}
            >
              {f === "all" ? "All" : f === "pass" ? "✅ Pass" : "❌ Fail"}
            </button>
          ))}
        </div>
        <input
          className="input"
          style={{ maxWidth: 220 }}
          placeholder="Filter by date…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <button className="btn btn-outline" onClick={refresh} style={{ padding: "7px 14px" }}>
          🔄 Refresh
        </button>
        <span style={{ marginLeft: "auto", fontSize: ".82rem", color: "var(--text-muted)" }}>
          {totalShown} records · Reject rate: <strong style={{ color: "var(--danger)" }}>{rateShown}%</strong>
        </span>
      </div>

      {/* Table */}
      <div className="card">
        <div className="card-body" style={{ padding: 0 }}>
          {loading ? (
            <div className="loading-wrap"><div className="spinner" /></div>
          ) : filtered.length === 0 ? (
            <div className="empty-state" style={{ padding: 60 }}>
              <div className="icon">📋</div>
              <p>No inspections found.<br />Go to <strong>Inspect</strong> to start.</p>
            </div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Timestamp</th>
                    <th>Result</th>
                    <th>Anomaly Score</th>
                    <th>Confidence</th>
                    <th>Threshold</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((row) => (
                    <tr key={row.id}>
                      <td style={{ color: "var(--text-muted)", fontSize: ".8rem" }}>#{row.id}</td>
                      <td style={{ fontSize: ".82rem", color: "var(--text-muted)" }}>{fmt(row.timestamp)}</td>
                      <td>
                        <span className={`badge ${row.result === "PASS" ? "badge-pass" : "badge-fail"}`}>
                          {row.result === "PASS" ? "✅ " : "❌ "}{row.result}
                        </span>
                      </td>
                      <td style={{ minWidth: 160 }}>
                        <ScoreBar value={row.anomaly_score} threshold={row.threshold} />
                      </td>
                      <td style={{ fontWeight: 600 }}>{row.confidence?.toFixed(1)}%</td>
                      <td style={{ color: "var(--text-muted)", fontSize: ".82rem" }}>
                        {(row.threshold * 100).toFixed(0)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Legend */}
      {filtered.length > 0 && (
        <p className="text-muted mt-4" style={{ fontSize: ".78rem" }}>
          Showing {totalShown} of {rows.length} total records. Auto-refreshes every 15s.
        </p>
      )}
    </div>
  );
}
