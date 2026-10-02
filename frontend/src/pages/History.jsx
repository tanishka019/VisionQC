// src/pages/History.jsx — v2
import { useEffect, useState, useCallback } from "react";
import { getHistory, deleteInspection, clearHistory, API_BASE } from "../api";

function fmt(ts) {
  try {
    return new Date(ts).toLocaleString("en-IN", {
      day: "2-digit", month: "short", year: "numeric",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  } catch { return ts; }
}

function ScoreBar({ value, threshold }) {
  const pct   = Math.min(Math.max(value * 100, 0), 100);
  const color = value >= threshold ? "var(--danger)" : "var(--success)";
  return (
    <div className="score-bar-wrap">
      <div className="score-bar-track" style={{ minWidth: 80 }}>
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="mono" style={{ fontSize: ".78rem", fontWeight: 600, minWidth: 38, color: "var(--text-secondary)" }}>
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

export default function History() {
  const [rows,    setRows]    = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter,  setFilter]  = useState("all");
  const [search,  setSearch]  = useState("");
  const [total,   setTotal]   = useState(0);
  const [deleting, setDeleting] = useState(null);

  const refresh = useCallback(() => {
    getHistory(500, filter === "all" ? null : filter)
      .then((r) => {
        setRows(r.data.inspections || []);
        setTotal(r.data.total || 0);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [filter]);

  useEffect(() => {
    setLoading(true);
    refresh();
    const t = setInterval(refresh, 20000);
    return () => clearInterval(t);
  }, [refresh]);

  const filtered = rows.filter((r) => {
    if (!search) return true;
    return (
      r.timestamp?.includes(search) ||
      r.filename?.toLowerCase().includes(search.toLowerCase())
    );
  });

  const failCount  = filtered.filter((r) => r.result === "FAIL").length;
  const passCount  = filtered.filter((r) => r.result === "PASS").length;
  const rateShown  = filtered.length > 0 ? ((failCount / filtered.length) * 100).toFixed(1) : "0.0";

  const handleDelete = async (id) => {
    if (!confirm("Delete this record?")) return;
    setDeleting(id);
    try {
      await deleteInspection(id);
      setRows((prev) => prev.filter((r) => r.id !== id));
    } catch { /* noop */ }
    finally { setDeleting(null); }
  };

  const handleClear = async () => {
    if (!confirm(`Clear ALL ${rows.length} inspection records? This cannot be undone.`)) return;
    try {
      await clearHistory();
      setRows([]);
    } catch { /* noop */ }
  };

  return (
    <div className="animate-fade-up">

      {/* Summary strip */}
      {!loading && rows.length > 0 && (
        <div className="card mb-4">
          <div className="card-body" style={{ display: "flex", gap: 28, flexWrap: "wrap", padding: "14px 22px" }}>
            <div>
              <div className="section-label">Total Records</div>
              <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text)" }}>{total.toLocaleString()}</div>
            </div>
            <div style={{ width: 1, background: "var(--border)" }} />
            <div>
              <div className="section-label">Showing</div>
              <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text)" }}>{filtered.length}</div>
            </div>
            <div style={{ width: 1, background: "var(--border)" }} />
            <div>
              <div className="section-label">Pass</div>
              <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--success-light)" }}>{passCount}</div>
            </div>
            <div style={{ width: 1, background: "var(--border)" }} />
            <div>
              <div className="section-label">Fail</div>
              <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--danger-light)" }}>{failCount}</div>
            </div>
            <div style={{ width: 1, background: "var(--border)" }} />
            <div>
              <div className="section-label">Reject Rate</div>
              <div style={{ fontSize: "1.4rem", fontWeight: 800, color: parseFloat(rateShown) > 20 ? "var(--danger-light)" : "var(--warning)" }}>
                {rateShown}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="history-controls">
        {/* Filter */}
        <div className="filter-group">
          {[
            { key: "all",  label: "All" },
            { key: "pass", label: "✓ Pass" },
            { key: "fail", label: "✕ Fail" },
          ].map(({ key, label }) => (
            <button
              key={key}
              id={`filter-${key}-btn`}
              className={`filter-btn${filter === key ? " active" : ""}`}
              onClick={() => setFilter(key)}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Search */}
        <input
          className="input"
          style={{ maxWidth: 240 }}
          placeholder="Search by date or filename…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />

        {/* Refresh */}
        <button className="btn btn-secondary btn-sm" onClick={refresh} style={{ padding: "8px 14px" }}>
          ↺ Refresh
        </button>

        <div style={{ marginLeft: "auto" }}>
          {rows.length > 0 && (
            <button className="btn btn-danger btn-sm" onClick={handleClear}>
              🗑 Clear All
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="card">
        {loading ? (
          <div className="loading-wrap">
            <div className="spinner" />
            <span>Loading history…</span>
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📋</div>
            <div className="empty-title">No records found</div>
            <p className="empty-sub">
              {rows.length === 0
                ? "Go to Inspect to start quality checks."
                : "Try changing the filter or search term."}
            </p>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Timestamp</th>
                  <th>File</th>
                  <th>Result</th>
                  <th>Anomaly Score</th>
                  <th>Confidence</th>
                  <th>Threshold</th>
                  <th>Heatmap</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((row) => (
                  <tr key={row.id}>
                    <td>
                      <span className="mono text-xs" style={{ color: "var(--text-muted)" }}>
                        #{row.id}
                      </span>
                    </td>
                    <td>
                      <span className="text-xs mono" style={{ color: "var(--text-muted)" }}>
                        {fmt(row.timestamp)}
                      </span>
                    </td>
                    <td>
                      <span
                        className="text-xs mono truncate"
                        title={row.filename}
                        style={{ maxWidth: 140, display: "block", color: "var(--text-secondary)" }}
                      >
                        {row.filename || "—"}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${row.result === "PASS" ? "badge-pass" : "badge-fail"}`}>
                        {row.result === "PASS" ? "✓ " : "✕ "}{row.result}
                      </span>
                    </td>
                    <td style={{ minWidth: 150 }}>
                      <ScoreBar value={row.anomaly_score} threshold={row.threshold} />
                    </td>
                    <td>
                      <span className="mono" style={{ fontWeight: 600, fontSize: ".85rem" }}>
                        {row.confidence?.toFixed(1)}%
                      </span>
                    </td>
                    <td>
                      <span className="mono text-xs" style={{ color: "var(--text-muted)" }}>
                        {(row.threshold * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td>
                      {row.heatmap_path ? (
                        <a
                          href={`${API_BASE}/results/${row.heatmap_path.split(/[\\/]/).pop()}`}
                          target="_blank"
                          rel="noreferrer"
                          className="btn btn-ghost btn-sm"
                          style={{ padding: "4px 8px", fontSize: ".72rem" }}
                        >
                          View
                        </a>
                      ) : (
                        <span className="text-xs" style={{ color: "var(--text-muted)" }}>—</span>
                      )}
                    </td>
                    <td>
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ padding: "4px 8px", color: "var(--danger-light)", fontSize: ".72rem" }}
                        onClick={() => handleDelete(row.id)}
                        disabled={deleting === row.id}
                      >
                        {deleting === row.id ? "…" : "✕"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {filtered.length > 0 && (
        <p className="text-muted mt-3 text-xs">
          Showing {filtered.length} of {rows.length} records · Auto-refreshes every 20s
        </p>
      )}
    </div>
  );
}
