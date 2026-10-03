// src/pages/History.jsx
import { useEffect, useState, useCallback } from "react";
import { getHistory, deleteInspection, clearHistory, API_BASE } from "../api";
import { COLORS } from "../theme";
import { Icon, PageHead, Segmented, Spinner, Status } from "../ui";

const PAGE = 50;

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
  const color = value >= threshold ? COLORS.fail : COLORS.pass;
  return (
    <div className="bar">
      <span><i style={{ width: `${pct}%`, background: color }} /></span>
      <span className="mono num" style={{ flex: "none", height: "auto", background: "none", minWidth: 34, fontSize: ".78rem", color: "var(--ink-2)" }}>
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
  const [shown,   setShown]   = useState(PAGE);

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
    <div>
      <PageHead title="History" sub="Every inspection, newest first." />

      {!loading && rows.length > 0 && (
        <div className="kpis compact" style={{ "--cols": 5 }}>
          <div className="kpi"><div className="label">Total records</div><div className="kpi-value">{total.toLocaleString()}</div></div>
          <div className="kpi"><div className="label">Showing</div><div className="kpi-value">{filtered.length}</div></div>
          <div className="kpi"><div className="label">Pass</div><div className="kpi-value pass">{passCount}</div></div>
          <div className="kpi"><div className="label">Fail</div><div className={`kpi-value ${failCount > 0 ? "fail" : ""}`}>{failCount}</div></div>
          <div className="kpi"><div className="label">Reject rate</div><div className={`kpi-value ${parseFloat(rateShown) > 20 ? "fail" : ""}`}>{rateShown}%</div></div>
        </div>
      )}

      <div className="controls-row">
        <Segmented
          idPrefix="filter"
          options={[{ key: "all", label: "All" }, { key: "pass", label: "Pass" }, { key: "fail", label: "Fail" }]}
          value={filter}
          onChange={(k) => { setFilter(k); setShown(PAGE); }}
        />
        <input
          className="input"
          placeholder="Search by date or filename"
          value={search}
          onChange={(e) => { setSearch(e.target.value); setShown(PAGE); }}
        />
        <button className="btn btn-secondary" onClick={refresh}><Icon name="refresh" size={14} /> Refresh</button>
        <div style={{ marginLeft: "auto" }}>
          {rows.length > 0 && (
            <button className="btn btn-danger" onClick={handleClear}><Icon name="trash" size={14} /> Clear all</button>
          )}
        </div>
      </div>

      <div className="card">
        {loading ? (
          <div className="loading"><Spinner /></div>
        ) : filtered.length === 0 ? (
          <div className="empty">
            <h3>No records</h3>
            <p>{rows.length === 0 ? "Inspect a part and it will be logged here." : "Try a different filter or search term."}</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Time</th>
                  <th>File</th>
                  <th>Result</th>
                  <th>Anomaly score</th>
                  <th>Confidence</th>
                  <th>Threshold</th>
                  <th>Heatmap</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, shown).map((row) => (
                  <tr key={row.id}>
                    <td><span className="mono small muted">{row.id}</span></td>
                    <td><span className="mono small">{fmt(row.timestamp)}</span></td>
                    <td><span className="mono small truncate" title={row.filename}>{row.filename || "—"}</span></td>
                    <td><Status result={row.result} /></td>
                    <td><ScoreBar value={row.anomaly_score} threshold={row.threshold} /></td>
                    <td><span className="mono num">{row.confidence?.toFixed(1)}%</span></td>
                    <td><span className="mono small muted num">{(row.threshold * 100).toFixed(0)}%</span></td>
                    <td>
                      {row.heatmap_path ? (
                        <a
                          href={`${API_BASE}/results/${row.heatmap_path.split(/[\\/]/).pop()}`}
                          target="_blank"
                          rel="noreferrer"
                          className="btn btn-quiet btn-sm"
                        >
                          View
                        </a>
                      ) : <span className="muted">—</span>}
                    </td>
                    <td>
                      <button
                        className="btn btn-quiet btn-sm"
                        onClick={() => handleDelete(row.id)}
                        disabled={deleting === row.id}
                        aria-label={`Delete record ${row.id}`}
                        title="Delete"
                      >
                        {deleting === row.id ? "…" : <Icon name="x" size={14} />}
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
        <div className="small muted" style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
          <span>Showing {Math.min(shown, filtered.length)} of {filtered.length} records</span>
          {shown < filtered.length && (
            <button className="btn btn-secondary btn-sm" onClick={() => setShown((n) => n + PAGE)}>
              Show {Math.min(PAGE, filtered.length - shown)} more
            </button>
          )}
        </div>
      )}
    </div>
  );
}
