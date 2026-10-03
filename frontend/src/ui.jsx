// src/ui.jsx — small shared building blocks (no state, no data fetching)

const PATHS = {
  check:   ["M5 12.5l4.5 4.5L19 7.5"],
  x:       ["M6 6l12 12M18 6L6 18"],
  upload:  ["M12 16V4m0 0L8 8m4-4l4 4", "M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3"],
  camera:  ["M4 8h3l1.5-2h7L17 8h3v11H4z", "M12 10.5a3.25 3.25 0 100 6.5 3.25 3.25 0 000-6.5z"],
  refresh: ["M20 11a8 8 0 10-2.3 5.7", "M20 5v6h-6"],
  trash:   ["M4 7h16M10 11v6M14 11v6", "M6 7l1 13h10l1-13M9 7V4h6v3"],
  arrow:   ["M5 12h14m-6-6l6 6-6 6"],
  folder:  ["M3 7h6l2 2h10v10H3z"],
  image:   ["M4 5h16v14H4z", "M4 16l5-5 4 4 3-3 4 4", "M9 9.5a.5.5 0 110-1 .5.5 0 010 1z"],
  play:    ["M7 5l12 7-12 7z"],
  stop:    ["M6 6h12v12H6z"],
  scan:    ["M3 8V3h5M16 3h5v5M21 16v5h-5M8 21H3v-5"],
};

export function Icon({ name, size = 16, ...rest }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth="1.6"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...rest}
    >
      {PATHS[name].map((d) => <path key={d} d={d} />)}
    </svg>
  );
}

// Viewfinder mark: four corner brackets and a centre point.
export function Mark({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="square" aria-hidden="true">
      <path d="M3 8V3h5M16 3h5v5M21 16v5h-5M8 21H3v-5" />
      <circle cx="12" cy="12" r="2.4" fill="currentColor" stroke="none" />
    </svg>
  );
}

// Wraps an image/video with outside corner brackets.
export function Frame({ children }) {
  return <div className="frame"><i /><i /><i /><i />{children}</div>;
}

export function Status({ result }) {
  const kind = result === "PASS" ? "pass" : result === "FAIL" ? "fail" : "neutral";
  return <span className={`pill ${kind}`}><span className="dot" />{result}</span>;
}

export function PageHead({ title, sub, children }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {sub && <p>{sub}</p>}
      </div>
      {children}
    </div>
  );
}

// Segmented control. `idPrefix` keeps stable element ids (`<prefix>-<key>-btn`).
export function Segmented({ options, value, onChange, idPrefix }) {
  return (
    <div className="seg" role="tablist">
      {options.map(({ key, label }) => (
        <button
          key={key}
          id={idPrefix ? `${idPrefix}-${key}-btn` : undefined}
          role="tab"
          aria-selected={value === key}
          className={value === key ? "active" : ""}
          onClick={() => onChange(key)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

// Anomaly score read off a ruler, with the decision limit marked.
const TICKS = 41;
export function TickGauge({ score, threshold }) {
  const clamp = (v) => Math.min(Math.max(v, 0), 1);
  const limitIdx = Math.round(clamp(threshold) * (TICKS - 1));
  const scoreIdx = Math.round(clamp(score) * (TICKS - 1));
  return (
    <div className="gauge">
      <span className="gauge-limit-label" style={{ left: `${(limitIdx / (TICKS - 1)) * 100}%` }}>
        LIMIT {threshold.toFixed(2)}
      </span>
      <div className="gauge-ticks">
        {Array.from({ length: TICKS }, (_, i) => (
          <i key={i} className={`${i <= scoreIdx ? "on" : ""}${i === limitIdx ? " limit" : ""}`} />
        ))}
      </div>
      <div className="gauge-scale"><span>0</span><span>0.5</span><span>1</span></div>
    </div>
  );
}

// 24-hour activity strip: stacked pass / fail bars, quiet ticks for hours with no data.
export function HourStrip({ rows, currentHour }) {
  const byHour = Object.fromEntries((rows || []).map((r) => [Number(r.hour), r]));
  const max = Math.max(1, ...Object.values(byHour).map((r) => (r.passed || 0) + (r.failed || 0)));
  return (
    <div>
      <div className="strip-bars">
        {Array.from({ length: 24 }, (_, h) => {
          const r = byHour[h];
          const p = r?.passed || 0;
          const f = r?.failed || 0;
          const future = h > currentHour;
          return (
            <div key={h} className="strip-col" title={`${String(h).padStart(2, "0")}:00 — ${p} passed, ${f} failed`}>
              {p + f === 0 || future ? (
                <i className="e" />
              ) : (
                <>
                  {f > 0 && <i className="f" style={{ height: `${(f / max) * 100}%` }} />}
                  {p > 0 && <i className="p" style={{ height: `${(p / max) * 100}%` }} />}
                </>
              )}
            </div>
          );
        })}
      </div>
      <div className="strip-ticks"><span>00</span><span>06</span><span>12</span><span>18</span><span>24</span></div>
    </div>
  );
}

export function Spinner() {
  return <span className="spin" aria-hidden="true" />;
}
