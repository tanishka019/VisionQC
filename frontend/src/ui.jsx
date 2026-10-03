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
export function Mark({ size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="square" aria-hidden="true">
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
  return <span className={`status ${kind}`}><span className="dot" />{result}</span>;
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

// Anomaly score drawn against the decision limit.
export function ScoreGauge({ score, threshold }) {
  const clamp = (v) => Math.min(Math.max(v, 0), 1);
  const fail = score >= threshold;
  return (
    <div className="gauge gauge-wrap">
      <div className="gauge-track">
        <div className={`gauge-fill ${fail ? "fail" : "pass"}`} style={{ width: `${clamp(score) * 100}%` }} />
        <div className="gauge-limit" style={{ left: `${clamp(threshold) * 100}%` }}>
          <span>limit {threshold.toFixed(2)}</span>
        </div>
      </div>
      <div className="gauge-scale"><span>0</span><span>0.5</span><span>1</span></div>
    </div>
  );
}

export function Spinner() {
  return <span className="spin" aria-hidden="true" />;
}
