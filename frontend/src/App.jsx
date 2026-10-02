// src/App.jsx — VisionQC v2 Shell
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { useEffect, useState, useCallback } from "react";
import Dashboard from "./pages/Dashboard";
import Train     from "./pages/Train";
import Inspect   from "./pages/Inspect";
import History   from "./pages/History";
import { getModelStatus } from "./api";
import "./index.css";

const NAV = [
  { to: "/",        icon: "⬡",  label: "Dashboard",   section: "MONITOR" },
  { to: "/inspect", icon: "◎",  label: "Inspect",      section: null },
  { to: "/train",   icon: "⬢",  label: "Train Model",  section: "CONFIGURE" },
  { to: "/history", icon: "≡",  label: "History",      section: null },
];

function Sidebar({ modelTrained, productName }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside className="sidebar" style={collapsed ? { width: "var(--sidebar-w-sm)" } : {}}>
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="logo-wrapper">
          <div className="logo-icon-wrap">🏭</div>
          {!collapsed && (
            <div className="logo-text">
              <h1>VisionQC</h1>
              <span>AI Quality Inspection</span>
            </div>
          )}
        </div>
      </div>

      {/* Nav */}
      <nav className="sidebar-nav">
        {NAV.map(({ to, icon, label, section }, i) => (
          <div key={to}>
            {section && !collapsed && (
              <div className="nav-section-label" style={i > 0 ? { marginTop: 16 } : {}}>
                {section}
              </div>
            )}
            <NavLink
              to={to}
              end={to === "/"}
              className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
            >
              <span className="nav-icon" style={{ fontSize: "1.1rem" }}>{icon}</span>
              {!collapsed && <span className="nav-label">{label}</span>}
            </NavLink>
          </div>
        ))}
      </nav>

      {/* Footer model status */}
      <div className="sidebar-footer">
        <div className="model-status-card">
          <div className="model-status-row">
            <div className={`status-dot${modelTrained ? " trained" : ""}`} />
            {!collapsed && (
              <span className="status-label">
                {modelTrained ? "Model Ready" : "No Model"}
              </span>
            )}
          </div>
          {!collapsed && modelTrained && (
            <div className="status-product">⬡ {productName}</div>
          )}
        </div>
      </div>
    </aside>
  );
}

function PageHeader({ modelTrained }) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const timeStr = time.toLocaleTimeString("en-IN", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
  const dateStr = time.toLocaleDateString("en-IN", {
    weekday: "short", day: "numeric", month: "short",
  });

  // Derive title from pathname
  const path = window.location.pathname;
  const META = {
    "/":        { title: "Dashboard",    sub: "Real-time quality overview" },
    "/train":   { title: "Train Model",  sub: "Upload reference images to teach the model" },
    "/inspect": { title: "Inspect",      sub: "Live AI-powered defect detection" },
    "/history": { title: "History",      sub: "Inspection log & analytics" },
  };
  const { title, sub } = META[path] || { title: "VisionQC", sub: "" };

  return (
    <header className="page-header">
      <div className="header-left">
        <h2>{title}</h2>
        <p className="subtitle">{sub}</p>
      </div>
      <div className="header-right">
        {modelTrained && (
          <div className="live-indicator">
            <span className="live-dot" />
            LIVE
          </div>
        )}
        <div className="header-time mono">
          {dateStr} &nbsp;·&nbsp; {timeStr}
        </div>
      </div>
    </header>
  );
}

export default function App() {
  const [modelTrained, setModelTrained] = useState(false);
  const [productName,  setProductName]  = useState("Product");

  const refreshStatus = useCallback(() => {
    getModelStatus()
      .then((r) => {
        setModelTrained(r.data.trained);
        setProductName(r.data.product_name);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    refreshStatus();
    const t = setInterval(refreshStatus, 10000);
    return () => clearInterval(t);
  }, [refreshStatus]);

  return (
    <BrowserRouter>
      <div className="app-layout">
        <Sidebar modelTrained={modelTrained} productName={productName} />
        <div className="main-content">
          <PageHeader modelTrained={modelTrained} />
          <main className="page-body animate-fade-up">
            <Routes>
              <Route path="/"        element={<Dashboard />} />
              <Route path="/train"   element={<Train onTrained={refreshStatus} />} />
              <Route path="/inspect" element={<Inspect modelTrained={modelTrained} />} />
              <Route path="/history" element={<History />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
