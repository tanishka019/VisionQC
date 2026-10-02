// src/App.jsx
import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import Dashboard from "./pages/Dashboard";
import Train     from "./pages/Train";
import Inspect   from "./pages/Inspect";
import History   from "./pages/History";
import { getModelStatus } from "./api";
import "./index.css";

const NAV = [
  { to: "/",        icon: "📊", label: "Dashboard" },
  { to: "/train",   icon: "🧠", label: "Train Model" },
  { to: "/inspect", icon: "🔍", label: "Inspect" },
  { to: "/history", icon: "📋", label: "History" },
];

function Sidebar({ modelTrained, productName }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">🏭</div>
        <h1>VisionQC</h1>
        <span>AI Quality Inspection</span>
      </div>

      <nav className="sidebar-nav">
        {NAV.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
          >
            <span className="nav-icon">{icon}</span>
            <span className="nav-label">{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="model-badge">
          <div className={`dot ${modelTrained ? "trained" : ""}`} />
          <span>
            {modelTrained
              ? `Model ready\n${productName}`
              : "No model trained"}
          </span>
        </div>
      </div>
    </aside>
  );
}

function PageHeader() {
  const loc = useLocation();
  const titles = {
    "/":        { title: "Dashboard",    sub: "Today's inspection overview" },
    "/train":   { title: "Train Model",  sub: "Upload 20–30 good product images" },
    "/inspect": { title: "Live Inspect", sub: "Webcam or image upload inspection" },
    "/history": { title: "History",      sub: "All inspection logs" },
  };
  const { title, sub } = titles[loc.pathname] || { title: "VisionQC", sub: "" };
  return (
    <header className="page-header">
      <div>
        <h2>{title}</h2>
        <p className="subtitle">{sub}</p>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: ".75rem", color: "var(--text-muted)" }}>
          {new Date().toLocaleDateString("en-IN", { weekday: "short", day: "numeric", month: "short" })}
        </span>
      </div>
    </header>
  );
}

export default function App() {
  const [modelTrained, setModelTrained] = useState(false);
  const [productName, setProductName]   = useState("Product");

  const refreshStatus = () => {
    getModelStatus()
      .then((r) => {
        setModelTrained(r.data.trained);
        setProductName(r.data.product_name);
      })
      .catch(() => {});
  };

  useEffect(() => {
    refreshStatus();
    const t = setInterval(refreshStatus, 10000);
    return () => clearInterval(t);
  }, []);

  return (
    <BrowserRouter>
      <div className="app-layout">
        <Sidebar modelTrained={modelTrained} productName={productName} />
        <div className="main-content">
          <PageHeader />
          <main className="page-body">
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
