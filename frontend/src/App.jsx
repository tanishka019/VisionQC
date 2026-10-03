// src/App.jsx — VisionQC shell
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { useEffect, useState, useCallback } from "react";
import Dashboard from "./pages/Dashboard";
import Train     from "./pages/Train";
import Inspect   from "./pages/Inspect";
import History   from "./pages/History";
import { Mark } from "./ui";
import { getModelStatus } from "./api";
import "./index.css";

const NAV = [
  { to: "/",        label: "Overview" },
  { to: "/inspect", label: "Inspect" },
  { to: "/train",   label: "Train" },
  { to: "/history", label: "History" },
];

function TopBar({ modelTrained, productName }) {
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <NavLink to="/" className="brand" aria-label="VisionQC">
          <Mark />
          <span>VisionQC</span>
        </NavLink>

        <nav className="nav">
          {NAV.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) => (isActive ? "active" : undefined)}
            >
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="model-chip" title={modelTrained ? "A trained model is loaded" : "Train a model to start inspecting"}>
          <span className={`dot${modelTrained ? " on" : ""}`} />
          {modelTrained ? (
            <span><span className="hide-sm">Model ready · </span><span className="mono">{productName}</span></span>
          ) : (
            <span>No model</span>
          )}
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
      <TopBar modelTrained={modelTrained} productName={productName} />
      <main className="page fade">
        <Routes>
          <Route path="/"        element={<Dashboard />} />
          <Route path="/train"   element={<Train onTrained={refreshStatus} />} />
          <Route path="/inspect" element={<Inspect modelTrained={modelTrained} />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
