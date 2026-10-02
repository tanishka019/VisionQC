// src/api.js — VisionQC v2 API layer
import axios from "axios";

export const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,  // 2 min for slow inference
});

// Response interceptor for unified error shape
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg =
      err.response?.data?.detail ||
      err.response?.data?.message ||
      err.message ||
      "Request failed";
    return Promise.reject({ ...err, userMessage: msg });
  }
);

// ─── Health & Status ───────────────────────────────
export const getHealth      = ()              => api.get("/health");
export const getModelStatus = ()              => api.get("/model/status");
export const resetModel     = ()              => api.delete("/model/reset");

// ─── Config ────────────────────────────────────────
export const getThreshold   = ()              => api.get("/threshold");
export const setThreshold   = (t)             => api.post("/threshold", { threshold: t });

// ─── Stats ─────────────────────────────────────────
export const getStats       = ()              => api.get("/stats");

// ─── History ───────────────────────────────────────
export const getHistory = (limit = 200, filter = null, page = 1) => {
  const params = new URLSearchParams({ limit, page });
  if (filter && filter !== "all") params.append("filter", filter.toUpperCase());
  return api.get(`/history?${params}`);
};

export const deleteInspection = (id)          => api.delete(`/history/${id}`);
export const clearHistory     = ()            => api.delete("/history");

// ─── Train ─────────────────────────────────────────
export const trainModel = (files, productName) => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("product_name", productName);
  return api.post("/train", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 600000,  // 10 min for training
  });
};

// ─── Inspect ───────────────────────────────────────
export const inspectImage = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/inspect", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const batchInspect = (files) => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return api.post("/inspect/batch", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300000,
  });
};
