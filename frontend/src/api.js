// src/api.js — Axios API wrapper
import axios from "axios";

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: BASE });

export const getHealth       = ()              => api.get("/health");
export const getThreshold    = ()              => api.get("/threshold");
export const setThreshold    = (t)             => api.post("/threshold", { threshold: t });
export const getModelStatus  = ()              => api.get("/model/status");
export const getStats        = ()              => api.get("/stats");
export const getHistory      = (limit = 200)   => api.get(`/history?limit=${limit}`);

export const trainModel = (files, productName) => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("product_name", productName);
  return api.post("/train", form, { headers: { "Content-Type": "multipart/form-data" } });
};

export const inspectImage = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/inspect", form, { headers: { "Content-Type": "multipart/form-data" } });
};

export const API_BASE = BASE;
