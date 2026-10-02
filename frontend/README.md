# VisionQC — Frontend

React 19 + Vite UI for VisionQC. Pages live in `src/pages/` (Dashboard, Train, Inspect, History) and all backend calls go through `src/api.js`.

```bash
cp .env.example .env   # set VITE_API_URL (and VITE_API_KEY if the backend uses one)
npm install
npm run dev            # http://localhost:5173
npm run lint           # oxlint
npm run build          # production build in dist/
```

See the [root README](../README.md) for the backend, the API reference and Docker setup.
