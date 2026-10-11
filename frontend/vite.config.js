import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the console runs on :5173 and proxies /api to the FastAPI service on :8000.
// In production `npm run build` writes dist/, which FastAPI serves at "/".
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist", sourcemap: false },
});
