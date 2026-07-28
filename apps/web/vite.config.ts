import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/fatura/",
  server: {
    port: 5173,
    proxy: {
      "/fatura-api": {
        target: "http://127.0.0.1:8105",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/fatura-api/, ""),
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
