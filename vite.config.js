import { defineConfig } from 'vite';

export default defineConfig(({ command, isPreview }) => ({
  // GitHub Pages serves this site from https://<user>.github.io/drivesight/, so built asset URLs
  // need the /drivesight/ prefix. The dev server keeps plain / for convenience.
  base: command === 'build' || isPreview ? '/drivesight/' : '/',
  build: {
    // Transformers.js (which bundles ONNX Runtime) is one large library, so a chunk above
    // Vite's default 500 kB warning size is expected here.
    chunkSizeWarningLimit: 1000,
  },
}));
