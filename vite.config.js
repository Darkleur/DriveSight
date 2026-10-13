import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset paths ("./assets/...") so the built site works from any folder, e.g.
  // https://<user>.github.io/DriveSight/ on GitHub Pages (whose paths are case-sensitive),
  // without hard-coding the repository name.
  base: './',
  build: {
    // Transformers.js (which bundles ONNX Runtime) is one large library, so a chunk above
    // Vite's default 500 kB warning size is expected here.
    chunkSizeWarningLimit: 1000,
  },
});
