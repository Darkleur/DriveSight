# DriveSight

Real-time object detection in the browser with RF-DETR and Transformers.js. Pick a model precision (fp32, fp16, q8, q4) and see how it changes speed and results on your own device. No server: everything runs client-side via WebGPU, with a WebAssembly (CPU) fallback.

**Live demo:** https://darkleur.github.io/drivesight/

> Work in progress: benchmark mode, measured results and a full write-up are coming next.

## Run locally

```bash
npm install
npm run dev
```

Then open http://localhost:5173 in Chrome.
