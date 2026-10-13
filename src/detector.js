import { pipeline } from '@huggingface/transformers';

// Checks what the GPU can do.
// - WebGPU runs the model on the graphics card: fast, but not every browser/device has it.
//   `navigator.gpu` existing is not enough: the browser must also hand us a real GPU "adapter".
// - `f16`: whether the GPU can do math on 16-bit floats ("shader-f16"). fp16 and q4f16 models need it.
// Without WebGPU we use WASM (WebAssembly), which runs the model on the CPU: slower, but works everywhere.
export async function checkWebGPU() {
  if (!navigator.gpu) return { available: false, f16: false };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { available: false, f16: false };
    return { available: true, f16: adapter.features.has('shader-f16') };
  } catch {
    return { available: false, f16: false };
  }
}

// Transformers.js keeps downloaded files in the browser's Cache Storage, keyed by their Hub URL.
// Checking it before loading tells us whether the load time includes a download.
export async function isCached(url) {
  try {
    const cache = await caches.open('transformers-cache');
    return Boolean(await cache.match(url));
  } catch {
    return false; // Cache Storage unavailable (e.g. some private browsing modes)
  }
}

// Downloads the model (or reads it from the browser cache) and prepares it to run.
//
// `dtype` is the numeric precision of the model's weights; it picks which ONNX file is downloaded
// (see models.js). Fewer bits = smaller download and usually faster math, but less precise numbers.
//
// Returns the detector, how long loading took, and the model file's size in bytes, measured from
// the download progress events (they fire for cached files too).
export async function loadDetector({ modelId, device, dtype, onProgress }) {
  let modelBytes = 0;
  const start = performance.now();
  const detector = await pipeline('object-detection', modelId, {
    device,
    dtype,
    progress_callback: (event) => {
      if (event.status === 'progress' && event.file.endsWith('.onnx')) modelBytes = event.total;
      onProgress?.(event);
    },
  });
  const loadTimeMs = performance.now() - start;
  return { detector, loadTimeMs, modelBytes };
}

// The size every image is resized to before it enters the model, read from the model's
// preprocessor_config.json (384x384 for RF-DETR nano, 512x512 for small).
export function getInputSize(detector) {
  const { width, height } = detector.processor.image_processor.size;
  return { width, height };
}

// The model can only work on one image at a time, so calls wait in line one after another.
let queue = Promise.resolve();

// Resolves once every queued inference has finished (needed before swapping models).
export function whenIdle() {
  return queue;
}

// Runs the model on one image and returns { detections, inferenceMs }.
// `detections` is a list of { label, score, box }.
//
// - `threshold`: the model scores every candidate box from 0 to 1 ("how sure am I this is a car?").
//   Boxes below the threshold are thrown away. Higher = fewer but more confident detections.
// - `percentage: true`: box coordinates come back as fractions of the image size (0 to 1)
//   instead of pixels, so we can scale them to whatever size the image is displayed at.
// - `inferenceMs` covers preprocessing + the model + turning its raw output into boxes.
export function detect(detector, image, threshold) {
  const result = queue.then(async () => {
    const start = performance.now();
    const detections = await detector(image, { threshold, percentage: true });
    return { detections, inferenceMs: performance.now() - start };
  });
  queue = result.catch(() => {}); // a failed run must not block the next ones
  return result;
}
