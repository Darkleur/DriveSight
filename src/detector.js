import { pipeline } from '@huggingface/transformers';

// RF-DETR "nano": a small, fast object detector trained on COCO (80 everyday classes:
// person, car, bus, truck, bicycle, traffic light, ...). This repo holds ONNX versions of
// the model, a portable format that ONNX Runtime can execute inside the browser.
export const MODEL_ID = 'onnx-community/rfdetr_nano-ONNX';

// Picks where the model runs.
// - WebGPU runs the math on the graphics card: fast, but not every browser/device has it.
// - WASM (WebAssembly) runs it on the CPU: slower, but works everywhere.
// `navigator.gpu` existing is not enough: we also need the browser to hand us a real GPU "adapter".
export async function pickDevice() {
  if (!navigator.gpu) return 'wasm';
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter ? 'webgpu' : 'wasm';
  } catch {
    return 'wasm';
  }
}

// Downloads the model (or reads it from the browser cache) and prepares it to run.
//
// `dtype` is the numeric precision of the model's weights, which picks which ONNX file is downloaded:
//   'fp32' -> onnx/model.onnx        (32-bit floats, the original model, 108 MB)
//   'fp16' -> onnx/model_fp16.onnx   (16-bit floats, half the size)
//   'q8'   -> onnx/model_quantized.onnx (weights stored as 8-bit integers, ~4x smaller)
// Fewer bits = smaller download and usually faster math, but slightly less precise numbers.
export async function loadDetector({ device, dtype, onProgress }) {
  const start = performance.now();
  const detector = await pipeline('object-detection', MODEL_ID, {
    device,
    dtype,
    progress_callback: onProgress,
  });
  const loadTimeMs = performance.now() - start;
  return { detector, loadTimeMs };
}

// Runs the model on one image and returns a list of { label, score, box }.
//
// - `threshold`: the model scores every candidate box from 0 to 1 ("how sure am I this is a car?").
//   Boxes below the threshold are thrown away. Higher = fewer but more confident detections.
// - `percentage: true`: box coordinates come back as fractions of the image size (0 to 1)
//   instead of pixels, so we can scale them to whatever size the image is displayed at.
export async function detect(detector, image, threshold) {
  return detector(image, { threshold, percentage: true });
}
