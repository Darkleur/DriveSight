import { detect } from './detector.js';

// Benchmark: run the same N frames through each selected precision and summarize speed and output.

export class BenchmarkCancelled extends Error {}

// IoU ("intersection over union") is the standard overlap score for two boxes:
// the area they share divided by the area they cover together. 1 = identical, 0 = no overlap.
export function iou(a, b) {
  const w = Math.max(0, Math.min(a.xmax, b.xmax) - Math.max(a.xmin, b.xmin));
  const h = Math.max(0, Math.min(a.ymax, b.ymax) - Math.max(a.ymin, b.ymin));
  const shared = w * h;
  const areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin);
  const areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin);
  const union = areaA + areaB - shared;
  return union > 0 ? shared / union : 0;
}

// Share of the fp32 boxes that another precision also found on the same frames: same label and
// IoU >= 0.5 (the usual "it's the same object" cutoff). Each box can only be matched once.
// This measures agreement with the full-precision model, NOT accuracy: we have no human labels,
// and fp32 itself makes mistakes.
function agreementWith(reference, candidate) {
  let total = 0;
  let matched = 0;
  reference.forEach((referenceBoxes, frame) => {
    const used = new Set();
    for (const ref of referenceBoxes) {
      total += 1;
      const hit = candidate[frame].findIndex(
        (box, i) => !used.has(i) && box.label === ref.label && iou(box.box, ref.box) >= 0.5,
      );
      if (hit >= 0) {
        used.add(hit);
        matched += 1;
      }
    }
  });
  return total > 0 ? matched / total : null;
}

const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
const round = (value, decimals) => Number(value.toFixed(decimals));

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Runs the benchmark. The caller supplies how to load a precision and how to get frame i, so this
// file only contains the measuring logic.
//   load(dtype)  -> loads + warms up the model (warm-up is never timed) and returns its info
//   getFrame(i)  -> frame i as an image (seeking the video is not timed either)
//   onFrame      -> called after every frame (progress display, drawing boxes)
//   onResult     -> called with one summary row per precision
export async function runBenchmark({ dtypes, frameCount, threshold, load, getFrame, onFrame, onResult, isCancelled }) {
  let reference = null; // fp32's detections per frame, for the "match fp32" column

  for (const [step, dtype] of dtypes.entries()) {
    const model = await load(dtype);
    const times = [];
    const perFrame = [];

    for (let i = 0; i < frameCount; i++) {
      if (isCancelled()) throw new BenchmarkCancelled('Benchmark cancelled');
      const frame = await getFrame(i);
      const { detections, inferenceMs } = await detect(model.detector, frame, threshold);
      times.push(inferenceMs);
      perFrame.push(detections);
      onFrame({ dtype, step: step + 1, steps: dtypes.length, frame: i + 1, frameCount, detections });
    }

    if (dtype === 'fp32') reference = perFrame;
    const avgMs = mean(times);
    const match = dtype === 'fp32' ? 1 : reference ? agreementWith(reference, perFrame) : null;
    onResult({
      precision: dtype,
      backend: model.device,
      sizeMB: round(model.modelBytes / 1e6, 1),
      loadS: round(model.loadTimeMs / 1000, 2),
      loadedFrom: model.cached ? 'cache' : 'download',
      frames: frameCount,
      avgMs: round(avgMs, 1),
      medianMs: round(median(times), 1),
      fps: round(1000 / avgMs, 2), // inference-only throughput: frames the model could process per second
      objectsPerFrame: round(mean(perFrame.map((d) => d.length)), 2),
      matchFp32: match == null ? null : round(match, 3),
      sanityOk: model.sanityOk,
    });
  }
}
