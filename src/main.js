import { RawImage } from '@huggingface/transformers';
import { MODEL_ID, pickDevice, loadDetector, detect } from './detector.js';
import { drawDetections } from './draw.js';

// M1 settings (they become UI controls in M3).
const DTYPE = 'fp32'; // full precision: the unmodified model, our reference point
const THRESHOLD = 0.5; // keep detections the model is at least 50% sure about

const statusEl = document.getElementById('status');
const imageEl = document.getElementById('image');
const overlayEl = document.getElementById('overlay');

function setStatus(text) {
  statusEl.textContent = text;
}

// Turns the library's progress events into a readable status line.
function onProgress(event) {
  if (event.status === 'progress' && event.file.endsWith('.onnx')) {
    const mb = (bytes) => (bytes / 1e6).toFixed(1);
    setStatus(`Downloading ${event.file}: ${mb(event.loaded)} / ${mb(event.total)} MB (${Math.round(event.progress)}%)`);
  }
}

async function run() {
  const device = await pickDevice();
  setStatus(`Loading ${MODEL_ID} (${DTYPE}) on ${device}…`);

  const { detector, loadTimeMs } = await loadDetector({ device, dtype: DTYPE, onProgress });

  // Decode the test image once, so the timing below measures only the model, not the download.
  const image = await RawImage.fromURL(imageEl.src);

  // Warm-up: the first run is slow (the GPU compiles its programs, memory gets allocated).
  // We run it once and ignore its time so it doesn't distort the measurement.
  setStatus('Warming up…');
  await detect(detector, image, THRESHOLD);

  const start = performance.now();
  const detections = await detect(detector, image, THRESHOLD);
  const inferenceMs = performance.now() - start;

  console.log(`Model: ${MODEL_ID} | dtype: ${DTYPE} | device: ${device}`);
  console.log(`Load time: ${(loadTimeMs / 1000).toFixed(2)} s | Inference: ${inferenceMs.toFixed(1)} ms`);
  console.table(
    detections.map(({ label, score, box }) => ({
      label,
      score: score.toFixed(3),
      xmin: box.xmin.toFixed(3),
      ymin: box.ymin.toFixed(3),
      xmax: box.xmax.toFixed(3),
      ymax: box.ymax.toFixed(3),
    })),
  );

  drawDetections(overlayEl, detections);
  // Redraw when the window resizes, since the image's displayed size changes.
  window.addEventListener('resize', () => drawDetections(overlayEl, detections));

  setStatus(
    `${detections.length} objects found · ${device} · ${DTYPE} · ` +
      `load ${(loadTimeMs / 1000).toFixed(2)} s · inference ${inferenceMs.toFixed(1)} ms`,
  );
}

// Wait for the <img> to be displayed so the overlay canvas has the right size.
if (imageEl.complete) run().catch(onError);
else imageEl.addEventListener('load', () => run().catch(onError));

function onError(err) {
  console.error(err);
  setStatus(`Error: ${err.message}`);
}
