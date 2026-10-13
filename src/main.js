import { checkWebGPU, isCached, loadDetector, detect, getInputSize, whenIdle } from './detector.js';
import { MODELS, PRECISIONS, findModel, findPrecision, modelFileUrl } from './models.js';
import { drawDetections, clearDetections } from './draw.js';
import {
  startWebcam,
  stopWebcam,
  playVideo,
  resetVideo,
  loadImage,
  objectUrlFor,
  captureFrame,
} from './sources.js';
import { RollingStats } from './metrics.js';

// Starting precision for each backend, picked from our measurements: on the GPU, fp32 is the
// accurate reference; on the CPU, q8 is the fastest (8-bit integer math is cheap on CPUs).
const DEFAULT_DTYPE = { webgpu: 'fp32', wasm: 'q8' };

// BASE_URL keeps this path working when the site is served from a sub-folder (GitHub Pages).
const TEST_IMAGE_URL = `${import.meta.env.BASE_URL}test-street.jpg`;

const $ = (id) => document.getElementById(id);
const el = {
  model: $('model'),
  dtype: $('dtype'),
  device: $('device'),
  threshold: $('threshold'),
  thresholdValue: $('threshold-value'),
  sourceButtons: document.querySelectorAll('button[data-source]'),
  sourceLabels: document.querySelectorAll('label[data-source]'),
  sourceControls: document.querySelectorAll('[data-source]'),
  videoFile: $('video-file'),
  imageFile: $('image-file'),
  toggle: $('toggle'),
  warning: $('warning'),
  status: $('status'),
  fps: $('m-fps'),
  ms: $('m-ms'),
  backend: $('m-backend'),
  size: $('m-size'),
  load: $('m-load'),
  loadNote: $('m-load-note'),
  image: $('image'),
  video: $('video'),
  overlay: $('overlay'),
};

const state = {
  webgpu: { available: false, f16: false },
  settings: { modelId: MODELS[0].id, dtype: 'fp32', device: 'webgpu', threshold: 0.5 },
  fellBackToWasm: false, // WebGPU was chosen but failed to load, so we switched to the CPU
  detector: null,
  loading: false,
  source: null, // 'test' | 'webcam' | 'video' | 'image'
  running: false, // is the live video loop running?
  loopId: 0, // changes on every start, so an old loop can tell it should quit
  stream: null, // the webcam stream, so we can switch the camera off
  detections: [], // last results, kept to redraw when the window resizes
};
const stats = new RollingStats(30);
const frameCanvas = document.createElement('canvas'); // reused, resized to the model's input size
const testImage = new Image(); // the bundled street photo, used for warm-up and the sanity check

function setStatus(text) {
  el.status.textContent = text;
}

function showStats({ live }) {
  el.fps.textContent = live && stats.fps ? stats.fps.toFixed(1) : '–';
  el.ms.textContent = stats.avgMs ? `${stats.avgMs.toFixed(0)} ms` : '–';
}

// Turns the library's download events into a readable status line.
function onProgress(event) {
  if (event.status === 'progress' && event.file.endsWith('.onnx')) {
    const mb = (bytes) => (bytes / 1e6).toFixed(1);
    setStatus(`Downloading ${event.file}: ${mb(event.loaded)} / ${mb(event.total)} MB (${Math.round(event.progress)}%)`);
  }
}

// ---------- Model loading ----------

async function loadModel() {
  const { modelId, dtype } = state.settings;
  let device = state.settings.device;
  const cached = await isCached(modelFileUrl(modelId, findPrecision(dtype).file));
  el.warning.hidden = true;
  setStatus(`Loading ${findModel(modelId).label} · ${dtype} · ${device}…`);

  let loaded;
  try {
    loaded = await loadDetector({ modelId, device, dtype, onProgress });
  } catch (err) {
    if (device !== 'webgpu') throw err;
    // WebGPU failed on this device: fall back to the CPU instead of crashing, and say so.
    console.warn('WebGPU failed, falling back to WASM:', err);
    device = 'wasm';
    state.settings.device = 'wasm';
    state.fellBackToWasm = true;
    el.device.value = 'wasm';
    loaded = await loadDetector({ modelId, device, dtype, onProgress });
  }
  const { detector, loadTimeMs, modelBytes } = loaded;
  state.detector = detector;

  const size = getInputSize(detector);
  frameCanvas.width = size.width;
  frameCanvas.height = size.height;

  // Warm-up + sanity check, using the bundled street photo.
  // - The first run is slow (the GPU compiles its programs, memory gets allocated), so we run it
  //   once here and leave it out of every measurement.
  // - The photo has lots of cars, people and traffic lights. If a precision finds nothing at all,
  //   its numbers have broken down on this device, and we say so instead of silently showing
  //   an empty screen. (fp16 on WebGPU does this on our test laptop.)
  setStatus('Warming up…');
  const { detections } = await detect(detector, captureFrame(testImage, frameCanvas), 0.5);
  if (detections.length === 0) {
    el.warning.textContent =
      `Warning: ${dtype} on ${device} found no objects in the built-in street test photo. ` +
      `Its results look broken on this device; try another precision.`;
    el.warning.hidden = false;
  }

  el.backend.textContent =
    device === 'wasm' && !state.webgpu.available
      ? 'wasm (no WebGPU)'
      : device === 'wasm' && state.fellBackToWasm
        ? 'wasm (WebGPU failed)'
        : device;
  el.size.textContent = `${(modelBytes / 1e6).toFixed(1)} MB`;
  el.load.textContent = `${(loadTimeMs / 1000).toFixed(1)} s`;
  el.loadNote.textContent = cached ? 'from cache' : 'incl. download';
  setStatus('Model ready');
}

// Swaps in a new model when the model, precision or backend changes.
async function changeModel() {
  const resume = state.running;
  state.running = false; // pause the live loop; the camera/video keeps playing meanwhile
  setLoading(true);
  setStatus('Switching model…');
  clearDetections(el.overlay);
  state.detections = [];
  stats.reset();
  showStats({ live: false });
  try {
    await whenIdle(); // let the frame that is being processed finish first
    await state.detector?.dispose(); // free the old model's memory (GPU memory especially)
    state.detector = null;
    await loadModel();
  } finally {
    setLoading(false);
  }
  if (resume) startLive();
  else if (isStill(state.source)) await detectStill();
}

// ---------- Input sources ----------

const isStill = (source) => source === 'test' || source === 'image';

function showMedia(kind) {
  el.image.hidden = kind !== 'image';
  el.video.hidden = kind !== 'video';
}

function markActiveSource(source) {
  for (const control of el.sourceControls) {
    control.classList.toggle('active', control.dataset.source === source);
  }
}

async function selectSource(source, file) {
  stopLive();
  resetVideo(el.video);
  clearDetections(el.overlay);
  state.detections = [];
  stats.reset();
  state.source = source;
  markActiveSource(source);
  updateToggle();

  if (isStill(source)) {
    showMedia('image');
    await loadImage(el.image, source === 'test' ? TEST_IMAGE_URL : objectUrlFor(file));
    await detectStill();
  } else {
    showMedia('video');
    if (source === 'webcam') {
      setStatus('Asking for camera permission…');
      state.stream = await startWebcam(el.video);
    } else {
      await playVideo(el.video, objectUrlFor(file));
    }
    startLive();
  }
}

// ---------- Detection ----------

// A still image only needs one pass.
async function detectStill() {
  if (!state.detector) return;
  const frame = captureFrame(el.image, frameCanvas);
  const { detections, inferenceMs } = await detect(state.detector, frame, state.settings.threshold);
  stats.add(inferenceMs);
  state.detections = detections;
  drawDetections(el.overlay, detections);
  showStats({ live: false });
  setStatus(`${detections.length} objects found in ${inferenceMs.toFixed(0)} ms`);
}

function startLive() {
  state.running = true;
  state.loopId += 1;
  el.video.play().catch(onError);
  updateToggle();
  setStatus(`Live: ${state.source === 'webcam' ? 'webcam' : 'video file'}`);
  liveLoop(state.loopId);
}

function stopLive() {
  if (!state.running) return;
  state.running = false;
  el.video.pause();
  if (state.stream) {
    stopWebcam(state.stream);
    state.stream = null;
  }
  updateToggle();
  setStatus('Stopped');
}

// The live loop: grab the current frame, run the model, draw the boxes, repeat.
// The video keeps playing smoothly underneath; the boxes update as fast as the model allows.
async function liveLoop(myId) {
  if (!state.running || myId !== state.loopId || !state.detector) return;

  // readyState >= 2 means the video has a frame we can grab.
  if (el.video.readyState >= 2) {
    const frame = captureFrame(el.video, frameCanvas);
    const { detections, inferenceMs } = await detect(state.detector, frame, state.settings.threshold);
    // The user may have pressed Stop (or changed source/model) while the model was busy.
    if (!state.running || myId !== state.loopId) return;
    stats.add(inferenceMs);
    state.detections = detections;
    drawDetections(el.overlay, detections);
    showStats({ live: true });
  }

  // Wait for the next screen refresh before grabbing another frame.
  requestAnimationFrame(() => liveLoop(myId));
}

// ---------- UI wiring ----------

function fillSettings() {
  for (const model of MODELS) el.model.add(new Option(model.label, model.id));
  el.model.value = state.settings.modelId;

  if (!state.webgpu.available) {
    const gpuOption = el.device.querySelector('option[value="webgpu"]');
    gpuOption.disabled = true;
    gpuOption.text += ' (not available)';
  }
  el.device.value = state.settings.device;
  fillPrecisions();
}

// Lists the precisions for the chosen backend. On WebGPU, fp16 and q4f16 need a GPU that can do
// 16-bit float math ("shader-f16"); those options are disabled when it can't.
function fillPrecisions() {
  el.dtype.replaceChildren();
  for (const precision of PRECISIONS) {
    const option = new Option(precision.label, precision.dtype);
    if (state.settings.device === 'webgpu' && precision.needsF16 && !state.webgpu.f16) {
      option.disabled = true;
      option.text += ' (GPU lacks fp16)';
    }
    el.dtype.add(option);
  }
  // Keep the current precision if it is still allowed, otherwise use the backend's default.
  const current = el.dtype.querySelector(`option[value="${state.settings.dtype}"]`);
  if (!current || current.disabled) state.settings.dtype = DEFAULT_DTYPE[state.settings.device];
  el.dtype.value = state.settings.dtype;
}

function showThreshold() {
  el.thresholdValue.textContent = state.settings.threshold.toFixed(2);
}

function setLoading(on) {
  state.loading = on;
  for (const control of [el.model, el.dtype, el.device, el.videoFile, el.imageFile, ...el.sourceButtons]) {
    control.disabled = on;
  }
  for (const label of el.sourceLabels) label.toggleAttribute('aria-disabled', on);
  updateToggle();
}

function updateToggle() {
  const isVideoSource = state.source === 'webcam' || state.source === 'video';
  el.toggle.disabled = state.loading || !isVideoSource;
  el.toggle.textContent = state.running ? 'Stop' : 'Start';
}

function onError(err) {
  // A video play() request was cancelled: another source was picked, or the browser paused a
  // video in a background tab. Nothing broke; the Start button can resume it.
  if (err.name === 'AbortError') {
    console.warn(err.message);
    setStatus('Video paused. Press Start to play.');
    return;
  }
  console.error(err);
  stopLive();
  const message =
    err.name === 'NotAllowedError'
      ? 'Camera access was blocked. Allow it in the browser address bar and try again.'
      : err.name === 'NotFoundError'
        ? 'No camera found on this device.'
        : `Error: ${err.message}`;
  setStatus(message);
}

function wireControls() {
  el.model.addEventListener('change', () => {
    state.settings.modelId = el.model.value;
    changeModel().catch(onError);
  });
  el.dtype.addEventListener('change', () => {
    state.settings.dtype = el.dtype.value;
    changeModel().catch(onError);
  });
  el.device.addEventListener('change', () => {
    state.settings.device = el.device.value;
    state.fellBackToWasm = false;
    fillPrecisions();
    changeModel().catch(onError);
  });

  el.threshold.addEventListener('input', () => {
    state.settings.threshold = Number(el.threshold.value);
    showThreshold();
  });
  // On release, re-run still images. Live video picks up the new value on its next frame.
  el.threshold.addEventListener('change', () => {
    if (!state.running && !state.loading && isStill(state.source)) detectStill().catch(onError);
  });

  for (const button of el.sourceButtons) {
    button.addEventListener('click', () => selectSource(button.dataset.source).catch(onError));
  }
  for (const [input, source] of [
    [el.videoFile, 'video'],
    [el.imageFile, 'image'],
  ]) {
    input.addEventListener('change', () => {
      const file = input.files[0];
      input.value = ''; // so picking the same file again still triggers 'change'
      if (file) selectSource(source, file).catch(onError);
    });
  }
  el.toggle.addEventListener('click', () => {
    if (state.running) stopLive();
    else if (state.source === 'webcam') selectSource('webcam').catch(onError); // camera was switched off
    else startLive();
  });

  // The displayed size changes with the window, so the boxes must be redrawn to match.
  window.addEventListener('resize', () => drawDetections(el.overlay, state.detections));
}

// ---------- Startup ----------

async function init() {
  wireControls();
  setLoading(true);

  state.webgpu = await checkWebGPU();
  state.settings.device = state.webgpu.available ? 'webgpu' : 'wasm';
  state.settings.dtype = DEFAULT_DTYPE[state.settings.device];
  fillSettings();
  showThreshold();

  // (Not img.decode(): Chrome delays that until the tab is visible, which would hang background tabs.)
  await loadImage(testImage, TEST_IMAGE_URL);

  try {
    await loadModel();
  } finally {
    setLoading(false);
  }
  await selectSource('test');
}

init().catch(onError);
