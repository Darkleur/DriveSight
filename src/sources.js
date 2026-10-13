import { RawImage } from '@huggingface/transformers';

// Everything about getting pixels into the app: webcam, video files, images, and grabbing frames.

export async function startWebcam(videoEl) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Webcam is not available here (it needs HTTPS or localhost).');
  }
  // facingMode 'environment' asks phones for the back camera (the one that sees the road).
  // Laptops just use their only camera.
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  videoEl.srcObject = stream;
  await videoEl.play();
  return stream;
}

export function stopWebcam(stream) {
  stream?.getTracks().forEach((track) => track.stop());
}

export async function playVideo(videoEl, url) {
  videoEl.srcObject = null;
  videoEl.src = url;
  await videoEl.play();
}

// Stops playback and detaches whatever the <video> was showing.
export function resetVideo(videoEl) {
  videoEl.pause();
  videoEl.srcObject = null;
  videoEl.removeAttribute('src');
  videoEl.load();
}

export function loadImage(imgEl, url) {
  return new Promise((resolve, reject) => {
    imgEl.onload = () => resolve();
    imgEl.onerror = () => reject(new Error('Could not load that image.'));
    imgEl.src = url;
  });
}

// Uploaded files get a temporary local URL. We free the previous one so memory doesn't pile up.
let currentObjectUrl = null;
export function objectUrlFor(file) {
  if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
  currentObjectUrl = URL.createObjectURL(file);
  return currentObjectUrl;
}

// Takes a snapshot of the current video frame (or image) at the model's input size.
//
// The model always works on a fixed-size square (384x384 for nano). Its own preprocessing
// resizes every input to that square without keeping the aspect ratio, so we do the same
// resize here with the browser's fast canvas drawing instead of passing a full-HD frame for
// JavaScript to shrink. Boxes come back as fractions (0 to 1), so they still line up with the
// original, un-squashed video when we draw them.
export function captureFrame(sourceEl, canvas) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(sourceEl, 0, 0, canvas.width, canvas.height);
  return RawImage.fromCanvas(canvas); // copies the pixels, so the canvas can be reused right away
}
