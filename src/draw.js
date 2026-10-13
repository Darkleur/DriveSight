// Draws detection boxes on a transparent canvas that sits exactly on top of the image/video.

// Gives each class a stable color (every "car" box gets the same color).
function colorFor(label) {
  let hash = 0;
  for (const ch of label) hash = (hash * 31 + ch.charCodeAt(0)) % 360;
  return `hsl(${hash}, 85%, 55%)`;
}

export function clearDetections(canvas) {
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

export function drawDetections(canvas, detections) {
  // Match the canvas's internal pixel grid to its on-screen size (times the screen's pixel
  // density, so lines stay sharp on phones and retina screens).
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.lineWidth = 2;
  ctx.font = '600 13px system-ui, sans-serif';
  ctx.textBaseline = 'top';

  for (const { label, score, box } of detections) {
    // Boxes are fractions (0 to 1) of the image, so multiplying by the displayed size
    // gives on-screen pixels, whatever size the image is shown at.
    const x = box.xmin * width;
    const y = box.ymin * height;
    const w = (box.xmax - box.xmin) * width;
    const h = (box.ymax - box.ymin) * height;
    const color = colorFor(label);

    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    // Label tag like "car 87%" above the box (or inside it, if the box touches the top edge).
    const text = `${label} ${Math.round(score * 100)}%`;
    const tagW = ctx.measureText(text).width + 8;
    const tagH = 18;
    const tagY = y - tagH >= 0 ? y - tagH : y;
    ctx.fillStyle = color;
    ctx.fillRect(x, tagY, tagW, tagH);
    ctx.fillStyle = '#000';
    ctx.fillText(text, x + 4, tagY + 3);
  }
}
