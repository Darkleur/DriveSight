// Rolling statistics over the last N frames: stable enough to read, but still reacts to changes.
export class RollingStats {
  constructor(size = 30) {
    this.size = size;
    this.reset();
  }

  reset() {
    this.durations = [];
    this.timestamps = [];
  }

  // Call once per processed frame with how long the model took on it.
  add(inferenceMs) {
    this.durations.push(inferenceMs);
    this.timestamps.push(performance.now());
    if (this.durations.length > this.size) {
      this.durations.shift();
      this.timestamps.shift();
    }
  }

  // Average time the model spends on one frame.
  get avgMs() {
    if (this.durations.length === 0) return 0;
    return this.durations.reduce((a, b) => a + b, 0) / this.durations.length;
  }

  // Frames actually processed per second, end to end: grabbing the frame + the model +
  // drawing + waiting for the next screen refresh. Always a bit lower than 1000 / avgMs.
  get fps() {
    const n = this.timestamps.length;
    if (n < 2) return 0;
    return ((n - 1) * 1000) / (this.timestamps[n - 1] - this.timestamps[0]);
  }
}
