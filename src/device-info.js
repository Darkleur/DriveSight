// A short description of the browser, OS and GPU, saved with every benchmark result so the
// numbers can always be traced back to the hardware that produced them.

function browserName(ua) {
  const patterns = [
    [/SamsungBrowser\/(\d+)/, 'Samsung Internet'],
    [/Edg\/(\d+)/, 'Edge'],
    [/Firefox\/(\d+)/, 'Firefox'],
    [/Chrome\/(\d+)/, 'Chrome'],
    [/Version\/(\d+)[\d.]* .*Safari/, 'Safari'],
  ];
  for (const [pattern, name] of patterns) {
    const match = ua.match(pattern);
    if (match) return `${name} ${match[1]}`;
  }
  return 'Unknown browser';
}

function osName(ua) {
  if (/Android/.test(ua)) return 'Android';
  if (/iPhone|iPad/.test(ua)) return 'iOS';
  if (/Windows/.test(ua)) return 'Windows';
  if (/Mac OS X/.test(ua)) return 'macOS';
  if (/Linux/.test(ua)) return 'Linux';
  return 'Unknown OS';
}

export async function describeDevice() {
  const ua = navigator.userAgent;
  let gpu = '–';
  try {
    // Browsers only reveal a rough GPU description (e.g. "intel gen-12lp"), for privacy reasons.
    const adapter = await navigator.gpu?.requestAdapter();
    const info = adapter?.info;
    if (info) gpu = [info.vendor, info.architecture].filter(Boolean).join(' ') || '–';
  } catch {
    // no WebGPU: leave '–'
  }
  return { browser: browserName(ua), os: osName(ua), gpu, cpuThreads: navigator.hardwareConcurrency || null };
}
