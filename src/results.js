// The benchmark results table: rendering, saving between visits, and exporting (CSV / JSON / Markdown).

const STORAGE_KEY = 'drivesight.results';

// [field, column title, formatter]
const COLUMNS = [
  ['device', 'Device', (v) => v],
  ['browser', 'Browser', (v) => v],
  ['model', 'Model', (v) => v],
  ['precision', 'Precision', (v) => v],
  ['backend', 'Backend', (v) => v],
  ['sizeMB', 'Size (MB)', (v) => v.toFixed(1)],
  ['loadS', 'Load (s)', (v, row) => `${v.toFixed(1)} (${row.loadedFrom})`],
  ['avgMs', 'Avg ms/frame', (v) => v.toFixed(0)],
  ['fps', 'FPS', (v) => v.toFixed(1)],
  ['objectsPerFrame', 'Objects/frame', (v) => v.toFixed(1)],
  ['matchFp32', 'Match fp32', (v, row) => (row.precision === 'fp32' ? 'ref' : v == null ? '–' : `${Math.round(v * 100)}%`)],
  ['frames', 'Frames', (v) => String(v)],
];

const cell = (row, [field, , format]) => format(row[field], row);

export function loadResults() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? [];
  } catch {
    return [];
  }
}

export function saveResults(rows) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(rows));
  } catch {
    // storage blocked or full: results stay in memory for this visit
  }
}

export function renderTable(table, rows) {
  const head = `<tr>${COLUMNS.map(([, title]) => `<th scope="col">${title}</th>`).join('')}</tr>`;
  const body = rows
    .map((row) => {
      const warn = row.sanityOk === false ? ' class="broken" title="Found nothing in the test photo"' : '';
      return `<tr${warn}>${COLUMNS.map((col) => `<td>${escapeHtml(cell(row, col))}</td>`).join('')}</tr>`;
    })
    .join('');
  table.innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

export function toMarkdown(rows) {
  const header = `| ${COLUMNS.map(([, title]) => title).join(' | ')} |`;
  const divider = `| ${COLUMNS.map(() => '---').join(' | ')} |`;
  const lines = rows.map((row) => `| ${COLUMNS.map((col) => cell(row, col)).join(' | ')} |`);
  return [header, divider, ...lines].join('\n');
}

// CSV keeps every raw field (not just the table columns), so nothing is lost for later analysis.
export function toCsv(rows) {
  if (rows.length === 0) return '';
  const fields = Object.keys(rows[0]);
  const quote = (value) => {
    const text = value == null ? '' : String(value);
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return [fields.join(','), ...rows.map((row) => fields.map((f) => quote(row[f])).join(','))].join('\n');
}

export function downloadFile(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
