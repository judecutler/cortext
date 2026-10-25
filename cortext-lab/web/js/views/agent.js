// Agent tab: everything between the headset and the answer, drawn as a map you can change.
//
//   Headset → filter layer → … → Epoching → Model → Output → Result        (live, in the bridge)
//   Recording → Dataset cleaning → train_p300 ↗ Model library            (offline, how models are made)
//
// The filter layers on the map are a DRAFT: add, remove, reorder, switch off or retune them, and every block shows
// the real signal after that stage (the bridge runs the draft on the last seconds of EEG). Nothing changes the live
// scorer until "Apply to live". Chains are shared through git like models (cortext-lab/filters/library), and a model
// only runs with the chain it was trained with: the map shows a broken wire when they differ.
import { h } from '../util.js';
import { toast } from '../ui.js';
import { record, snapshot } from '../undo.js';
import { loadSettings, loadParticipant } from '../settings.js';
import { evidenceBars } from '../brainviz.js';
import { modelCard } from './modelCard.js';
import { predictorCard, narratorCard, predictorName, row } from './rig.js';

const DRAFT_KEY = 'cortext-lab.filterDraft.v1';
const CHAN_KEY = 'cortext-lab.agentChannel';

async function api(path, body) {
  const r = await fetch(path, body === undefined ? { cache: 'no-store' } : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return r.json();
}

/** Same rule as filters.signature() in the bridge: enabled layers + their settings; names, notes, reject_uv don't count. */
export function chainSignature(spec) {
  const core = (spec?.layers || []).filter((l) => l.enabled !== false).map((l) => {
    const o = {};
    for (const k of Object.keys(l).sort()) if (k !== 'enabled' && k !== 'note') o[k] = l[k];
    return o;
  });
  return JSON.stringify(core);
}
const sameChain = (a, b) => chainSignature(a) === chainSignature(b);
const sameDraft = (a, b) => sameChain(a, b) && (a?.reject_uv ?? 100) === (b?.reject_uv ?? 100) && (a?.name || '') === (b?.name || '');

function layerSummary(l) {
  const f = (x) => String(+Number(x).toFixed(2));
  switch (l.type) {
    case 'bandpass': return `${f(l.low)}–${f(l.high)} Hz · order ${l.order}`;
    case 'highpass': return `above ${f(l.cutoff)} Hz · order ${l.order}`;
    case 'lowpass': return `below ${f(l.cutoff)} Hz · order ${l.order}`;
    case 'notch': return `${f(l.freq)} Hz${l.harmonics > 1 ? ` + ${l.harmonics - 1} harmonic${l.harmonics > 2 ? 's' : ''}` : ''} · Q ${f(l.q)}`;
    case 'reref': return l.channels?.length ? `minus ${l.channels.join('+')}` : 'common average';
    default: return l.type;
  }
}

function cssVar(name, fallback) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

/** A small symmetric trace (one channel). Returns the half-range in µV that it was scaled to. */
function sparkline(canvas, data, { color, scale = null } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 140, hgt = canvas.clientHeight || 34;
  canvas.width = w * dpr; canvas.height = hgt * dpr;
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, hgt);
  if (!data || !data.length) return null;
  const m = data.reduce((a, v) => a + v, 0) / data.length;
  const amp = scale ?? Math.max(1, ...data.map((v) => Math.abs(v - m)));
  g.strokeStyle = cssVar('--line', 'rgba(125,249,255,.16)');
  g.beginPath(); g.moveTo(0, hgt / 2); g.lineTo(w, hgt / 2); g.stroke();
  g.strokeStyle = color || cssVar('--accent', '#00e5ff');
  g.lineWidth = 1.2;
  g.beginPath();
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = hgt / 2 - ((v - m) / amp) * (hgt / 2 - 2);
    if (i) g.lineTo(x, y); else g.moveTo(x, y);
  });
  g.stroke();
  return amp;
}

/** Bigger before/after plot: traces (top) and spectra (bottom). series: [{label, trace, f, db, color}] */
function comparePlot(canvas, series, { traceHz = 125 } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, hgt = canvas.clientHeight || 260;
  canvas.width = w * dpr; canvas.height = hgt * dpr;
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, hgt);
  const muted = cssVar('--muted', '#8aa0bd'), line = cssVar('--line', 'rgba(125,249,255,.16)');
  g.font = '11px "JetBrains Mono", monospace';
  const topH = hgt * 0.5, pad = 34;
  // traces, one shared scale so the size difference is visible
  const centred = series.map((s) => { const m = s.trace.reduce((a, v) => a + v, 0) / (s.trace.length || 1); return s.trace.map((v) => v - m); });
  const amp = Math.max(1, ...centred.flat().map(Math.abs));
  g.strokeStyle = line; g.beginPath(); g.moveTo(pad, topH / 2); g.lineTo(w, topH / 2); g.stroke();
  g.fillStyle = muted; g.fillText(`±${amp.toFixed(0)} µV`, 2, 12); g.fillText(`${((series[0]?.trace.length || 0) / traceHz).toFixed(0)} s`, w - 24, topH - 4);
  centred.forEach((d, si) => {
    g.strokeStyle = series[si].color; g.lineWidth = si === centred.length - 1 ? 1.4 : 1; g.beginPath();
    d.forEach((v, i) => { const x = pad + (i / (d.length - 1)) * (w - pad); const y = topH / 2 - (v / amp) * (topH / 2 - 4); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
    g.stroke();
  });
  // spectra (dB), shared axis
  const y0 = topH + 14, sh = hgt - y0 - 16;
  const allDb = series.flatMap((s) => s.db);
  if (!allDb.length) return;
  const lo = Math.min(...allDb), hi = Math.max(...allDb);
  const fmax = Math.max(...series.flatMap((s) => s.f));
  g.strokeStyle = line; g.strokeRect(pad, y0, w - pad, sh);
  g.fillStyle = muted;
  [10, 20, 30, 40, 50, 60, 70].filter((f) => f <= fmax).forEach((f) => { const x = pad + (f / fmax) * (w - pad); g.fillText(String(f), x - 6, y0 + sh + 12); });
  g.fillText('Hz', w - 16, y0 + sh + 12); g.fillText('dB', 2, y0 + 10); g.fillText(`${(hi - lo).toFixed(0)} dB`, 2, y0 + 22); g.fillText('range', 2, y0 + 34);
  series.forEach((s) => {
    g.strokeStyle = s.color; g.lineWidth = 1.2; g.beginPath();
    s.f.forEach((f, i) => { const x = pad + (f / fmax) * (w - pad); const y = y0 + sh - ((s.db[i] - lo) / (hi - lo || 1)) * sh; if (i) g.lineTo(x, y); else g.moveTo(x, y); });
    g.stroke();
  });
}

export async function renderAgent(root, app, focus) {
  let info = null;           // /api/filters
  let modelInfo = null;      // /api/models (from the model card)
  let preview = null;        // /api/filters/preview for the draft
  let previewErr = null;
  let draft = null;
  let selected = focus || 'filters';
  let channel = localStorage.getItem(CHAN_KEY) || 'Pz';
  let live = true;
  let disposed = false;

  try { draft = JSON.parse(localStorage.getItem(DRAFT_KEY) || 'null'); } catch { draft = null; }
  const saveDraft = () => { try { localStorage.setItem(DRAFT_KEY, JSON.stringify(draft)); } catch { /* private mode */ } };
  const types = () => info?.layer_types || {};
  const running = () => info?.running?.spec || null;
  const dirty = () => !!(draft && running() && !sameDraft(draft, running()));

  /** Every draft change goes through here so it can be undone. quiet = no toast (small visible tweaks). */
  function editDraft(label, fn, { quiet = false } = {}) {
    const before = snapshot(draft);
    if (fn() === false) return;
    // a changed chain is a new chain: don't let it carry the old one's name and description
    if (before && draft && chainSignature(before) !== chainSignature(draft) && draft.name === before.name) {
      if (!/\(edited\)$/.test(draft.name)) draft.name = `${draft.name} (edited)`;
      draft.description = '';
    }
    const after = snapshot(draft);
    saveDraft(); paint(); schedulePreview();
    const put = (v) => () => { draft = snapshot(v); saveDraft(); paint(); schedulePreview(); };
    record(label, put(before), put(after), { toast: !quiet });
  }

  // ---------------------------------------------------------------- DOM skeleton
  const banner = h('div', { class: 'agent-banner' });
  const lane = h('div', { class: 'lane', role: 'list', 'aria-label': 'Live pipeline' });
  const offLane = h('div', { class: 'lane lane-off', role: 'list', 'aria-label': 'How models are made (offline)' });
  const chanSel = h('select', { 'aria-label': 'Channel to preview', onchange: () => { channel = chanSel.value; localStorage.setItem(CHAN_KEY, channel); runPreview(); } });
  const liveBox = h('input', { type: 'checkbox', checked: true, onchange: () => { live = liveBox.checked; if (live) runPreview(); } });
  const previewNote = h('span', { class: 'muted small' });
  const inspector = h('div', { class: 'card inspector', id: 'inspector' });
  const libHost = h('div', {});
  const libCard = h('div', { class: 'card lib-card', id: 'filterLibrary' });

  const model = modelCard(() => loadParticipant().participant, {
    onInfo: (mi) => { modelInfo = mi; paintLane(); paintLibrary(); },
    onChange: () => refresh(),
  });
  const predictor = predictorCard({ onChange: () => paint() });
  const narrator = narratorCard();

  root.append(
    h('div', { class: 'exp-head' }, h('h1', {}, 'Agent'),
      h('span', { class: 'muted' }, 'How the EEG becomes an answer. Click a block to inspect or change it.')),
    banner,
    h('div', { class: 'card workzone' },
      h('div', { class: 'wz-tools' },
        h('h2', {}, 'System map'),
        h('span', { class: 'spacer' }),
        h('label', { class: 'inline' }, 'Signal shown ', chanSel),
        h('label', { class: 'inline', title: 'Run the draft chain on the last seconds of EEG every 2 s' }, liveBox, ' live preview'),
        previewNote),
      h('div', { class: 'lane-label' }, 'Live, in the bridge: every trial goes through this'),
      lane,
      h('div', { class: 'lane-label' }, 'Offline: how a trained model is made'),
      offLane),
    inspector,
    libCard);

  // ---------------------------------------------------------------- data
  async function refresh() {
    try {
      info = await api('/api/filters');
      if (!draft) { draft = snapshot(info.running.spec); saveDraft(); }
    } catch {
      info = null;
    }
    paint();
  }

  let previewTimer = null;
  let previewBusy = false;
  function schedulePreview() { clearTimeout(previewTimer); previewTimer = setTimeout(runPreview, 250); }
  async function runPreview() {
    if (disposed || previewBusy || !draft || !info) return;
    if (!info.live) { preview = null; previewErr = 'No EEG stream: connect the headset (or run eeg/fake_eeg.py) to see the signal through each layer.'; paintSignals(); return; }
    previewBusy = true;
    try {
      const r = await api('/api/filters/preview', { spec: draft, channel, seconds: 4 });
      if (r.ok) { preview = r; previewErr = null; } else { preview = null; previewErr = r.error; }
    } catch { preview = null; previewErr = 'Bridge not reachable.'; }
    previewBusy = false;
    paintSignals();
  }
  const pvTimer = setInterval(() => { if (live && document.visibilityState === 'visible') runPreview(); }, 2000);
  const infoTimer = setInterval(() => { refresh(); }, 6000);

  // ---------------------------------------------------------------- the map
  const stageIndex = (layerIdx) => {
    // preview stages: 0 = raw input, then one per ENABLED layer
    if (!draft.layers[layerIdx] || draft.layers[layerIdx].enabled === false) return null;
    return 1 + draft.layers.slice(0, layerIdx).filter((l) => l.enabled !== false).length;
  };

  function node({ key, kind, title, sub, cls = '', stage = null, extra = null, draggable = false, idx = null }) {
    const canvas = stage != null ? h('canvas', { class: 'spark', dataset: { stage: String(stage) } }) : null;
    const el = h('div', { class: `node ${cls}${selected === key ? ' sel' : ''}`, role: 'listitem', tabindex: '0', dataset: { key },
      onclick: (e) => { if (e.target.closest('button')) return; select(key); },
      onkeydown: (e) => { if ((e.key === 'Enter' || e.key === ' ') && e.target === el) { e.preventDefault(); select(key); } } },
    h('div', { class: 'node-kind' }, kind), h('div', { class: 'node-title' }, title),
    sub ? h('div', { class: 'node-sub' }, sub) : null,
    canvas, stage != null ? h('div', { class: 'node-std', dataset: { stage: String(stage) } }) : null,
    extra);
    if (draggable) {
      el.draggable = true;
      el.addEventListener('dragstart', (e) => { e.dataTransfer.setData('text/x-layer', String(idx)); e.dataTransfer.effectAllowed = 'move'; el.classList.add('dragging'); });
      el.addEventListener('dragend', () => el.classList.remove('dragging'));
    }
    return el;
  }
  const wire = (cls = '', title = '') => h('div', { class: `wire ${cls}`, title, 'aria-hidden': 'true' });

  function insertSlot(at) {
    const menu = h('div', { class: 'ins-menu', hidden: true },
      ...Object.entries(types()).map(([t, d]) => h('button', { type: 'button', class: 'btn small', onclick: () => {
        menu.hidden = true; addLayer(t, at);
      } }, `+ ${d.label}`)));
    const btn = h('button', { type: 'button', class: 'ins-btn', title: 'Add a filter layer here', 'aria-label': 'Add a filter layer here',
      onclick: (e) => { e.stopPropagation(); document.querySelectorAll('.ins-menu').forEach((m) => { if (m !== menu) m.hidden = true; }); menu.hidden = !menu.hidden; } }, '+');
    const slot = h('div', { class: 'ins', dataset: { at: String(at) } }, btn, menu);
    slot.addEventListener('dragover', (e) => { if ([...e.dataTransfer.types].includes('text/x-layer')) { e.preventDefault(); slot.classList.add('over'); } });
    slot.addEventListener('dragleave', () => slot.classList.remove('over'));
    slot.addEventListener('drop', (e) => {
      e.preventDefault(); slot.classList.remove('over');
      const from = Number(e.dataTransfer.getData('text/x-layer'));
      moveLayer(from, at);
    });
    return slot;
  }

  function addLayer(type, at) {
    const d = types()[type];
    const layer = { type, enabled: true };
    for (const [k, p] of Object.entries(d.params)) layer[k] = Array.isArray(p.default) ? [...p.default] : p.default;
    editDraft(`Added ${d.label} to the draft`, () => { draft.layers.splice(at, 0, layer); }, { quiet: true });
    select(`layer:${at}`);
  }
  function moveLayer(from, to) {
    if (Number.isNaN(from) || from === to || from + 1 === to) return;
    const l = draft.layers[from];
    editDraft(`Moved ${types()[l.type]?.label || l.type}`, () => {
      draft.layers.splice(from, 1);
      draft.layers.splice(to > from ? to - 1 : to, 0, l);
    }, { quiet: true });
    select(`layer:${to > from ? to - 1 : to}`);
  }
  function removeLayer(i) {
    const l = draft.layers[i];
    editDraft(`Removed ${types()[l.type]?.label || l.type} from the draft`, () => { draft.layers.splice(i, 1); });
    if (selected === `layer:${i}`) select(draft.layers.length ? `layer:${Math.min(i, draft.layers.length - 1)}` : 'filters');
  }
  function toggleLayer(i) {
    const l = draft.layers[i];
    editDraft(`${l.enabled === false ? 'Switched on' : 'Bypassed'} ${types()[l.type]?.label || l.type}`, () => { l.enabled = l.enabled === false; }, { quiet: true });
  }

  function modelState() {
    const sc = info?.model;
    const mi = sc?.model_info;
    if (!sc) return { title: 'Scorer', sub: 'needs the bridge’s EEG engine', cls: 'off' };
    const kind = sc.active;
    const mismatchDraft = mi && !sameChain(draft, mi.filters || {});
    if (mi) {
      return { title: mi.file, sub: `${(mi.participants || []).join(', ') || 'participant ?'} · ${kind === 'model' ? 'running' : 'paused'}`,
        cls: sc.model_filter_mismatch ? 'warn' : '', mismatchDraft, mismatchRunning: !!sc.model_filter_mismatch, kind };
    }
    return { title: kind === 'adaptive' ? 'Live calibration' : 'Untrained rule', sub: kind === 'adaptive' ? `${sc.adaptive_trials} trials of this session` : `adapts after ${sc.adaptive_needs}`,
      cls: kind === 'heuristic' ? 'warn' : '', kind };
  }

  function paintLane() {
    if (!draft) { lane.replaceChildren(h('div', { class: 'muted' }, info === null ? 'The bridge is not running: start python cortext-lab/bridge/bridge.py.' : 'Loading…')); return; }
    const b = app.bridge.status();
    const eeg = b.eeg || {};
    const ms = modelState();
    const s = loadSettings();
    const T = types();
    const parts = [];
    parts.push(node({ key: 'input', kind: 'Input', title: eeg.found ? (eeg.simulated ? 'Simulated EEG' : (eeg.device || eeg.name || 'Headset')) : 'No headset',
      sub: eeg.found ? `${eeg.channels} ch · ${eeg.measured_srate ?? eeg.nominal_srate ?? '?'} Hz` : (b.connected ? 'start eeg/stream_eeg.py' : 'bridge offline'),
      cls: eeg.found ? (eeg.simulated ? 'warn' : '') : 'off', stage: 0 }));
    parts.push(wire());
    parts.push(insertSlot(0));
    draft.layers.forEach((l, i) => {
      const d = T[l.type] || { label: l.type };
      const off = l.enabled === false;
      parts.push(node({ key: `layer:${i}`, kind: `Filter ${i + 1}${off ? ' · bypassed' : ''}`, title: d.label, sub: layerSummary(l),
        cls: `layer${off ? ' bypassed' : ''}`, stage: off ? null : stageIndex(i), draggable: true, idx: i,
        extra: h('div', { class: 'node-acts' },
          h('button', { type: 'button', class: 'chip-btn', title: off ? 'Switch this layer on' : 'Bypass this layer (keep its settings)', 'aria-label': off ? 'Switch on' : 'Bypass', onclick: () => toggleLayer(i) }, off ? '○' : '●'),
          h('button', { type: 'button', class: 'chip-btn', title: 'Remove this layer', 'aria-label': 'Remove layer', onclick: () => removeLayer(i) }, '✕')) }));
      parts.push(insertSlot(i + 1));
    });
    if (!draft.layers.length) parts.push(h('div', { class: 'node ghost' }, h('div', { class: 'node-kind' }, 'Filters'), h('div', { class: 'node-sub' }, 'none: the raw signal goes straight to the epochs')));
    parts.push(wire());
    parts.push(node({ key: 'epoch', kind: 'Epoching', title: '0–800 ms after each flash', sub: `20 × 40 ms bins · reject > ${draft.reject_uv ?? 100} µV`, extra: h('div', { class: 'node-std', dataset: { reject: '1' } }) }));
    const broken = ms.mismatchDraft;
    parts.push(wire(broken ? 'broken' : '', broken ? 'The model was trained with other filters than the draft' : ''));
    parts.push(node({ key: 'model', kind: 'Model', title: ms.title, sub: ms.sub, cls: `${ms.cls || ''}${broken ? ' broken' : ''}`,
      extra: broken ? h('div', { class: 'node-flag' }, 'trained with other filters') : null }));
    parts.push(wire());
    const mode = s.predictor_mode;
    parts.push(node({ key: 'output', kind: 'Output', title: predictorName(mode), sub: mode === 'eeg' ? 'evidence per option → %' : mode === 'mock' ? 'uses the true answer!' : mode === 'ws' ? s.predictor_url : 'no prediction',
      cls: mode === 'mock' ? 'warn' : mode === 'off' ? 'off' : '' }));
    parts.push(wire());
    const shows = [s.show_model_probs && s.trial_type === 'mcq_sequential' && '% on choices', s.narrate !== 'off' && `narrator ${s.narrate === 'before_click' ? 'before' : 'after'} click`, s.feedback && 'feedback'].filter(Boolean);
    parts.push(node({ key: 'result', kind: 'Result', title: 'Answer + narrator', sub: shows.join(' · ') || 'click only' }));
    lane.replaceChildren(...parts);

    offLane.replaceChildren(
      node({ key: 'off:record', kind: 'Recording', title: 'Raw EEG + markers', sub: 'every session, untouched', cls: 'offline' }), wire(),
      node({ key: 'off:clean', kind: 'Dataset cleaning', title: 'Zero-phase, offline', sub: info?.offline_cleaning ? `notch · ${info.offline_cleaning.bandpass_hz.join('–')} Hz · artifacts flagged` : 'pipeline/build_session.py', cls: 'offline' }), wire(),
      node({ key: 'off:train', kind: 'Training', title: 'train_p300.py', sub: `raw EEG → ${draft && running() && sameChain(draft, running()) ? 'the running chain' : 'filters/active chain'}`, cls: 'offline' }), wire('up'),
      node({ key: 'model', kind: 'Model library', title: `${(modelInfo?.library || []).length} shared · ${(modelInfo?.calibration || []).length} on this laptop`, sub: 'pick one in the Model block', cls: 'offline' }));
    paintSignals();
  }

  function paintSignals() {
    const muted = cssVar('--muted', '#8aa0bd');
    lane.querySelectorAll('canvas.spark').forEach((c) => {
      const st = preview?.stages?.[Number(c.dataset.stage)];
      sparkline(c, st?.trace, { color: Number(c.dataset.stage) === 0 ? muted : undefined });
    });
    lane.querySelectorAll('.node-std[data-stage]').forEach((el) => {
      const st = preview?.stages?.[Number(el.dataset.stage)];
      const ci = preview ? preview.labels.indexOf(preview.channel) : -1;
      el.textContent = st && ci >= 0 ? `${preview.channel} ${st.std_uv[ci]} µV` : '';
    });
    const rj = lane.querySelector('.node-std[data-reject]');
    if (rj) rj.textContent = preview?.reject_estimate?.windows ? `${preview.reject_estimate.rejected}/${preview.reject_estimate.windows} windows over the limit` : '';
    previewNote.textContent = previewErr ? previewErr : preview ? `draft chain on the last ${preview.filtered_seconds} s of ${preview.channel}` : '';
    previewNote.className = `small ${previewErr ? 'warn' : 'muted'}`;
    if (selected.startsWith('layer:') || selected === 'input' || selected === 'epoch') paintInspectorSignals();
  }

  // ---------------------------------------------------------------- banner (draft vs running)
  function paintBanner() {
    if (!info) { banner.replaceChildren(h('div', { class: 'note-banner warn-banner small' }, 'The bridge is not running, so the live chain and models are unknown. Start ', h('code', {}, 'python cortext-lab/bridge/bridge.py'), '.')); return; }
    const run = info.running;
    const sc = info.model;
    const mi = sc?.model_info;
    const bits = [h('span', {}, h('b', {}, 'Running: '), run.name, h('span', { class: 'muted' }, ` · ${(run.summary || []).join(' → ')}`))];
    if (run.error) bits.push(h('div', { class: 'warn small' }, run.error));
    if (!dirty()) {
      bits.push(h('div', { class: 'muted small' }, 'The map shows the running chain. Edit any layer to start a draft; nothing changes live until you apply it.'));
      banner.replaceChildren(h('div', { class: 'note-banner small' }, ...bits));
      return;
    }
    const pauses = mi && !sameChain(draft, mi.filters || {});
    const resumes = mi && sc.model_filter_mismatch && sameChain(draft, mi.filters || {});
    const sessionOn = !!(app.runner && app.runner.phase !== 'finished');
    const apply = h('button', { class: 'btn small primary', type: 'button', onclick: applyDraft }, 'Apply to live');
    const discard = h('button', { class: 'btn small', type: 'button', onclick: () => editDraft('Discarded the draft', () => { draft = snapshot(running()); }) }, 'Discard draft');
    const save = h('button', { class: 'btn small', type: 'button', onclick: () => document.getElementById('saveDraftName')?.focus() }, 'Save to library…');
    banner.replaceChildren(h('div', { class: 'note-banner warn-banner small draft-banner' },
      h('div', {}, h('b', {}, `Draft “${draft.name}”`), h('span', { class: 'muted' }, ` · ${draft.layers.filter((l) => l.enabled !== false).map(layerSummaryLong).join(' → ') || 'no filtering'}`)),
      h('div', { class: 'muted' }, ...bits),
      pauses ? h('div', { class: 'warn' }, `Applying pauses ${mi.file}: it was trained with other filters (${(mi.filter_summary || []).join(' → ')}). The scorer then calibrates on this session instead, or pick a model trained with these filters.`) : null,
      resumes ? h('div', { class: 'good' }, `Applying resumes ${mi.file}: these are the filters it was trained with.`) : null,
      sessionOn ? h('div', { class: 'warn' }, 'A session is running: applying rebuilds this session’s live calibration from the last 90 s of EEG.') : null,
      h('div', { class: 'draft-acts' }, apply, discard, save)));
  }
  const layerSummaryLong = (l) => `${types()[l.type]?.label || l.type} ${layerSummary(l)}`;

  async function applyDraft() {
    const prev = snapshot(running());
    const r = await api('/api/filters/apply', { spec: draft }).catch((e) => ({ ok: false, error: e.message }));
    if (!r.ok) { toast(r.error, 'bad'); return; }
    info = r;
    draft = snapshot(r.running.spec); saveDraft();
    paint(); model.refresh();
    const back = async () => { const x = await api('/api/filters/apply', { spec: prev, source: 'Undo (Agent page)' }); if (!x.ok) throw new Error(x.error); info = x; draft = snapshot(x.running.spec); saveDraft(); paint(); model.refresh(); };
    const again = async () => { const x = await api('/api/filters/apply', { spec: r.running.spec, source: 'Redo (Agent page)' }); if (!x.ok) throw new Error(x.error); info = x; draft = snapshot(x.running.spec); saveDraft(); paint(); model.refresh(); };
    record(`Applied filters “${r.running.name}” to live`, back, again, { kind: 'good' });
  }

  // ---------------------------------------------------------------- inspector
  function select(key) {
    selected = key;
    lane.querySelectorAll('.node').forEach((n) => n.classList.toggle('sel', n.dataset.key === key));
    offLane.querySelectorAll('.node').forEach((n) => n.classList.toggle('sel', n.dataset.key === key));
    paintInspector();
  }

  let insPlot = null;
  function paintInspector() {
    insPlot = null;
    const head = (kind, title, help) => [h('div', { class: 'eyebrow' }, kind), h('h2', { style: { marginTop: '4px' } }, title), help ? h('p', { class: 'muted small' }, help) : null];
    const T = types();
    if (!draft || !info) { inspector.replaceChildren(...head('Agent', 'Waiting for the bridge')); return; }
    if (selected.startsWith('layer:')) {
      const i = Number(selected.split(':')[1]);
      const l = draft.layers[i];
      if (!l) { selected = 'filters'; return paintInspector(); }
      const d = T[l.type];
      const fields = Object.entries(d.params).map(([k, p]) => {
        if (p.list) {
          const labels = info.labels?.length ? info.labels : (app.bridge.status().eeg?.labels || []);
          if (!labels.length) {
            return row(p.label, h('input', { type: 'text', value: (l[k] || []).join(', '), placeholder: 'e.g. Cz, or empty', onchange: (e) => editDraft(`Changed ${d.label} ${p.label.toLowerCase()}`, () => { l[k] = e.target.value.split(',').map((x) => x.trim()).filter(Boolean); }, { quiet: true }) }), h('small', {}, p.hint || ''));
          }
          return row(p.label, h('div', { class: 'chanpick' }, ...labels.map((c) => h('label', { class: 'inline' },
            h('input', { type: 'checkbox', checked: (l[k] || []).includes(c), onchange: (e) => editDraft(`${e.target.checked ? 'Added' : 'Removed'} ${c} ${e.target.checked ? 'to' : 'from'} the reference`, () => {
              l[k] = e.target.checked ? [...(l[k] || []), c] : (l[k] || []).filter((x) => x !== c);
            }, { quiet: true }) }), c))), h('small', {}, p.hint || ''));
        }
        if (p.choices) {
          return row(p.label, h('select', { onchange: (e) => editDraft(`Set ${d.label} ${p.label.toLowerCase()} to ${e.target.value}`, () => { l[k] = Number(e.target.value); }, { quiet: true }) },
            ...p.choices.map((c) => h('option', { value: c, selected: Number(l[k]) === c }, `${c} Hz${c === 60 ? ' (Americas)' : ' (Europe, Asia)'}`))));
        }
        const inp = h('input', { type: 'number', min: p.min, max: p.max, step: p.step, value: l[k],
          onchange: () => {
            const v = Number(inp.value);
            if (!Number.isFinite(v) || v < p.min || v > p.max) { toast(`${p.label} must be between ${p.min} and ${p.max}.`, 'warn'); inp.value = l[k]; return; }
            editDraft(`Set ${d.label} ${p.label.toLowerCase()} to ${v}`, () => { l[k] = p.int ? Math.round(v) : v; }, { quiet: true });
          } });
        return row(p.label, inp);
      });
      const enabled = h('input', { type: 'checkbox', checked: l.enabled !== false, onchange: () => toggleLayer(i) });
      const plot = h('canvas', { class: 'cmp-plot' });
      insPlot = { canvas: plot, kind: 'layer', i, legend: h('div', { class: 'cmp-legend small' }) };
      inspector.replaceChildren(...head(`Filter layer ${i + 1} of ${draft.layers.length}`, d.label, d.help),
        h('div', { class: 'ins-grid' },
          h('div', {}, row('On', enabled, h('small', {}, 'Off = bypassed: the layer stays on the map with its settings but does nothing.')), ...fields,
            h('div', { class: 'ctrls', style: { display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '8px' } },
              h('button', { class: 'btn small', type: 'button', disabled: i === 0, onclick: () => moveLayer(i, i - 1) }, '◀ Earlier'),
              h('button', { class: 'btn small', type: 'button', disabled: i === draft.layers.length - 1, onclick: () => moveLayer(i, i + 2) }, 'Later ▶'),
              h('button', { class: 'btn small danger', type: 'button', onclick: () => removeLayer(i) }, 'Remove'))),
          h('div', {}, h('h3', {}, `Before → after, ${channel}`), plot, insPlot.legend)));
      paintInspectorSignals();
      return;
    }
    if (selected === 'filters') {
      inspector.replaceChildren(...head('Filters', 'The filter chain', 'Layers run left to right on every sample, causally (only past samples), the same live and in training. Use + on the map to add one, drag a layer onto a + to move it, ● to bypass it. Each block shows the real signal after that stage.'),
        h('div', { class: 'muted small' }, 'Pick a layer on the map to change its settings.'));
      return;
    }
    if (selected === 'input') {
      const eeg = app.bridge.status().eeg || {};
      const q = eeg.quality || [];
      const plot = h('canvas', { class: 'cmp-plot' });
      insPlot = { canvas: plot, kind: 'input', legend: h('div', { class: 'cmp-legend small' }) };
      inspector.replaceChildren(...head('Input', eeg.found ? (eeg.device || eeg.name) : 'No headset stream',
        'What arrives from the headset over LSL, before any filtering. Electrode quality is always judged with a fixed 0.5–20 Hz reference filter, so the chips mean the same thing whatever chain runs.'),
        h('div', { class: 'ins-grid' },
          h('div', {},
            eeg.found ? h('div', { class: 'mono small' }, `${eeg.channels} channels · ${eeg.measured_srate ?? '…'} Hz measured · delay ${eeg.delay_ms ?? '?'} ms${eeg.simulated ? ' · SIMULATED' : ''}`) : h('div', { class: 'warn small' }, 'Start eeg/stream_eeg.py (or eeg/fake_eeg.py).'),
            h('div', { class: 'qchips' }, ...q.map((x) => h('span', { class: `qchip ${x.status}`, title: `${x.std_uv} µV` }, x.label, h('b', {}, x.status === 'good' ? '' : ` ${x.status}`)))),
            h('div', { style: { marginTop: '8px' } }, h('a', { class: 'btn small', href: '#/experiment' }, 'Headset card on Experiment →'), ' ', h('a', { class: 'btn small', href: '#/eeg' }, 'EEG monitor →'))),
          h('div', {}, h('h3', {}, `Raw signal, ${channel}`), plot, insPlot.legend)));
      paintInspectorSignals();
      return;
    }
    if (selected === 'epoch') {
      const ep = info.epoching || {};
      const reject = h('input', { type: 'number', min: 10, max: 2000, step: 5, value: draft.reject_uv ?? 100, onchange: () => {
        const v = Number(reject.value);
        if (!(v >= 10 && v <= 2000)) { toast('The artifact limit must be between 10 and 2000 µV.', 'warn'); reject.value = draft.reject_uv; return; }
        editDraft(`Set the artifact limit to ${v} µV`, () => { draft.reject_uv = v; }, { quiet: true });
      } });
      const est = h('div', { class: 'small muted', dataset: { estimate: '1' } });
      const n = (info.labels || []).length || '?';
      inspector.replaceChildren(...head('Epoching', 'Cut one window per flash',
        'Fixed in code (bridge/p300.py), because every trained model depends on it. Each flash becomes one feature vector.'),
        h('div', { class: 'ins-grid' },
          h('div', {},
            h('table', { class: 'kv' }, h('tbody', {},
              ...[['Window', `${(ep.tmin_s ?? 0) * 1000}–${(ep.tmax_s ?? 0.8) * 1000} ms after the flash`], ['Baseline', `${(ep.baseline_s || [-0.1, 0]).map((x) => x * 1000).join(' to ')} ms, subtracted`],
                ['Bins', `${ep.bin_ms ?? 40} ms mean amplitude → ${Math.round(((ep.tmax_s ?? 0.8) - (ep.tmin_s ?? 0)) * 1000 / (ep.bin_ms ?? 40))} per channel`],
                ['Features', `${n} channels × 20 bins = ${typeof n === 'number' ? n * 20 : '?'} numbers per flash`],
                ['Untrained rule', `mean ${((ep.heuristic_window_s || [0.25, 0.45]).map((x) => x * 1000)).join('–')} ms over ${(ep.parietal || []).slice(0, 5).join(', ')}…`]]
                .map(([k, v]) => h('tr', {}, h('td', { class: 'muted' }, k), h('td', {}, v)))))),
          h('div', {}, row('Artifact limit (µV)', reject, h('small', {}, 'An epoch whose filtered signal goes beyond ± this anywhere is skipped (blink, movement, loose electrode). Part of the chain, but it changes which epochs count, not the features, so models stay compatible.')), est)));
      paintInspectorSignals();
      return;
    }
    if (selected === 'model') {
      const sc = info.model;
      const kinds = h('div', { class: 'small muted' },
        h('div', {}, h('b', {}, 'model'), ': a trained model from the table below (only while its filters run).'),
        h('div', {}, h('b', {}, 'adaptive'), ': retrained in the bridge after every valid trial of this session (after 5 trials / 30 target epochs).'),
        h('div', {}, h('b', {}, 'heuristic'), ': untrained, mean 250–450 ms amplitude over parietal channels; only works for a strong P300.'));
      inspector.replaceChildren(...head('Model', sc ? `Scoring with: ${sc.active}` : 'Scorer unavailable', 'Scores each flash epoch; the evidence per option is summed over repeats. Pick which trained model runs here.'),
        kinds, h('div', { style: { marginTop: '10px' } }, model.el));
      return;
    }
    if (selected === 'output') {
      const evHost = h('div', { class: 'evidence' }, 'Loading…');
      inspector.replaceChildren(...head('Output', 'From evidence to an answer', 'The option with the most P300 evidence is the guess; a softmax turns the evidence into percentages.'),
        h('div', { class: 'ins-grid' },
          h('div', {}, predictor.el,
            h('div', { class: 'note-banner warn-banner small', style: { marginTop: '10px' } }, 'The percentages are NOT calibrated probabilities. In a no-signal dry run the live model said 100% and was wrong (runbook gap 3). Say “confidence score”, and quote the hit rate over trials instead.')),
          h('div', {}, h('h3', {}, 'Last prediction from the bridge'), evHost)));
      fetch('/api/eeg/trace?seconds=1', { cache: 'no-store' }).then((r) => r.json()).then((t) => evidenceBars(evHost, t.evidence)).catch(() => { evHost.textContent = 'Bridge not reachable.'; });
      return;
    }
    if (selected === 'result') {
      const s = loadSettings();
      const line = (k, v) => h('tr', {}, h('td', { class: 'muted' }, k), h('td', {}, v));
      inspector.replaceChildren(...head('Result', 'What the room sees and hears', 'Set per experiment on the Experiment tab; the voice and how the narrator writes are set here.'),
        h('table', { class: 'kv' }, h('tbody', {},
          line('Probabilities on the choices', s.trial_type === 'mcq_sequential' ? (s.show_model_probs ? 'shown after the EEG window' : 'hidden') : 'flash mode: none'),
          line('Narrator', { off: 'off', after_click: 'after the click', before_click: 'before the click (show mode: those clicks are not evaluation data)' }[s.narrate] || s.narrate),
          line('Correct / incorrect', s.feedback ? 'shown after the click' : 'hidden'))),
        h('div', { style: { margin: '8px 0 12px' } }, h('a', { class: 'btn small', href: '#/experiment' }, 'Change on Experiment →')),
        narrator.el);
      return;
    }
    if (selected === 'off:record') {
      inspector.replaceChildren(...head('Offline', 'Recording', 'Every session saves the raw EEG (CSV, with LSL and page timestamps) and every marker, untouched. Filters never touch the recording, so any chain can be tried on old sessions later.'));
      return;
    }
    if (selected === 'off:clean') {
      const oc = info.offline_cleaning;
      inspector.replaceChildren(...head('Offline', 'Dataset cleaning (pipeline/build_session.py)', 'Runs after each session to make the shared dataset. It is zero-phase (forwards and backwards), which the live scorer cannot do, so it is NOT the live chain; it is for analysis, epochs.npz and artifact flags.'),
        oc ? h('table', { class: 'kv' }, h('tbody', {},
          ...[['Mains', `detects 50 or 60 Hz, notches it and its harmonics (Q ${oc.notch_q})`], ['Band-pass', `${oc.bandpass_hz.join('–')} Hz, order ${oc.bandpass_order}, zero-phase`],
            ['Bad electrodes', 'flagged and interpolated from neighbours'], ['Artifacts', `spans over ${oc.artifact_uv} µV flagged; epochs touching them marked rejected (kept, not deleted)`]]
            .map(([k, v]) => h('tr', {}, h('td', { class: 'muted' }, k), h('td', {}, v))))) : h('div', { class: 'muted small' }, 'Needs numpy + scipy in the bridge to show its settings.'));
      return;
    }
    if (selected === 'off:train') {
      const runName = info.running?.name;
      inspector.replaceChildren(...head('Offline', 'Training a model (pipeline/train_p300.py)', 'Re-filters the RAW recordings with a filter chain (causally, exactly like the bridge), cross-validates by trial with a shuffled-label control, and saves the model with the chain inside it.'),
        h('pre', { class: 'mono small codeblock' }, `python cortext-lab/pipeline/train_p300.py                 # uses the running chain (${runName})\npython cortext-lab/pipeline/train_p300.py --filters p300-60-hz-notch\npython cortext-lab/pipeline/train_p300.py --activate      # and make it the running model`),
        h('div', { class: 'muted small' }, 'To compare chains honestly, train one model per chain on the same sessions and compare their held-out accuracy and shuffled control in the Model block.'));
      return;
    }
    inspector.replaceChildren(...head('Agent', 'Select a block'));
  }

  function paintInspectorSignals() {
    if (!insPlot || !insPlot.canvas.isConnected) return;
    const est = inspector.querySelector('[data-estimate]');
    if (est) est.textContent = preview?.reject_estimate?.windows ? `With this limit and the draft filters, ${preview.reject_estimate.rejected} of ${preview.reject_estimate.windows} epoch-length windows in the last ${preview.seconds.toFixed(0)} s (any channel) would be skipped.` : (previewErr || '');
    if (!preview) { insPlot.legend.textContent = previewErr || 'Waiting for signal…'; return; }
    const accent = cssVar('--accent', '#00e5ff'), muted = cssVar('--muted', '#8aa0bd');
    const ci = preview.labels.indexOf(preview.channel);
    let series;
    if (insPlot.kind === 'input') {
      const s0 = preview.stages[0];
      series = [{ label: 'raw', trace: s0.trace, f: s0.psd_f, db: s0.psd_db, color: accent }];
      insPlot.legend.textContent = `${preview.channel}: ${s0.std_uv[ci]} µV std (mean removed). Spectrum 0.5–${Math.round(Math.max(...s0.psd_f))} Hz.`;
    } else {
      const st = stageIndex(insPlot.i);
      if (st == null) { insPlot.legend.textContent = 'This layer is bypassed: switch it on to see what it does.'; comparePlot(insPlot.canvas, []); return; }
      const a = preview.stages[st - 1], b = preview.stages[st];
      if (!a || !b) return;
      series = [{ label: 'before', trace: a.trace, f: a.psd_f, db: a.psd_db, color: muted }, { label: 'after', trace: b.trace, f: b.psd_f, db: b.psd_db, color: accent }];
      insPlot.legend.replaceChildren(h('span', { style: { color: muted } }, `■ before (${a.label}) ${a.std_uv[ci]} µV`), '  ', h('span', { style: { color: accent } }, `■ after ${b.std_uv[ci]} µV`), h('span', { class: 'muted' }, `  · ${preview.channel}, last ${preview.seconds.toFixed(0)} s`));
    }
    comparePlot(insPlot.canvas, series, { traceHz: preview.trace_hz });
  }

  // ---------------------------------------------------------------- filter library
  function paintLibrary() {
    if (!info) { libCard.replaceChildren(h('h2', {}, 'Filter library'), h('div', { class: 'muted' }, 'Needs the bridge.')); return; }
    const all = [info.builtin, ...info.library];
    const modelsFor = (sig) => [...(modelInfo?.library || []), ...(modelInfo?.calibration || [])].filter((m) => m.filter_signature === sig);
    const draftSig = chainSignature(draft);
    const rows = all.map((c) => {
      if (c.error) return h('tr', {}, h('td', { class: 'mono small' }, c.file), h('td', { colspan: 4, class: 'bad small' }, c.error), h('td', {}));
      const ms = modelsFor(c.signature);
      const isRun = c.is_running;
      const inDraft = chainSignature(c.spec) === draftSig;
      return h('tr', { class: isRun ? 'model-on' : '' },
        h('td', {}, h('div', {}, h('b', {}, c.name), c.group === 'builtin' ? h('span', { class: 'pill', style: { marginLeft: '6px' } }, 'built in') : null),
          h('div', { class: 'muted small mono' }, c.file || 'in the code (bridge/filters.py)'), c.description ? h('div', { class: 'muted small' }, c.description) : null),
        h('td', { class: 'small' }, (c.summary || []).join(' → ')),
        h('td', { class: 'small' }, ms.length ? h('span', { title: ms.map((m) => m.file).join(', ') }, `${ms.length} model${ms.length === 1 ? '' : 's'}`) : h('span', { class: 'muted' }, 'none')),
        h('td', {}, isRun ? h('span', { class: 'pill good' }, '● running') : null, ' ', inDraft && dirty() ? h('span', { class: 'pill warn' }, 'draft') : null),
        h('td', { class: 'model-acts' },
          h('button', { class: 'btn small', type: 'button', title: 'Put this chain on the map as the draft (nothing changes live)', onclick: () => editDraft(`Loaded “${c.name}” into the draft`, () => { draft = snapshot(c.spec); }) }, 'Edit'),
          isRun ? null : h('button', { class: 'btn small', type: 'button', title: 'Run this chain live now', onclick: async () => { draft = snapshot(c.spec); saveDraft(); await applyDraft(); } }, 'Use this'),
          c.group === 'library' ? h('button', { class: 'btn small danger', type: 'button', title: 'Delete from cortext-lab/filters/library (Undo puts it back)', onclick: async () => {
            const r = await api('/api/filters/delete', { id: c.id });
            if (!r.ok) { toast(r.error, 'bad'); return; }
            info = r; paint();
            const name = c.file.replace(/\.json$/, '');
            record(`Deleted filter chain “${c.name}”`, async () => { const x = await api('/api/filters/save', { spec: r.deleted, name: r.deleted.name, overwrite: true }); if (!x.ok) throw new Error(x.error); info = x; paint(); },
              async () => { const x = await api('/api/filters/delete', { id: `library/${name}.json` }); if (!x.ok) throw new Error(x.error); info = x; paint(); });
          } }, 'Delete') : null));
    });
    const nameIn = h('input', { type: 'text', id: 'saveDraftName', placeholder: 'name for the draft, e.g. P300 + 60 Hz notch', value: draft && dirty() ? draft.name : '' });
    const fileIn = h('input', { type: 'file', accept: '.json', style: { display: 'none' }, onchange: async () => {
      const f = fileIn.files[0]; fileIn.value = '';
      if (!f) return;
      try {
        const spec = JSON.parse(await f.text());
        if (!Array.isArray(spec.layers)) throw new Error('not a filter chain (no "layers" list)');
        editDraft(`Imported “${spec.name || f.name}” into the draft`, () => { draft = { name: spec.name || f.name.replace(/\.json$/i, ''), description: spec.description || '', reject_uv: spec.reject_uv ?? 100, layers: spec.layers }; });
      } catch (e) { toast(`Could not import: ${e.message}`, 'bad'); }
    } });
    libHost.replaceChildren(
      h('div', { class: 'table-wrap' }, h('table', { class: 'models' }, h('thead', {}, h('tr', {}, ...['Chain', 'Layers', 'Models trained with it', '', ''].map((t) => h('th', {}, t)))), h('tbody', {}, ...rows))),
      h('div', { class: 'lib-save' },
        nameIn,
        h('button', { class: 'btn small', type: 'button', onclick: async () => {
          const name = nameIn.value.trim() || draft.name;
          const r = await api('/api/filters/save', { spec: { ...draft, name } });
          if (!r.ok) { toast(r.error, 'bad'); return; }
          info = r; draft.name = name; saveDraft(); paint();
          const id = r.id;
          record(`Saved “${name}” to the filter library`, async () => { const x = await api('/api/filters/delete', { id }); if (!x.ok) throw new Error(x.error); info = x; paint(); },
            async () => { const x = await api('/api/filters/save', { spec: { ...draft, name }, overwrite: true }); if (!x.ok) throw new Error(x.error); info = x; paint(); }, { kind: 'good' });
        } }, 'Save draft to library'),
        h('button', { class: 'btn small', type: 'button', onclick: () => {
          const blob = new Blob([JSON.stringify(draft, null, 2)], { type: 'application/json' });
          const a = h('a', { href: URL.createObjectURL(blob), download: `${(draft.name || 'chain').toLowerCase().replace(/[^a-z0-9]+/g, '-')}.json` });
          document.body.append(a); a.click(); a.remove();
        } }, 'Export draft…'),
        h('button', { class: 'btn small', type: 'button', onclick: () => fileIn.click() }, 'Import…'), fileIn),
      h('p', { class: 'muted small', style: { margin: '8px 0 0' } },
        'Chains live in cortext-lab/filters/library and are shared through git, like models: commit the folder to share one. The running chain is copied into cortext-lab/filters/active so the bridge starts with it. A model stores the chain it was trained with and only runs with that chain.'));
    libCard.replaceChildren(h('div', { class: 'card-head' }, h('h2', {}, 'Filter library'), h('span', { class: 'muted small mono' }, info.filters_dir)), libHost);
  }

  function paintChannels() {
    const labels = info?.labels?.length ? info.labels : [];
    if (!labels.length) { chanSel.replaceChildren(h('option', { value: channel }, channel)); return; }
    if (!labels.includes(channel)) channel = labels.includes('Pz') ? 'Pz' : labels[0];
    if ([...chanSel.options].map((o) => o.value).join() !== labels.join()) chanSel.replaceChildren(...labels.map((l) => h('option', { value: l }, l)));
    chanSel.value = channel;
  }

  /** Don't rebuild the map under the operator's hands (a drag, an open + menu). */
  const busyMap = () => !!lane.querySelector('.node.dragging, .ins-menu:not([hidden])');
  function paint() {
    const typing = inspector.contains(document.activeElement);
    paintChannels(); paintBanner();
    if (!busyMap()) paintLane();
    paintLibrary();
    if (!typing) paintInspector();
  }

  const offBridge = app.bridge.onChange(() => { if (!busyMap()) paintLane(); });
  const closeMenus = (e) => { if (!e.target.closest('.ins')) document.querySelectorAll('.ins-menu').forEach((m) => { m.hidden = true; }); };
  document.addEventListener('click', closeMenus);

  await refresh();
  paintInspector();
  runPreview();
  if (focus) setTimeout(() => inspector.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);

  return () => {
    disposed = true;
    clearInterval(pvTimer); clearInterval(infoTimer); clearTimeout(previewTimer);
    offBridge(); document.removeEventListener('click', closeMenus);
    model.dispose(); narrator.dispose?.();
  };
}
