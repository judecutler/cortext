// Agent → Model block: which trained model predicts. Lists the team library (cortext-lab/models/library, shared
// through git) and the models calibrated live on this laptop (bridge/logs), and switches the running model in the
// bridge without a restart. The choice is also written to cortext-lab/models/active so the next start uses it.
// A model only works with the filter chain it was trained with, so "Use this" also switches the filters to that chain
// (and Undo switches both back).
import { h } from '../util.js';
import { toast } from '../ui.js';

const pct = (x) => (x == null ? null : `${Math.round(x * 100)}%`);

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return r.json();
}

function who(m) { return (m.participants || []).join(', ') || 'unknown'; }

function dataText(m) {
  const bits = [];
  if (m.kind === 'calibration') bits.push(`${m.n_trials ?? '?'} trials live`);
  else if (m.sessions?.length) bits.push(`${m.sessions.length} session${m.sessions.length === 1 ? '' : 's'}`);
  if (m.n_epochs != null) bits.push(`${m.n_epochs} epochs`);
  return bits.join(' · ') || '–';
}

function accText(m) {
  if (m.cv_answer_accuracy == null) return h('span', { class: 'muted', title: 'Trained during a session; no held-out test. Judge it by the next few trials.' }, 'not tested');
  const chance = m.chance ? ` vs ${pct(m.chance)} chance` : '';
  const shuf = m.shuffled_accuracy != null ? ` · shuffled ${pct(m.shuffled_accuracy)}` : '';
  return h('span', { title: 'Cross-validated answer accuracy (held-out trials). Shuffled = the same pipeline on scrambled labels: should sit at chance.' },
    `${pct(m.cv_answer_accuracy)}${chance}`, h('span', { class: 'muted small' }, shuf));
}

function filterPill(m) {
  if (m.error || m.fits_filters == null) return h('span', { class: 'muted small' }, m.filter_name || '–');
  return h('span', { title: (m.filter_summary || []).join(' → ') },
    m.fits_filters ? h('span', { class: 'pill good' }, 'running') : h('span', { class: 'pill warn', title: 'Trained with other filters. Use this switches the filters to its chain.' }, 'other'),
    ' ', h('span', { class: 'small' }, m.filter_name));
}

function fitPill(m, live) {
  if (m.error) return h('span', { class: 'pill bad', title: m.error }, 'unusable');
  if (!live) return h('span', { class: 'pill', title: 'Connect the headset to check' }, 'no headset');
  return m.fits_headset ? h('span', { class: 'pill good' }, 'fits')
    : h('span', { class: 'pill bad', title: `This headset has no ${(m.missing_channels || []).join(', ')}` }, 'wrong channels');
}

export function modelCard(getParticipant, { onChange, onInfo } = {}) {
  const running = h('div', { class: 'model-running' }, 'Loading…');
  const warnings = h('div', {});
  const tbody = h('tbody', {});
  const msg = h('div', { class: 'small', style: { marginTop: '6px' } });
  const file = h('input', { type: 'file', accept: '.npz', style: { display: 'none' }, onchange: () => upload(file.files[0]) });
  let last = null;
  let busy = false;

  const say = (text, kind = 'muted') => { msg.className = `small ${kind}`; msg.textContent = text; };

  async function act(fn) {
    if (busy) return;
    busy = true;
    try {
      const r = await fn();
      if (r && r.ok === false) say(r.error, 'bad');
      else if (r?.running !== undefined) render(r);
      return r;
    } catch (err) { say(`Bridge not reachable: ${err.message || err}`, 'bad'); } finally { busy = false; }
  }

  function runningId(info) {
    const file = info.running?.model_file?.split(/[\\/]/).pop();
    if (!file) return null;
    const hit = [...info.library, ...info.calibration].find((m) => m.file === file);
    return hit ? hit.id : null;
  }

  async function use(id, label, restoreFilters = null) {
    const prev = last ? runningId(last) : null;
    let prevFilters = null;
    try { prevFilters = (await api('/api/filters')).running?.spec || null; } catch { /* bridge gone: act() reports it */ }
    const r = await act(() => api('/api/models/select', { id, follow_filters: !restoreFilters }));
    if (!r?.ok) return;
    if (restoreFilters) await api('/api/filters/apply', { spec: restoreFilters, source: 'Undo (Agent page)' }).catch(() => null);
    say('');
    const switched = r.filters_switched ? ' (filters switched to its chain)' : '';
    toast(id ? `Now predicting with ${label}${switched}` : 'No trained model: the scorer will adapt to this session', 'good', null,
      { label: 'Undo', fn: () => use(prev, prev ? prev.split('/').pop() : 'none', r.filters_switched ? prevFilters : null) }, 'model-switch');
    if (r.note) say(r.note, r.filters_switched ? 'good' : 'warn');
    if (restoreFilters || r.filters_switched) refresh();
    onChange?.();
  }

  async function saveCal(m) {
    const r = await act(() => api('/api/models/save-calibration', { id: m.id }));
    if (r?.ok) say(`Saved to the team library as ${r.id.split('/').pop()}. Commit cortext-lab/models/library to share it.`, 'good');
  }

  async function upload(f) {
    file.value = '';
    if (!f) return;
    if (!/\.npz$/i.test(f.name)) return say('Pick a .npz model file (from train_p300.py).', 'bad');
    const buf = new Uint8Array(await f.arrayBuffer());
    let bin = '';
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
    const r = await act(() => api('/api/models/upload', { filename: f.name, data: btoa(bin) }));
    if (r?.ok) say(`Added ${r.id.split('/').pop()} to the library. Press “Use this” to run it.`, 'good');
  }

  function row(m, live, runId) {
    const isRunning = m.id === runId;
    const sim = m.simulated ? h('span', { class: 'pill warn', title: 'Trained on simulated EEG (fake_eeg.py): not a real brain model' }, 'SIM') : null;
    const useBtn = isRunning ? h('span', { class: 'pill good' }, '● running')
      : h('button', { class: 'btn small', type: 'button', disabled: !!m.error || (live && !m.fits_headset),
        title: m.error || (live && !m.fits_headset ? `Made for other electrodes: this headset has no ${(m.missing_channels || []).join(', ')}`
          : m.fits_filters === false ? 'Predict with this model from the next trial on. Also switches the filters to the chain it was trained with.' : 'Predict with this model from the next trial on'),
        onclick: () => use(m.id, m.file) }, 'Use this');
    const save = m.group === 'calibration' ? h('button', { class: 'btn small', type: 'button', disabled: !!m.error, title: 'Copy into cortext-lab/models/library so teammates get it through git', onclick: () => saveCal(m) }, 'Save to library') : null;
    return h('tr', { class: isRunning ? 'model-on' : '', dataset: { id: m.id } },
      h('td', {}, h('div', { class: 'mono small' }, m.file, ' ', sim), h('div', { class: 'muted small' }, m.trained_at ? m.trained_at.replace('T', ' ').slice(0, 16) : m.modified)),
      h('td', {}, who(m)), h('td', { class: 'small' }, dataText(m)), h('td', { class: 'small' }, accText(m)), h('td', {}, filterPill(m)), h('td', {}, fitPill(m, live)),
      h('td', { class: 'model-acts' }, useBtn, save));
  }

  function render(info) {
    last = info;
    onInfo?.(info);
    const run = info.running;
    const live = !!(info.live_channels && info.live_channels.length);
    const runId = runningId(info);
    const mi = run?.model_info;
    const pid = (getParticipant() || '').trim();
    if (!info.scoring_available) {
      running.innerHTML = '<span class="dot amber"></span>The bridge cannot score here (needs <code>pylsl</code> and <code>numpy</code>). Your choice is still saved for the next start.';
    } else if (run.model_file) {
      running.replaceChildren(h('span', { class: `dot ${run.model_error ? 'red' : 'green'}` }), 'Predicting with ',
        h('b', { class: 'mono' }, run.model_file.split(/[\\/]/).pop()),
        mi ? h('span', { class: 'muted' }, ` · trained on ${who(mi)}${mi.n_epochs ? ` · ${mi.n_epochs} epochs` : ''}${mi.cv_answer_accuracy != null ? ` · ${pct(mi.cv_answer_accuracy)} held-out` : ''}`) : null,
        h('span', { class: 'muted small' }, `  (from ${run.model_source})`));
    } else {
      running.replaceChildren(h('span', { class: 'dot amber' }), h('b', {}, 'No trained model. '),
        h('span', { class: 'muted' }, `The scorer adapts to this session after ${run.adaptive_needs}, using a rough untrained rule until then. Fine for calibrating; for the demo, pick a model.`));
    }
    const w = [];
    if (run?.model_error) w.push(['bad', run.model_error]);
    if (run?.model_filter_mismatch) w.push(['warn', `This model is ${run.model_filter_mismatch}`]);
    if (run?.filters_reset_note) w.push(['warn', run.filters_reset_note]);
    if (mi?.simulated) w.push(['warn', 'This model was trained on SIMULATED EEG. It cannot read a real brain.']);
    if (mi && pid && mi.participants?.length && !mi.participants.includes(pid)) {
      w.push(['warn', `This model was trained on ${who(mi)}, but the participant on the Experiment tab is ${pid}. P300 models are personal: use one trained on ${pid}, or calibrate first.`]);
    }
    if (info.active_warning) w.push(['warn', info.active_warning]);
    warnings.replaceChildren(...w.map(([k, t]) => h('div', { class: `note-banner ${k === 'warn' ? 'warn-banner' : ''} small` }, h('span', { class: k }, t))));
    const section = (title, list, empty) => [h('tr', { class: 'model-group' }, h('td', { colspan: 7 }, title)),
      ...(list.length ? list.map((m) => row(m, live, runId)) : [h('tr', {}, h('td', { colspan: 7, class: 'muted small' }, empty))])];
    tbody.replaceChildren(
      ...section('Team library · cortext-lab/models/library (shared through GitHub)', info.library,
        'Empty. Train with pipeline/train_p300.py (saves here), add a file below, or git pull to get teammates’ models.'),
      ...section('Calibrated on this laptop · bridge/logs (not shared)', info.calibration,
        'None yet. Run a session with the live EEG predictor: after 5 good trials the bridge trains one and saves it here.'));
  }

  async function refresh() {
    if (busy) return;
    try { render(await api('/api/models')); } catch { running.textContent = 'The bridge is not running, so models cannot be listed.'; tbody.replaceChildren(); }
  }

  const el = h('div', { class: 'model-card' },
    h('div', { class: 'card-head' }, h('h3', {}, 'P300 model (what predicts)'), h('button', { class: 'btn small', type: 'button', onclick: refresh }, 'Refresh')),
    running, warnings,
    h('div', { class: 'table-wrap' }, h('table', { class: 'models' },
      h('thead', {}, h('tr', {}, ...['Model', 'Participant', 'Trained on', 'Held-out accuracy', 'Filters', 'Headset', ''].map((t) => h('th', {}, t)))), tbody)),
    h('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '10px' } },
      h('button', { class: 'btn small', type: 'button', onclick: () => use(null, 'none') }, 'No trained model (calibrate live)'),
      h('button', { class: 'btn small', type: 'button', onclick: () => file.click() }, 'Add a model file…'), file),
    h('p', { class: 'muted small', style: { margin: '8px 0 0' } },
      '“Use this” switches immediately, no restart, and becomes the default for the next start. It also switches the filters to the chain the model was trained with. Models are personal: pick one trained on the person in the chair. To share a calibration, Save to library, then commit cortext-lab/models/library.'),
    msg);
  refresh();
  const timer = setInterval(refresh, 8000);       // calibrations appear during a session; teammates' models after a git pull
  return { el, refresh, get info() { return last; }, dispose: () => clearInterval(timer) };
}
