// Setup tab: the rig. Headset and electrode quality, who is being recorded, how answers are predicted, display
// hardware, and where recordings are stored. The experiment itself is configured on the Experiment tab.
import { h } from '../util.js';
import { DEVICES } from '../schema.js';
import { loadSettings, patchSettings, loadParticipant, saveParticipant } from '../settings.js';
import { HeadMap, Traces, EEGFeed, mapValues } from '../brainviz.js';
import { storageInfo, uploadPending, downloadFromHub, createHubRepo } from '../dataset.js';
import { toast } from '../ui.js';

export async function renderSetup(root, app) {
  const s = loadSettings();
  const who = loadParticipant();
  const persistWho = () => saveParticipant({ participant: participant.value.trim(), device: device.value, notes: notes.value });

  // ---------- participant & recording ----------
  const participant = h('input', { type: 'text', value: who.participant, placeholder: 'e.g. P01', autocomplete: 'off', oninput: () => { persistWho(); idHint(); } });
  const device = h('select', { onchange: persistWho }, h('option', { value: '' }, '— choose —'),
    ...DEVICES.map((d) => h('option', { value: d, selected: who.device === d }, d)));
  const notes = h('textarea', { placeholder: 'electrode gel, impedance notes, caffeine, lighting …', oninput: persistWho }, who.notes);
  notes.value = who.notes;
  const idWarn = h('small', { class: 'warn' });
  function idHint() {
    const v = participant.value.trim();
    idWarn.textContent = v && !/\d/.test(v) ? 'Use a code like P01, not a name: this ends up in the dataset.' : '';
  }
  idHint();
  const row = (label, el, hint) => h('div', { class: 'field' }, h('label', {}, label), el, hint || null);
  const whoCard = h('div', { class: 'card' }, h('h2', {}, 'Participant & recording'),
    row('Participant ID', participant, idWarn), row('Device used', device), row('Notes', notes),
    h('small', { class: 'muted', style: { display: 'block' } }, 'Remembered between visits. Notes are cleared after each session starts.'));

  // ---------- prediction & display hardware ----------
  const mode = h('select', { onchange: savePred }, ...[['eeg', 'Live EEG (headset → bridge P300 scorer)'], ['mock', 'Mock (simulated; demo only)'],
    ['ws', 'External model server (WebSocket)'], ['off', 'Off']].map(([v, l]) => h('option', { value: v, selected: s.predictor_mode === v }, l)));
  const url = h('input', { type: 'text', value: s.predictor_url, onchange: savePred });
  const pd = h('input', { type: 'checkbox', checked: !!s.photodiode, onchange: savePred });
  const corner = h('select', { onchange: savePred }, ...['bottom-left', 'bottom-right', 'top-left', 'top-right'].map((c) => h('option', { value: c, selected: s.photodiode_corner === c }, c.replace('-', ' '))));
  function savePred() { patchSettings({ predictor_mode: mode.value, predictor_url: url.value, photodiode: pd.checked, photodiode_corner: corner.value }); }
  const predCard = h('div', { class: 'card' }, h('h2', {}, 'Prediction'),
    row('Predictor', mode, h('small', {}, 'Live EEG scores the real headset in the bridge. Mock uses the ground truth, so it is NOT evidence.')),
    row('Predictor URL', url));
  const hwCard = h('div', { class: 'card' }, h('h2', {}, 'Display hardware'),
    row('Photodiode square', pd, h('small', {}, 'White square in a screen corner during every stimulus, for hardware timing checks.')),
    row('Photodiode corner', corner),
    h('div', {}, h('a', { class: 'btn small', href: '#/timing' }, 'Run the timing test →')));

  // ---------- headset (auto-detected from the bridge) ----------
  const hsStatus = h('div', { class: 'hs-status' });
  const hsChips = h('div', { class: 'qchips' });
  const hsNote = h('div', { class: 'muted small' });
  const hsMap = h('canvas', { class: 'headmap-sm' });
  const hsTrace = h('canvas', { class: 'traces-sm' });
  const headsetCard = h('div', { class: 'card headset-card span2' },
    h('div', { class: 'card-head' }, h('h2', {}, 'Headset'), h('a', { class: 'btn small', href: '#/eeg' }, 'Open the EEG monitor →')),
    h('div', { class: 'hs-grid' }, h('div', {}, hsStatus, hsChips, hsNote), hsMap, hsTrace));
  const headMap = new HeadMap(hsMap, { compact: true });
  const traces = new Traces(hsTrace);
  const feed = new EEGFeed({ traceSeconds: 4, traceMs: 300, erpMs: 60000 }).start();
  let autofilled = false;
  function deviceFor(eeg) {
    if (eeg.simulated) return 'Simulated EEG (fake_eeg.py)';
    if (/unicorn|bci core/i.test(eeg.device || '')) return 'g.tec Unicorn / BCI Core-8';
    return eeg.channels >= 24 ? '32-node EEG' : eeg.channels >= 6 ? '8-node EEG' : eeg.channels === 4 ? 'Muse' : 'Other';
  }
  function updateHeadset() {
    const b = app.bridge.status();
    const eeg = b.eeg || {};
    if (!b.connected) {
      hsStatus.innerHTML = '<span class="dot red"></span><b>Bridge offline</b> — run <code>python cortext-lab/bridge/bridge.py</code> and open this page from http://localhost:8000';
    } else if (!eeg.found) {
      hsStatus.innerHTML = '<span class="dot red"></span><b>No headset stream</b> — run <code>python cortext-lab/eeg/stream_eeg.py</code> (or <code>eeg/fake_eeg.py</code> to simulate)' +
        (eeg.error && !/pylsl/.test(eeg.error) ? '' : eeg.error ? `<div class="bad small">${eeg.error}</div>` : '');
    } else {
      const sc = eeg.scorer || {};
      hsStatus.innerHTML = `<span class="dot ${eeg.simulated ? 'amber' : 'green'}"></span><b>${eeg.device || eeg.name}</b>${eeg.simulated ? ' <span class="pill warn">SIMULATED</span>' : ''}` +
        `<div class="muted small mono">${eeg.channels} ch · ${eeg.measured_srate ?? '…'} Hz · ${(eeg.labels || []).join(' ')}</div>` +
        `<div class="muted small mono">P300 scorer: ${sc.active ?? '–'}${sc.model_file ? ` (${sc.model_file})` : ''} · adapts after ${sc.adaptive_needs || '5 trials'}</div>` +
        (eeg.analysis_error ? `<div class="warn small">${eeg.analysis_error}</div>` : '');
      const q = eeg.quality || [];
      hsChips.replaceChildren(...q.map((x) => h('span', { class: `qchip ${x.status}`, title: `${x.std_uv} µV (0.5–20 Hz std)` }, x.label, h('b', {}, x.status === 'good' ? '' : ` ${x.status}`))));
      const bad = q.filter((x) => x.status !== 'good');
      hsNote.textContent = !q.length ? '' : bad.length
        ? `${bad.length} electrode(s) need attention: ${bad.map((x) => `${x.label} ${x.status} (${x.std_uv} µV)`).join(', ')}. Add gel / press the electrode on, sit still, then re-check.`
        : 'All electrodes look good (5–40 µV in the 0.5–20 Hz band).';
      if (!autofilled) {
        autofilled = true;
        if (!device.value) device.value = deviceFor(eeg);
        if (['mock', 'off'].includes(mode.value)) mode.value = 'eeg';
        persistWho(); savePred();
        hsNote.textContent = `Auto-filled: device = ${device.value}, predictor = Live EEG. ` + hsNote.textContent;
      }
    }
  }
  feed.on(() => {
    const tr = feed.trace;
    headMap.draw(mapValues('alpha', tr, null) || {}, { quality: tr?.quality, title: 'alpha' });
    traces.draw(tr);
  });
  const offHs = app.bridge.onChange(updateHeadset);
  updateHeadset();

  // ---------- data storage ----------
  const stBody = h('div', { class: 'stbody muted' }, 'Loading…');
  const stMsg = h('div', { class: 'small', style: { marginTop: '8px' } });
  const auto = h('input', { type: 'checkbox', checked: localStorage.getItem('cortext-lab.autoupload') === '1',
    onchange: () => localStorage.setItem('cortext-lab.autoupload', auto.checked ? '1' : '0') });
  const btn = (label, fn, cls = '') => h('button', { class: `btn small ${cls}`, type: 'button', onclick: async (e) => {
    const b = e.currentTarget; b.disabled = true; stMsg.className = 'small muted'; stMsg.textContent = 'Working… (large files can take a few minutes)';
    try { await fn(); } catch (err) { stMsg.className = 'small bad'; stMsg.textContent = String(err.message || err); }
    b.disabled = false;
  } }, label);
  const actions = h('div', { class: 'ctrls', style: { display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '10px' } });
  async function loadStorage(check = false) {
    let info;
    try { info = await storageInfo(check); } catch { stBody.replaceChildren('The bridge is not running, so storage cannot be managed from here.'); actions.replaceChildren(); return; }
    const hf = info.hf;
    const line = (k, v) => h('div', { class: 'row' }, h('span', { class: 'muted' }, k), h('span', {}, v));
    const state = !hf.installed ? h('span', { class: 'bad' }, 'huggingface_hub is not installed (pip install huggingface_hub)')
      : !hf.configured ? h('span', { class: 'warn' }, 'No repo set: put its id in datasets/cortext-p300/dataset.json ("hf_repo")')
      : !hf.has_token ? h('span', { class: 'warn' }, 'No token: run  hf auth login')
      : hf.checked ? (hf.reachable ? h('span', { class: 'good' }, `connected as ${hf.user} · ${hf.remote_sessions.length} session(s) on the server`) : h('span', { class: 'bad' }, hf.error))
      : h('span', { class: 'muted' }, 'configured; press “Check connection” to test it');
    stBody.replaceChildren(
      line('This computer', `${info.data_root} · ${info.local_sessions} session(s), ${info.size_mb} MB${info.simulated_sessions ? ` (+${info.simulated_sessions} simulated, never uploaded)` : ''}`),
      line('Hugging Face', hf.repo ? `${hf.repo} (${hf.private ? 'private' : 'PUBLIC'})` : 'not set up'), line('Status', state),
      line('Not uploaded yet', info.pending_upload.length ? `${info.pending_upload.length} session(s)` : 'nothing'),
      info.remote_only?.length ? line('On the server only', `${info.remote_only.length} session(s) you have not downloaded`) : null,
      h('label', { class: 'inline' }, auto, ' Upload each session automatically when it ends'));
    const ready = hf.installed && hf.configured && hf.has_token;
    const up = btn(`Upload ${info.pending_upload.length} pending`, async () => {
      const r = await uploadPending(); const bad = r.results.filter((x) => !x.ok);
      stMsg.className = `small ${bad.length ? 'bad' : 'good'}`;
      stMsg.textContent = bad.length ? bad.map((x) => `${x.session_id}: ${x.error}`).join(' | ') : `Uploaded ${r.results.length} session(s).`;
      loadStorage(false);
    }, 'primary');
    up.disabled = !ready || !info.pending_upload.length;
    const dl = btn('Download sessions (epochs only)', async () => { const r = await downloadFromHub('epochs'); stMsg.className = `small ${r.ok ? 'good' : 'bad'}`; stMsg.textContent = r.ok ? `Downloaded ${r.sessions.length} session(s).` : r.error; loadStorage(false); });
    const dlAll = btn('Download everything (raw EEG too)', async () => { const r = await downloadFromHub('all'); stMsg.className = `small ${r.ok ? 'good' : 'bad'}`; stMsg.textContent = r.ok ? `Downloaded ${r.sessions.length} session(s).` : r.error; loadStorage(false); });
    const init = btn('Create the repo', async () => { const r = await createHubRepo(); stMsg.className = `small ${r.ok ? 'good' : 'bad'}`; stMsg.textContent = r.ok ? `Ready: ${r.url}` : r.error; loadStorage(true); });
    dl.disabled = dlAll.disabled = init.disabled = !ready;
    actions.replaceChildren(btn('Check connection', () => loadStorage(true)), up, dl, dlAll, init);
  }
  const storeCard = h('div', { class: 'card span2' }, h('h2', {}, 'Data storage'),
    h('p', { class: 'muted small' }, 'Recordings are stored on this computer and shared through a private Hugging Face dataset. GitHub holds code only.'),
    stBody, actions, stMsg);
  loadStorage(false);

  root.append(h('h1', {}, 'Setup'), h('div', { class: 'grid2' }, headsetCard, whoCard, predCard, hwCard, storeCard));
  return () => { feed.stop(); offHs(); };
}
