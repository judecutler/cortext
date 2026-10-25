// Setup tab: the rig. Headset and electrode quality, who is being recorded, how answers are predicted, display
// hardware, and where recordings are stored. The experiment itself is configured on the Experiment tab.
import { h } from '../util.js';
import { DEVICES } from '../schema.js';
import { loadSettings, patchSettings, loadParticipant, saveParticipant } from '../settings.js';
import { HeadMap, Traces, EEGFeed, mapValues } from '../brainviz.js';
import { storageInfo, uploadPending, downloadFromHub, createHubRepo } from '../dataset.js';
import { toast } from '../ui.js';
import { modelCard } from './modelCard.js';
import { Voice, NARRATOR_MOODS, EDGE_VOICES, browserVoices, narrationFacts, composeNarration } from '../voice.js';

export async function renderSetup(root, app) {
  const s = loadSettings();
  const who = loadParticipant();
  const persistWho = () => saveParticipant({ participant: participant.value.trim(), device: device.value, notes: notes.value, display_name: displayName.value.trim() });

  // ---------- participant & recording ----------
  const participant = h('input', { type: 'text', value: who.participant, placeholder: 'e.g. P01', autocomplete: 'off', oninput: () => { persistWho(); idHint(); } });
  const device = h('select', { onchange: persistWho }, h('option', { value: '' }, '— choose —'),
    ...DEVICES.map((d) => h('option', { value: d, selected: who.device === d }, d)));
  const displayName = h('input', { type: 'text', value: who.display_name || '', placeholder: 'e.g. Jude (optional)', autocomplete: 'off', oninput: () => persistWho() });
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
    row('Participant ID', participant, idWarn), row('Device used', device),
    row('Name for the narrator', displayName, h('small', {}, 'Demo only: the narrator says it out loud. Kept in this browser; never written to the session, markers or dataset.')),
    row('Notes', notes),
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
        `<div class="muted small mono">P300 scorer: ${sc.active ?? '–'}${sc.model_file ? ` (${sc.model_file.split(/[\\/]/).pop()})` : ''} · adapts after ${sc.adaptive_needs || '5 trials'}</div>` +
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

  // ---------- voice & narrator (demo) ----------
  const vStatus = h('div', { class: 'small muted' }, 'Checking the bridge…');
  const vMsg = h('div', { class: 'small', style: { marginTop: '6px' } });
  const engine = h('select', { onchange: saveVoice }, h('option', { value: 'auto', selected: s.tts_engine !== 'browser' }, 'Bridge voice if installed, else the browser’s'),
    h('option', { value: 'browser', selected: s.tts_engine === 'browser' }, 'Browser voice only'));
  const edgeList = h('datalist', { id: 'edgeVoices' }, ...EDGE_VOICES.map(([v, l]) => h('option', { value: v }, l)));
  const tVoice = h('input', { type: 'text', list: 'edgeVoices', value: s.tts_voice || '', placeholder: 'default: en-US-AndrewMultilingualNeural', onchange: saveVoice });
  const bVoice = h('select', { onchange: saveVoice }, h('option', { value: '' }, 'Best English voice found'));
  browserVoices().then((vs) => {
    bVoice.append(...vs.filter((v) => /^en/i.test(v.lang)).concat(vs.filter((v) => !/^en/i.test(v.lang)))
      .map((v) => h('option', { value: v.name, selected: v.name === s.browser_voice }, `${v.name} (${v.lang})`)));
  });
  const rate = h('input', { type: 'number', min: 0.5, max: 2, step: 0.05, value: s.tts_rate ?? 1, onchange: saveVoice });
  const llm = h('select', { onchange: saveVoice }, ...[['template', 'Built-in lines (offline, instant)'], ['ollama', 'Ollama (free, runs on this computer)'],
    ['openai', 'OpenAI-compatible API (OpenAI, Groq free tier, OpenRouter, LM Studio)'], ['anthropic', 'Claude (Anthropic API)']]
    .map(([v, l]) => h('option', { value: v, selected: s.narrator_llm === v }, l)));
  const mood = h('select', { onchange: () => { const m = NARRATOR_MOODS.find(([id]) => id === mood.value); if (m) ctxBox.value = m[2]; saveVoice(); } },
    ...NARRATOR_MOODS.map(([id, label]) => h('option', { value: id, selected: s.narrator_mood === id }, label)),
    h('option', { value: 'custom', selected: s.narrator_mood === 'custom' }, 'Custom (edited below)'));
  const moodText = (id) => (NARRATOR_MOODS.find(([m]) => m === id) || NARRATOR_MOODS[0])[2];
  const ctxBox = h('textarea', { class: 'narr-context', oninput: () => { if (ctxBox.value.trim() !== moodText(mood.value)) mood.value = 'custom'; saveVoice(); } });
  ctxBox.value = (s.narrator_context || '').trim() || moodText(s.narrator_mood);
  const rulesEl = h('pre', { class: 'mono small', style: { whiteSpace: 'pre-wrap' } }, 'Loading…');
  function saveVoice() {
    const txt = ctxBox.value.trim();
    patchSettings({ tts_engine: engine.value, tts_voice: tVoice.value.trim(), browser_voice: bVoice.value, tts_rate: Number(rate.value) || 1,
      narrator_llm: llm.value, narrator_mood: mood.value,
      // store the text only when it differs from a preset, so editing a preset later still applies
      narrator_context: mood.value === 'custom' || txt !== moodText(mood.value) ? txt : '' });
  }
  const mkVoice = () => new Voice({ engine: engine.value, voice: tVoice.value.trim(), browserVoice: bVoice.value, rate: Number(rate.value) || 1 });
  let testVoice = null;
  const testBtn = (label, fn) => h('button', { class: 'btn small', type: 'button', onclick: async (e) => {
    const b = e.currentTarget; b.disabled = true; testVoice?.stopAll();
    try { await fn(); } catch (err) { vMsg.className = 'small bad'; vMsg.textContent = String(err.message || err); }
    b.disabled = false;
  } }, label);
  const sayHi = testBtn('▶ Test voice', async () => {
    testVoice = mkVoice(); testVoice.unlock();
    vMsg.className = 'small muted'; vMsg.textContent = 'Speaking…';
    const r = await testVoice.speak(`Hi${displayName.value.trim() ? ' ' + displayName.value.trim() : ''}. I'm the CorText narrator. Which one is the dog?`);
    vMsg.className = `small ${r.engine ? 'good' : 'bad'}`;
    vMsg.textContent = r.engine ? `Spoken with: ${r.engine === 'browser' ? 'browser speech' : r.engine}${testVoice.lastServerError ? ` (bridge voice failed: ${testVoice.lastServerError})` : ''}` : 'No speech available in this browser.';
  });
  const sayNarr = testBtn('▶ Test narrator', async () => {
    saveVoice(); testVoice = mkVoice(); testVoice.unlock();
    const trial = { prompt: 'Which one is the dog?', trial_index: 0, clicked_answer_id: null,
      options: [['cat', 'A'], ['dog', 'B'], ['cup', 'C'], ['circle', 'D']].map(([id, letter]) => ({ id, letter, label: id })) };
    const facts = narrationFacts({ trial, name: displayName.value.trim(), when: 'before_click', modality: loadSettings().stim_modality,
      pred: { source: 'eeg', best_option_id: 'dog', probabilities: { cat: 0.12, dog: 0.78, cup: 0.06, circle: 0.04 } } });
    vMsg.className = 'small muted'; vMsg.textContent = 'Writing the line…';
    const n = await composeNarration(facts, { provider: llm.value, context: ctxBox.value.trim(), plain: mood.value === 'scientist' });
    vMsg.className = `small ${n.fallback_error ? 'warn' : 'good'}`;
    vMsg.textContent = `“${n.text}” — ${n.provider === 'template' ? 'built-in line' : `${n.provider}${n.model ? ' · ' + n.model : ''}, ${n.ms} ms`}` +
      (n.fallback_error ? ` (LLM failed, used the built-in line: ${n.fallback_error})` : '');
    await testVoice.speak(n.text);
  });
  async function loadVoiceStatus(check = false) {
    try {
      const v = await (await fetch(`/api/voice/status${check ? '?check=1' : ''}`)).json();
      const L = v.llm || {};
      const yes = (ok, txt) => `<span class="${ok ? 'good' : ok === false ? 'warn' : 'muted'}">${txt}</span>`;
      vStatus.innerHTML = `Bridge voices: ${v.tts_engines.length ? yes(true, v.tts_engines.join(', ')) : yes(false, 'none — <code>pip install edge-tts</code> for neural voices (free), or <code>pyttsx3</code> for offline ones')}<br>` +
        `LLMs: Ollama ${L.ollama?.configured == null ? yes(null, 'not checked') : L.ollama.configured ? yes(true, `running (${L.ollama.model})`) : yes(false, 'not running')}` +
        `${L.ollama?.warning ? ` <span class="warn">${L.ollama.warning}</span>` : ''} · ` +
        `OpenAI-compatible ${L.openai?.configured ? yes(true, `${L.openai.model} @ ${L.openai.base_url}`) : yes(false, 'no key')} · ` +
        `Claude ${L.anthropic?.configured ? yes(true, L.anthropic.model) : yes(false, 'no key')}` +
        (v.config_error ? `<br><span class="bad">${v.config_error}</span>` : '');
      rulesEl.textContent = v.rules || '';
    } catch { vStatus.textContent = 'The bridge is not running: only the browser voice and the built-in lines are available.'; rulesEl.textContent = '(bridge offline)'; }
  }
  loadVoiceStatus(true);
  const voiceCard = h('div', { class: 'card span2' }, h('div', { class: 'card-head' }, h('h2', {}, 'Voice & narrator (demo)'),
    h('button', { class: 'btn small', type: 'button', onclick: () => loadVoiceStatus(true) }, 'Re-check')),
    h('p', { class: 'muted small' }, 'What is read aloud and the stimulus modality are chosen per experiment (Experiment → Voice & audio). Here: which voice speaks, and how the narrator writes its lines.'),
    vStatus,
    h('div', { class: 'grid2', style: { marginTop: '10px' } },
      h('div', {}, row('Speech', engine), row('Bridge voice', tVoice, h('small', {}, 'Any edge-tts voice name; pick one from the list or type it.')), edgeList,
        row('Browser voice', bVoice), row('Speed', rate, h('small', {}, '1 = normal.'))),
      h('div', {}, row('Narrator writes with', llm, h('small', {}, 'Any failure or a reply over 5 s falls back to the built-in line, so the demo never stalls.')),
        row('Mood', mood),
        row('Narrator context', ctxBox, h('small', {}, 'The LLM’s persona and mood, plus anything about the experiment it should know (e.g. “This run uses spoken words instead of pictures”). Saved with each session’s settings. The built-in lines are warm and theatrical (plain for the Calm lab scientist mood); other moods and your own context need an LLM.')),
        h('details', {}, h('summary', { class: 'small muted' }, 'Rules always added after the context (keeps the numbers honest)'), rulesEl))),
    h('div', { class: 'voice-test' }, sayHi, sayNarr), vMsg);

  // ---------- trained P300 model ----------
  const models = modelCard(() => participant.value);
  participant.addEventListener('change', () => models.refresh());

  root.append(h('h1', {}, 'Setup'), h('div', { class: 'grid2' }, headsetCard, whoCard, predCard, hwCard, models.el, voiceCard, storeCard));
  return () => { feed.stop(); offHs(); testVoice?.stopAll(); models.dispose(); };
}
