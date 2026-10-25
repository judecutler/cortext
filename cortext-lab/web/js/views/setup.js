import { h } from '../util.js';
import { DEVICES } from '../schema.js';
import { loadSettings, saveSettings, estimateDurations } from '../settings.js';
import { loadStimuli, buildTrialSpecs } from '../stimuli.js';
import { newSessionRecord } from '../session.js';
import { listTrialTypes, getTrialType } from '../trialtypes/index.js';
import { buildSetFromFiles, saveSet, listSets, getSet, materializeSet, stimuliSnapshot } from '../stimulusSets.js';
import { db } from '../db.js';
import { HeadMap, Traces, EEGFeed, mapValues } from '../brainviz.js';

const MODE_HELP = {
  mcq_flash: 'The answer tiles stay on screen and light up one at a time (like a P300 speller). Afterwards you click the answer.',
  mcq_sequential: 'The question stays on screen. Each option’s picture/text appears alone, one after another. Then the full multiple-choice grid appears with the model’s probability on every choice, and you click the correct one.',
};

export async function renderSetup(root, app) {
  const s = loadSettings();
  const prev = JSON.parse(localStorage.getItem('cortext-lab.last-participant') || '{}');
  const err = h('div', { class: 'bad' });
  const estimate = h('div', { class: 'muted' });
  const stimInfo = h('div', { class: 'muted' }, 'Loading stimulus set…');
  const uploadMsg = h('div', { class: 'muted' });
  let stimuli = null;

  const inputs = {};
  const field = (key, label, el, hint) => {
    inputs[key] = el;
    return h('div', { class: 'field' }, h('label', {}, label), el, hint ? h('small', {}, hint) : null);
  };
  const num = (key, label, hint, attrs = {}) => field(key, label, h('input', { type: 'number', value: s[key], ...attrs, oninput: refresh }), hint);
  const sel = (key, label, options, hint, onchange) => field(key, label, h('select', { onchange: () => { refresh(); onchange?.(); } }, ...options.map(([v, l, dis]) => h('option', { value: v, selected: s[key] === v, disabled: dis }, l))), hint);
  const chk = (key, label, hint) => field(key, label, h('input', { type: 'checkbox', checked: !!s[key], onchange: refresh }), hint);

  const participant = h('input', { type: 'text', value: prev.participant || '', placeholder: 'e.g. P01', autocomplete: 'off', style: { width: '100%' } });
  const device = h('select', {}, ...DEVICES.map((d) => h('option', { value: d, selected: prev.device === d }, d)));
  const notes = h('textarea', { placeholder: 'electrode gel, impedance notes, caffeine, lighting …' });
  const modeHelp = h('small', {});

  const trialTypeOpts = listTrialTypes().map((t) => [t.id, t.label + (t.implemented ? '' : ' (not implemented yet)'), !t.implemented]);

  // ---------- stimulus set card ----------
  const sourceSel = h('select', { onchange: () => { updateSourceUi(); loadChosen(); } });
  inputs.stimuli_source = sourceSel;
  const pathInput = h('input', { type: 'text', value: s.stimuli_url, style: { width: '100%' }, onchange: loadChosen });
  inputs.stimuli_url = pathInput;
  const pathRow = h('div', { class: 'field' }, h('label', {}, 'Config file path'), pathInput, h('small', {}, 'Relative to the web folder. Default: the built-in 12-item sample set.'));
  const nameInput = h('input', { type: 'text', placeholder: 'name for this set (optional)', style: { width: '100%' } });
  const fileInput = h('input', { type: 'file', multiple: true, accept: '.json,image/*,audio/*', style: { display: 'none' }, onchange: (e) => { handleFiles(e.target.files); e.target.value = ''; } });
  const dirInput = h('input', { type: 'file', multiple: true, webkitdirectory: true, style: { display: 'none' }, onchange: (e) => { handleFiles(e.target.files); e.target.value = ''; } });
  const drop = h('div', { class: 'dropzone', tabindex: '0' }, 'Drop stimulus files here, or use the buttons. A config .json and/or images.');
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));
  const stimCard = h('div', { class: 'card' }, h('h2', {}, 'Stimulus set'),
    h('div', { class: 'field' }, h('label', {}, 'Use'), sourceSel),
    pathRow,
    h('h3', { style: { marginTop: '12px' } }, 'Upload new stimuli'),
    drop,
    h('div', { class: 'ctrls', style: { margin: '8px 0', display: 'flex', gap: '8px', flexWrap: 'wrap' } },
      h('button', { class: 'btn small', type: 'button', onclick: () => fileInput.click() }, 'Choose files…'),
      h('button', { class: 'btn small', type: 'button', onclick: () => dirInput.click() }, 'Choose folder…'),
      h('a', { class: 'btn small', href: 'config/stimuli.json', download: 'stimuli_example.json' }, 'Download example config')),
    fileInput, dirInput,
    h('div', { class: 'field' }, h('label', {}, 'Set name'), nameInput),
    h('small', { class: 'muted', style: { display: 'block', marginBottom: '6px' } },
      'Images are matched to config items by file name (folders are ignored). With no .json, a folder of images becomes the stimulus list: file name = item, sub-folder name = category. Uploaded sets stay in this browser for reuse.'),
    uploadMsg, stimInfo);

  // ---------- other cards ----------
  const sessionCard = h('div', { class: 'card' }, h('h2', {}, 'Session'),
    h('div', { class: 'field' }, h('label', {}, 'Participant ID'), participant),
    h('div', { class: 'field' }, h('label', {}, 'Device used'), device),
    h('div', { class: 'field' }, h('label', {}, 'Notes'), notes),
    sel('trial_type', 'Presentation mode', trialTypeOpts, null, updateModeUi),
    h('div', { class: 'field' }, h('label', {}), modeHelp));
  const designCard = h('div', { class: 'card' }, h('h2', {}, 'Design (both modes)'),
    num('n_trials', 'Questions', 'Each target is used once before any repeats.', { min: 1 }),
    num('options_per_trial', 'Options per question', '2–6; the grid adapts.', { min: 2, max: 6 }),
    num('prompt_ms', 'Prompt shown first (ms)', 'Time to read the question before anything else happens.', { min: 0 }),
    sel('tile_content', 'What each stimulus shows', [['image_text', 'Image + text'], ['image', 'Image only'], ['text', 'Text only']], 'Applies to the one-at-a-time view and to the answer tiles.'),
    estimate);
  const flashCard = h('div', { class: 'card' }, h('h2', {}, 'Flash timing (flash mode)'),
    num('repeats', 'Repeats per option', 'Default 10.', { min: 1 }),
    num('on_ms', 'Flash on (ms)', '≈500. Rounded to display frames.', { min: 50 }),
    num('off_min_ms', 'Gap min (ms)', 'Random gap between flashes.', { min: 0 }),
    num('off_max_ms', 'Gap max (ms)', '100–200 typical.', { min: 0 }));
  const seqCard = h('div', { class: 'card' }, h('h2', {}, 'One-at-a-time timing & model (sequential mode)'),
    num('seq_on_ms', 'Each option on screen (ms)', '1000 = one second each.', { min: 100 }),
    num('seq_gap_ms', 'Gap between options (ms)', 'Blank gap; the question stays on screen.', { min: 0 }),
    num('seq_repeats', 'Times each option is shown', 'More repeats = more EEG epochs per option.', { min: 1 }),
    chk('show_model_probs', 'Show model probability on each choice', 'Shown AFTER the EEG phase, on the final multiple-choice screen. Needs a predictor (mock or ws).'),
    chk('send_feedback', 'Send answer back to model server', 'ws predictor only. After each trial the server gets the true answer, the click and the validity flag, so it can learn from them.'),
    h('small', { class: 'warn', style: { display: 'block' } },
      'Seeing the model’s probabilities can bias the click (anchoring). Every trial records exactly what was shown (model_shown_before_click), so keep those clicks out of any evaluation of the model and use them as feedback only.'));
  const respCard = h('div', { class: 'card' }, h('h2', {}, 'Response & cleaning'),
    chk('feedback', 'Show correct / incorrect', 'Shown after the click, never while stimuli are on screen.'),
    sel('post_mode', 'After each trial', [['manual', 'Wait for operator (Space = next)'], ['auto', 'Auto-advance']], 'Discard / Redo stay available while waiting.'),
    num('auto_advance_ms', 'Auto-advance after (ms)', 'Only for auto mode; Discard cancels it.', { min: 500 }),
    num('response_timeout_ms', 'Response timeout (ms)', '0 = wait forever. A timeout marks the trial invalid (no_response).', { min: 0 }),
    chk('log_flash_off', 'Also send flash_off markers'));
  const hwCard = h('div', { class: 'card' }, h('h2', {}, 'Hardware & prediction'),
    chk('photodiode', 'Photodiode square', 'White square in a screen corner during every stimulus (for hardware timing checks).'),
    sel('photodiode_corner', 'Photodiode corner', [['bottom-left', 'Bottom left'], ['bottom-right', 'Bottom right'], ['top-left', 'Top left'], ['top-right', 'Top right']]),
    sel('predictor_mode', 'Predictor', [['eeg', 'Live EEG (headset → bridge P300 scorer)'], ['mock', 'Mock (simulated; demo only)'], ['ws', 'External model server (WebSocket)'], ['off', 'Off']], 'Live EEG scores the real headset in the bridge. Mock uses the ground truth, so it is NOT evidence.'),
    field('predictor_url', 'Predictor URL', h('input', { type: 'text', value: s.predictor_url, style: { width: '100%' } })),
    chk('show_bars', 'Live score bars under tiles', 'Flash mode, demo use only: it adds visible changes during flashing. Toggle with Shift+B.'));

  // ---------- headset card (auto-detected from the bridge) ----------
  const hsStatus = h('div', { class: 'hs-status' });
  const hsChips = h('div', { class: 'qchips' });
  const hsNote = h('div', { class: 'muted small' });
  const hsMap = h('canvas', { class: 'headmap-sm' });
  const hsTrace = h('canvas', { class: 'traces-sm' });
  const headsetCard = h('div', { class: 'card headset-card span2' },
    h('div', { class: 'card-head' }, h('h2', {}, 'Headset'), h('a', { class: 'btn small', href: '#/brain', target: '_blank', rel: 'noopener' }, 'Open neural monitor ↗')),
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
        device.value = deviceFor(eeg);
        if (['mock', 'off'].includes(inputs.predictor_mode.value)) inputs.predictor_mode.value = 'eeg';
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

  const startBtn = h('button', { class: 'btn primary', onclick: start }, 'Start session');
  root.append(h('h1', {}, 'New session'),
    h('div', { class: 'grid2' }, headsetCard, sessionCard, stimCard, designCard, flashCard, seqCard, respCard, hwCard),
    err, h('div', {}, startBtn, ' ', h('a', { class: 'btn', href: 'timing.html', target: '_blank', rel: 'noopener' }, 'Run timing test ↗')));

  function read() {
    const out = { ...s };
    for (const [k, el] of Object.entries(inputs)) {
      out[k] = el.type === 'checkbox' ? el.checked : el.type === 'number' ? Number(el.value) : el.value;
    }
    return out;
  }
  function updateModeUi() {
    const t = inputs.trial_type.value;
    flashCard.style.display = t === 'mcq_sequential' ? 'none' : '';
    seqCard.style.display = t === 'mcq_sequential' ? '' : 'none';
    modeHelp.textContent = MODE_HELP[t] || (getTrialType(t)?.plan ?? '');
  }
  function updateSourceUi() { pathRow.style.display = sourceSel.value === 'path' ? '' : 'none'; }
  function refresh() {
    const cur = read();
    const d = estimateDurations(cur);
    estimate.textContent = `${d.flashes} stimulus onsets per question ≈ ${d.flashSec.toFixed(0)} s; about ${d.trialSec.toFixed(0)} s per question and ${d.totalMin.toFixed(1)} min in total.`;
  }

  async function refreshSources(select) {
    const sets = await listSets().catch(() => []);
    sourceSel.replaceChildren(h('option', { value: 'path' }, 'Config file path (built-in sample by default)'),
      ...sets.map((x) => h('option', { value: `set:${x.name}` }, `Uploaded: ${x.name} (${x.config.items.length} items)`)));
    const want = select || s.stimuli_source;
    sourceSel.value = [...sourceSel.options].some((o) => o.value === want) ? want : 'path';
    updateSourceUi();
  }

  async function loadChosen() {
    stimInfo.replaceChildren('Loading…');
    try {
      if (sourceSel.value.startsWith('set:')) {
        const set = await getSet(sourceSel.value.slice(4));
        if (!set) throw new Error('That uploaded set no longer exists.');
        stimuli = materializeSet(set);
      } else {
        stimuli = await loadStimuli(pathInput.value);
        stimuli.source = { type: 'path', url: pathInput.value };
      }
      const cats = [...new Set(stimuli.items.map((i) => i.category))];
      stimInfo.replaceChildren(`${stimuli.items.length} items in ${cats.length} categories (${cats.join(', ')}).`);
      if (stimuli.warnings.length) stimInfo.append(h('div', { class: 'warn' }, 'Warnings: ' + stimuli.warnings.join('; ')));
    } catch (e) { stimuli = null; stimInfo.replaceChildren(h('span', { class: 'bad' }, String(e.message))); }
  }

  async function handleFiles(fileList) {
    uploadMsg.replaceChildren();
    const files = Array.from(fileList || []);
    if (!files.length) return;
    try {
      const built = await buildSetFromFiles(files);
      if (built.config.items.length < 2) throw new Error('A stimulus set needs at least 2 items.' + (built.warnings.length ? ' ' + built.warnings.join(' ') : ''));
      const firstRel = (files[0].webkitRelativePath || '').split('/')[0];
      const name = nameInput.value.trim() || firstRel || (built.config_name ? built.config_name.replace(/\.json$/i, '') : `Upload ${new Date().toLocaleString()}`);
      const rec = await saveSet(name, built);
      nameInput.value = '';
      await refreshSources(`set:${rec.name}`);
      await loadChosen();
      uploadMsg.replaceChildren(h('span', { class: 'good' }, `Saved "${rec.name}": ${built.config.items.length} items, ${Object.keys(built.files).length} media files.`));
      if (built.warnings.length) uploadMsg.append(h('div', { class: 'warn' }, 'Notes: ' + built.warnings.join(' ')));
    } catch (e) { uploadMsg.replaceChildren(h('span', { class: 'bad' }, String(e.message))); }
  }

  updateModeUi();
  refresh();
  await refreshSources();
  await loadChosen();
  return () => { feed.stop(); offHs(); };

  async function start() {
    err.textContent = '';
    const cur = read();
    const pid = participant.value.trim();
    if (!pid) { err.textContent = 'Enter a participant ID.'; return; }
    if (!stimuli) { err.textContent = 'The stimulus set could not be loaded.'; return; }
    if (cur.off_max_ms < cur.off_min_ms) { err.textContent = 'Gap max must be ≥ gap min.'; return; }
    if (cur.options_per_trial > stimuli.items.length) { err.textContent = 'More options per question than stimulus items.'; return; }
    if (cur.predictor_mode === 'eeg' && !(app.bridge.status().eeg || {}).found) { err.textContent = 'Predictor is Live EEG but no headset stream is reaching the bridge. Start eeg/stream_eeg.py, or choose Mock.'; return; }
    if (cur.trial_type === 'mcq_sequential' && cur.show_model_probs && cur.predictor_mode === 'off') { err.textContent = 'Model probabilities need a predictor: choose Mock or WebSocket, or untick "Show model probability".'; return; }
    try { await db.getAll('sessions'); } catch (e) { err.textContent = 'Local storage (IndexedDB) is unavailable: ' + e.message; return; }
    saveSettings(cur);
    localStorage.setItem('cortext-lab.last-participant', JSON.stringify({ participant: pid, device: device.value }));
    cur.seed = (Math.random() * 2 ** 31) >>> 0;
    let specs;
    try {
      specs = buildTrialSpecs(stimuli, { n: cur.n_trials, seed: cur.seed, optionsPerTrial: cur.options_per_trial, tileContent: cur.tile_content, trialType: cur.trial_type });
    } catch (e) { err.textContent = e.message; return; }
    const session = newSessionRecord({ participant_id: pid, device: device.value, notes: notes.value, settings: cur, stimuli, plannedTrials: specs.length, snapshot: stimuliSnapshot(stimuli) });
    app.pending = { session, settings: cur, stimuli, specs };
    location.hash = '#/run';
  }
}
