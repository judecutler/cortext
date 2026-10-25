import { h } from '../util.js';
import { loadSettings, saveSettings, patchSettings, loadPlan, savePlan, loadParticipant, saveParticipant, estimateDurations, DEFAULT_SETTINGS } from '../settings.js';
import { loadStimuli, buildTrialSpecs, generatePlan, rephrase, specsFromPlan, shuffled } from '../stimuli.js';
import { listPresets, getPreset, savePreset, deletePreset, exportPreset, importPresetFile, isBuiltin, PRESET_KEYS } from '../presets.js';
import { newSessionRecord } from '../session.js';
import { listTrialTypes, getTrialType } from '../trialtypes/index.js';
import { buildSetFromFiles, saveSet, listSets, getSet, materializeSet, stimuliSnapshot } from '../stimulusSets.js';
import { db } from '../db.js';
import { subTabs, toast, confirmDialog } from '../ui.js';

const MODE_HELP = {
  mcq_flash: 'The answer tiles stay on screen and light up one at a time (like a P300 speller). Afterwards you click the answer.',
  mcq_sequential: 'The question stays on screen. Each option’s picture/text appears alone, one after another. Then the full multiple-choice grid appears with the model’s probability on every choice, and you click the correct one.',
};

export async function renderExperiment(root, app) {
  const s = loadSettings();
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

  const modeHelp = h('small', {});

  const trialTypeOpts = listTrialTypes().map((t) => [t.id, t.label + (t.implemented ? '' : ' (not implemented yet)'), !t.implemented]);

  // ---------- stimulus set card ----------
  const sourceSel = h('select', { onchange: () => { updateSourceUi(); loadChosen(); refresh(); } });
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
  const typeCard = h('div', { class: 'card' }, h('h2', {}, 'Experiment'),
    sel('trial_type', 'Presentation mode', trialTypeOpts, null, updateModeUi),
    h('div', { class: 'field' }, h('label', {}), modeHelp));
  const designCard = h('div', { class: 'card' }, h('h2', {}, 'Design (both modes)'),
    num('n_trials', 'Questions', 'Each target is used once before any repeats.', { min: 1 }),
    num('options_per_trial', 'Options per question', '2–6; the grid adapts.', { min: 2, max: 6 }),
    num('prompt_ms', 'Prompt shown first (ms)', 'Time to read the question before anything else happens.', { min: 0 }),
    num('imagery_ms', 'Imagery window (ms)', '0 = off. Otherwise, after the question the screen goes dark with a + for this long while the participant pictures the answer; then the options return for 1 s before flashing. Saved as imagery.npz for the image model. 2000–3000 typical.', { min: 0, step: 100 }),
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
    chk('log_flash_off', 'Also send flash_off markers'),
    chk('show_bars', 'Live score bars under tiles', 'Flash mode, demo use only: it adds visible changes during flashing. Toggle with Shift+B.'));

  // ---------- question list & image order ----------
  let plan = loadPlan();
  const planHost = h('div', { class: 'planlist' });
  const planMsg = h('div', { class: 'muted' });
  const wordingInput = h('input', { type: 'text', placeholder: 'Which one is the {label}?', style: { width: '100%' } });
  const itemName = (id) => { const it = stimuli?.byId?.[id] || stimuli?.items?.find((i) => i.id === id); return it ? (it.label || it.text || id) : `${id} (missing)`; };
  const persistPlan = () => { savePlan(plan); refresh(); };
  function renderPlan() {
    planMsg.textContent = plan.length ? `${plan.length} question${plan.length === 1 ? '' : 's'} in the list.` : 'The list is empty. Use “Generate from stimulus set” or “Add question”.';
    planHost.replaceChildren(...plan.map((q, i) => {
      const move = (arr, a, b) => { [arr[a], arr[b]] = [arr[b], arr[a]]; persistPlan(); renderPlan(); };
      const chips = q.options.map((id, j) => h('span', { class: `chip${id === q.target ? ' answer' : ''}`, title: id === q.target ? 'Correct answer' : '' },
        h('span', { class: 'chip-name' }, itemName(id)),
        h('button', { class: 'chip-btn', type: 'button', 'aria-label': 'Move earlier', disabled: j === 0, onclick: () => move(q.options, j, j - 1) }, '◀'),
        h('button', { class: 'chip-btn', type: 'button', 'aria-label': 'Move later', disabled: j === q.options.length - 1, onclick: () => move(q.options, j, j + 1) }, '▶'),
        h('button', { class: 'chip-btn', type: 'button', 'aria-label': 'Remove image', disabled: q.options.length <= 2, onclick: () => { q.options.splice(j, 1); if (!q.options.includes(q.target)) q.target = q.options[0]; persistPlan(); renderPlan(); } }, '✕')));
      const free = (stimuli?.items || []).filter((it) => !q.options.includes(it.id));
      const addSel = h('select', { 'aria-label': 'Add an image', onchange: (e) => { if (e.target.value) { q.options.push(e.target.value); persistPlan(); renderPlan(); } } },
        h('option', { value: '' }, '+ image'), ...free.map((it) => h('option', { value: it.id }, it.label || it.text || it.id)));
      const ansSel = h('select', { 'aria-label': 'Correct answer', onchange: (e) => { q.target = e.target.value; persistPlan(); renderPlan(); } },
        ...q.options.map((id) => h('option', { value: id, selected: id === q.target }, itemName(id))));
      return h('div', { class: 'planrow' },
        h('div', { class: 'planrow-head' }, h('b', {}, `Q${i + 1}`),
          h('input', { type: 'text', value: q.prompt, class: 'plan-prompt', 'aria-label': `Question ${i + 1} wording`, oninput: (e) => { q.prompt = e.target.value; persistPlan(); } }),
          h('button', { class: 'btn small', type: 'button', title: 'Move question up', disabled: i === 0, onclick: () => { const t = plan[i - 1]; plan[i - 1] = plan[i]; plan[i] = t; persistPlan(); renderPlan(); } }, '↑'),
          h('button', { class: 'btn small', type: 'button', title: 'Move question down', disabled: i === plan.length - 1, onclick: () => { const t = plan[i + 1]; plan[i + 1] = plan[i]; plan[i] = t; persistPlan(); renderPlan(); } }, '↓'),
          h('button', { class: 'btn small', type: 'button', title: 'Shuffle this question’s images', onclick: () => { q.options = shuffled(q.options); persistPlan(); renderPlan(); } }, '🔀'),
          h('button', { class: 'btn small danger', type: 'button', title: 'Remove question', onclick: () => { plan.splice(i, 1); persistPlan(); renderPlan(); } }, '✕')),
        h('div', { class: 'planrow-body' }, h('div', { class: 'chips' }, ...chips, addSel), h('label', { class: 'plan-answer' }, 'Answer ', ansSel)));
    }));
  }
  const needStim = () => { if (!stimuli) { toast('Load a stimulus set first.', 'warn'); return false; } return true; };
  const planBtn = (label, fn, title) => h('button', { class: 'btn small', type: 'button', title, onclick: fn }, label);
  const planCard = h('div', { class: 'card wide' }, h('h2', {}, 'Questions & image order'),
    chk('use_plan', 'Use this list for the session', 'Off: questions are generated automatically (random targets and distractors). On: exactly the questions below are asked, with the wording and images listed.'),
    sel('order_mode', 'Image order in each question', [['random', 'Random every trial (recommended for analysis)'], ['fixed', 'Exactly as listed']], 'Random removes position effects. “As listed” keeps tile positions and the one-at-a-time sequence in the order you set, so position and order are confounded with the item.'),
    chk('shuffle_on_start', 'Re-shuffle this list every time a session starts', 'Randomizes question order and image order at Start, so each participant sees a different pattern. The pattern actually used is saved with the session.'),
    h('div', { class: 'ctrls plan-tools' },
      planBtn('Generate from stimulus set', () => {
        if (!needStim()) return;
        const o = read();
        try { plan = generatePlan(stimuli, { n: o.n_trials, optionsPerTrial: o.options_per_trial }); } catch (e) { toast(e.message, 'bad'); return; }
        persistPlan(); renderPlan();
      }, 'Fills the list using “Questions” and “Options per question” from Design'),
      planBtn('🔀 Shuffle questions', () => { plan = shuffled(plan); persistPlan(); renderPlan(); }, 'Randomize the order of the questions'),
      planBtn('🔀 Shuffle images', () => { plan = plan.map((q) => ({ ...q, options: shuffled(q.options) })); persistPlan(); renderPlan(); }, 'Randomize image order inside every question'),
      planBtn('🔀 Shuffle both', () => { plan = shuffled(plan).map((q) => ({ ...q, options: shuffled(q.options) })); persistPlan(); renderPlan(); }, 'Randomize question order and image order'),
      planBtn('Add question', () => {
        if (!needStim()) return;
        const o = read(); const ids = shuffled(stimuli.items.map((x) => x.id)).slice(0, Math.max(2, Math.min(o.options_per_trial, stimuli.items.length)));
        plan.push({ prompt: 'Which one is correct?', target: ids[0], options: ids }); persistPlan(); renderPlan();
      }),
      planBtn('Clear list', async () => { if (plan.length && await confirmDialog({ title: 'Clear the question list?', body: 'This only empties the list on this page; saved presets are not touched.', confirmLabel: 'Clear', danger: true })) { plan = []; persistPlan(); renderPlan(); } })),
    h('div', { class: 'field' }, h('label', {}, 'Wording for all questions'), wordingInput,
      h('small', {}, 'Placeholders: {label} {text} {category} {id}, e.g. “Which one is the {label}?”.')),
    h('div', { class: 'ctrls' }, planBtn('Apply wording to all', () => {
      if (!needStim() || !wordingInput.value.trim()) return;
      try { plan = rephrase(stimuli, plan, wordingInput.value.trim()); } catch (e) { toast(e.message, 'bad'); return; }
      persistPlan(); renderPlan();
    })),
    planMsg, planHost);

  // ---------- presets ----------
  const presetSel = h('select', { 'aria-label': 'Preset' });
  const presetName = h('input', { type: 'text', placeholder: 'name for a new preset', style: { width: '100%' } });
  const presetDesc = h('small', { class: 'muted' });
  const presetFile = h('input', { type: 'file', accept: '.json', style: { display: 'none' }, onchange: async (e) => {
    const f = e.target.files[0]; e.target.value = '';
    if (!f) return;
    try { const n = await importPresetFile(f); fillPresetList(n); toast(`Imported “${n}”.`, 'good'); } catch (er) { toast(er.message, 'bad'); }
  } });
  function fillPresetList(select) {
    const { builtin, mine } = listPresets();
    presetSel.replaceChildren(
      h('optgroup', { label: 'Built-in' }, ...builtin.map((n) => h('option', { value: n }, n))),
      h('optgroup', { label: 'Mine (this browser)' }, ...(mine.length ? mine.map((n) => h('option', { value: n }, n)) : [h('option', { value: '', disabled: true }, '— none saved yet —')])));
    if (select) presetSel.value = select;
    showPresetInfo();
  }
  function showPresetInfo() {
    const p = getPreset(presetSel.value);
    presetDesc.textContent = p ? (p.description || `${p.plan?.length || 0} listed question(s), ${Object.keys(p.settings).length} settings.`) : '';
  }
  presetSel.addEventListener('change', showPresetInfo);
  function snapshotPreset() { const cur = read(); return { settings: Object.fromEntries(PRESET_KEYS.map((k) => [k, cur[k]])), plan }; }
  async function loadPreset() {
    const p = getPreset(presetSel.value);
    if (!p) return;
    // A preset that predates the imagery window has no imagery_ms: treat that as OFF so a value left in the form can't leak into it.
    for (const [k, v] of Object.entries({ imagery_ms: 0, ...p.settings })) {
      const el = inputs[k];
      if (!el) continue;
      if (k === 'stimuli_source') { if ([...el.options].some((o) => o.value === v)) el.value = v; else toast(`The stimulus set “${String(v).replace(/^set:/, '')}” is not in this browser; upload it again to use this preset's list.`, 'warn'); continue; }
      if (el.type === 'checkbox') el.checked = !!v; else el.value = v;
    }
    if (Array.isArray(p.plan) && p.plan.length) plan = p.plan.map((q) => ({ prompt: q.prompt, target: q.target, options: q.options.slice() }));
    else if (p.settings.use_plan === false) { /* keep the current list around, just not in use */ }
    updateModeUi(); updateSourceUi(); refresh(); savePlan(plan); renderPlan();
    await loadChosen();
    toast(`Loaded “${presetSel.value}”.`, 'good');
  }
  const presetCard = h('div', { class: 'card wide' }, h('h2', {}, 'Presets'),
    h('div', { class: 'field' }, h('label', {}, 'Preset'), presetSel, presetDesc),
    h('div', { class: 'ctrls' },
      planBtn('Load', loadPreset, 'Apply this preset to the settings below'),
      planBtn('Update', () => {
        const n = presetSel.value;
        if (!n || isBuiltin(n)) { toast('Built-in presets cannot be changed; use “Save as new”.', 'warn'); return; }
        try { savePreset(n, snapshotPreset()); toast(`Updated “${n}”.`, 'good'); fillPresetList(n); } catch (e) { toast(e.message, 'bad'); }
      }, 'Overwrite the selected preset with the current settings and list'),
      planBtn('Delete', async () => {
        const n = presetSel.value;
        if (!n || isBuiltin(n)) { toast('Built-in presets cannot be deleted.', 'warn'); return; }
        if (await confirmDialog({ title: `Delete preset “${n}”?`, body: 'This removes it from this browser. Recorded sessions are not affected.', confirmLabel: 'Delete', danger: true })) { deletePreset(n); fillPresetList(); }
      }),
      planBtn('Export…', () => { try { exportPreset(presetSel.value); } catch (e) { toast(e.message, 'bad'); } }, 'Download as a .json file to share'),
      planBtn('Import…', () => presetFile.click(), 'Load a preset .json from a teammate'), presetFile),
    h('div', { class: 'field' }, h('label', {}, 'Save current as'), presetName,
      h('small', {}, 'Saves the experiment settings (not the Setup tab) and the question list.')),
    h('div', { class: 'ctrls' }, planBtn('Save as new', () => {
      try { const n = savePreset(presetName.value, snapshotPreset()); presetName.value = ''; fillPresetList(n); toast(`Saved “${n}”.`, 'good'); } catch (e) { toast(e.message, 'bad'); }
    })));
  // ---------- rig strip: what Setup decided (read-only here) ----------
  const rig = h('div', { class: 'rigstrip card' });
  function updateRig() {
    const p = loadParticipant(), cur = loadSettings(), b = app.bridge.status(), eeg = b.eeg || {};
    const chip = (ok, text) => h('span', { class: `rigchip ${ok === true ? 'ok' : ok === false ? 'bad' : 'warn'}` }, text);
    const q = eeg.quality || [];
    const bad = q.filter((x) => x.status !== 'good').length;
    rig.replaceChildren(
      chip(!!p.participant, p.participant ? `Participant ${p.participant}` : 'No participant ID'),
      chip(!!p.device, p.device || 'No device chosen'),
      chip(!b.connected ? false : eeg.found ? (bad ? null : true) : (cur.predictor_mode === 'eeg' ? false : null),
        !b.connected ? 'Bridge offline' : eeg.found ? `Headset${eeg.simulated ? ' (SIMULATED)' : ''}: ${bad ? bad + ' electrode(s) need attention' : 'all electrodes good'}` : 'No headset stream'),
      chip(cur.predictor_mode === 'off' ? null : true, `Predictor: ${({ eeg: 'Live EEG', mock: 'Mock', ws: 'Model server', off: 'Off' })[cur.predictor_mode] || cur.predictor_mode}`),
      h('a', { class: 'btn small', href: '#/setup' }, 'Change in Setup'));
  }
  const offRig = app.bridge.onChange(updateRig);
  updateRig();

  const startBtn = h('button', { class: 'btn primary', onclick: start }, 'Start session');
  const resetBtn = h('button', { class: 'btn', type: 'button', onclick: resetDefaults }, 'Reset experiment settings');
  root.append(subTabs('configure'), h('h1', {}, 'Experiment'), rig,
    h('div', { class: 'grid2' }, presetCard, typeCard, stimCard, designCard, planCard, flashCard, seqCard, respCard),
    h('div', { class: 'startbar' }, startBtn, resetBtn, err));

  function read() {
    const out = { ...loadSettings() };
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
    const d = estimateDurations(cur.use_plan ? { ...cur, n_trials: plan.length } : cur);
    estimate.textContent = `${d.flashes} stimulus onsets per question ≈ ${d.flashSec.toFixed(0)} s; about ${d.trialSec.toFixed(0)} s per question and ${d.totalMin.toFixed(1)} min in total.`;
    patchSettings(Object.fromEntries(Object.keys(inputs).map((k) => [k, cur[k]])));   // remembered as you type; no need to start a session
  }
  function resetDefaults() {
    for (const [k, el] of Object.entries(inputs)) {
      const v = DEFAULT_SETTINGS[k];
      if (el.type === 'checkbox') el.checked = !!v; else el.value = v;
    }
    updateModeUi(); updateSourceUi(); refresh(); loadChosen();
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
      renderPlan();
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
  fillPresetList();
  await refreshSources();
  await loadChosen();
  renderPlan();
  return () => { offRig(); };

  async function start() {
    err.textContent = '';
    const cur = read();
    const who = loadParticipant();
    const pid = (who.participant || '').trim();
    if (!pid) { err.replaceChildren('Enter a participant ID in ', h('a', { href: '#/setup' }, 'Setup'), ' first.'); return; }
    if (!who.device) { err.replaceChildren('Choose the recording device in ', h('a', { href: '#/setup' }, 'Setup'), ' first.'); return; }
    if (!stimuli) { err.textContent = 'The stimulus set could not be loaded.'; return; }
    if (cur.off_max_ms < cur.off_min_ms) { err.textContent = 'Gap max must be ≥ gap min.'; return; }
    if (cur.options_per_trial > stimuli.items.length) { err.textContent = 'More options per question than stimulus items.'; return; }
    if (cur.predictor_mode === 'eeg' && !(app.bridge.status().eeg || {}).found) { err.textContent = 'Predictor is Live EEG but no headset stream is reaching the bridge. Start eeg/stream_eeg.py, or choose Mock under Setup → Prediction.'; return; }
    if (cur.trial_type === 'mcq_sequential' && cur.show_model_probs && cur.predictor_mode === 'off') { err.textContent = 'Model probabilities need a predictor: choose Mock or WebSocket, or untick "Show model probability".'; return; }
    try { await db.getAll('sessions'); } catch (e) { err.textContent = 'Local storage (IndexedDB) is unavailable: ' + e.message; return; }
    saveSettings(cur);
    saveParticipant({ ...who, notes: '' });   // notes belong to one session; the participant and device carry over
    cur.seed = (Math.random() * 2 ** 31) >>> 0;
    let specs;
    try {
      if (cur.use_plan) {
        if (!plan.length) throw new Error('“Use this list” is on but the question list is empty.');
        if (cur.shuffle_on_start) { plan = shuffled(plan).map((q) => ({ ...q, options: shuffled(q.options) })); savePlan(plan); renderPlan(); }
        specs = specsFromPlan(stimuli, plan, { trialType: cur.trial_type, tileContent: cur.tile_content, fixedOrder: cur.order_mode === 'fixed' });
        cur.n_trials = specs.length;
        cur.question_plan = plan.map((q) => ({ prompt: q.prompt, target: q.target, options: q.options.slice() }));
      } else specs = buildTrialSpecs(stimuli, { n: cur.n_trials, seed: cur.seed, optionsPerTrial: cur.options_per_trial, tileContent: cur.tile_content, trialType: cur.trial_type });
    } catch (e) { err.textContent = e.message; return; }
    const session = newSessionRecord({ participant_id: pid, device: who.device, notes: who.notes, settings: cur, stimuli, plannedTrials: specs.length, snapshot: stimuliSnapshot(stimuli) });
    app.pending = { session, settings: cur, stimuli, specs };
    location.hash = '#/run';
  }
}
