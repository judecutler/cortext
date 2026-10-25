import { h } from '../util.js';
import { loadSettings, saveSettings, patchSettings, loadPlan, savePlan, loadParticipant, saveParticipant, estimateDurations, DEFAULT_SETTINGS } from '../settings.js';
import { loadStimuli, generatePlan, rephrase, shuffled, categoriesOf } from '../stimuli.js';
import { listPresets, getPreset, savePreset, deletePreset, exportPreset, importPresetFile, isBuiltin, PRESET_KEYS } from '../presets.js';
import { prepareLaunch, goRun } from '../launch.js';
import { record, snapshot } from '../undo.js';
import { listTrialTypes, getTrialType } from '../trialtypes/index.js';
import { buildSetFromFiles, saveSet, listSets, getSet, materializeSet } from '../stimulusSets.js';
import { db } from '../db.js';
import { subTabs, toast } from '../ui.js';
import { participantCard, headsetCard, hardwareCard, storageCard, rawEegCard, autoLiveEeg, predictorName } from './rig.js';

const MODE_HELP = {
  mcq_flash: 'The answer tiles stay on screen and light up one at a time (like a P300 speller). Afterwards you click the answer.',
  mcq_sequential: 'The question stays on screen. Each option’s picture/text appears alone, one after another. Then the full multiple-choice grid appears with the model’s probability on every choice, and you click the correct one.',
};

// The Experiment tab: who is in the chair and whether the headset is good, the experiment design, and the lab's
// display / storage settings, all on one page with one Start button. (This used to be two tabs, Setup + Experiment.)
// How the EEG becomes an answer (filters, model, predictor, narrator voice) is on the Agent tab.
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
  const stimCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', {}, 'Stimulus set'),
      h('a', { class: 'btn small', href: '#/library', title: 'See every category with its images; drag to rearrange, add or remove' }, 'Organise images & categories →')),
    h('div', { class: 'field' }, h('label', {}, 'Use'), sourceSel),
    stimInfo,
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
    uploadMsg);

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
  const voiceCard = h('div', { class: 'card' }, h('h2', {}, 'Voice & audio'),
    chk('voice_on', 'Voice on (demo)', 'One switch for the stage: reads the question aloud and says each option’s name as it appears. Same as ticking “Read the question aloud” with “Audio + visual” below; the session records it that way.'),
    chk('speak_question', 'Read the question aloud', 'Spoken during the prompt; the stimuli start only after it has finished, so the voice never overlaps the EEG window.'),
    sel('stim_modality', 'Stimulus modality', [['visual', 'Visual (pictures / text)'], ['audio', 'Audio: each option is spoken, nothing changes on screen'], ['audiovisual', 'Audio + visual together']],
      'Plays the item’s own audio file if the stimulus set has one (upload dog.mp3 next to dog.png), otherwise speaks its text. Sounds are pre-loaded and start on the marker frame; each flash_on marker records the audio latency. One-at-a-time mode (1 s each) suits spoken words best.'),
    sel('narrate', 'Narrator', [['off', 'Off'], ['after_click', 'After the answer: announces the model’s guess'], ['before_click', 'Before the answer (show mode)']],
      'Voice, mood and LLM are set on the Agent tab (Result block). “Before the answer” tells the participant the guess, so those clicks are flagged model_shown_before_click and must not be used to evaluate the model.'),
    h('div', {}, h('a', { class: 'btn small', href: '#/agent/result' }, 'Voice & narrator settings →')));
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
  /** Every change to the list goes through here so it can be undone (Ctrl/Cmd+Z). quiet: no toast for small, visible moves. */
  function editPlan(label, fn, { quiet = false } = {}) {
    const before = snapshot(plan);
    if (fn() === false) return;
    const after = snapshot(plan);
    persistPlan(); renderPlan();
    const put = (v) => () => { plan = snapshot(v); persistPlan(); renderPlan(); };
    record(label, put(before), put(after), { toast: !quiet });
  }
  function renderPlan() {
    planMsg.textContent = plan.length ? `${plan.length} question${plan.length === 1 ? '' : 's'} in the list.` : 'The list is empty. Use “Generate from stimulus set” or “Add question”.';
    planHost.replaceChildren(...plan.map((q, i) => {
      const swap = (arr, a, b) => { [arr[a], arr[b]] = [arr[b], arr[a]]; };
      const chips = q.options.map((id, j) => h('span', { class: `chip${id === q.target ? ' answer' : ''}`, title: id === q.target ? 'Correct answer' : '' },
        h('span', { class: 'chip-name' }, itemName(id)),
        h('button', { class: 'chip-btn', type: 'button', 'aria-label': 'Move earlier', disabled: j === 0, onclick: () => editPlan(`Moved ${itemName(id)} earlier in Q${i + 1}`, () => swap(q.options, j, j - 1), { quiet: true }) }, '◀'),
        h('button', { class: 'chip-btn', type: 'button', 'aria-label': 'Move later', disabled: j === q.options.length - 1, onclick: () => editPlan(`Moved ${itemName(id)} later in Q${i + 1}`, () => swap(q.options, j, j + 1), { quiet: true }) }, '▶'),
        h('button', { class: 'chip-btn', type: 'button', 'aria-label': 'Remove image', title: q.options.length <= 2 ? 'A question needs at least 2 images' : 'Remove this image', disabled: q.options.length <= 2,
          onclick: () => editPlan(`Removed ${itemName(id)} from Q${i + 1}`, () => { q.options.splice(j, 1); if (!q.options.includes(q.target)) q.target = q.options[0]; }) }, '✕')));
      // "+ image" grouped by category, so a long set stays easy to scan
      const free = (stimuli?.items || []).filter((it) => !q.options.includes(it.id));
      const groups = stimuli ? categoriesOf({ items: free }) : [];
      const addSel = h('select', { 'aria-label': 'Add an image', onchange: (e) => { const v = e.target.value; if (v) editPlan(`Added ${itemName(v)} to Q${i + 1}`, () => { q.options.push(v); }, { quiet: true }); } },
        h('option', { value: '' }, '+ image'),
        ...groups.map((c) => h('optgroup', { label: c }, ...free.filter((it) => it.category === c).map((it) => h('option', { value: it.id }, it.label || it.text || it.id)))));
      const ansSel = h('select', { 'aria-label': 'Correct answer', onchange: (e) => { const v = e.target.value; editPlan(`Set Q${i + 1}’s answer to ${itemName(v)}`, () => { q.target = v; }, { quiet: true }); } },
        ...q.options.map((id) => h('option', { value: id, selected: id === q.target }, itemName(id))));
      return h('div', { class: 'planrow' },
        h('div', { class: 'planrow-head' }, h('b', {}, `Q${i + 1}`),
          h('input', { type: 'text', value: q.prompt, class: 'plan-prompt', 'aria-label': `Question ${i + 1} wording`, oninput: (e) => { q.prompt = e.target.value; persistPlan(); } }),
          h('button', { class: 'btn small', type: 'button', title: 'Move question up', disabled: i === 0, onclick: () => editPlan(`Moved Q${i + 1} up`, () => swap(plan, i, i - 1), { quiet: true }) }, '↑'),
          h('button', { class: 'btn small', type: 'button', title: 'Move question down', disabled: i === plan.length - 1, onclick: () => editPlan(`Moved Q${i + 1} down`, () => swap(plan, i, i + 1), { quiet: true }) }, '↓'),
          h('button', { class: 'btn small', type: 'button', title: 'Shuffle this question’s images', onclick: () => editPlan(`Shuffled Q${i + 1}’s images`, () => { q.options = shuffled(q.options); }) }, '🔀'),
          h('button', { class: 'btn small', type: 'button', title: 'Duplicate this question', onclick: () => editPlan(`Duplicated Q${i + 1}`, () => { plan.splice(i + 1, 0, snapshot(q)); }) }, '⧉'),
          h('button', { class: 'btn small danger', type: 'button', title: 'Remove question', onclick: () => editPlan(`Removed Q${i + 1} (“${q.prompt}”)`, () => { plan.splice(i, 1); }) }, '✕')),
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
        let fresh;
        try { fresh = generatePlan(stimuli, { n: o.n_trials, optionsPerTrial: o.options_per_trial }); } catch (e) { toast(e.message, 'bad'); return; }
        editPlan(`Generated ${fresh.length} questions${plan.length ? ' (replacing the old list)' : ''}`, () => { plan = fresh; });
      }, 'Fills the list using “Questions” and “Options per question” from Design'),
      planBtn('🔀 Shuffle questions', () => editPlan('Shuffled the question order', () => { plan = shuffled(plan); }), 'Randomize the order of the questions'),
      planBtn('🔀 Shuffle images', () => editPlan('Shuffled the images in every question', () => { plan = plan.map((q) => ({ ...q, options: shuffled(q.options) })); }), 'Randomize image order inside every question'),
      planBtn('🔀 Shuffle both', () => editPlan('Shuffled questions and images', () => { plan = shuffled(plan).map((q) => ({ ...q, options: shuffled(q.options) })); }), 'Randomize question order and image order'),
      planBtn('Add question', () => {
        if (!needStim()) return;
        const o = read(); const ids = shuffled(stimuli.items.map((x) => x.id)).slice(0, Math.max(2, Math.min(o.options_per_trial, stimuli.items.length)));
        editPlan('Added a question', () => { plan.push({ prompt: 'Which one is correct?', target: ids[0], options: ids }); }, { quiet: true });
      }),
      planBtn('Clear list', () => { if (plan.length) editPlan(`Cleared the list (${plan.length} questions)`, () => { plan = []; }); }, 'Empty the list (undo with Ctrl/Cmd+Z); saved presets are not touched')),
    h('div', { class: 'field' }, h('label', {}, 'Wording for all questions'), wordingInput,
      h('small', {}, 'Placeholders: {label} {text} {category} {id}, e.g. “Which one is the {label}?”.')),
    h('div', { class: 'ctrls' }, planBtn('Apply wording to all', () => {
      if (!needStim() || !wordingInput.value.trim()) return;
      let next;
      try { next = rephrase(stimuli, plan, wordingInput.value.trim()); } catch (e) { toast(e.message, 'bad'); return; }
      editPlan('Applied the wording to every question', () => { plan = next; });
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
  /** Everything a preset can change on this page, so loading one can be undone. */
  function formState() { return { values: Object.fromEntries(Object.entries(inputs).map(([k, el]) => [k, el.type === 'checkbox' ? el.checked : el.value])), plan: snapshot(plan) }; }
  async function restoreForm(st) {
    for (const [k, v] of Object.entries(st.values)) { const el = inputs[k]; if (!el) continue; if (el.type === 'checkbox') el.checked = !!v; else el.value = v; }
    plan = snapshot(st.plan);
    updateModeUi(); updateSourceUi(); refresh(); savePlan(plan); renderPlan();
    await loadChosen();
  }
  async function loadPreset() {
    const p = getPreset(presetSel.value);
    if (!p) return;
    const before = formState();
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
    const after = formState();
    record(`Loaded preset “${presetSel.value}”`, () => restoreForm(before), () => restoreForm(after), { kind: 'good' });
  }
  const presetCard = h('div', { class: 'card wide' }, h('h2', {}, 'Presets'),
    h('div', { class: 'field' }, h('label', {}, 'Preset'), presetSel, presetDesc),
    h('div', { class: 'ctrls' },
      planBtn('Load', loadPreset, 'Apply this preset to the settings below'),
      planBtn('Update', () => {
        const n = presetSel.value;
        if (!n || isBuiltin(n)) { toast('Built-in presets cannot be changed; use “Save as new”.', 'warn'); return; }
        const old = getPreset(n);
        try { savePreset(n, snapshotPreset()); fillPresetList(n); } catch (e) { toast(e.message, 'bad'); return; }
        const now = getPreset(n);
        record(`Updated preset “${n}”`, () => { savePreset(n, old); fillPresetList(n); }, () => { savePreset(n, now); fillPresetList(n); }, { kind: 'good' });
      }, 'Overwrite the selected preset with the current settings and list'),
      planBtn('Delete', async () => {
        const n = presetSel.value;
        if (!n || isBuiltin(n)) { toast('Built-in presets cannot be deleted.', 'warn'); return; }
        const p = getPreset(n);
        deletePreset(n); fillPresetList();
        record(`Deleted preset “${n}”`, () => { savePreset(n, p); fillPresetList(n); }, () => { deletePreset(n); fillPresetList(); });
      }),
      planBtn('Export…', () => { try { exportPreset(presetSel.value); } catch (e) { toast(e.message, 'bad'); } }, 'Download as a .json file to share'),
      planBtn('Import…', () => presetFile.click(), 'Load a preset .json from a teammate'), presetFile),
    h('div', { class: 'field' }, h('label', {}, 'Save current as'), presetName,
      h('small', {}, 'Saves the experiment design and the question list (not the participant, headset, storage or Agent settings).')),
    h('div', { class: 'ctrls' }, planBtn('Save as new', () => {
      try { const n = savePreset(presetName.value, snapshotPreset()); presetName.value = ''; fillPresetList(n); toast(`Saved “${n}”.`, 'good'); } catch (e) { toast(e.message, 'bad'); }
    })));
  // ---------- participant, headset, and what the Agent will do ----------
  const agentBody = h('div', { class: 'agent-sum' });
  const agentCard = h('div', { class: 'card', id: 'agentSummary' },
    h('div', { class: 'card-head' }, h('h2', {}, 'Agent'), h('a', { class: 'btn small', href: '#/agent' }, 'Open the Agent →')),
    h('p', { class: 'muted small', style: { margin: '0 0 6px' } }, 'What turns the EEG into an answer. Change it on the Agent tab.'), agentBody);
  function updateAgent() {
    const cur = loadSettings(), b = app.bridge.status(), eeg = b.eeg || {}, sc = eeg.scorer || {}, fl = eeg.filters || {};
    const line = (k, v, cls = '') => h('div', { class: 'row' }, h('span', { class: 'muted' }, k), h('span', { class: cls }, v));
    const mode = cur.predictor_mode;
    agentBody.replaceChildren(
      line('Predictor', predictorName(mode), mode === 'mock' ? 'warn' : mode === 'off' ? 'muted' : ''),
      mode === 'eeg' ? line('Filters', b.connected ? (fl.summary || ['–']).join(' → ') : 'bridge offline', fl.error ? 'warn' : '') : null,
      mode === 'eeg' ? line('Model', !b.connected ? '–' : sc.active === 'model' ? (sc.model_file || '').split(/[\\/]/).pop()
        : sc.active === 'adaptive' ? 'calibrated on this session' : `none yet: adapts after ${sc.adaptive_needs || '5 trials'}`,
        sc.model_filter_mismatch || sc.model_error ? 'warn' : sc.active === 'model' ? 'good' : '') : null,
      sc.model_filter_mismatch ? h('div', { class: 'warn small' }, `Model ${sc.model_filter_mismatch}`) : null,
      mode === 'mock' ? h('div', { class: 'warn small' }, 'Mock predictions use the true answer: fine for rehearsing, not evidence.') : null,
      // the p3000 function reads the spike 240-480 ms after each picture: in flash mode the next flash lands inside that
      // window, so the spike is credited to the wrong option (measured: chance on simulated EEG). One at a time only.
      mode === 'eeg' && sc.active === 'model' && /p3000/i.test(sc.model_file || '') && cur.trial_type === 'mcq_flash'
        ? h('div', { class: 'warn small' }, 'The p3000 function needs “One at a time, then choose”: in flash mode the flashes come too fast for it (chance accuracy).') : null);
    [...agentBody.childNodes].forEach((n) => { if (n.nodeType === 3 && n.textContent === 'null') n.remove(); });
  }

  // the headset card can report a stream synchronously (bridge already connected), so it comes after the Agent card
  const whoCard = participantCard();
  const headset = headsetCard(app, { onFirstStream: (eeg) => {
    const dev = whoCard.autofillDevice(eeg);
    const live = autoLiveEeg();
    if (live) { updateAgent(); }
    const bits = [dev && `device = ${dev}`, live && 'predictor = Live EEG'].filter(Boolean);
    return bits.length ? `Auto-filled: ${bits.join(', ')}.` : null;
  } });
  // ---------- readiness: shown next to Start ----------
  const rig = h('div', { class: 'rigstrip' });
  function updateRig() {
    const p = loadParticipant(), cur = loadSettings(), b = app.bridge.status(), eeg = b.eeg || {};
    const chip = (ok, text, go) => h('button', { type: 'button', class: `rigchip ${ok === true ? 'ok' : ok === false ? 'bad' : 'warn'}`, onclick: go }, text);
    const toCard = (id) => () => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    const q = eeg.quality || [];
    const bad = q.filter((x) => x.status !== 'good').length;
    rig.replaceChildren(
      chip(!!p.participant, p.participant ? `Participant ${p.participant}` : 'No participant ID', () => { toCard('participant')(); whoCard.focus(); }),
      chip(!!p.device, p.device || 'No device chosen', toCard('participant')),
      chip(!b.connected ? false : eeg.found ? (bad ? null : true) : (cur.predictor_mode === 'eeg' ? false : null),
        !b.connected ? 'Bridge offline' : eeg.found ? `Headset${eeg.simulated ? ' (SIMULATED)' : ''}: ${bad ? bad + ' electrode(s) need attention' : 'all electrodes good'}` : 'No headset stream', toCard('headset')),
      chip(cur.predictor_mode === 'off' || cur.predictor_mode === 'mock' ? null : (eeg.scorer?.model_filter_mismatch ? null : true),
        `Agent: ${predictorName(cur.predictor_mode)}`, () => { location.hash = '#/agent'; }));
  }
  const tick = () => { updateRig(); updateAgent(); };
  const offRig = app.bridge.onChange(tick);
  const rigTimer = setInterval(tick, 1500);     // participant / device edits happen on this page
  tick();

  const rawCard = rawEegCard();
  const lab = [hardwareCard(), storageCard(), rawCard];
  const section = (id, title, hint) => h('div', { class: 'section-title', id }, h('h2', {}, title), hint ? h('span', { class: 'muted small' }, hint) : null);
  const jump = (id, label) => h('button', { type: 'button', class: 'jump', onclick: () => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }, label);
  const startBtn = h('button', { class: 'btn primary', onclick: start }, 'Start session');
  const resetBtn = h('button', { class: 'btn', type: 'button', onclick: resetDefaults, title: 'Experiment design only: participant, headset, storage and Agent settings are kept' }, 'Reset design');
  root.append(subTabs('configure'),
    h('div', { class: 'exp-head' }, h('h1', {}, 'Experiment'),
      h('nav', { class: 'jumps', 'aria-label': 'Sections' }, jump('sec-who', 'Participant & headset'), jump('sec-design', 'Design'),
        jump('sec-questions', 'Questions'), jump('sec-lab', 'Lab & data'))),
    section('sec-who', '1 · Participant & headset', 'check every electrode is green before you start'),
    h('div', { class: 'grid2' }, headset.el, whoCard.el, agentCard),
    section('sec-design', '2 · Design', 'saved as you type; presets hold all of this'),
    h('div', { class: 'grid2' }, presetCard, typeCard, stimCard, designCard, flashCard, seqCard, voiceCard, respCard),
    section('sec-questions', '3 · Questions & image order'),
    h('div', { class: 'grid2' }, planCard),
    section('sec-lab', '4 · Lab & data', 'this computer: display timing and where recordings go'),
    h('div', { class: 'grid2' }, ...lab.map((c) => c.el)),
    h('div', { class: 'startbar' }, startBtn, resetBtn, rig, err));

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
    const before = formState();
    for (const [k, el] of Object.entries(inputs)) {
      const v = DEFAULT_SETTINGS[k];
      if (el.type === 'checkbox') el.checked = !!v; else el.value = v;
    }
    updateModeUi(); updateSourceUi(); refresh(); loadChosen();
    const after = formState();
    record('Reset the experiment settings to defaults', () => restoreForm(before), () => restoreForm(after));
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
      const cats = categoriesOf(stimuli);
      renderPlan();
      stimInfo.replaceChildren(`${stimuli.items.length} items in ${cats.length} categories: `,
        h('span', { class: 'catchips' }, ...cats.map((c) => h('a', { class: 'catchip', href: '#/library', title: stimuli.items.filter((i) => i.category === c).map((i) => i.label || i.id).join(', ') },
          c, h('b', {}, String(stimuli.items.filter((i) => i.category === c).length))))));
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
  return () => { offRig(); clearInterval(rigTimer); headset.dispose(); rawCard.dispose(); };

  async function start() {
    err.textContent = '';
    const cur = read();
    const who = loadParticipant();
    const pid = (who.participant || '').trim();
    const toWho = () => { document.getElementById('participant')?.scrollIntoView({ behavior: 'smooth', block: 'center' }); whoCard.focus(); };
    if (!pid) { err.textContent = 'Enter a participant ID first (section 1).'; toWho(); return; }
    if (!who.device) { err.textContent = 'Choose the recording device first (section 1).'; toWho(); return; }
    if (!stimuli) { err.textContent = 'The stimulus set could not be loaded.'; return; }
    if (cur.off_max_ms < cur.off_min_ms) { err.textContent = 'Gap max must be ≥ gap min.'; return; }
    if (cur.options_per_trial > stimuli.items.length) { err.textContent = 'More options per question than stimulus items.'; return; }
    if (cur.predictor_mode === 'eeg' && !(app.bridge.status().eeg || {}).found) { err.replaceChildren(app.bridge.status().connected ? 'No EEG yet: the bridge switches to its built-in simulated EEG within 5 s of not finding a headset. Press Start again in a moment, or choose Mock on the ' : 'The bridge is not running: start python cortext-lab/bridge/bridge.py, or choose Mock on the ', h('a', { href: '#/agent/output' }, 'Agent tab'), '.'); return; }
    if (cur.trial_type === 'mcq_sequential' && cur.show_model_probs && cur.predictor_mode === 'off') { err.textContent = 'Model probabilities need a predictor: choose Mock or WebSocket, or untick "Show model probability".'; return; }
    if (cur.narrate !== 'off' && cur.predictor_mode === 'off') { err.replaceChildren('The narrator announces the model’s guess, so it needs a predictor: choose one on the ', h('a', { href: '#/agent/output' }, 'Agent tab'), ', or turn the narrator off.'); return; }
    try { await db.getAll('sessions'); } catch (e) { err.textContent = 'Local storage (IndexedDB) is unavailable: ' + e.message; return; }
    saveSettings(cur);
    saveParticipant({ ...who, notes: '' });   // notes belong to one session; the participant and device carry over
    let launched;
    try {
      launched = prepareLaunch({ settings: cur, stimuli, plan, reshuffle: cur.use_plan && cur.shuffle_on_start, participant: pid, device: who.device, notes: who.notes });
    } catch (e) { err.textContent = e.message; return; }
    if (launched.plan && cur.shuffle_on_start) { plan = launched.plan; savePlan(plan); renderPlan(); }
    goRun(app, launched.pending);
  }
}
