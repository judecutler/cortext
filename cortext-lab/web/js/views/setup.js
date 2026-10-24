import { h } from '../util.js';
import { DEVICES } from '../schema.js';
import { loadSettings, saveSettings, estimateDurations } from '../settings.js';
import { loadStimuli, buildTrialSpecs } from '../stimuli.js';
import { newSessionRecord } from '../session.js';
import { listTrialTypes } from '../trialtypes/index.js';
import { db } from '../db.js';

export async function renderSetup(root, app) {
  const s = loadSettings();
  const prev = JSON.parse(localStorage.getItem('cortext-lab.last-participant') || '{}');
  const err = h('div', { class: 'bad' });
  const estimate = h('div', { class: 'muted' });
  const stimInfo = h('div', { class: 'muted' }, 'Loading stimulus config…');
  let stimuli = null;

  const inputs = {};
  const field = (key, label, el, hint) => {
    inputs[key] = el;
    return h('div', { class: 'field' }, h('label', {}, label), el, hint ? h('small', {}, hint) : null);
  };
  const num = (key, label, hint, attrs = {}) => field(key, label, h('input', { type: 'number', value: s[key], ...attrs, oninput: refresh }), hint);
  const sel = (key, label, options, hint) => field(key, label, h('select', { onchange: refresh }, ...options.map(([v, l, dis]) => h('option', { value: v, selected: s[key] === v, disabled: dis }, l))), hint);
  const chk = (key, label, hint) => field(key, label, h('input', { type: 'checkbox', checked: !!s[key], onchange: refresh }), hint);

  const participant = h('input', { type: 'text', value: prev.participant || '', placeholder: 'e.g. P01', autocomplete: 'off', style: { width: '100%' } });
  const device = h('select', {}, ...DEVICES.map((d) => h('option', { value: d, selected: prev.device === d }, d)));
  const notes = h('textarea', { placeholder: 'electrode gel, impedance notes, caffeine, lighting …' });

  const trialTypeOpts = listTrialTypes().map((t) => [t.id, t.label + (t.implemented ? '' : ' (not implemented yet)'), !t.implemented]);

  const form = h('div', { class: 'grid2' },
    h('div', { class: 'card' }, h('h2', {}, 'Session'),
      h('div', { class: 'field' }, h('label', {}, 'Participant ID'), participant),
      h('div', { class: 'field' }, h('label', {}, 'Device used'), device),
      h('div', { class: 'field' }, h('label', {}, 'Notes'), notes),
      sel('trial_type', 'Trial type', trialTypeOpts, 'Other types are stubs (see js/trialtypes/stubs.js).'),
      field('stimuli_url', 'Stimulus config', h('input', { type: 'text', value: s.stimuli_url, style: { width: '100%' }, onchange: reloadStimuli }), 'Path relative to the web folder.'),
      stimInfo),
    h('div', { class: 'card' }, h('h2', {}, 'Flash design'),
      num('n_trials', 'Questions', 'Each target is used once before any repeats.', { min: 1 }),
      num('options_per_trial', 'Options per question', '2–6; the grid adapts.', { min: 2, max: 6 }),
      num('repeats', 'Repeats per option', 'Default 10.', { min: 1 }),
      num('on_ms', 'Flash on (ms)', '≈500. Rounded to display frames.', { min: 50 }),
      num('off_min_ms', 'Gap min (ms)', 'Random gap between flashes.', { min: 0 }),
      num('off_max_ms', 'Gap max (ms)', '100–200 typical.', { min: 0 }),
      num('prompt_ms', 'Prompt before flashing (ms)', 'Time to read the question and find the answer.', { min: 0 }),
      sel('tile_content', 'Tile content', [['image_text', 'Image + text'], ['image', 'Image only'], ['text', 'Text only']]),
      estimate),
    h('div', { class: 'card' }, h('h2', {}, 'Response & cleaning'),
      chk('feedback', 'Show correct / incorrect', 'Shown after the click, never during flashing.'),
      sel('post_mode', 'After each trial', [['manual', 'Wait for operator (Space = next)'], ['auto', 'Auto-advance']], 'Discard / Redo stay available while waiting.'),
      num('auto_advance_ms', 'Auto-advance after (ms)', 'Only for auto mode; Discard cancels it.', { min: 500 }),
      num('response_timeout_ms', 'Response timeout (ms)', '0 = wait forever. A timeout marks the trial invalid (no_response).', { min: 0 }),
      chk('log_flash_off', 'Also send flash_off markers')),
    h('div', { class: 'card' }, h('h2', {}, 'Hardware & prediction'),
      chk('photodiode', 'Photodiode square', 'White square in a screen corner during every flash (for hardware timing checks).'),
      sel('photodiode_corner', 'Photodiode corner', [['bottom-left', 'Bottom left'], ['bottom-right', 'Bottom right'], ['top-left', 'Top left'], ['top-right', 'Top right']]),
      sel('predictor_mode', 'Predictor', [['mock', 'Mock (simulated; demo only)'], ['ws', 'WebSocket model server'], ['off', 'Off']], 'Mock uses the ground truth, so it is NOT evidence. Marked as such everywhere.'),
      field('predictor_url', 'Predictor URL', h('input', { type: 'text', value: s.predictor_url, style: { width: '100%' } })),
      chk('show_bars', 'Live score bars under tiles', 'Demo use only: it adds visible changes during flashing. Toggle with Shift+B.')));

  const startBtn = h('button', { class: 'btn primary', onclick: start }, 'Start session');
  root.append(h('h1', {}, 'New session'), form, err,
    h('div', {}, startBtn, ' ', h('a', { class: 'btn', href: 'timing.html', target: '_blank', rel: 'noopener' }, 'Run timing test ↗')));

  function read() {
    const out = { ...s };
    for (const [k, el] of Object.entries(inputs)) {
      out[k] = el.type === 'checkbox' ? el.checked : el.type === 'number' ? Number(el.value) : el.value;
    }
    return out;
  }
  function refresh() {
    const cur = read();
    const d = estimateDurations(cur);
    estimate.textContent = `${d.flashes} flashes per question ≈ ${d.flashSec.toFixed(0)} s of flashing; about ${d.trialSec.toFixed(0)} s per question and ${d.totalMin.toFixed(1)} min in total.`;
  }
  async function reloadStimuli() {
    try {
      stimuli = await loadStimuli(inputs.stimuli_url.value);
      const cats = [...new Set(stimuli.items.map((i) => i.category))];
      stimInfo.innerHTML = '';
      stimInfo.append(`${stimuli.items.length} items in ${cats.length} categories (${cats.join(', ')}).`);
      if (stimuli.warnings.length) stimInfo.append(h('div', { class: 'warn' }, 'Warnings: ' + stimuli.warnings.join('; ')));
    } catch (e) { stimuli = null; stimInfo.innerHTML = ''; stimInfo.append(h('span', { class: 'bad' }, String(e.message))); }
  }
  refresh();
  await reloadStimuli();

  async function start() {
    err.textContent = '';
    const cur = read();
    const pid = participant.value.trim();
    if (!pid) { err.textContent = 'Enter a participant ID.'; return; }
    if (!stimuli) { err.textContent = 'The stimulus config could not be loaded.'; return; }
    if (cur.off_max_ms < cur.off_min_ms) { err.textContent = 'Gap max must be ≥ gap min.'; return; }
    if (cur.options_per_trial > stimuli.items.length) { err.textContent = 'More options per question than stimulus items.'; return; }
    try { await db.getAll('sessions'); } catch (e) { err.textContent = 'Local storage (IndexedDB) is unavailable: ' + e.message; return; }
    saveSettings(cur);
    localStorage.setItem('cortext-lab.last-participant', JSON.stringify({ participant: pid, device: device.value }));
    cur.seed = (Math.random() * 2 ** 31) >>> 0;
    let specs;
    try {
      specs = buildTrialSpecs(stimuli, { n: cur.n_trials, seed: cur.seed, optionsPerTrial: cur.options_per_trial, tileContent: cur.tile_content, trialType: cur.trial_type });
    } catch (e) { err.textContent = e.message; return; }
    const session = newSessionRecord({ participant_id: pid, device: device.value, notes: notes.value, settings: cur, stimuli, plannedTrials: specs.length });
    app.pending = { session, settings: cur, stimuli, specs };
    location.hash = '#/run';
  }
}
