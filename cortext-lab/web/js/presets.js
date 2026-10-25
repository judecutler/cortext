// Experiment presets: a named bundle of experiment settings plus (optionally) a custom question list.
// Stored in this browser (localStorage); can be exported to / imported from a small .json file to share with the team.
import { downloadBlob } from './util.js';

const KEY = 'cortext-lab.presets.v1';
export const PRESET_VERSION = 1;

/** Settings that belong to the experiment (the rig settings on the Setup tab are not part of a preset). */
export const PRESET_KEYS = ['trial_type', 'stimuli_source', 'stimuli_url', 'n_trials', 'options_per_trial', 'tile_content', 'prompt_ms',
  'repeats', 'on_ms', 'off_min_ms', 'off_max_ms', 'seq_on_ms', 'seq_gap_ms', 'seq_repeats', 'show_model_probs', 'send_feedback',
  'feedback', 'post_mode', 'auto_advance_ms', 'response_timeout_ms', 'log_flash_off', 'show_bars', 'use_plan', 'order_mode', 'shuffle_on_start'];

export const BUILTIN_PRESETS = {
  'Quick test (3 questions, fast)': {
    description: 'Three short flash trials, to check that everything works.',
    settings: { trial_type: 'mcq_flash', n_trials: 3, options_per_trial: 4, repeats: 2, on_ms: 200, off_min_ms: 100, off_max_ms: 150, prompt_ms: 400, use_plan: false },
    plan: [],
  },
  'Demo: one at a time (5 questions)': {
    description: 'Question stays up, each picture appears alone for 1 s, three rounds, then the choice grid with model probabilities.',
    settings: { trial_type: 'mcq_sequential', n_trials: 5, options_per_trial: 4, seq_on_ms: 1000, seq_gap_ms: 300, seq_repeats: 3, prompt_ms: 3000, show_model_probs: true, use_plan: false },
    plan: [],
  },
};

function readAll() { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch { return {}; } }
function writeAll(all) { localStorage.setItem(KEY, JSON.stringify(all)); }

export const isBuiltin = (name) => name in BUILTIN_PRESETS;
export function listPresets() {
  const mine = readAll();
  return { builtin: Object.keys(BUILTIN_PRESETS), mine: Object.keys(mine).sort((a, b) => a.localeCompare(b)) };
}
export function getPreset(name) { return BUILTIN_PRESETS[name] || readAll()[name] || null; }

export function savePreset(name, { settings, plan }) {
  name = String(name || '').trim();
  if (!name) throw new Error('Give the preset a name.');
  if (isBuiltin(name)) throw new Error(`“${name}” is a built-in preset; choose another name.`);
  const all = readAll();
  all[name] = {
    version: PRESET_VERSION, saved_iso: new Date().toISOString(),
    settings: Object.fromEntries(PRESET_KEYS.filter((k) => k in settings).map((k) => [k, settings[k]])),
    plan: (plan || []).map((q) => ({ prompt: q.prompt, target: q.target, options: q.options.slice() })),
  };
  writeAll(all);
  return name;
}
export function deletePreset(name) {
  if (isBuiltin(name)) throw new Error('Built-in presets cannot be deleted.');
  const all = readAll(); delete all[name]; writeAll(all);
}

export function exportPreset(name) {
  const p = getPreset(name);
  if (!p) throw new Error('No such preset.');
  downloadBlob(new Blob([JSON.stringify({ cortext_preset: PRESET_VERSION, name, ...p }, null, 2)], { type: 'application/json' }), `${name.replace(/[^\w.-]+/g, '_')}.preset.json`);
}

/** Read a preset file; returns the saved name. Never overwrites a built-in; a clash with your own preset gets “ (2)”. */
export async function importPresetFile(file) {
  let data;
  try { data = JSON.parse(await file.text()); } catch { throw new Error('That file is not valid JSON.'); }
  if (!data || !data.cortext_preset || typeof data.settings !== 'object') throw new Error('That file is not a CorText preset.');
  const all = readAll();
  let name = String(data.name || file.name.replace(/\.preset\.json$|\.json$/i, '')).trim() || 'Imported preset';
  if (isBuiltin(name) || name in all) { let n = 2; while (`${name} (${n})` in all) n++; name = `${name} (${n})`; }
  const plan = Array.isArray(data.plan) ? data.plan.filter((q) => q && q.target && Array.isArray(q.options)) : [];
  savePreset(name, { settings: data.settings, plan });
  return name;
}
