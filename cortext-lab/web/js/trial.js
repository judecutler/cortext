// Pure trial-record logic (no DOM): creation, validity bookkeeping, summaries. Unit-tested in tests/logic.test.mjs.
import { shuffle, mean, LETTERS } from './util.js';
import { SCHEMA_VERSION } from './schema.js';
import { flashParamsFor } from './settings.js';

export function makeTrialRecord({ sessionId, spec, trialId, attempt, rng, settings, byId, wallMs = Date.now() }) {
  const fixed = !!spec.fixed_order;   // a preset can fix the order: tile positions and the one-at-a-time sequence follow the list
  const positions = fixed ? spec.option_ids.slice() : shuffle(spec.option_ids, rng);
  const options = positions.map((id, position) => {
    const it = byId[id];
    return {
      id, position, letter: LETTERS[position], category: it.category, label: it.label ?? it.text ?? id,
      text: it.text ?? it.label ?? id, image: it.image ?? null, audio: it.audio ?? null, is_target: id === spec.target_id,
    };
  });
  return {
    key: `${sessionId}:${trialId}`, schema_version: SCHEMA_VERSION,
    session_id: sessionId, trial_id: trialId, trial_index: spec.trial_index, attempt, type: spec.type,
    prompt: spec.prompt, tile_content: spec.tile_content, options, fixed_order: fixed,
    true_answer_id: spec.target_id, true_answer_category: byId[spec.target_id].category,
    clicked_answer_id: null, correct: null, rt_ms: null,
    flash_params: flashParamsFor(spec.type, settings),
    flash_order: [], flash_timing: null, early_clicks: [],
    valid: true, invalid_reason: null, invalid_note: '', validity_log: [],
    status: 'pending', prediction: null,
    model_display: null,   // set when the model's probabilities were shown to the participant BEFORE the click
    notes: '',
    created_wall_ms: wallMs, started_wall_ms: null, ended_wall_ms: null,
  };
}

/** Mark valid/invalid. Never removes anything; appends to validity_log. */
export function applyValidity(trial, { valid, reason = null, note = '', by = 'operator', wallMs = Date.now() }) {
  trial.valid = !!valid;
  trial.invalid_reason = valid ? null : reason || 'unspecified';
  if (note !== undefined && note !== null) trial.invalid_note = note;
  trial.validity_log.push({ at_wall_ms: wallMs, valid: trial.valid, reason: trial.invalid_reason, note: trial.invalid_note, by });
  return trial;
}

export function recordClick(trial, optionId, rtMs) {
  trial.clicked_answer_id = optionId;
  trial.correct = optionId === trial.true_answer_id;
  trial.rt_ms = rtMs;
  return trial;
}

/** Flash counts for VALID trials, per category and per target/non-target. */
export function flashCounts(trials) {
  const out = {};
  for (const t of trials) {
    if (!t.valid) continue;
    const cat = Object.fromEntries(t.options.map((o) => [o.id, o]));
    for (const f of t.flash_order || []) {
      const o = cat[f.option_id];
      if (!o) continue;
      const c = (out[o.category] ||= { target: 0, nontarget: 0 });
      c[o.is_target ? 'target' : 'nontarget']++;
    }
  }
  return out;
}

export function summarizeTrials(trials) {
  const total = trials.length;
  const valid = trials.filter((t) => t.valid);
  const clicked = valid.filter((t) => t.clicked_answer_id != null);
  const correct = clicked.filter((t) => t.correct);
  const byCat = {};
  for (const t of clicked) {
    const c = (byCat[t.true_answer_category] ||= { n: 0, correct: 0 });
    c.n++; if (t.correct) c.correct++;
  }
  const reasons = {};
  trials.filter((t) => !t.valid).forEach((t) => { reasons[t.invalid_reason || 'unspecified'] = (reasons[t.invalid_reason || 'unspecified'] || 0) + 1; });
  const predStat = (isMock) => {
    const ps = clicked.filter((t) => t.prediction && t.prediction.best_option_id != null && (t.prediction.source === 'mock' || !!t.prediction.simulated) === isMock);
    return ps.length ? { n: ps.length, accuracy_vs_truth: ps.filter((t) => t.prediction.best_option_id === t.true_answer_id).length / ps.length } : null;
  };
  return {
    attempts: total, valid: valid.length, invalid: total - valid.length,
    discard_rate: total ? (total - valid.length) / total : null,
    clicked: clicked.length, click_accuracy: clicked.length ? correct.length / clicked.length : null,
    mean_rt_ms: mean(clicked.map((t) => t.rt_ms).filter((x) => x != null)),
    early_clicks: trials.reduce((a, t) => a + (t.early_clicks || []).length, 0),
    by_category: byCat, invalid_reasons: reasons,
    prediction_real: predStat(false), prediction_mock: predStat(true),
  };
}
