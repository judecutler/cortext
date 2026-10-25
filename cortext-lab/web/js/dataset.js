// Talks to the bridge's dataset pipeline (pipeline/build_session.py): build the clean, shareable copy of a session
// in datasets/cortext-p300/ and optionally git-push it so teammates can pull it and train.
import { loadSession } from './exporter.js';

async function post(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return r.json();
}

/** Strip blob URLs and other bulky/irrelevant fields before sending the trial table. */
function slimTrial(t) {
  const { options, flash_order, ...rest } = t;
  return {
    ...rest,
    options: (options || []).map(({ id, category, position, label, is_target }) => ({ id, category, position, label, is_target })),
    option_ids_by_position: (options || []).slice().sort((a, b) => a.position - b.position).map((o) => o.id).join('|'),
    prediction_best_option_id: t.prediction?.best_option_id ?? '', prediction_source: t.prediction?.source ?? '',
    model_shown_before_click: !!t.model_display,
  };
}

async function payload(sessionId) {
  const { session, trials } = await loadSession(sessionId);
  const { stimuli_snapshot, ...s } = session || {};
  return { session_id: sessionId, session: { ...s, stimuli_items: stimuli_snapshot?.items?.length }, trials: trials.map(slimTrial) };
}

export async function buildDataset(sessionId) { return post('/api/dataset/build', await payload(sessionId)); }
export async function pushDataset(sessionId) { return post('/api/dataset/push', await payload(sessionId)); }
export async function datasetStatus(sessionId) {
  const r = await fetch(`/api/dataset?session_id=${encodeURIComponent(sessionId)}`, { cache: 'no-store' });
  return r.json();
}
