// Talks to the bridge: build the clean dataset entry of a session (pipeline/build_session.py), upload it to the
// Hugging Face dataset repo, list / delete / label what the bridge holds, and manage storage.
import { loadSession } from './exporter.js';

async function post(path, body = {}) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return r.json();
}
async function get(path) {
  const r = await fetch(path, { cache: 'no-store' });
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
    prediction_sim_p300: t.prediction?.sim_p300 ?? '',   // decoder input: off | overlay | replace (simulated P300)
    model_shown_before_click: !!t.model_display,
  };
}

/** Decoder input: 'off' (headset EEG) | 'overlay' (headset + simulated P300) | 'replace' (simulated P300 on
 * simulated background EEG; the headset still drives the visuals). uv = simulated P300 amplitude. */
export async function setSimP300({ mode, uv } = {}) { return post('/api/eeg/sim', { mode, uv }); }

async function payload(sessionId) {
  const { session, trials } = await loadSession(sessionId);
  const { stimuli_snapshot, ...s } = session || {};
  return { session_id: sessionId, session: { ...s, stimuli_items: stimuli_snapshot?.items?.length }, trials: trials.map(slimTrial) };
}

export async function buildDataset(sessionId) { return post('/api/dataset/build', await payload(sessionId)); }
/** Build (if needed) and upload to Hugging Face. */
export async function pushDataset(sessionId) { return post('/api/dataset/push', await payload(sessionId)); }
export async function datasetStatus(sessionId) { return get(`/api/dataset?session_id=${encodeURIComponent(sessionId)}`); }

// ---- what the bridge machine holds (dataset folders + raw logs), deletion, buckets, storage ----
export const bridgeSessions = () => get('/api/sessions');
export const deleteOnBridge = (sessionId, { remote = false, purge_remote = true } = {}) => post('/api/session/delete', { session_id: sessionId, remote, purge_remote });
export const setBucket = (sessionId, bucket) => post('/api/session/bucket', { session_id: sessionId, bucket });
export const storageInfo = (check = false) => get(`/api/storage${check ? '?check=1' : ''}`);
export const uploadPending = () => post('/api/storage/push-pending');
export const downloadFromHub = (profile = 'epochs') => post('/api/storage/pull', { profile });
export const createHubRepo = () => post('/api/storage/init');

// ---- raw EEG (raw_eeg/ on Hugging Face): free recordings with no stimuli, or files recorded with another program ----
export const rawEegInfo = () => get('/api/storage/raw');
export const startRawRecording = (participant, note) => post('/api/storage/raw/start', { participant, note });
export const stopRawRecording = (upload = true) => post('/api/storage/raw/stop', { upload });
export const pushRawFile = (name) => post('/api/storage/raw/push', { name });
/** Upload a file picked in the browser, untouched. */
export async function uploadRawFile(file, { note = '', participant = '' } = {}) {
  const q = new URLSearchParams({ name: file.name, note, participant });
  const r = await fetch(`/api/storage/raw-upload?${q}`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file });
  try { return await r.json(); } catch { return { ok: false, error: `HTTP ${r.status}` }; }
}
