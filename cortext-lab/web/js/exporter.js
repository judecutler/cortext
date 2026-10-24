// Export: trial table, marker table, session JSON, and a tiny store-only ZIP writer (no dependencies).
import { db } from './db.js';
import { toCSV, downloadBlob } from './util.js';
import { MARKER_COLUMNS, TRIAL_COLUMNS, FEEDBACK_COLUMNS, SCHEMA_VERSION } from './schema.js';

export function trialRow(t) {
  const clicked = (t.options || []).find((o) => o.id === t.clicked_answer_id);
  const pred = t.prediction || {};
  const by = (t.options || []).slice().sort((a, b) => a.position - b.position).map((o) => o.id).join('|');
  return {
    session_id: t.session_id, trial_id: t.trial_id, trial_index: t.trial_index, attempt: t.attempt, type: t.type,
    status: t.status, valid: t.valid, invalid_reason: t.invalid_reason || '', invalid_note: t.invalid_note || '',
    prompt: t.prompt, tile_content: t.tile_content, option_ids_by_position: by,
    true_answer_id: t.true_answer_id, true_answer_category: t.true_answer_category,
    clicked_answer_id: t.clicked_answer_id ?? '', clicked_category: clicked ? clicked.category : '',
    correct: t.correct ?? '', rt_ms: t.rt_ms ?? '', early_clicks: (t.early_clicks || []).length,
    n_flashes: (t.flash_order || []).length, flash_order: (t.flash_order || []).map((f) => f.option_id).join('|'),
    flash_on_ms: t.flash_params?.on_ms, flash_off_min_ms: t.flash_params?.off_min_ms, flash_off_max_ms: t.flash_params?.off_max_ms,
    repeats: t.flash_params?.repeats,
    prediction_source: pred.source ?? '', prediction_best_option_id: pred.best_option_id ?? '', prediction_confidence: pred.confidence ?? '',
    model_probabilities: pred.probabilities ?? '', model_shown_before_click: !!t.model_display,
    started_wall_ms: t.started_wall_ms ?? '', ended_wall_ms: t.ended_wall_ms ?? '', notes: t.notes || '',
  };
}

/** Per (trial, option) rows pairing the model's probability with the true answer and the participant's click. */
export function feedbackRows(trials) {
  const rows = [];
  for (const t of trials) {
    const p = t.prediction;
    if (!p || !p.probabilities) continue;
    const top = Object.entries(p.probabilities).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '';
    for (const o of t.options) {
      rows.push({
        session_id: t.session_id, trial_id: t.trial_id, trial_index: t.trial_index, valid: t.valid, invalid_reason: t.invalid_reason || '',
        presentation_mode: t.type, option_id: o.id, option_position: o.position, category: o.category,
        model_probability: p.probabilities[o.id] ?? '', model_score: p.p300 ? (p.p300[o.id] ?? '') : '', model_source: p.source,
        model_probability_calibrated: !!p.probabilities_calibrated,
        is_true_answer: o.id === t.true_answer_id, is_clicked: o.id === t.clicked_answer_id, is_model_top_choice: o.id === top,
        true_answer_id: t.true_answer_id, clicked_answer_id: t.clicked_answer_id ?? '', model_top_option_id: top,
        human_agrees_with_model: t.clicked_answer_id == null ? '' : t.clicked_answer_id === top,
        model_shown_before_click: !!t.model_display,
      });
    }
  }
  return rows;
}
export const feedbackCsv = (trials) => toCSV(feedbackRows(trials), FEEDBACK_COLUMNS);

export const trialsCsv = (trials) => toCSV(trials.map(trialRow), TRIAL_COLUMNS);
export const markersCsv = (markers) => toCSV(markers, MARKER_COLUMNS);

export async function loadSession(sessionId) {
  const [session, trials, markers] = await Promise.all([
    db.get('sessions', sessionId), db.getAllBySession('trials', sessionId), db.getAllBySession('markers', sessionId),
  ]);
  trials.sort((a, b) => a.trial_id - b.trial_id);
  markers.sort((a, b) => a.seq - b.seq);
  return { session, trials, markers };
}

export async function sessionFiles(sessionId, prefix = '') {
  const { session, trials, markers } = await loadSession(sessionId);
  const bundle = { schema_version: SCHEMA_VERSION, exported_at: new Date().toISOString(), session, trials, markers_count: markers.length };
  return [
    { name: `${prefix}session.json`, data: JSON.stringify(bundle, null, 2) },
    { name: `${prefix}trials.csv`, data: trialsCsv(trials) },
    { name: `${prefix}markers.csv`, data: markersCsv(markers) },
    { name: `${prefix}model_feedback.csv`, data: feedbackCsv(trials) },
  ];
}

export async function exportSessionZip(sessionId) {
  downloadBlob(makeZip(await sessionFiles(sessionId)), `cortext_${sessionId}.zip`);
}
export async function exportAllZip() {
  const sessions = await db.getAll('sessions');
  const files = [];
  for (const s of sessions) files.push(...(await sessionFiles(s.session_id, `${s.session_id}/`)));
  downloadBlob(makeZip(files), `cortext_all_sessions_${new Date().toISOString().slice(0, 10)}.zip`);
}
export async function exportSingleFile(sessionId, which) {
  const files = await sessionFiles(sessionId);
  const f = files.find((x) => x.name === which);
  downloadBlob(new Blob([f.data], { type: 'text/plain;charset=utf-8' }), `${sessionId}_${which}`);
}

// ---- minimal ZIP (store, no compression) ----
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
export function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
export function makeZip(files) {
  const enc = new TextEncoder();
  const parts = [], central = [];
  let offset = 0;
  const now = new Date();
  const dosTime = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
  const dosDate = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  for (const f of files) {
    const name = enc.encode(f.name);
    const data = typeof f.data === 'string' ? enc.encode(f.data) : f.data;
    const crc = crc32(data);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true); local.setUint16(4, 20, true); local.setUint16(6, 0x0800, true);
    local.setUint16(8, 0, true); local.setUint16(10, dosTime, true); local.setUint16(12, dosDate, true);
    local.setUint32(14, crc, true); local.setUint32(18, data.length, true); local.setUint32(22, data.length, true);
    local.setUint16(26, name.length, true); local.setUint16(28, 0, true);
    parts.push(new Uint8Array(local.buffer), name, data);
    const cen = new DataView(new ArrayBuffer(46));
    cen.setUint32(0, 0x02014b50, true); cen.setUint16(4, 20, true); cen.setUint16(6, 20, true); cen.setUint16(8, 0x0800, true);
    cen.setUint16(10, 0, true); cen.setUint16(12, dosTime, true); cen.setUint16(14, dosDate, true);
    cen.setUint32(16, crc, true); cen.setUint32(20, data.length, true); cen.setUint32(24, data.length, true);
    cen.setUint16(28, name.length, true); cen.setUint32(42, offset, true);
    central.push(new Uint8Array(cen.buffer), name);
    offset += 30 + name.length + data.length;
  }
  const cenSize = central.reduce((a, b) => a + b.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
  end.setUint32(12, cenSize, true); end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, new Uint8Array(end.buffer)], { type: 'application/zip' });
}
