// MarkerBus: one place where every marker is stamped, kept in memory, queued for IndexedDB and sent to the bridge.
import { db, markerKey } from './db.js';

export class MarkerBus {
  constructor(bridge) {
    this.bridge = bridge;
    this.sessionId = null;
    this.seq = 0;
    this.pending = [];
    this.recent = [];          // last 300 markers for the dev log
    this.listeners = new Set();
    this.failed = 0;
    this._timer = setInterval(() => this.flush(), 250);
    const flushNow = () => this.flush();
    window.addEventListener('pagehide', flushNow);
    document.addEventListener('visibilitychange', () => { if (document.hidden) flushNow(); });
  }

  setSession(sessionId) { this.sessionId = sessionId; this.seq = 0; this.recent = []; }
  onMarker(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }

  /**
   * Emit a marker. Call this from inside the requestAnimationFrame callback that draws the change, passing that
   * callback's timestamp as frameMs. t_ms is performance.now() at the moment of the call.
   */
  emit(event, fields = {}, frameMs = null) {
    const t = performance.now();
    const rec = {
      seq: this.seq++, session_id: this.sessionId, event,
      t_ms: t, frame_ms: frameMs, wall_ms: performance.timeOrigin + t,
      trial_id: null, trial_index: null, option_id: null, option_position: null, category: null,
      is_target: null, repeat_index: null, flash_index: null, true_answer_id: null, extra: null,
      ...fields,
    };
    rec.bridge_sent = this.bridge ? this.bridge.sendMarker(rec) : false;
    this.pending.push({ key: markerKey(rec.session_id, rec.seq), ...rec });
    this.recent.push(rec);
    if (this.recent.length > 300) this.recent.shift();
    this.listeners.forEach((fn) => fn(rec));
    return rec;
  }

  async flush() {
    if (!this.pending.length) return;
    const batch = this.pending;
    this.pending = [];
    try { await db.bulkPut('markers', batch); }
    catch (e) { this.failed += batch.length; this.pending = batch.concat(this.pending); console.error('marker flush failed', e); }
  }
}
