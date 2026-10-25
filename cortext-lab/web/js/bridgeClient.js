// WebSocket client for bridge/bridge.py: reconnects, estimates the page->bridge clock offset, forwards markers.

export class BridgeClient {
  constructor(url) {
    this.url = url || defaultUrl();
    this.ws = null;
    this.connected = false;
    this.info = {};          // last 'stats' message from the bridge
    this.offset_ms = null;   // bridge_time - page_time
    this.rtt_ms = null;
    this.sentCount = 0;
    this.quiet = false;      // true while flashing: skip re-sync traffic
    this.listeners = new Set();
    this._syncSeq = 0;
    this._pending = new Map();
    this._hello = null;
    this._timer = null;
  }

  onChange(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  _emit() { this.listeners.forEach((fn) => { try { fn(this.status()); } catch (e) { console.error(e); } }); }

  status() {
    return { connected: this.connected, url: this.url, offset_ms: this.offset_ms, rtt_ms: this.rtt_ms, sent: this.sentCount, ...this.info };
  }

  start() {
    this._connect();
    this._timer = setInterval(() => {
      if (this.connected && !this.quiet) this.sync(6).catch(() => {});
    }, 30000);
  }

  _connect() {
    let ws;
    try { ws = new WebSocket(this.url); } catch { return setTimeout(() => this._connect(), 2000); }
    this.ws = ws;
    ws.onopen = async () => {
      this.connected = true;
      this._emit();
      if (this._hello) this._send(this._hello);
      else this._send({ type: 'hello' });
      try { await this.sync(16); } catch { /* ignore */ }
    };
    ws.onmessage = (ev) => {
      let m; try { m = JSON.parse(ev.data); } catch { return; }
      if (m.type === 'sync_reply') {
        const p = this._pending.get(m.id);
        if (p) { this._pending.delete(m.id); p(m); }
      } else if (m.type === 'stats') { this.info = { ...this.info, ...m }; this._emit(); }
      else if (m.type === 'hello_ack') { this.info = { ...this.info, ...m }; this._emit(); }
    };
    ws.onclose = () => {
      this.connected = false;
      this.info = {};
      this._emit();
      setTimeout(() => this._connect(), 2000);
    };
    ws.onerror = () => { try { ws.close(); } catch { /* ignore */ } };
  }

  _send(obj) {
    if (this.ws && this.ws.readyState === 1) { this.ws.send(JSON.stringify(obj)); return true; }
    return false;
  }

  /** Tell the bridge which session is starting (opens a fresh CSV backup on the bridge side). */
  hello(sessionInfo) {
    this._hello = { type: 'hello', ...sessionInfo };
    this._send(this._hello);
  }

  /** Forget the session announced with hello() (e.g. after the timing test) so a reconnect does not re-announce it. */
  clearHello() { this._hello = null; }

  /** n ping-pong rounds; keep the min-RTT sample (NTP-style) and tell the bridge the offset. */
  async sync(n = 10) {
    let best = null;
    for (let k = 0; k < n; k++) {
      const id = ++this._syncSeq;
      const t0 = performance.now();
      const reply = await new Promise((resolve, reject) => {
        const to = setTimeout(() => { this._pending.delete(id); reject(new Error('sync timeout')); }, 1000);
        this._pending.set(id, (m) => { clearTimeout(to); resolve(m); });
        if (!this._send({ type: 'sync', id, t_c: t0 })) { clearTimeout(to); reject(new Error('not connected')); }
      });
      const t1 = performance.now();
      const rtt = t1 - t0;
      if (!best || rtt < best.rtt) best = { rtt, offset: reply.t_b - (t0 + t1) / 2 };
      await new Promise((r) => setTimeout(r, 20));
    }
    if (best) {
      this.offset_ms = best.offset;
      this.rtt_ms = best.rtt;
      this._send({ type: 'sync_result', offset_ms: best.offset, rtt_ms: best.rtt });
      this._emit();
    }
    return best;
  }

  sendMarker(rec) {
    const ok = this._send({ type: 'marker', ...rec });
    if (ok) this.sentCount++;
    return ok;
  }
}

function defaultUrl() {
  const q = new URLSearchParams(location.search).get('bridge');
  if (q) return q;
  if (location.protocol === 'http:' || location.protocol === 'https:') {
    return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
  }
  return 'ws://localhost:8000/ws';
}
