// Timing tab: flashes a square at fixed intervals through the same scheduler + marker path as the experiment and
// reports how closely the browser kept to the schedule. (This used to be the separate timing.html page.)
import { runFlashPlan } from '../flasher.js';
import { h, fmt, mean, stdev, percentile, toCSV, downloadBlob } from '../util.js';

const STYLE = `
    #flashbox { width: min(46vh, 90%); aspect-ratio: 1; margin: 10px auto; background: #1c1c1f; border: 4px solid #3a3a40; border-radius: 18px; }
    #flashbox.flash { background: #fff; border-color: #fff; }
    .big { font-size: 22px; font-weight: 700; }
  `;

export function renderTiming(root, app) {
  const bridge = app.bridge;
  root.innerHTML = `<style>${STYLE}</style><h1>Timing test</h1><div id="conn" class="muted mono" style="margin-bottom:8px"></div><div class="grid2">
      <div class="card">
        <h2>Run</h2>
        <p class="muted">Flashes the square (and the optional photodiode corner) at fixed intervals using the same scheduler as the experiment, sends a <code>timing_flash</code> marker on every flash, then reports how closely the browser kept to the schedule. Use the same browser window size, display, and fullscreen state as a real session, with other tabs closed.</p>
        <div class="field"><label>Interval (ms)</label><input id="interval" type="number" value="1000"></div>
        <div class="field"><label>Flash duration (ms)</label><input id="ondur" type="number" value="500"></div>
        <div class="field"><label>Number of flashes</label><input id="count" type="number" value="60"></div>
        <div class="field"><label>Photodiode square</label><input id="pd" type="checkbox" checked></div>
        <div class="field"><label>Corner</label><select id="corner"><option>bottom-left</option><option>bottom-right</option><option>top-left</option><option>top-right</option></select></div>
        <button class="btn primary" id="go">Start test</button>
        <span id="state" class="muted"></span>
      </div>
      <div class="card"><div id="flashbox"></div></div>
    </div>
    <div class="card" id="results"><h2>Results</h2><p class="muted">Run the test to see results.</p></div>
    <div class="card">
      <h2>Measuring the real screen lag (hardware)</h2>
      <ol>
        <li>Tape a photodiode (or a phone camera in slow motion) over the corner square; feed the diode into an EEG aux/trigger channel or a second LSL stream.</li>
        <li>Run this test with the bridge running and your recorder (LabRecorder) saving both the <code>CorTextMarkers</code> stream and the photodiode channel.</li>
        <li>Extract photodiode onsets (threshold crossing) to a CSV with an <code>lsl_timestamp</code> column, then run <code>python cortext-lab/tools/analyze_timing.py cortext-lab/bridge/logs/markers_….csv --event timing_flash --photodiode pd_onsets.csv</code>.</li>
        <li>The mean marker→light lag is the constant offset to subtract; its SD is the jitter you cannot remove. See <code>cortext-lab/TIMING_CHECKLIST.md</code> for pass/fail thresholds.</li>
      </ol>
    </div>`;
  const pdsq = document.createElement('div');
  pdsq.id = 'pdsq'; pdsq.className = 'pd bottom-left enabled';
  document.body.append(pdsq);
  const $ = (id) => id === 'pdsq' ? pdsq : root.querySelector('#' + id);
  const showConn = (s) => {
    $('conn').innerHTML = `<span class="dot ${s.connected ? 'green' : 'red'}"></span>bridge ${s.connected ? 'connected' : 'offline'} · LSL ${s.lsl_available ? 'on' : 'off'} · offset ${fmt(s.offset_ms, 1)} ms · RTT ${fmt(s.rtt_ms, 2)} ms`;
  };
  const off = bridge.onChange(showConn);
  showConn(bridge.status());

  const SESSION = 'timing-test';
    let lastRows = [];
  
  $('pd').addEventListener('change', () => $('pdsq').classList.toggle('enabled', $('pd').checked));
    $('corner').addEventListener('change', () => { $('pdsq').className = `pd ${$('corner').value}${$('pd').checked ? ' enabled' : ''}`; });
  
  const onGo = async () => {
    const interval = Number($('interval').value), onMs = Number($('ondur').value), n = Number($('count').value);
    if (onMs >= interval) { $('state').textContent = 'Flash duration must be shorter than the interval.'; return; }
    $('go').disabled = true; $('state').textContent = 'running…';
    bridge.hello({ session_id: SESSION, participant_id: 'timing', device: 'timing-test' });
    await new Promise((r) => setTimeout(r, 200));
    try { await bridge.sync(16); } catch { /* bridge may be offline */ }
    bridge.quiet = true;
    const events = Array.from({ length: n }, (_, i) => ({ flash_index: i, option_id: 'square', repeat_index: i, on_at: i * interval, off_at: i * interval + onMs }));
    const plan = { events, total_ms: events[n - 1].off_at };
    const box = $('flashbox'), pd = $('pdsq');
    const sentBefore = bridge.sentCount;
    let seq = 0;
    const res = await runFlashPlan(plan, {
      onOn: (ev, ts, r) => {
        box.classList.add('flash'); pd.classList.add('on');
        const t = performance.now();
        r.t_on_ms = t;
        bridge.sendMarker({ seq: seq++, session_id: SESSION, event: 'timing_flash', t_ms: t, frame_ms: ts, wall_ms: performance.timeOrigin + t,
          flash_index: ev.flash_index, extra: { intended_ms: ev.on_at, interval, on_ms: onMs } });
      },
      onOff: () => { box.classList.remove('flash'); pd.classList.remove('on'); },
    });
    bridge.quiet = false;
    const sent = bridge.sentCount - sentBefore;
    await new Promise((r) => setTimeout(r, 1500)); // let the bridge's stats message arrive
    const onset = res.results.map((r) => r.onset_err_ms);
    const dur = res.results.map((r) => r.dur_err_ms).filter((x) => x != null);
    const ioi = res.results.slice(1).map((r, i) => r.frame_on_ms - res.results[i].frame_on_ms - interval);
    const f = res.frame;
    const stat = (name, xs, note = '') => h('tr', {}, h('td', {}, name), h('td', {}, fmt(mean(xs), 2)), h('td', {}, fmt(stdev(xs), 2)), h('td', {}, fmt(percentile(xs.map(Math.abs), 95), 2)), h('td', {}, fmt(Math.max(...xs.map(Math.abs)), 2)), h('td', { class: 'muted' }, note));
    const b = bridge.status();
    const verdict = (ok, text) => h('span', { class: ok ? 'good' : 'bad' }, (ok ? 'PASS ' : 'CHECK ') + text);
    const frameMs = f.median_ms;
    $('results').replaceChildren(
      h('h2', {}, 'Results'),
      h('p', {}, `Display refresh ≈ ${fmt(1000 / frameMs, 1)} Hz (median frame ${fmt(frameMs, 2)} ms, p95 ${fmt(f.p95_ms, 2)} ms, max ${fmt(f.max_ms, 2)} ms). Dropped/long frames: `, h('b', { class: f.dropped ? 'warn' : 'good' }, f.dropped), ` of ${f.n}.`),
      h('table', {}, h('thead', {}, h('tr', {}, ...['measure (ms)', 'mean', 'SD', '|p95|', '|max|', 'meaning'].map((c) => h('th', {}, c)))),
        h('tbody', {}, stat('Onset error (frame time − scheduled)', onset, 'quantisation to frames; should stay within ±1 frame'),
          stat('Duration error', dur, 'whole frames; flash lasts ≥ 1 frame'),
          stat('Onset-to-onset interval error', ioi, 'drift/jitter between consecutive flashes'))),
      h('p', {}, 'Verdicts: ',
        verdict(Math.max(...onset.map(Math.abs)) <= frameMs * 1.5, `max onset error ≤ 1.5 frames (${fmt(frameMs * 1.5, 1)} ms)`), ' · ',
        verdict(f.dropped === 0, 'no dropped frames'), ' · ',
        verdict(sent === n, `markers sent ${sent}/${n}`), ' · ',
        verdict(b.markers_received != null && b.markers_received >= n, `bridge received ${b.markers_received ?? '–'}`)),
      h('p', { class: 'muted' }, `Bridge: clock offset ${fmt(b.offset_ms, 2)} ms, sync RTT ${fmt(b.rtt_ms, 2)} ms (transit error of the page→bridge timestamp is bounded by ~RTT/2). CSV backup: ${b.csv_path || 'n/a'}. This page cannot see the physical screen: use the photodiode procedure below for display latency.`),
      h('button', { class: 'btn', onclick: () => downloadBlob(new Blob([toCSV(lastRows, ['flash_index', 'scheduled_ms', 'frame_on_ms', 't_on_ms', 'onset_err_ms', 'dur_ms'])], { type: 'text/csv' }), 'timing_test.csv') }, 'Download per-flash CSV'));
    lastRows = res.results.map((r) => ({ flash_index: r.flash_index, scheduled_ms: r.on_at, frame_on_ms: r.frame_on_ms, t_on_ms: r.t_on_ms, onset_err_ms: r.onset_err_ms, dur_ms: r.dur_ms }));
    $('go').disabled = false; $('state').textContent = 'done';
  };
    $('go').addEventListener('click', onGo);
  
  return () => { off(); bridge.quiet = false; bridge.clearHello(); pdsq.remove(); };
}
