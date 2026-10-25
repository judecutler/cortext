// Neural monitor: full-screen live view of the headset for the audience (open it on a second screen / window).
// Never shown to the participant during flashing: moving visuals in their field of view would add their own brain
// responses to the recording.
import { h, fmt } from '../util.js';
import { HeadMap, Traces, ErpPlot, EEGFeed, evidenceBars, mapValues, regionBands, BAND_INFO } from '../brainviz.js';

export function renderBrain(root, app) {
  let mode = localStorage.getItem('cortext-lab.mapmode') || 'alpha';
  const status = h('span', { class: 'live-pill' }, 'connecting…');
  const meta = h('div', { class: 'muted mono' });
  const tabs = h('div', { class: 'tabs' }, ...Object.entries(BAND_INFO).map(([k, v]) =>
    h('button', { class: 'tab' + (k === mode ? ' on' : ''), dataset: { mode: k }, onclick: () => setMode(k) }, v.label)));
  const mapCanvas = h('canvas', { class: 'headmap-lg' });
  const legend = h('div', { class: 'legend' });
  const hint = h('div', { class: 'muted small' });
  const traceCanvas = h('canvas', { class: 'traces-lg' });
  const erpCanvas = h('canvas', { class: 'erp' });
  const erpSel = h('select', { onchange: () => draw() }, ...['Pz', 'POz', 'Cz', 'PO7', 'PO8', 'Fz', 'C3', 'C4'].map((l) => h('option', { value: l }, l)));
  const regions = h('div', { class: 'regions' });
  const evidence = h('div', { class: 'evidence' });
  const quality = h('div', { class: 'qchips' });

  root.append(h('div', { class: 'monitor' },
    h('div', { class: 'mon-head' },
      h('div', {}, h('div', { class: 'eyebrow' }, 'CorText · live'), h('h1', { class: 'glow' }, 'Neural activity monitor')),
      h('div', { class: 'mon-status' }, status, meta)),
    h('div', { class: 'mon-grid' },
      h('section', { class: 'panel map-panel' }, h('div', { class: 'panel-title' }, 'Scalp map', h('span', { class: 'muted small' }, '8 electrodes, interpolated · top view, nose up')),
        tabs, mapCanvas, legend, hint, quality),
      h('section', { class: 'panel' }, h('div', { class: 'panel-title' }, 'Brain regions', h('span', { class: 'muted small' }, 'relative power by band')), regions),
      h('section', { class: 'panel' }, h('div', { class: 'panel-title' }, 'Answer evidence', h('span', { class: 'muted small' }, 'P300 scorer, current trial')), evidence),
      h('section', { class: 'panel wide' }, h('div', { class: 'panel-title' }, 'Live EEG', h('span', { class: 'muted small' }, 'last 5 s')), traceCanvas),
      h('section', { class: 'panel wide' }, h('div', { class: 'panel-title' }, 'P300 waveform', h('span', {}, erpSel)), erpCanvas))));

  const head = new HeadMap(mapCanvas);
  const traces = new Traces(traceCanvas);
  const erpPlot = new ErpPlot(erpCanvas);
  const feed = new EEGFeed({ traceMs: 200 }).start();

  function setMode(k) {
    mode = k;
    try { localStorage.setItem('cortext-lab.mapmode', k); } catch { /* ignore */ }
    tabs.querySelectorAll('.tab').forEach((b) => b.classList.toggle('on', b.dataset.mode === k));
    draw();
  }

  function draw() {
    const tr = feed.trace, erp = feed.erp;
    const eeg = app.bridge.status().eeg || {};
    const live = tr && tr.data && tr.data.length;
    status.className = 'live-pill ' + (!app.bridge.status().connected ? 'off' : !live ? 'off' : tr.simulated ? 'sim' : 'on');
    status.textContent = !app.bridge.status().connected ? 'bridge offline' : !live ? 'no EEG signal' : tr.simulated ? 'SIMULATED EEG' : 'LIVE';
    meta.textContent = eeg.found ? `${eeg.device || eeg.name} · ${eeg.channels} ch · ${fmt(eeg.measured_srate, 0)} Hz · scorer: ${eeg.scorer?.active ?? '–'}` +
      (eeg.recording ? ` · recording ${eeg.recording.session_id}` : '') : 'Start the headset streamer: python cortext-lab/eeg/stream_eeg.py';
    const vals = mapValues(mode, tr, erp);
    const info = BAND_INFO[mode];
    let range = null;
    if (mode === 'p300' && vals) { const m = Math.max(2, ...Object.values(vals).map(Math.abs)); range = [-m, m]; }
    const [lo, hi] = head.draw(vals || {}, { quality: tr?.quality, range });
    legend.replaceChildren(h('span', {}, `${fmt(lo, 1)} ${info.unit}`), h('i', { class: 'cbar' }), h('span', {}, `${fmt(hi, 1)} ${info.unit}`));
    hint.textContent = vals ? info.hint : mode === 'p300' ? 'Appears once a few trials of a session have finished.' : 'Waiting for signal…';
    quality.replaceChildren(...(tr?.quality || []).map((q) => h('span', { class: `qchip ${q.status}`, title: `${q.std_uv} µV` }, q.label, h('b', {}, q.status === 'good' ? '' : ` ${q.status}`))));
    traces.draw(tr);
    erpPlot.draw(erp, erpSel.value);
    const rb = regionBands(tr);
    regions.replaceChildren(...(rb.length ? rb.map((r) => h('div', { class: 'region' },
      h('div', { class: 'rname' }, r.name, h('span', { class: 'muted small' }, ` ${r.role} · ${r.channels.join(' ')}`)),
      h('div', { class: 'stack' }, ...['delta', 'theta', 'alpha', 'beta'].map((b) => h('i', { class: `b-${b}`, style: { width: `${(r[b] * 100).toFixed(1)}%` }, title: `${b} ${(r[b] * 100).toFixed(0)}%` }))))) :
      [h('div', { class: 'muted' }, 'Waiting for signal…')]),
    h('div', { class: 'bandkey' }, ...['delta', 'theta', 'alpha', 'beta'].map((b) => h('span', {}, h('i', { class: `b-${b}` }), b))));
    evidenceBars(evidence, tr?.evidence);
  }

  feed.on(draw);
  const offBridge = app.bridge.onChange(() => {});
  const onResize = () => draw();
  window.addEventListener('resize', onResize);
  return () => { feed.stop(); offBridge(); window.removeEventListener('resize', onResize); };
}
