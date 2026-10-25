// EEG tab (the neural monitor): live view of the headset for the audience (open it on a second screen / window).
// Never shown to the participant during flashing: moving visuals in their field of view would add their own brain
// responses to the recording.
import { h, fmt } from '../util.js';
import { HeadMap, Traces, ErpPlot, EEGFeed, evidenceBars, mapValues, regionBands, BAND_INFO } from '../brainviz.js';

export function renderEeg(root, app) {
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
  // P300 sites first; the list follows the connected headset (8-ch Unicorn, or TP9/TP10 first on a Muse)
  const ERP_ORDER = ['Pz', 'POz', 'Cz', 'PO7', 'PO8', 'TP9', 'TP10', 'Fz', 'C3', 'C4', 'AF7', 'AF8'];
  const erpOptions = (labels) => [...ERP_ORDER.filter((l) => labels.includes(l)), ...labels.filter((l) => !ERP_ORDER.includes(l))];
  const erpSel = h('select', { onchange: () => draw() }, ...erpOptions(['Pz', 'POz', 'Cz', 'PO7', 'PO8', 'Fz', 'C3', 'C4']).map((l) => h('option', { value: l }, l)));
  let erpLabelsKey = '';
  const mapNote = h('span', { class: 'muted small' }, '8 electrodes, interpolated · top view, nose up');
  function syncLabels(labels) {
    const key = (labels || []).join(',');
    if (!labels || !labels.length || key === erpLabelsKey) return;
    erpLabelsKey = key;
    const keep = erpSel.value;
    const opts = erpOptions(labels);
    erpSel.replaceChildren(...opts.map((l) => h('option', { value: l }, l)));
    erpSel.value = opts.includes(keep) ? keep : opts[0];
    mapNote.textContent = `${labels.length} electrodes${labels.length < 8 ? ' (sparse map)' : ', interpolated'} · top view, nose up`;
  }
  const regions = h('div', { class: 'regions' });
  const evidence = h('div', { class: 'evidence' });
  const quality = h('div', { class: 'qchips' });

  root.append(h('div', { class: 'monitor' },
    h('div', { class: 'mon-head' },
      h('div', {}, h('div', { class: 'eyebrow' }, 'CorText · live'), h('h1', { class: 'glow' }, 'Neural activity monitor')),
      h('div', { class: 'mon-status' }, status, meta,
        h('button', { class: 'btn small', type: 'button', title: 'Open the monitor in its own window for a second screen',
          onclick: () => window.open(location.pathname + '#/eeg', 'cortext-monitor', 'popup,width=1400,height=900') }, 'Open in its own window ↗'))),
    h('div', { class: 'mon-grid' },
      h('section', { class: 'panel map-panel' }, h('div', { class: 'panel-title' }, 'Scalp map', mapNote),
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
    syncLabels(eeg.labels || erp?.labels);
    const live = tr && tr.data && tr.data.length;
    status.className = 'live-pill ' + (!app.bridge.status().connected ? 'off' : !live ? 'off' : tr.simulated ? 'sim' : 'on');
    status.textContent = !app.bridge.status().connected ? 'bridge offline' : !live ? 'no EEG signal' : tr.simulated ? 'SIMULATED EEG' : 'LIVE';
    meta.textContent = eeg.found ? `${eeg.device || eeg.name} · ${eeg.channels} ch · ${fmt(eeg.measured_srate, 0)} Hz · scorer: ${eeg.scorer?.active ?? '–'}` +
      (eeg.recording ? ` · recording ${eeg.recording.session_id}` : '') : 'Start the headset streamer: python cortext-lab/eeg/stream_eeg.py (Unicorn) or eeg/stream_muse.py (Muse)';
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
  // Optional "vibe meter" (just-for-fun band-power estimates). Loaded on its own and fully isolated: read-only, own
  // try/catch, appended after the existing panels, so if it fails the rest of the monitor is unaffected.
  let detachVibes = null, gone = false;
  import('../statepanel.js').then(({ createStatePanel }) => {
    if (gone) return;
    const p = createStatePanel(feed);
    detachVibes = p.detach;
    root.querySelector('.mon-grid')?.append(p.el);
  }).catch((e) => console.warn('vibe meter unavailable', e));
  const offBridge = app.bridge.onChange(() => {});
  const onResize = () => draw();
  window.addEventListener('resize', onResize);
  return () => { gone = true; if (detachVibes) detachVibes(); feed.stop(); offBridge(); window.removeEventListener('resize', onResize); };
}
