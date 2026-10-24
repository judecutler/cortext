// App shell: hash router, shared singletons (bridge, markers), Researcher/Demo mode toggle.
import { BridgeClient } from './bridgeClient.js';
import { MarkerBus } from './markers.js';
import { renderSetup } from './views/setup.js';
import { renderRun } from './views/run.js';
import { renderReview } from './views/review.js';
import { renderSummary } from './views/summary.js';
import { renderSessions } from './views/sessions.js';
import { h } from './util.js';

const bridge = new BridgeClient();
const markers = new MarkerBus(bridge);
export const app = {
  bridge, markers, runner: null, pending: null,
  mode: localStorage.getItem('cortext-lab.mode') || 'researcher',
  toggleMode() { app.setMode(app.mode === 'demo' ? 'researcher' : 'demo'); },
  setMode(m) {
    app.mode = m;
    document.body.dataset.mode = m;
    localStorage.setItem('cortext-lab.mode', m);
    const b = document.getElementById('modeToggle');
    if (b) b.textContent = m === 'demo' ? 'Mode: Demo' : 'Mode: Researcher';
  },
};
window.cortext = app; // handy in the console

const view = document.getElementById('view');
let cleanup = null;
let currentHash = null;

function topStatus() {
  const s = bridge.status();
  const el = document.getElementById('topstatus');
  if (!el) return;
  el.innerHTML = `<span class="dot ${s.connected ? 'green' : 'red'}"></span>bridge ${s.connected ? 'connected' : 'offline'}` +
    (s.connected ? ` · LSL ${s.lsl_available ? 'on' : 'off'}` : '');
}

async function route() {
  const hash = location.hash || '#/setup';
  // Leaving a running session: confirm first.
  if (app.runner && app.runner.phase !== 'finished' && !hash.startsWith('#/run') && currentHash && currentHash.startsWith('#/run')) {
    if (!confirm('A session is running. Leave and end it? (Data recorded so far is kept.)')) {
      history.replaceState(null, '', currentHash);
      return;
    }
  }
  if (cleanup) { try { cleanup(); } catch (e) { console.error(e); } cleanup = null; }
  currentHash = hash;
  view.replaceChildren();
  const [, name, arg] = hash.split('/');
  try {
    if (name === 'run') cleanup = renderRun(view, app);
    else if (name === 'review') await renderReview(view, app, decodeURIComponent(arg));
    else if (name === 'summary') await renderSummary(view, app, decodeURIComponent(arg));
    else if (name === 'sessions') await renderSessions(view, app);
    else await renderSetup(view, app);
  } catch (e) {
    console.error(e);
    view.replaceChildren(h('div', { class: 'card' }, h('h2', { class: 'bad' }, 'Something went wrong'), h('pre', { class: 'mono' }, String(e.stack || e))));
  }
}

document.getElementById('modeToggle').addEventListener('click', () => app.toggleMode());
document.addEventListener('keydown', (e) => {
  // Mode toggle outside the run view (the run view handles it itself).
  if (!location.hash.startsWith('#/run') && (e.key === 'm' || e.key === 'M') && !/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) app.toggleMode();
});
window.addEventListener('hashchange', route);
app.setMode(app.mode);
bridge.onChange(topStatus);
topStatus();
bridge.start();
route();
