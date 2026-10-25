// Run: node --test tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import { rawIndices, StateTracker, STATES } from '../web/js/statelogic.js';

const LABELS = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'PO8', 'Oz'];
function trace({ delta = 0.1, theta = 0.2, alpha = 0.4, beta = 0.3, bad = [] } = {}) {
  const col = (v) => LABELS.map(() => v);
  return {
    labels: LABELS,
    quality: LABELS.map((label) => ({ label, status: bad.includes(label) ? 'noisy' : 'good' })),
    bands: { delta: col(delta), theta: col(theta), alpha: col(alpha), beta: col(beta) },
  };
}
const run = (tr, seconds, tracker = new StateTracker(), t0 = 0) => {
  let out;
  for (let i = 0; i < seconds * 4; i++) out = tracker.update(tr, t0 + i * 250);
  return [out, tracker];
};

test('no usable data never throws and never reports numbers', () => {
  assert.equal(rawIndices(null), null);
  assert.equal(rawIndices({}), null);
  assert.equal(rawIndices({ labels: ['Fz'], bands: {} }), null);
  const t = new StateTracker();
  assert.equal(t.update(null, 0).status, 'waiting');
  assert.equal(t.update(trace({ bad: LABELS.slice(1) }), 250).status, 'contact');
  assert.equal(t.update(trace({ bad: LABELS.slice(1) }), 500).scores, null);
});

test('calibrates first, then shows 0-100 scores centred on 50 for a steady signal', () => {
  const tr = trace();
  const [early] = run(tr, 5);
  assert.equal(early.status, 'calibrating');
  assert.equal(early.scores, null);
  const [out] = run(tr, 40);
  assert.equal(out.status, 'ok');
  for (const { key } of STATES) assert.ok(Math.abs(out.scores[key] - 50) <= 3, `${key} ${out.scores[key]}`);
});

test('beta rising moves focus/stress/excitement up and tiredness down, relative to baseline', () => {
  const [, tracker] = run(trace(), 40);
  const [out] = run(trace({ theta: 0.15, alpha: 0.2, beta: 0.55 }), 20, tracker, 40000);
  assert.ok(out.scores.focus > 60, `focus ${out.scores.focus}`);
  assert.ok(out.scores.stress > 60, `stress ${out.scores.stress}`);
  assert.ok(out.scores.excitement > 60, `excitement ${out.scores.excitement}`);
  assert.ok(out.scores.tiredness < 40, `tiredness ${out.scores.tiredness}`);
});

test('slow waves rising moves tiredness up', () => {
  const [, tracker] = run(trace(), 40);
  const [out] = run(trace({ theta: 0.4, alpha: 0.4, beta: 0.1 }), 20, tracker, 40000);
  assert.ok(out.scores.tiredness > 60, `tiredness ${out.scores.tiredness}`);
});

test('movement holds the last values and is not learned as baseline', () => {
  const [ok, tracker] = run(trace(), 40);
  const before = JSON.stringify(ok.scores);
  const [mv] = run(trace({ delta: 0.9, theta: 0.03, alpha: 0.03, beta: 0.04 }), 10, tracker, 40000);
  assert.equal(mv.status, 'moving');
  assert.equal(JSON.stringify(mv.scores), before);
});

test('scores stay within 0-100 even for extreme input', () => {
  const [, tracker] = run(trace(), 40);
  const [out] = run(trace({ theta: 0.0001, alpha: 0.0001, beta: 0.9 }), 30, tracker, 40000);
  for (const { key } of STATES) assert.ok(out.scores[key] >= 0 && out.scores[key] <= 100);
});
