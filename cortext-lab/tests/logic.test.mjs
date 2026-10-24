// Run: node --test tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

import { makeFlashPlan, runFlashPlan } from '../web/js/flasher.js';
import { mulberry32, toCSV, csvEscape } from '../web/js/util.js';
import { makeTrialRecord, applyValidity, recordClick, summarizeTrials, flashCounts } from '../web/js/trial.js';
import { buildTrialSpecs, validateStimuli } from '../web/js/stimuli.js';
import { makeZip, crc32, trialRow } from '../web/js/exporter.js';
import { TRIAL_COLUMNS } from '../web/js/schema.js';

const here = dirname(fileURLToPath(import.meta.url));
const cfg = JSON.parse(readFileSync(join(here, '../web/config/stimuli.json'), 'utf8'));
cfg.byId = Object.fromEntries(cfg.items.map((i) => [i.id, i]));

test('stimulus config is valid and has the promised 6/3/3 split', () => {
  assert.deepEqual(validateStimuli(cfg), []);
  const n = (c) => cfg.items.filter((i) => i.category === c).length;
  assert.equal(cfg.items.length, 12);
  assert.deepEqual([n('animal'), n('object'), n('shape')], [6, 3, 3]);
  for (const it of cfg.items) readFileSync(join(here, '../web', it.image)); // image files exist
});

test('flash plan: every option exactly `repeats` times, never the same option twice in a row', () => {
  for (let seed = 1; seed <= 200; seed++) {
    const ids = ['a', 'b', 'c', 'd'];
    const plan = makeFlashPlan({ optionIds: ids, repeats: 10, rng: mulberry32(seed) });
    assert.equal(plan.events.length, 40);
    for (const id of ids) assert.equal(plan.events.filter((e) => e.option_id === id).length, 10);
    for (let i = 1; i < plan.events.length; i++) assert.notEqual(plan.events[i].option_id, plan.events[i - 1].option_id, `seed ${seed} idx ${i}`);
    // each block of 4 is a permutation
    for (let r = 0; r < 10; r++) assert.deepEqual(plan.events.slice(r * 4, r * 4 + 4).map((e) => e.option_id).sort(), ids);
  }
});

test('flash plan: timing respects on/off configuration', () => {
  const plan = makeFlashPlan({ optionIds: ['a', 'b', 'c'], repeats: 5, onMs: 500, offMinMs: 100, offMaxMs: 200, rng: mulberry32(7) });
  for (let i = 0; i < plan.events.length; i++) {
    const e = plan.events[i];
    assert.equal(e.off_at - e.on_at, 500);
    assert.ok(e.off_gap_ms >= 100 && e.off_gap_ms <= 200);
    if (i) assert.ok(Math.abs(e.on_at - (plan.events[i - 1].off_at + plan.events[i - 1].off_gap_ms)) < 1e-9);
  }
  assert.throws(() => makeFlashPlan({ optionIds: ['only'], repeats: 2 }));
  const two = makeFlashPlan({ optionIds: ['x', 'y'], repeats: 50, rng: mulberry32(3) });
  for (let i = 1; i < two.events.length; i++) assert.notEqual(two.events[i].option_id, two.events[i - 1].option_id);
});

// ---- run the scheduler on a simulated 60 Hz display ----
function installFakeRaf({ hz = 60, jitter = 0, dropEvery = 0 } = {}) {
  let now = 1000, queue = [], n = 0;
  globalThis.requestAnimationFrame = (cb) => { queue.push(cb); return queue.length; };
  globalThis.performance = { now: () => now, timeOrigin: 0 };
  const frameMs = 1000 / hz;
  return {
    async pump(maxFrames = 100000) {
      for (let i = 0; i < maxFrames && queue.length; i++) {
        const cbs = queue; queue = [];
        n++;
        now += frameMs * (dropEvery && n % dropEvery === 0 ? 3 : 1) + (Math.random() - 0.5) * jitter;
        cbs.forEach((cb) => cb(now));
        await Promise.resolve();
      }
    },
  };
}

test('scheduler on a 60 Hz clock: onset within one frame, one change per frame, ordered callbacks', async () => {
  const clock = installFakeRaf({ hz: 60 });
  const plan = makeFlashPlan({ optionIds: ['a', 'b', 'c', 'd'], repeats: 3, rng: mulberry32(11) });
  const log = [];
  const p = runFlashPlan(plan, { onOn: (ev, ts) => log.push(['on', ev.flash_index, ts]), onOff: (ev, ts) => log.push(['off', ev.flash_index, ts]) });
  await clock.pump();
  const res = await p;
  assert.equal(res.aborted, false);
  assert.equal(res.results.length, 12);
  for (const r of res.results) {
    assert.ok(Math.abs(r.onset_err_ms) <= 1000 / 60 + 1e-6, `onset err ${r.onset_err_ms}`);
    assert.ok(r.dur_ms >= 1000 / 60 - 1e-6);                 // at least one frame visible
    assert.ok(Math.abs(r.dur_err_ms) <= 1000 / 60 + 1e-6);   // 500 ms => 30 frames +-1
  }
  // strictly alternating on/off, never two on the same frame timestamp
  for (let i = 0; i < log.length; i++) assert.equal(log[i][0], i % 2 === 0 ? 'on' : 'off');
  for (let i = 1; i < log.length; i++) assert.ok(log[i][2] > log[i - 1][2]);
  assert.equal(res.frame.dropped, 0);
});

test('scheduler survives dropped frames and reports them; abort stops cleanly and switches the flash off', async () => {
  let clock = installFakeRaf({ hz: 60, dropEvery: 50 });
  const plan = makeFlashPlan({ optionIds: ['a', 'b'], repeats: 6, rng: mulberry32(5) });
  let p = runFlashPlan(plan, { onOn: () => {}, onOff: () => {} });
  await clock.pump();
  let res = await p;
  assert.ok(res.frame.dropped > 0);
  assert.equal(res.results.length, 12);

  clock = installFakeRaf({ hz: 60 });
  const token = { aborted: false };
  let on = false, offs = 0;
  p = runFlashPlan(plan, { token, onOn: (ev, ts, r) => { on = true; if (ev.flash_index === 3) token.aborted = true; }, onOff: () => { on = false; offs++; } });
  await clock.pump();
  res = await p;
  assert.equal(res.aborted, true);
  assert.equal(on, false, 'highlight must be off after abort');
});

test('trial records: shuffled positions, ground truth, validity log never loses history', () => {
  const spec = buildTrialSpecs(cfg, { n: 1, seed: 4 })[0];
  const t = makeTrialRecord({ sessionId: 'S', spec, trialId: 0, attempt: 0, rng: mulberry32(1), settings: { on_ms: 500, off_min_ms: 100, off_max_ms: 200, repeats: 10 }, byId: cfg.byId });
  assert.equal(t.options.length, 4);
  assert.deepEqual(t.options.map((o) => o.position), [0, 1, 2, 3]);
  assert.equal(t.options.filter((o) => o.is_target).length, 1);
  assert.equal(t.options.find((o) => o.is_target).id, t.true_answer_id);
  recordClick(t, t.true_answer_id, 812);
  assert.equal(t.correct, true);
  applyValidity(t, { valid: false, reason: 'misclick' });
  applyValidity(t, { valid: true });
  applyValidity(t, { valid: false, reason: 'distracted', note: 'looked away' });
  assert.equal(t.valid, false);
  assert.equal(t.validity_log.length, 3);
  assert.equal(t.validity_log[0].reason, 'misclick');
  assert.equal(t.clicked_answer_id, t.true_answer_id, 'click data is kept when a trial is invalidated');
});

test('trial specs: balanced targets, 4 unique options incl. target, prompt filled', () => {
  const specs = buildTrialSpecs(cfg, { n: 24, seed: 9 });
  assert.equal(specs.length, 24);
  const count = {};
  for (const s of specs) {
    count[s.target_id] = (count[s.target_id] || 0) + 1;
    assert.equal(new Set(s.option_ids).size, 4);
    assert.ok(s.option_ids.includes(s.target_id));
    assert.match(s.prompt, /^Which one is the \w+\?$/);
  }
  assert.ok(Object.values(count).every((c) => c === 2), JSON.stringify(count));
  const same = buildTrialSpecs({ ...cfg, defaults: { ...cfg.defaults, distractor_set: 'same_category' } }, { n: 6, seed: 1, categories: ['animal'] });
  for (const s of same) assert.ok(s.option_ids.every((id) => cfg.byId[id].category === 'animal'));
  const plan = buildTrialSpecs({ ...cfg, trial_plan: [{ target: 'dog', options: ['dog', 'cat', 'cup', 'circle'] }] }, { seed: 1 });
  assert.equal(plan.length, 1); assert.equal(plan[0].prompt, 'Which one is the dog?');
});

test('summary & counters count only valid trials and keep invalid reasons', () => {
  const settings = { on_ms: 500, off_min_ms: 100, off_max_ms: 200, repeats: 2 };
  const specs = buildTrialSpecs(cfg, { n: 4, seed: 2 });
  const trials = specs.map((spec, i) => {
    const t = makeTrialRecord({ sessionId: 'S', spec, trialId: i, attempt: 0, rng: mulberry32(i + 1), settings, byId: cfg.byId });
    t.flash_order = makeFlashPlan({ optionIds: t.options.map((o) => o.id), repeats: 2, rng: mulberry32(i) }).events;
    recordClick(t, i < 3 ? t.true_answer_id : t.options.find((o) => !o.is_target).id, 700 + i);
    return t;
  });
  applyValidity(trials[0], { valid: false, reason: 'blink_or_movement' });
  const S = summarizeTrials(trials);
  assert.equal(S.attempts, 4); assert.equal(S.valid, 3); assert.equal(S.invalid, 1);
  assert.equal(S.discard_rate, 0.25);
  assert.equal(S.click_accuracy, 2 / 3);
  assert.deepEqual(S.invalid_reasons, { blink_or_movement: 1 });
  const fc = flashCounts(trials);
  const tot = Object.values(fc).reduce((a, v) => a + v.target + v.nontarget, 0);
  assert.equal(tot, 3 * 8);
  assert.equal(Object.values(fc).reduce((a, v) => a + v.target, 0), 3 * 2);
});

test('CSV escaping and trial row columns', () => {
  assert.equal(csvEscape('a,b'), '"a,b"');
  assert.equal(csvEscape('say "hi"'), '"say ""hi"""');
  assert.equal(csvEscape(null), '');
  assert.equal(toCSV([{ a: 1, b: 'x,y' }], ['a', 'b']), 'a,b\n1,"x,y"\n');
  const spec = buildTrialSpecs(cfg, { n: 1, seed: 1 })[0];
  const t = makeTrialRecord({ sessionId: 'S', spec, trialId: 3, attempt: 1, rng: mulberry32(1), settings: { on_ms: 500, off_min_ms: 100, off_max_ms: 200, repeats: 10 }, byId: cfg.byId });
  const row = trialRow(t);
  for (const c of TRIAL_COLUMNS) assert.ok(c in row, `missing column ${c}`);
  assert.equal(row.option_ids_by_position.split('|').length, 4);
});

test('ZIP writer: correct CRC and a structure Python can read back', () => {
  assert.equal(crc32(new TextEncoder().encode('123456789')), 0xcbf43926);
  return makeZip([{ name: 'a/session.json', data: '{"x":1}' }, { name: 'a/trials.csv', data: 'h\n1,2\n' }, { name: 'é.txt', data: 'ünï' }]).arrayBuffer().then((buf) => {
    const dir = mkdtempSync(join(tmpdir(), 'zip-'));
    writeFileSync(join(dir, 't.zip'), Buffer.from(buf));
    writeFileSync(join(dir, 'path.txt'), join(dir, 't.zip'));
    globalThis.__zipPath = join(dir, 't.zip');
    console.log('zip written to', globalThis.__zipPath);
  });
});
