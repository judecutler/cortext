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

test('stimulus config is valid and has the expected 4/4/3 split', () => {
  assert.deepEqual(validateStimuli(cfg), []);
  const n = (c) => cfg.items.filter((i) => i.category === c).length;
  assert.equal(cfg.items.length, 11);
  assert.deepEqual([n('animal'), n('object'), n('shape')], [4, 4, 3]);
  assert.ok(!cfg.byId.fish && !cfg.byId.frog, 'fish and frog were removed');
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
  const N = cfg.items.length * 2;   // every item twice as target
  const specs = buildTrialSpecs(cfg, { n: N, seed: 9 });
  assert.equal(specs.length, N);
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

// ================= new in this change: upload, sequential mode, model probabilities, feedback export =================
import { softmax } from '../web/js/predictor.js';
import { estimateDurations, flashParamsFor, DEFAULT_SETTINGS } from '../web/js/settings.js';
import { feedbackRows, feedbackCsv } from '../web/js/exporter.js';
import { FEEDBACK_COLUMNS } from '../web/js/schema.js';
import { buildSetFromFiles, materializeSet, stimuliSnapshot } from '../web/js/stimulusSets.js';

const svg = (name, rel = '') => { const f = new File(['<svg xmlns="http://www.w3.org/2000/svg"/>'], name, { type: 'image/svg+xml' }); if (rel) f._relPath = rel; return f; };

test('softmax: sums to 1, orders like the scores, stable for large values', () => {
  const p = softmax({ a: 2, b: 1, c: -1, d: 0 }, 1);
  assert.ok(Math.abs(Object.values(p).reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(p.a > p.b && p.b > p.d && p.d > p.c);
  assert.ok(Math.abs(softmax({ x: 1000, y: 999 }, 1).x - 1 / (1 + Math.exp(-1))) < 1e-9);
  assert.deepEqual(softmax({}), {});
});

test('sequential mode: timing parameters and duration estimate use the sequential settings', () => {
  const s = { ...DEFAULT_SETTINGS, trial_type: 'mcq_sequential', options_per_trial: 4, seq_repeats: 2, seq_on_ms: 1000, seq_gap_ms: 300, prompt_ms: 2000, n_trials: 10 };
  assert.deepEqual(flashParamsFor('mcq_sequential', s), { on_ms: 1000, off_min_ms: 300, off_max_ms: 300, repeats: 2 });
  assert.deepEqual(flashParamsFor('mcq_flash', s), { on_ms: 500, off_min_ms: 100, off_max_ms: 200, repeats: 10 });
  const d = estimateDurations(s);
  assert.equal(d.flashes, 8); assert.ok(Math.abs(d.flashSec - 8 * 1.3) < 1e-9);
  // the original mode is unchanged: 40 flashes x (500 + 150) ms
  assert.ok(Math.abs(estimateDurations({ ...DEFAULT_SETTINGS, trial_type: 'mcq_flash' }).flashSec - 26) < 1e-9);
  // the default is the demo: 4 options x 3 rounds x 1.3 s, after a 5 s question
  const demo = estimateDurations({ ...DEFAULT_SETTINGS });
  assert.equal(DEFAULT_SETTINGS.trial_type, 'mcq_sequential'); assert.equal(demo.flashes, 12);
  assert.ok(Math.abs(demo.flashSec - 15.6) < 1e-9);
});

test('sequential plan: one-at-a-time with fixed gap, every option shown `repeats` times, no immediate repeats', () => {
  const plan = makeFlashPlan({ optionIds: ['a', 'b', 'c', 'd'], repeats: 3, onMs: 1000, offMinMs: 300, offMaxMs: 300, rng: mulberry32(2) });
  assert.equal(plan.events.length, 12);
  plan.events.forEach((e, i) => { assert.equal(e.off_at - e.on_at, 1000); if (i) assert.equal(e.on_at - plan.events[i - 1].off_at, 300); });
  for (let i = 1; i < 12; i++) assert.notEqual(plan.events[i].option_id, plan.events[i - 1].option_id);
});

test('upload: config + images are matched by file name; missing files produce warnings, unused files too', async () => {
  const conf = { items: [
    { id: 'dog', category: 'animal', label: 'dog', text: 'Dog', image: 'stimuli/dog.svg' },
    { id: 'cat', category: 'animal', label: 'cat', text: 'Cat', image: 'imgs/CAT.SVG' },
    { id: 'cup', category: 'object', label: 'cup', text: 'Cup', image: 'cup.png' },     // not uploaded
    { id: 'sun', category: 'object', label: 'sun', text: 'Sun' },                        // matched by id
  ] };
  const built = await buildSetFromFiles([new File([JSON.stringify(conf)], 'my.json'), svg('dog.svg'), svg('cat.svg'), svg('sun.svg'), svg('extra.svg'), new File(['x'], 'notes.txt')]);
  assert.equal(built.config.items.find((i) => i.id === 'dog').image, 'dog.svg');
  assert.equal(built.config.items.find((i) => i.id === 'cat').image, 'cat.svg');           // case-insensitive
  assert.equal(built.config.items.find((i) => i.id === 'sun').image, 'sun.svg');           // auto-attached by id
  assert.equal(built.config.items.find((i) => i.id === 'cup').image, null);
  assert.deepEqual(Object.keys(built.files).sort(), ['cat.svg', 'dog.svg', 'sun.svg']);
  const w = built.warnings.join(' | ');
  assert.match(w, /cup.*cup\.png.*not uploaded/); assert.match(w, /extra\.svg.*not used/); assert.match(w, /notes\.txt/);
});

test('upload: a folder of images becomes a stimulus set (file = item, sub-folder = category)', async () => {
  const built = await buildSetFromFiles([
    svg('dog.svg', 'pets/animals/dog.svg'), svg('Red_Cat.svg', 'pets/animals/Red_Cat.svg'),
    svg('cup.svg', 'pets/objects/cup.svg'), svg('dog.svg', 'pets/objects/dog.svg')]);
  const items = built.config.items;
  assert.equal(items.length, 3, 'duplicate file name is ignored with a warning');
  assert.match(built.warnings.join(' '), /Two files are named "dog.svg"/);
  const byId = Object.fromEntries(items.map((i) => [i.id, i]));
  assert.equal(byId.dog.category, 'animals'); assert.equal(byId.red_cat.category, 'animals'); assert.equal(byId.cup.category, 'objects');
  assert.equal(byId.red_cat.label, 'red cat'); assert.equal(byId.red_cat.text, 'Red Cat'); assert.equal(byId.red_cat.image, 'Red_Cat.svg');
  // and it plugs straight into trial generation
  const specs = buildTrialSpecs({ ...built.config, byId }, { n: 3, seed: 1, optionsPerTrial: 3 });
  assert.equal(specs.length, 3); specs.forEach((s) => assert.equal(new Set(s.option_ids).size, 3));
  // flat upload (no folders) is allowed but flagged
  const flat = await buildSetFromFiles([svg('a.svg'), svg('b.svg')]);
  assert.ok(flat.config.items.every((i) => i.category === 'uncategorized')); assert.match(flat.warnings.join(' '), /uncategorized/);
  const empty = await buildSetFromFiles([new File(['x'], 'readme.txt')]);
  assert.equal(empty.config.items.length, 0);
});

test('materialize: blob URLs for display, file names kept in the reproducible snapshot', async () => {
  const built = await buildSetFromFiles([svg('a.svg', 'x/g/a.svg'), svg('b.svg', 'x/g/b.svg')]);
  const cfg = materializeSet({ name: 'n', config: built.config, files: built.files });
  assert.ok(cfg.items.every((i) => i.image.startsWith('blob:')));
  const snap = stimuliSnapshot(cfg);
  assert.deepEqual(snap.items.map((i) => i.image), ['a.svg', 'b.svg']);
  assert.deepEqual(snap.source, { type: 'upload', name: 'n' });
  assert.ok(!JSON.stringify(snap).includes('blob:'));
});

test('model feedback rows: one per option, probabilities next to truth/click, anchoring flag', () => {
  const spec = buildTrialSpecs(cfg, { n: 1, seed: 3 })[0];
  const settings = { ...DEFAULT_SETTINGS };
  const t = makeTrialRecord({ sessionId: 'S', spec: { ...spec, type: 'mcq_sequential' }, trialId: 5, attempt: 0, rng: mulberry32(1), settings, byId: cfg.byId });
  const others = t.options.filter((o) => !o.is_target);
  t.prediction = { source: 'ws', probabilities: { [t.true_answer_id]: 0.2, [others[0].id]: 0.6, [others[1].id]: 0.1, [others[2].id]: 0.1 }, p300: { [t.true_answer_id]: 0.1 }, probabilities_calibrated: true };
  t.model_display = { probabilities: t.prediction.probabilities };
  recordClick(t, t.true_answer_id, 900);
  const rows = feedbackRows([t]);
  assert.equal(rows.length, 4);
  assert.ok(Math.abs(rows.reduce((a, r) => a + r.model_probability, 0) - 1) < 1e-9);
  assert.equal(rows.filter((r) => r.is_true_answer).length, 1); assert.equal(rows.filter((r) => r.is_clicked).length, 1);
  assert.equal(rows.filter((r) => r.is_model_top_choice).length, 1);
  assert.equal(rows.find((r) => r.is_model_top_choice).option_id, others[0].id);
  assert.ok(rows.every((r) => r.human_agrees_with_model === false && r.model_shown_before_click === true && r.presentation_mode === 'mcq_sequential'));
  assert.equal(t.flash_params.on_ms, 1000, 'sequential trial records carry the sequential timing');
  const csv = feedbackCsv([t]).split('\n');
  assert.equal(csv[0].split(',').length, FEEDBACK_COLUMNS.length); assert.equal(csv.length, 6);
  assert.deepEqual(feedbackRows([{ ...t, prediction: null }]), [], 'no model output, no feedback rows');
  const row = trialRow(t); assert.equal(row.model_shown_before_click, true);
});

// ---------- presets: custom question lists, fixed order, randomising ----------
import { generatePlan, rephrase, specsFromPlan } from '../web/js/stimuli.js';

const pcfg = (() => {
  const items = ['dog', 'cat', 'cup', 'house', 'circle', 'square'].map((id) => ({ id, label: id, category: id.length === 3 ? 'a' : 'b', text: id }));
  return { items, byId: Object.fromEntries(items.map((i) => [i.id, i])), defaults: {} };
})();

test('fixedOrder flash plans repeat the listed order exactly; random ones do not', () => {
  const ids = ['a', 'b', 'c', 'd'];
  const fixed = makeFlashPlan({ optionIds: ids, repeats: 3, rng: mulberry32(1), fixedOrder: true });
  assert.deepEqual(fixed.events.map((e) => e.option_id), [...ids, ...ids, ...ids]);
  for (let i = 1; i < fixed.events.length; i++) assert.notEqual(fixed.events[i].option_id, fixed.events[i - 1].option_id);
  const random = makeFlashPlan({ optionIds: ids, repeats: 6, rng: mulberry32(1) });
  assert.notDeepEqual(random.events.map((e) => e.option_id), [...ids, ...ids, ...ids, ...ids, ...ids, ...ids]);
});

test('trial records keep the listed option order only when the spec says fixed', () => {
  const spec = (fixed) => ({ type: 'mcq_sequential', target_id: 'cat', option_ids: ['dog', 'cat', 'cup', 'house'], tile_content: 'image_text', prompt: 'q', trial_index: 0, fixed_order: fixed });
  const mk = (s, seed) => makeTrialRecord({ sessionId: 's', spec: s, trialId: 0, attempt: 0, rng: mulberry32(seed), settings: { ...{ seq_on_ms: 1, seq_gap_ms: 1, seq_repeats: 1, on_ms: 1, off_min_ms: 1, off_max_ms: 1, repeats: 1 } }, byId: pcfg.byId });
  assert.deepEqual(mk(spec(true), 5).options.map((o) => o.id), ['dog', 'cat', 'cup', 'house']);
  assert.equal(mk(spec(true), 5).fixed_order, true);
  const orders = new Set([1, 2, 3, 4, 5, 6].map((s) => mk(spec(false), s).options.map((o) => o.id).join()));
  assert.ok(orders.size > 1, 'random order should vary');
});

test('plans: generate, rephrase, validate, convert to specs', () => {
  const plan = generatePlan(pcfg, { n: 6, optionsPerTrial: 3, seed: 4 });
  assert.equal(plan.length, 6);
  for (const q of plan) { assert.ok(q.options.includes(q.target)); assert.equal(q.options.length, 3); }
  const re = rephrase(pcfg, plan, 'Find the {label} please');
  assert.equal(re[0].prompt, `Find the ${plan[0].target} please`);
  const specs = specsFromPlan(pcfg, re, { trialType: 'mcq_sequential', fixedOrder: true });
  assert.equal(specs[2].trial_index, 2); assert.equal(specs[2].fixed_order, true); assert.deepEqual(specs[2].option_ids, plan[2].options);
  assert.throws(() => specsFromPlan(pcfg, [{ prompt: 'x', target: 'dog', options: ['cat', 'cup'] }]), /not among its images/);
  assert.throws(() => specsFromPlan(pcfg, [{ prompt: 'x', target: 'nope', options: ['nope', 'cup'] }]), /not in the chosen stimulus set/);
  assert.throws(() => specsFromPlan(pcfg, [{ prompt: 'x', target: 'dog', options: ['dog', 'dog'] }]), /twice/);
});

// ---------- Images & categories (library.js) ----------
import { groups, categoryOrder, addCategory, renameCategory, deleteCategory, moveCategory, moveItem, removeItem, updateItem, addItems, planUses } from '../web/js/library.js';
import { snapshot } from '../web/js/undo.js';

// Fixed fixture (the original 6/3/3 sample) so these tests check the editing logic, not whatever images the
// shipped sample set happens to contain.
const FIXTURE = [['animal', ['dog', 'cat', 'bird', 'fish', 'rabbit', 'frog']], ['object', ['cup', 'umbrella', 'house']], ['shape', ['circle', 'square', 'triangle']]]
  .flatMap(([category, ids]) => ids.map((id) => ({ id, category, label: id, text: id[0].toUpperCase() + id.slice(1), image: `stimuli/${id}.svg` })));
const lib = () => ({ items: FIXTURE.map((i) => ({ ...i })) });
const names = (c, cat) => groups(c).find((g) => g.name === cat).items.map((i) => i.id);

test('library: categories in first-appearance order; empty ones kept once listed', () => {
  const c = lib();
  assert.deepEqual(categoryOrder(c), ['animal', 'object', 'shape']);
  const { cfg: c2, name } = addCategory(c, 'tools');
  assert.equal(name, 'tools');
  assert.deepEqual(categoryOrder(c2), ['animal', 'object', 'shape', 'tools']);
  assert.equal(addCategory(c2, 'Tools').name, 'Tools 2', 'names are unique ignoring case');
  assert.equal(c.categories, undefined, 'the input is never changed');
});

test('library: move an item to another category and position, and reorder inside one', () => {
  let c = moveItem(lib(), 'dog', 'shape', 1);
  assert.deepEqual(names(c, 'shape'), ['circle', 'dog', 'square', 'triangle']);
  assert.ok(!names(c, 'animal').includes('dog'));
  c = moveItem(c, 'triangle', 'shape', 0);
  assert.deepEqual(names(c, 'shape'), ['triangle', 'circle', 'dog', 'square']);
  c = moveItem(c, 'triangle', 'shape', null);
  assert.deepEqual(names(c, 'shape'), ['circle', 'dog', 'square', 'triangle']);
  assert.equal(c.items.length, 12, 'nothing lost or duplicated');
  // emptying a category keeps it
  let e = lib();
  for (const id of ['cup', 'umbrella', 'house']) e = moveItem(e, id, 'shape');
  assert.deepEqual(categoryOrder(e), ['animal', 'object', 'shape']);
  assert.deepEqual(names(e, 'object'), []);
});

test('library: rename, reorder and delete categories', () => {
  let c = renameCategory({ ...lib(), prompts: { animal: 'Which {label}?' } }, 'animal', 'Animals');
  assert.ok(c.items.filter((i) => i.category === 'Animals').length === 6);
  assert.equal(c.prompts.Animals, 'Which {label}?');
  assert.throws(() => renameCategory(c, 'Animals', 'shape'), /already a category/);
  assert.throws(() => renameCategory(c, 'Animals', '  '), /needs a name/);
  c = moveCategory(c, 'shape', -1);
  assert.deepEqual(categoryOrder(c), ['Animals', 'shape', 'object']);
  assert.equal(moveCategory(c, 'Animals', -1), c, 'moving past the edge does nothing');
  const moved = deleteCategory(c, 'shape', { moveTo: 'object' });
  assert.deepEqual(categoryOrder(moved), ['Animals', 'object']);
  assert.equal(moved.items.length, 12);
  const dropped = deleteCategory(c, 'shape');
  assert.equal(dropped.items.length, 9);
  assert.throws(() => deleteCategory(c, 'shape', { moveTo: 'nope' }), /No category/);
});

test('library: edit, remove and add items; ids never change', () => {
  let c = updateItem(lib(), 'cat', { label: ' kitty ', text: 'Kitty', category: 'object' });
  const cat = c.items.find((i) => i.id === 'cat');
  assert.deepEqual([cat.label, cat.text, cat.category], ['kitty', 'Kitty', 'object']);
  c = removeItem(c, 'cat');
  assert.ok(!c.items.some((i) => i.id === 'cat'));
  const r = addItems(c, [{ name: 'dog.png', image: 'dog.png' }, { name: 'red-car.jpg', image: 'red-car.jpg' }], 'vehicles');
  assert.deepEqual(r.added.map((i) => i.id), ['dog_2', 'red_car'], 'a clashing id gets a suffix');
  assert.equal(r.added[1].text, 'Red car');
  assert.deepEqual(categoryOrder(r.cfg).at(-1), 'vehicles');
  assert.deepEqual(planUses([{ target: 'dog', options: ['dog', 'cup'] }, { target: 'cup', options: ['cup', 'fish'] }], 'cup'), [1, 2]);
});

test('undo snapshot: deep copy of plain data, blobs kept by reference', () => {
  const blob = new Blob(['x']);
  const a = { plan: [{ options: ['a', 'b'] }], files: { f: blob } };
  const b = snapshot(a);
  b.plan[0].options.push('c');
  assert.deepEqual(a.plan[0].options, ['a', 'b']);
  assert.equal(b.files.f, blob);
});

// 2026-10-25: per-trial seeds used to overflow double precision, so the correct answer kept landing in the same slot.
test('per-trial seeds give an even answer position (no bias toward A)', async () => {
  const u = await import('../web/js/util.js');
  const { makeTrialRecord } = await import('../web/js/trial.js');
  const byId = Object.fromEntries(['dog', 'cat', 'bird', 'car'].map((id) => [id, { id, category: 'x', label: id }]));
  const c = [0, 0, 0, 0];
  for (let s = 0; s < 100; s++) {
    const S = (s * 2147483629 + 12345) >>> 0;
    for (let tid = 0; tid < 20; tid++) {
      const tr = makeTrialRecord({ sessionId: 's', spec: { type: 'mcq_sequential', target_id: 'dog', option_ids: ['dog', 'cat', 'bird', 'car'], trial_index: tid },
        trialId: tid, attempt: 0, rng: u.mulberry32(u.mixSeed(S, tid)), settings: {}, byId });
      c[tr.options.find((o) => o.is_target).position]++;
    }
  }
  for (const n of c) assert.ok(n > 400 && n < 600, `answer positions ${c}`);
});

test('one-at-a-time: first round follows letter order A, B, C, D; later rounds are random', async () => {
  const { makeFlashPlan } = await import('../web/js/flasher.js');
  const { mulberry32 } = await import('../web/js/util.js');
  const ids = ['a', 'b', 'c', 'd'];
  const p = makeFlashPlan({ optionIds: ids, repeats: 3, rng: mulberry32(7), firstRoundInOrder: true });
  assert.deepEqual(p.events.slice(0, 4).map((e) => e.option_id), ids);
  assert.notEqual(p.events[4].option_id, 'd');
});
