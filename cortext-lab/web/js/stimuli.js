// Stimulus config loading and trial-spec generation. Adding categories/items never needs code changes.
import { shuffle, mulberry32 } from './util.js';

export async function loadStimuli(url = 'config/stimuli.json') {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Cannot load ${url}: HTTP ${res.status}`);
  const cfg = await res.json();
  cfg.warnings = validateStimuli(cfg);
  cfg.byId = Object.fromEntries(cfg.items.map((it) => [it.id, it]));
  return cfg;
}

export function validateStimuli(cfg) {
  const warn = [];
  const ids = new Set();
  for (const it of cfg.items || []) {
    if (!it.id) warn.push('item without id');
    if (ids.has(it.id)) warn.push(`duplicate item id: ${it.id}`);
    ids.add(it.id);
    if (!it.category) warn.push(`${it.id}: missing category`);
    if (!it.image && !it.text) warn.push(`${it.id}: needs an image or a text`);
  }
  for (const [name, set] of Object.entries(cfg.distractor_sets || {})) {
    for (const m of set.members || []) if (!ids.has(m)) warn.push(`distractor set ${name}: unknown item ${m}`);
  }
  for (const t of cfg.trial_plan || []) {
    for (const id of [t.target, ...(t.options || [])]) if (!ids.has(id)) warn.push(`trial_plan: unknown item ${id}`);
  }
  return warn;
}

export const categoriesOf = (cfg) => [...new Set(cfg.items.map((i) => i.category))];

function fillPrompt(tpl, item) {
  return tpl.replace(/\{(\w+)\}/g, (_, k) => (item[k] != null ? item[k] : ''));
}

function distractorPool(cfg, target, setName) {
  const set = (cfg.distractor_sets || {})[setName] || { all: true };
  let pool;
  if (set.members) pool = cfg.items.filter((i) => set.members.includes(i.id));
  else if (set.same_category) pool = cfg.items.filter((i) => i.category === target.category);
  else pool = cfg.items.slice();
  return pool.filter((i) => i.id !== target.id);
}

/**
 * Turn the config into trial specs (one per question). Option POSITIONS are shuffled later, per trial.
 * opts: { n, seed, optionsPerTrial, tileContent, categories (filter targets), trialType }
 */
export function buildTrialSpecs(cfg, opts = {}) {
  const rng = mulberry32(opts.seed ?? 1);
  const d = cfg.defaults || {};
  const nOpt = opts.optionsPerTrial || d.options_per_trial || 4;
  const tile = opts.tileContent || d.tile_content || 'image_text';
  const specs = [];
  const byId = cfg.byId || Object.fromEntries(cfg.items.map((i) => [i.id, i]));

  if (cfg.trial_plan && cfg.trial_plan.length) {
    const reps = opts.n ? Math.ceil(opts.n / cfg.trial_plan.length) : 1;
    for (let r = 0; r < reps; r++) for (const t of cfg.trial_plan) {
      specs.push({
        type: opts.trialType || 'mcq_flash', target_id: t.target, option_ids: t.options.slice(),
        tile_content: t.tile_content || tile, prompt: t.prompt || fillPrompt(byId[t.target].prompt || (cfg.prompts || {})[byId[t.target].category] || d.prompt || 'Which one is the {label}?', byId[t.target]),
      });
    }
  } else {
    const targets = cfg.items.filter((i) => !opts.categories || opts.categories.includes(i.category));
    const n = opts.n || targets.length;
    let order = [];
    while (order.length < n) order = order.concat(shuffle(targets, rng));   // balanced: every target before any repeats
    for (const target of order.slice(0, n)) {
      const pool = distractorPool(cfg, target, target.distractor_set || d.distractor_set || 'mixed');
      const dis = shuffle(pool, rng).slice(0, nOpt - 1);
      if (dis.length < nOpt - 1) throw new Error(`Not enough distractors for ${target.id} (need ${nOpt - 1}, have ${dis.length})`);
      const tpl = target.prompt || (cfg.prompts || {})[target.category] || d.prompt || 'Which one is the {label}?';
      specs.push({
        type: opts.trialType || 'mcq_flash', target_id: target.id, option_ids: [target.id, ...dis.map((x) => x.id)],
        tile_content: tile, prompt: fillPrompt(tpl, target),
      });
    }
  }
  return specs.map((s, i) => ({ ...s, trial_index: i }));
}

// ---------- custom question lists (presets) ----------
// A plan is [{ prompt, target, options: [ids in display order] }]. It is plain data, so it can be saved in a preset.

/** Fisher-Yates on a copy, so randomising never changes the original list. */
export const shuffled = (arr, rng = Math.random) => shuffle(arr, rng);

/** Generate a plan from the stimulus set (same rules as the automatic questions), options listed target-first-shuffled. */
export function generatePlan(cfg, { n, optionsPerTrial = 4, seed = (Math.random() * 2 ** 31) >>> 0, categories } = {}) {
  const rng = mulberry32(seed);
  return buildTrialSpecs(cfg, { n, seed, optionsPerTrial, categories }).map((s) => ({ prompt: s.prompt, target: s.target_id, options: shuffle(s.option_ids, rng) }));
}

/** Apply a wording template such as "Which one is the {label}?" to every question ({label} {text} {category} {id}). */
export function rephrase(cfg, plan, template) {
  const byId = cfg.byId || Object.fromEntries(cfg.items.map((i) => [i.id, i]));
  return plan.map((q) => ({ ...q, prompt: fillPrompt(template, { ...byId[q.target], id: q.target }) }));
}

/** Turn a plan into the trial specs the runner uses. Throws a readable error if the plan does not fit the stimulus set. */
export function specsFromPlan(cfg, plan, { trialType = 'mcq_flash', tileContent = 'image_text', fixedOrder = false } = {}) {
  const byId = cfg.byId || Object.fromEntries(cfg.items.map((i) => [i.id, i]));
  const bad = new Set();
  plan.forEach((q, i) => {
    for (const id of [q.target, ...q.options]) if (!byId[id]) bad.add(id);
    if (!q.options.includes(q.target)) throw new Error(`Question ${i + 1}: the answer “${q.target}” is not among its images.`);
    if (q.options.length < 2) throw new Error(`Question ${i + 1} needs at least 2 images.`);
    if (new Set(q.options).size !== q.options.length) throw new Error(`Question ${i + 1} lists the same image twice.`);
  });
  if (bad.size) throw new Error(`The question list uses items that are not in the chosen stimulus set: ${[...bad].join(', ')}. Pick the stimulus set this list was made for.`);
  return plan.map((q, i) => ({ type: trialType, target_id: q.target, option_ids: q.options.slice(), tile_content: tileContent, prompt: q.prompt || 'Which one is correct?',
    fixed_order: fixedOrder, trial_index: i }));
}
