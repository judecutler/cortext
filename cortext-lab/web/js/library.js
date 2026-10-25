// Editing a stimulus set's categories and images. Pure functions on a config ({ items, categories? }): no DOM, no
// IndexedDB, so they are unit-tested in tests/logic.test.mjs. Each returns a NEW config; the caller saves it and
// records the old one for undo.
//
// Item ids never change here (only label, text and category), so question lists and recorded sessions that refer
// to an item by id keep working after you reorganise.
//
// Category order: config.categories lists the categories in display order and also keeps EMPTY categories (a
// category that only exists through its items would vanish when its last item moves out). Older configs without
// it get the order in which categories first appear in items.

const clone = (cfg) => ({ ...cfg, items: cfg.items.map((i) => ({ ...i })), categories: cfg.categories ? cfg.categories.slice() : undefined });

/** Category names in display order, including empty ones. */
export function categoryOrder(cfg) {
  const out = (cfg.categories || []).filter((c, i, a) => c && a.indexOf(c) === i);
  for (const it of cfg.items) if (it.category && !out.includes(it.category)) out.push(it.category);
  return out;
}

/** [{ name, items }] in display order. Items keep their order inside the items list. */
export function groups(cfg) {
  return categoryOrder(cfg).map((name) => ({ name, items: cfg.items.filter((i) => i.category === name) }));
}

/** Rebuild items so they are stored grouped by category in display order (keeps everything tidy for exports). */
function normalize(cfg) {
  const order = categoryOrder(cfg);
  cfg.categories = order;
  cfg.items = order.flatMap((c) => cfg.items.filter((i) => i.category === c));
  return cfg;
}

export function cleanCategoryName(name) { return String(name ?? '').trim().replace(/\s+/g, ' ').slice(0, 60); }

function uniqueName(cfg, base) {
  const taken = new Set(categoryOrder(cfg).map((c) => c.toLowerCase()));
  let name = base, n = 2;
  while (taken.has(name.toLowerCase())) name = `${base} ${n++}`;
  return name;
}

export function addCategory(cfg, name = 'new category') {
  const c = clone(cfg);
  const clean = cleanCategoryName(name) || 'new category';
  const final = uniqueName(c, clean);
  c.categories = [...categoryOrder(c), final];
  return { cfg: normalize(c), name: final };
}

export function renameCategory(cfg, from, to) {
  const clean = cleanCategoryName(to);
  if (!clean) throw new Error('A category needs a name.');
  if (clean === from) return cfg;
  if (categoryOrder(cfg).some((x) => x !== from && x.toLowerCase() === clean.toLowerCase())) throw new Error(`There is already a category called “${clean}”. Drag the images across instead to merge them.`);
  const c = clone(cfg);
  c.categories = categoryOrder(c).map((x) => (x === from ? clean : x));
  c.items.forEach((it) => { if (it.category === from) it.category = clean; });
  if (c.prompts && from in c.prompts) { c.prompts = { ...c.prompts, [clean]: c.prompts[from] }; delete c.prompts[from]; }
  return normalize(c);
}

/** Delete a category. moveTo: another category for its images, or null to delete the images as well. */
export function deleteCategory(cfg, name, { moveTo = null } = {}) {
  const c = clone(cfg);
  if (moveTo != null && !categoryOrder(c).includes(moveTo)) throw new Error(`No category called “${moveTo}”.`);
  c.categories = categoryOrder(c).filter((x) => x !== name);
  if (moveTo != null) c.items.forEach((it) => { if (it.category === name) it.category = moveTo; });
  else c.items = c.items.filter((it) => it.category !== name);
  return normalize(c);
}

/** Move a whole category left (-1) or right (+1) in the display order. */
export function moveCategory(cfg, name, delta) {
  const order = categoryOrder(cfg);
  const i = order.indexOf(name), j = i + delta;
  if (i < 0 || j < 0 || j >= order.length) return cfg;
  [order[i], order[j]] = [order[j], order[i]];
  const c = clone(cfg);
  c.categories = order;
  return normalize(c);
}

/**
 * Move one item to a category, at a position inside that category (0 = first; null or too large = last).
 * Works for reordering inside a category too.
 */
export function moveItem(cfg, id, toCategory, toIndex = null) {
  const c = clone(cfg);
  const k = c.items.findIndex((i) => i.id === id);
  if (k < 0) throw new Error(`No item “${id}”.`);
  if (!categoryOrder(c).includes(toCategory)) c.categories = [...categoryOrder(c), toCategory];
  const [it] = c.items.splice(k, 1);
  it.category = toCategory;
  const order = categoryOrder({ ...c, items: [...c.items, it] });
  const g = Object.fromEntries(order.map((n) => [n, c.items.filter((x) => x.category === n)]));
  const dest = g[toCategory];
  const at = toIndex == null ? dest.length : Math.max(0, Math.min(dest.length, toIndex));
  dest.splice(at, 0, it);
  c.categories = order;
  c.items = order.flatMap((n) => g[n]);
  return c;
}

export function removeItem(cfg, id) {
  const c = clone(cfg);
  c.categories = categoryOrder(c);   // the category stays even if this was its last image
  c.items = c.items.filter((i) => i.id !== id);
  return normalize(c);
}

/** Change what is shown / used for an item: label ({label} in prompts), text (shown on screen), category. */
export function updateItem(cfg, id, changes) {
  let c = clone(cfg);
  const it = c.items.find((i) => i.id === id);
  if (!it) throw new Error(`No item “${id}”.`);
  const { category, ...rest } = changes;
  for (const [k, v] of Object.entries(rest)) if (['label', 'text', 'image'].includes(k)) it[k] = typeof v === 'string' ? v.trim() : v;
  if (category != null && cleanCategoryName(category) && cleanCategoryName(category) !== it.category) c = moveItem(c, id, cleanCategoryName(category));
  return c;
}

const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'item';

/** An item id not used yet in this set, derived from a name (file name or label). */
export function uniqueId(cfg, name) {
  const used = new Set(cfg.items.map((i) => i.id));
  const base = slug(name);
  let id = base, n = 2;
  while (used.has(id)) id = `${base}_${n++}`;
  return id;
}

/** Add new items (from images) to a category. entries: [{ name, image }] where image is the stored file name. */
export function addItems(cfg, entries, category) {
  let c = clone(cfg);
  const added = [];
  if (!categoryOrder(c).includes(category)) c.categories = [...categoryOrder(c), category];
  for (const e of entries) {
    const pretty = String(e.name).replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim() || 'item';
    const id = uniqueId(c, pretty);
    const item = { id, category, label: pretty.toLowerCase(), text: pretty.charAt(0).toUpperCase() + pretty.slice(1), image: e.image ?? null };
    c.items.push(item);
    added.push(item);
  }
  c = normalize(c);
  return { cfg: c, added };
}

/** Which question-list entries use an item (to warn before it disappears). */
export function planUses(plan, id) {
  return (plan || []).map((q, i) => (q.target === id || q.options.includes(id) ? i + 1 : null)).filter(Boolean);
}
