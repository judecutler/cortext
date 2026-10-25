// Experiment → Images & categories: the stimulus set as a board, one column per category with its images.
// Drag images between categories or to reorder them, drop image files onto a column to add them, rename / add /
// delete / reorder categories, edit an image's label and on-screen text. Every change saves at once and can be undone.
//
// The built-in sample set is files on disk, so it cannot be changed in place: the first change makes an editable copy
// in this browser (and switches the experiment to it if the experiment was using the built-in set).
import { h } from '../util.js';
import { db } from '../db.js';
import { loadStimuli } from '../stimuli.js';
import { listSets, getSet, saveSet, isImage } from '../stimulusSets.js';
import { loadSettings, patchSettings, loadPlan } from '../settings.js';
import { groups, categoryOrder, addCategory, renameCategory, deleteCategory, moveCategory, moveItem, removeItem, updateItem, addItems, planUses } from '../library.js';
import { record, snapshot } from '../undo.js';
import { subTabs, toast, confirmDialog, promptDialog } from '../ui.js';

const BUILTIN = 'path';
const ITEM_MIME = 'application/x-cortext-item';
const base = (p) => String(p || '').split(/[\\/]/).pop();
const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');

export async function renderLibrary(root, app) {
  let sel = loadSettings().stimuli_source || BUILTIN;   // which set is on screen
  let rec = null;            // { name, config, files, readOnly, url }
  let filter = '';
  let focusId = null;        // card to focus after a re-render (keyboard moves)
  const urlCache = new Map(); // File -> object URL
  const thumb = (img) => {
    if (!img) return null;
    if (rec.readOnly) return img;
    const f = rec.files?.[img];
    if (!f) return null;
    if (!urlCache.has(f)) urlCache.set(f, URL.createObjectURL(f));
    return urlCache.get(f);
  };

  // ---------- chrome ----------
  const setSel = h('select', { 'aria-label': 'Stimulus set', onchange: (e) => open(e.target.value) });
  const useBtn = h('button', { class: 'btn small', type: 'button', onclick: useInExperiment });
  const usedChip = h('span', { class: 'rigchip ok' }, 'Used by the experiment');
  const saved = h('span', { class: 'muted small savestate', 'aria-live': 'polite' });
  const search = h('input', { type: 'search', placeholder: 'find an image…', 'aria-label': 'Find an image', oninput: (e) => { filter = e.target.value.trim().toLowerCase(); draw(); } });
  const banner = h('div');
  const board = h('div', { class: 'board' });
  const fileIn = h('input', { type: 'file', multiple: true, accept: 'image/*', style: { display: 'none' } });
  let fileTarget = null;
  fileIn.addEventListener('change', (e) => { const files = [...e.target.files]; e.target.value = ''; if (fileTarget && files.length) addFiles(files, fileTarget); });

  root.append(subTabs('library'),
    h('div', { class: 'card-head' }, h('h1', {}, 'Images & categories'), saved),
    h('p', { class: 'muted' }, 'Drag images between categories or up and down to reorder. Drop image files onto a category to add them. ',
      `Every change saves immediately and can be undone with ${isMac ? '⌘Z' : 'Ctrl+Z'}.`),
    h('div', { class: 'toolbar' },
      h('label', { class: 'inline' }, 'Set ', setSel), usedChip, useBtn,
      h('span', { class: 'spacer' }), search,
      h('button', { class: 'btn small', type: 'button', onclick: () => newCategory() }, '+ Category'),
      h('button', { class: 'btn small', type: 'button', onclick: newSet, title: 'Start an empty set and drop images into it' }, 'New set…'),
      h('button', { class: 'btn small', type: 'button', onclick: duplicateSet }, 'Duplicate…'),
      h('button', { class: 'btn small danger', type: 'button', onclick: deleteSet, id: 'delSetBtn' }, 'Delete set')),
    banner, board, fileIn,
    h('p', { class: 'kbdhelp' }, 'Keyboard: Tab to an image, then Alt+↑/↓ to reorder, Alt+←/→ to move it to the next category, Enter to edit, Delete to remove.'));

  // ---------- loading ----------
  async function fillSetList() {
    const sets = await listSets().catch(() => []);
    setSel.replaceChildren(h('option', { value: BUILTIN }, 'Built-in sample (read-only)'),
      ...sets.map((x) => h('option', { value: `set:${x.name}` }, `${x.name} (${x.config.items.length})`)));
    if (![...setSel.options].some((o) => o.value === sel)) sel = BUILTIN;
    setSel.value = sel;
  }
  async function open(value) {
    sel = value;
    if (sel.startsWith('set:')) {
      const r = await getSet(sel.slice(4));
      if (!r) { toast('That set is no longer in this browser.', 'warn'); sel = BUILTIN; return open(BUILTIN); }
      rec = { ...r, readOnly: false };
    } else {
      const url = loadSettings().stimuli_url || 'config/stimuli.json';
      const cfg = await loadStimuli(url);
      delete cfg.byId; delete cfg.warnings;
      rec = { name: null, url, config: cfg, files: {}, readOnly: true };
    }
    await fillSetList();
    draw();
  }

  // ---------- saving + undo ----------
  /** Built-in set → editable copy in this browser (images fetched from the web folder). Returns false if it failed. */
  async function ensureEditable() {
    if (!rec.readOnly) return true;
    try {
      const files = {};
      const missing = [];
      const cfg = snapshot(rec.config);
      for (const it of cfg.items) {
        if (!it.image) continue;
        // A missing or unreadable image must not block editing the whole set (e.g. a typo in stimuli.json):
        // keep the item without its picture, say which ones, and carry on.
        let res = null;
        try { res = await fetch(it.image); } catch { /* treated as missing */ }
        if (!res || !res.ok) { missing.push(`${label(it)} (${it.image})`); it.image = null; continue; }
        const blob = await res.blob();
        const name = base(it.image);
        files[name] = new File([blob], name, { type: blob.type || 'image/svg+xml' });
        it.image = name;
      }
      if (missing.length) toast(`Could not find the picture for ${missing.join(', ')}. Kept as text only; use ✎ to give it an image.`, 'warn', 12000);
      cfg.categories = categoryOrder(cfg);
      const made = await saveSet('Sample set (my copy)', { config: cfg, files, source_files: [] });
      const wasUsed = loadSettings().stimuli_source === BUILTIN || !loadSettings().stimuli_source;
      if (wasUsed) patchSettings({ stimuli_source: `set:${made.name}` });
      sel = `set:${made.name}`;
      rec = { ...made, readOnly: false };
      await fillSetList();
      toast(`The built-in set can’t be changed, so you are now editing a copy, “${made.name}”${wasUsed ? ', and the experiment uses it' : ''}.`, 'good', 9000);
      return true;
    } catch (e) { toast('Could not make an editable copy: ' + e.message, 'bad'); return false; }
  }

  async function put(r) {
    const { readOnly, ...stored } = r;
    stored.updated_iso = new Date().toISOString();
    await db.put('stimulus_sets', stored);
    saved.textContent = `All changes saved · ${new Date().toLocaleTimeString()}`;
  }
  /** Show a stored version of a set (used by undo / redo). */
  async function restore(r) {
    await put(r);
    sel = `set:${r.name}`;
    rec = { ...r, readOnly: false };
    await fillSetList();
    draw();
  }

  /**
   * Apply one change. fn(config, files) returns a new config, or { cfg, files }. quiet: no toast (moves you can see).
   * Errors (e.g. a duplicate category name) are shown and nothing changes.
   */
  async function change(label, fn, { quiet = false, kind = '' } = {}) {
    if (!(await ensureEditable())) return;
    const before = { ...rec, config: snapshot(rec.config), files: { ...rec.files } };
    let out;
    try { out = fn(rec.config, { ...rec.files }); } catch (e) { toast(e.message, 'bad'); return; }
    if (!out || out === rec.config) return;
    const cfg = out.cfg || out;
    const files = out.files || pruneFiles(cfg, rec.files);
    rec = { ...rec, config: cfg, files };
    await put(rec);
    draw();
    const after = { ...rec, config: snapshot(rec.config), files: { ...rec.files } };
    record(typeof label === 'function' ? label() : label, () => restore(before), () => restore(after), { toast: !quiet, kind });
  }
  const pruneFiles = (cfg, files) => {
    const used = new Set(cfg.items.flatMap((i) => [i.image, i.audio]).filter(Boolean));
    return Object.fromEntries(Object.entries(files || {}).filter(([k]) => used.has(k)));
  };

  // ---------- set-level actions ----------
  async function useInExperiment() {
    const prev = loadSettings().stimuli_source || BUILTIN;
    patchSettings({ stimuli_source: sel });
    drawHead();
    record(`The experiment now uses “${setName()}”`, () => { patchSettings({ stimuli_source: prev }); drawHead(); }, () => { patchSettings({ stimuli_source: sel }); drawHead(); }, { kind: 'good' });
  }
  const setName = () => (rec?.readOnly ? 'Built-in sample' : rec?.name);
  async function newSet() {
    const name = await promptDialog({ title: 'New stimulus set', label: 'Name', placeholder: 'e.g. animals vs tools', confirmLabel: 'Create',
      hint: 'You get one empty category to start with. Drop images onto it, or add more categories.' });
    if (!name) return;
    const made = await saveSet(name, { config: { version: 1, description: 'Made in Images & categories', defaults: { options_per_trial: 4, tile_content: 'image_text', prompt: 'Which one is the {label}?', distractor_set: 'mixed' },
      items: [], categories: ['category 1'], distractor_sets: { mixed: { all: true }, same_category: { same_category: true } }, trial_plan: null }, files: {}, source_files: [] });
    const prevSel = sel;
    await open(`set:${made.name}`);
    record(`Created set “${made.name}”`, async () => { await db.delete('stimulus_sets', made.name); await open(prevSel); }, () => restore(made), { kind: 'good' });
  }
  async function duplicateSet() {
    const name = await promptDialog({ title: 'Duplicate this set', label: 'Name of the copy', value: `${setName()} (copy)`, confirmLabel: 'Duplicate' });
    if (!name) return;
    let files = rec.files, cfg = snapshot(rec.config);
    if (rec.readOnly) { if (!(await ensureEditable())) return; files = rec.files; cfg = snapshot(rec.config); }
    const made = await saveSet(name, { config: cfg, files: { ...files }, source_files: [] });
    const prevSel = sel;
    await open(`set:${made.name}`);
    record(`Duplicated as “${made.name}”`, async () => { await db.delete('stimulus_sets', made.name); await open(prevSel); }, () => restore(made), { kind: 'good' });
  }
  async function deleteSet() {
    if (rec.readOnly) { toast('The built-in set cannot be deleted.', 'warn'); return; }
    const gone = { ...rec };
    const wasUsed = loadSettings().stimuli_source === sel;
    await db.delete('stimulus_sets', gone.name);
    if (wasUsed) patchSettings({ stimuli_source: BUILTIN });
    await open(BUILTIN);
    record(`Deleted set “${gone.name}”${wasUsed ? ' (the experiment is back on the built-in set)' : ''}`,
      async () => { await restore(gone); if (wasUsed) { patchSettings({ stimuli_source: `set:${gone.name}` }); drawHead(); } },
      async () => { await db.delete('stimulus_sets', gone.name); if (wasUsed) patchSettings({ stimuli_source: BUILTIN }); await open(BUILTIN); }, { kind: 'warn' });
  }

  // ---------- category actions ----------
  async function newCategory(name) {
    if (name == null) {
      name = await promptDialog({ title: 'New category', label: 'Name', placeholder: 'e.g. tools', confirmLabel: 'Add' });
      if (!name) return;
    }
    let made;
    await change(() => `Added category “${made}”`, (cfg) => { const r = addCategory(cfg, name); made = r.name; return r.cfg; }, { quiet: true });
  }
  async function askDeleteCategory(name) {
    const g = groups(rec.config).find((x) => x.name === name);
    const others = categoryOrder(rec.config).filter((c) => c !== name);
    if (!g.items.length) return change(`Deleted the empty category “${name}”`, (cfg) => deleteCategory(cfg, name));
    const target = h('select', {}, ...others.map((c) => h('option', { value: c }, c)));
    const res = await confirmDialog({
      title: `Delete category “${name}”?`, confirmLabel: 'Delete category', danger: true,
      body: h('div', {}, others.length ? h('p', {}, `Its ${g.items.length} image${g.items.length === 1 ? '' : 's'} will move to `, target, '.') : h('p', {}, `Its ${g.items.length} image(s) will be deleted with it.`),
        h('p', { class: 'muted small' }, 'You can undo this afterwards.')),
      options: others.length ? [{ key: 'drop', label: 'Delete the images too', hint: 'Instead of moving them', danger: true }] : [],
    });
    if (!res) return;
    const moveTo = others.length && !res.options.drop ? target.value : null;
    await change(moveTo ? `Deleted “${name}”; its images moved to “${moveTo}”` : `Deleted “${name}” and its ${g.items.length} image(s)`,
      (cfg) => deleteCategory(cfg, name, { moveTo }), { kind: moveTo ? '' : 'warn' });
  }

  // ---------- item actions ----------
  const label = (it) => it.label || it.text || it.id;
  function usageNote(id) {
    if (loadSettings().stimuli_source !== sel) return '';
    const qs = planUses(loadPlan(), id);
    return qs.length ? ` It is used in question${qs.length > 1 ? 's' : ''} ${qs.map((q) => 'Q' + q).join(', ')} of your list.` : '';
  }
  async function removeOne(id) {
    const it = rec.config.items.find((i) => i.id === id);
    const note = usageNote(id);
    await change(`Removed “${label(it)}”.${note}`, (cfg) => removeItem(cfg, id), { kind: note ? 'warn' : '' });
  }
  async function moveOne(id, toCat, beforeId = null) {
    const it = rec.config.items.find((i) => i.id === id);
    const dest = rec.config.items.filter((i) => i.category === toCat && i.id !== id);
    const idx = beforeId ? Math.max(0, dest.findIndex((i) => i.id === beforeId)) : null;
    const moved = it.category !== toCat;
    if (!moved) {
      const cur = rec.config.items.filter((i) => i.category === toCat);
      const from = cur.findIndex((i) => i.id === id);
      const to = idx == null ? cur.length - 1 : idx;
      if (from === to) return;
    }
    focusId = id;
    await change(moved ? `Moved “${label(it)}” to “${toCat}”` : `Reordered “${toCat}”`, (cfg) => moveItem(cfg, id, toCat, idx), { quiet: !moved });
  }
  function storeFile(files, f) {
    let key = base(f.name), n = 2;
    while (files[key] && files[key] !== f) key = base(f.name).replace(/(\.[^.]+)?$/, `_${n++}$1`);
    files[key] = f;
    return key;
  }
  async function addFiles(fileList, cat) {
    const imgs = [...fileList].filter(isImage);
    const skipped = fileList.length - imgs.length;
    if (!imgs.length) { toast('Only image files can be added here (png, jpg, svg, webp, gif…).', 'warn'); return; }
    let n = 0;
    await change(() => `Added ${n} image${n === 1 ? '' : 's'} to “${cat}”${skipped ? ` (${skipped} non-image file${skipped === 1 ? '' : 's'} skipped)` : ''}`, (cfg, files) => {
      const entries = imgs.map((f) => ({ name: f.name, image: storeFile(files, f) }));
      const r = addItems(cfg, entries, cat);
      n = r.added.length;
      return { cfg: r.cfg, files };
    }, { kind: 'good' });
  }
  function editItem(id) {
    const it = rec.config.items.find((i) => i.id === id);
    if (!it) return;
    const labelIn = h('input', { type: 'text', value: it.label || '', style: { width: '100%' } });
    const textIn = h('input', { type: 'text', value: it.text || '', style: { width: '100%' } });
    const catSel = h('select', { style: { width: '100%' } }, ...categoryOrder(rec.config).map((c) => h('option', { value: c, selected: c === it.category }, c)));
    let newFile = null;
    const preview = h('div', { class: 'edit-preview' }, it.image ? h('img', { src: thumb(it.image), alt: '' }) : h('span', { class: 'muted' }, 'no image'));
    const fileBtn = h('input', { type: 'file', accept: 'image/*', onchange: (e) => {
      newFile = e.target.files[0] || null;
      if (newFile) preview.replaceChildren(h('img', { src: URL.createObjectURL(newFile), alt: '' }));
    } });
    const row = (l, el, hint) => h('div', { class: 'field' }, h('label', {}, l), h('div', {}, el, hint ? h('small', { class: 'muted' }, hint) : null));
    const dlg = h('dialog', { class: 'modal' },
      h('form', { method: 'dialog', onsubmit: (e) => { e.preventDefault(); save(); } },
        h('h2', {}, `Edit “${label(it)}”`), preview,
        row('Label', labelIn, 'Used in question wording, e.g. “Which one is the {label}?”'),
        row('Text on screen', textIn, 'Shown under or instead of the image.'),
        row('Category', catSel),
        row('Replace image', fileBtn),
        h('p', { class: 'muted small' }, `id: ${it.id} (stays the same, so question lists and recorded sessions still match)`),
        h('div', { class: 'dlg-actions' }, h('button', { class: 'btn', type: 'button', onclick: () => close() }, 'Cancel'), h('button', { class: 'btn primary', type: 'submit' }, 'Save'))));
    function close() { dlg.close(); dlg.remove(); }
    async function save() {
      close();
      focusId = id;
      await change(`Edited “${label(it)}”`, (cfg, files) => {
        const changes = { label: labelIn.value, text: textIn.value, category: catSel.value };
        if (newFile) changes.image = storeFile(files, newFile);
        return { cfg: updateItem(cfg, id, changes), files: newFile ? files : undefined };
      }, { quiet: true });
    }
    dlg.addEventListener('cancel', (e) => { e.preventDefault(); close(); });
    document.body.append(dlg);
    dlg.showModal();
    labelIn.select();
  }

  // ---------- drawing ----------
  function drawHead() {
    const used = (loadSettings().stimuli_source || BUILTIN) === sel;
    usedChip.style.display = used ? '' : 'none';
    useBtn.style.display = used ? 'none' : '';
    useBtn.textContent = 'Use this set in the experiment';
    root.querySelector('#delSetBtn').disabled = !!rec?.readOnly;
    banner.replaceChildren(rec?.readOnly ? h('p', { class: 'note-banner' }, 'This is the built-in sample set, stored as files with the app. Your first change makes an editable copy in this browser; the built-in set itself never changes.') : '');
  }
  function draw() {
    drawHead();
    const gs = groups(rec.config);
    const match = (it) => !filter || [it.id, it.label, it.text, it.category].join(' ').toLowerCase().includes(filter);
    board.replaceChildren(...gs.map((g, gi) => column(g, gi, gs.length, match)),
      h('div', { class: 'catcol newcol' },
        h('form', { onsubmit: (e) => { e.preventDefault(); const v = e.target.elements.c.value.trim(); if (v) newCategory(v); } },
          h('input', { name: 'c', type: 'text', placeholder: 'New category name', 'aria-label': 'New category name', style: { width: '100%' } }),
          h('button', { class: 'btn small', type: 'submit', style: { marginTop: '6px' } }, '+ Add category'))));
    if (focusId) { board.querySelector(`.icard[data-id="${CSS.escape(focusId)}"]`)?.focus(); focusId = null; }
    if (!gs.length) board.prepend(h('p', { class: 'muted' }, 'This set has no categories yet. Add one, then drop images onto it.'));
  }

  function column(g, gi, n, match) {
    const nameBtn = h('button', { class: 'catname', type: 'button', title: 'Rename', onclick: () => startRename() }, g.name);
    const head = h('div', { class: 'cathead' }, nameBtn, h('span', { class: 'count' }, String(g.items.length)),
      h('span', { class: 'spacer' }),
      h('button', { class: 'chip-btn', type: 'button', title: 'Move category left', 'aria-label': `Move ${g.name} left`, disabled: gi === 0, onclick: () => change(`Moved “${g.name}” left`, (cfg) => moveCategory(cfg, g.name, -1), { quiet: true }) }, '◀'),
      h('button', { class: 'chip-btn', type: 'button', title: 'Move category right', 'aria-label': `Move ${g.name} right`, disabled: gi === n - 1, onclick: () => change(`Moved “${g.name}” right`, (cfg) => moveCategory(cfg, g.name, 1), { quiet: true }) }, '▶'),
      h('button', { class: 'chip-btn danger-x', type: 'button', title: 'Delete category', 'aria-label': `Delete ${g.name}`, onclick: () => askDeleteCategory(g.name) }, '✕'));
    function startRename() {
      const inp = h('input', { type: 'text', value: g.name, class: 'catname-in', 'aria-label': 'Category name' });
      let done = false;
      const finish = async (commit) => {
        if (done) return; done = true;
        const v = inp.value.trim();
        if (commit && v && v !== g.name) await change(`Renamed “${g.name}” to “${v}”`, (cfg) => renameCategory(cfg, g.name, v), { quiet: true });
        else draw();
      };
      inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); finish(true); } else if (e.key === 'Escape') { e.preventDefault(); finish(false); } });
      inp.addEventListener('blur', () => finish(true));
      nameBtn.replaceWith(inp);
      inp.select();
    }
    const body = h('div', { class: 'catbody', dataset: { cat: g.name } }, ...g.items.map((it) => card(it, g, gi, n, match(it))),
      g.items.length ? null : h('div', { class: 'empty muted small' }, 'Empty: drop images here'));
    const indicator = h('div', { class: 'drop-line' });
    const beforeIdAt = (y) => {
      const cards = [...body.querySelectorAll('.icard:not(.dragging):not(.hidden)')];
      const c = cards.find((el) => y < el.getBoundingClientRect().top + el.offsetHeight / 2);
      return c ? c.dataset.id : null;
    };
    body.addEventListener('dragover', (e) => {
      const types = [...e.dataTransfer.types];
      if (!types.includes(ITEM_MIME) && !types.includes('Files')) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = types.includes('Files') ? 'copy' : 'move';
      col.classList.add('over');
      const b = beforeIdAt(e.clientY);
      const ref = b ? body.querySelector(`.icard[data-id="${CSS.escape(b)}"]`) : null;
      if (ref) body.insertBefore(indicator, ref); else body.append(indicator);
    });
    body.addEventListener('dragleave', (e) => { if (!col.contains(e.relatedTarget)) { col.classList.remove('over'); indicator.remove(); } });
    body.addEventListener('drop', (e) => {
      e.preventDefault();
      col.classList.remove('over'); indicator.remove();
      const id = e.dataTransfer.getData(ITEM_MIME);
      if (id) moveOne(id, g.name, beforeIdAt(e.clientY));
      else if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files, g.name);
    });
    const foot = h('button', { class: 'btn small addimg', type: 'button', onclick: () => { fileTarget = g.name; fileIn.click(); } }, '+ Add images');
    const col = h('section', { class: 'catcol', 'aria-label': `Category ${g.name}` }, head, body, foot);
    return col;
  }

  function card(it, g, gi, n, visible) {
    const src = thumb(it.image);
    const el = h('div', { class: `icard${visible ? '' : ' hidden'}`, tabindex: '0', draggable: 'true', dataset: { id: it.id },
      'aria-label': `${label(it)}, in ${g.name}. Alt+arrows to move, Enter to edit, Delete to remove.` },
      h('div', { class: 'ithumb' }, src ? h('img', { src, alt: '', draggable: 'false', loading: 'lazy' }) : h('span', { class: 'muted small' }, it.text || 'no image')),
      h('div', { class: 'imeta' }, h('div', { class: 'ilabel' }, label(it)), it.text && it.text.toLowerCase() !== (it.label || '').toLowerCase() ? h('div', { class: 'muted small' }, `shows “${it.text}”`) : null),
      h('div', { class: 'iacts' },
        h('button', { class: 'chip-btn', type: 'button', title: 'Edit', 'aria-label': `Edit ${label(it)}`, tabindex: '-1', onclick: () => editItem(it.id) }, '✎'),
        h('button', { class: 'chip-btn danger-x', type: 'button', title: 'Remove', 'aria-label': `Remove ${label(it)}`, tabindex: '-1', onclick: () => removeOne(it.id) }, '✕')));
    el.addEventListener('dragstart', (e) => { e.dataTransfer.setData(ITEM_MIME, it.id); e.dataTransfer.effectAllowed = 'move'; requestAnimationFrame(() => el.classList.add('dragging')); });
    el.addEventListener('dragend', () => el.classList.remove('dragging'));
    el.addEventListener('dblclick', () => editItem(it.id));
    el.addEventListener('keydown', (e) => {
      if (e.target !== el) return;
      const cats = categoryOrder(rec.config);
      const idx = g.items.findIndex((x) => x.id === it.id);
      if (e.key === 'Enter') { e.preventDefault(); editItem(it.id); }
      else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); removeOne(it.id); }
      else if (e.altKey && e.key === 'ArrowUp' && idx > 0) { e.preventDefault(); moveOne(it.id, g.name, g.items[idx - 1].id); }
      else if (e.altKey && e.key === 'ArrowDown' && idx < g.items.length - 1) { e.preventDefault(); moveOne(it.id, g.name, g.items[idx + 2]?.id ?? null); }
      else if (e.altKey && e.key === 'ArrowLeft' && gi > 0) { e.preventDefault(); moveOne(it.id, cats[gi - 1]); }
      else if (e.altKey && e.key === 'ArrowRight' && gi < n - 1) { e.preventDefault(); moveOne(it.id, cats[gi + 1]); }
      else if (!e.altKey && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
        e.preventDefault();
        const all = [...board.querySelectorAll('.icard:not(.hidden)')];
        all[all.indexOf(el) + (e.key === 'ArrowDown' ? 1 : -1)]?.focus();
      }
    });
    return el;
  }

  await open(sel);
  return () => { urlCache.forEach((u) => URL.revokeObjectURL(u)); };
}
