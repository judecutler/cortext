// Uploaded stimulus sets: parse a mix of files into a config, keep them in IndexedDB, turn them into a usable config.
//
// Accepted uploads (any mix, drag-and-drop, multi-select or a whole folder):
//   * one stimuli .json in the same format as web/config/stimuli.json  ("items" required)
//   * image files (png/jpg/jpeg/gif/webp/svg/bmp/avif), matched to items by FILE NAME (folders are ignored)
//   * audio files are stored for later use (not played yet)
// If there is no .json, items are generated from the images: id/label = file name, category = parent folder name
// (or "uncategorized" when files were not uploaded as a folder).
import { db } from './db.js';
import { validateStimuli } from './stimuli.js';

const IMG_EXT = /\.(png|jpe?g|gif|webp|svg|bmp|avif)$/i;
const AUD_EXT = /\.(mp3|wav|ogg|m4a|aac|flac)$/i;
const base = (p) => String(p).split(/[\\/]/).pop();
const stem = (n) => n.replace(/\.[^.]+$/, '');
export const isImage = (f) => (f.type || '').startsWith('image/') || IMG_EXT.test(f.name);
export const isAudio = (f) => (f.type || '').startsWith('audio/') || AUD_EXT.test(f.name);
const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'item';
const pretty = (s) => s.replace(/[_-]+/g, ' ').trim();
const relPathOf = (f) => f.webkitRelativePath || f._relPath || '';

/** Pure (no DOM/IndexedDB): files -> { config, files:{basename: File}, warnings, source_files }. */
export async function buildSetFromFiles(fileList) {
  const files = Array.from(fileList);
  const warnings = [];
  const media = {};
  const mediaMeta = {};
  let config = null, configName = null;

  for (const f of files) {
    if (/\.json$/i.test(f.name)) {
      try {
        const parsed = JSON.parse(await f.text());
        if (parsed && Array.isArray(parsed.items)) {
          if (config) warnings.push(`Several config files found; using "${configName}", ignoring "${f.name}".`);
          else { config = parsed; configName = f.name; }
        } else warnings.push(`"${f.name}" is JSON but has no "items" list, ignored.`);
      } catch (e) { warnings.push(`"${f.name}" is not valid JSON (${e.message}).`); }
    } else if (isImage(f) || isAudio(f)) {
      const key = base(f.name);
      if (media[key]) warnings.push(`Two files are named "${key}"; using the first one.`);
      else { media[key] = f; mediaMeta[key] = relPathOf(f); }
    } else warnings.push(`Ignored "${f.name}" (not a .json, image or audio file).`);
  }

  const findMedia = (name) => {
    if (!name) return null;
    if (media[base(name)]) return base(name);
    const lower = base(name).toLowerCase();
    return Object.keys(media).find((k) => k.toLowerCase() === lower) || null;
  };

  if (config) {
    config = JSON.parse(JSON.stringify(config));
    config.items.forEach((it) => {
      if (it.image) {
        const k = findMedia(it.image);
        if (k) it.image = k;
        else { warnings.push(`Item "${it.id}": image "${base(it.image)}" was not uploaded.`); it.image = null; }
      } else {
        const k = Object.keys(media).find((m) => isImage(media[m]) && stem(m).toLowerCase() === String(it.id).toLowerCase());
        if (k) it.image = k;
      }
      if (it.audio) {
        const k = findMedia(it.audio);
        if (k) it.audio = k; else { warnings.push(`Item "${it.id}": audio "${base(it.audio)}" was not uploaded.`); it.audio = null; }
      }
    });
  } else {
    const imgs = Object.keys(media).filter((k) => isImage(media[k]));
    if (!imgs.length) warnings.push('No .json config and no image files: nothing to build a stimulus set from.');
    const used = new Set();
    const items = imgs.map((k) => {
      const parts = (mediaMeta[k] || '').split('/').filter(Boolean);
      const category = parts.length >= 3 ? parts[parts.length - 2] : parts.length === 2 ? parts[0] : 'uncategorized';
      let id = slug(stem(k)), n = 2;
      while (used.has(id)) id = `${slug(stem(k))}_${n++}`;
      used.add(id);
      const label = pretty(stem(k));
      return { id, category: slug(category), label: label.toLowerCase(), text: label.charAt(0).toUpperCase() + label.slice(1), image: k };
    });
    config = {
      version: 1, description: 'Generated from uploaded images', defaults: { options_per_trial: 4, tile_content: 'image_text', prompt: 'Which one is the {label}?', distractor_set: 'mixed' },
      items, distractor_sets: { mixed: { all: true }, same_category: { same_category: true } }, trial_plan: null,
    };
    if (items.length && new Set(items.map((i) => i.category)).size === 1 && items[0].category === 'uncategorized') {
      warnings.push('All items are "uncategorized". Upload a folder with one sub-folder per category (animals/dog.png …) to get categories, or upload a .json config.');
    }
  }

  const used = new Set(config.items.map((i) => i.image).concat(config.items.map((i) => i.audio)).filter(Boolean));
  const kept = {};
  for (const k of Object.keys(media)) { if (used.has(k)) kept[k] = media[k]; else warnings.push(`File "${k}" is not used by any item.`); }
  warnings.push(...validateStimuli(config));
  return { config, files: kept, warnings, source_files: files.map((f) => f.name), config_name: configName };
}

// ---- persistence ----
export async function listSets() {
  return (await db.getAll('stimulus_sets')).sort((a, b) => (b.created_iso || '').localeCompare(a.created_iso || ''));
}
export async function getSet(name) { return db.get('stimulus_sets', name); }

export async function saveSet(name, built) {
  const existing = new Set((await listSets()).map((s) => s.name));
  let final = name, n = 2;
  while (existing.has(final)) final = `${name} (${n++})`;   // never overwrite an earlier set
  const rec = { name: final, created_iso: new Date().toISOString(), config: built.config, files: built.files, source_files: built.source_files };
  await db.put('stimulus_sets', rec);
  return rec;
}

/** Stored set -> config the app can use (image paths become blob: URLs). Also builds a reproducible snapshot. */
export function materializeSet(set) {
  const cfg = JSON.parse(JSON.stringify(set.config));
  cfg.items.forEach((it) => {
    it.image_file = it.image || null;
    if (it.image && set.files[it.image]) it.image = URL.createObjectURL(set.files[it.image]);
    else if (it.image) it.image = null;
  });
  cfg.byId = Object.fromEntries(cfg.items.map((i) => [i.id, i]));
  cfg.warnings = validateStimuli(cfg);
  cfg.source = { type: 'upload', name: set.name };
  return cfg;
}

/** What goes into session.json: the items without blob URLs, so a session can be reproduced from file names. */
export function stimuliSnapshot(cfg) {
  return {
    source: cfg.source || { type: 'path' },
    defaults: cfg.defaults, prompts: cfg.prompts, distractor_sets: cfg.distractor_sets, trial_plan: cfg.trial_plan || null,
    items: cfg.items.map((i) => ({ id: i.id, category: i.category, label: i.label, text: i.text, image: i.image_file ?? i.image ?? null, audio: i.audio ?? null })),
  };
}
