// Undo / redo for the operator's edits (golden rule: easy reversal of actions).
//
// Every reversible action calls record(label, undoFn, redoFn). The action happens immediately, with no "are you sure?"
// dialog; a toast says what happened and offers Undo. Ctrl/Cmd+Z undoes, Ctrl/Cmd+Shift+Z (or Ctrl+Y) redoes, and the
// ↶ button in the top bar shows what the next undo will do.
//
// Undo entries hold closures over the page that made them, so the history is cleared when you move to another page.
// Things that cannot be reversed (deleting a session from disk / Hugging Face, starting a recording) keep their
// confirmation dialog and never go on this stack.
import { toast } from './ui.js';

const MAX = 60;
const done = [];
const undone = [];
const listeners = new Set();
let busy = false;

const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');
export const UNDO_KEY = isMac ? '⌘Z' : 'Ctrl+Z';

function emit() { listeners.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } }); }
export function onUndoChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }
export const canUndo = () => done.length > 0;
export const canRedo = () => undone.length > 0;
export const nextUndoLabel = () => done[done.length - 1]?.label || null;
export const nextRedoLabel = () => undone[undone.length - 1]?.label || null;

/**
 * Remember an action that has just been done.
 *   label  short past-tense description shown to the operator ("Moved dog to shapes")
 *   undoFn / redoFn  may be async; they must restore / re-apply the state and refresh the page
 *   opts.toast  false to stay silent (the caller already shows feedback)
 */
export function record(label, undoFn, redoFn, opts = {}) {
  done.push({ label, undo: undoFn, redo: redoFn });
  if (done.length > MAX) done.shift();
  undone.length = 0;
  emit();
  if (opts.toast !== false) toast(label, opts.kind || '', null, { label: 'Undo', kbd: UNDO_KEY, fn: undo }, 'undo');
}

export async function undo() {
  if (busy) return;
  const a = done.pop();
  if (!a) { toast('Nothing to undo.', '', 2000, null, 'undo'); return; }
  busy = true;
  try { await a.undo(); undone.push(a); toast(`Undone: ${a.label}`, '', null, { label: 'Redo', kbd: isMac ? '⇧⌘Z' : 'Ctrl+Y', fn: redo }, 'undo'); }
  catch (e) { console.error(e); toast(`Could not undo “${a.label}”: ${e.message}`, 'bad', null, null, 'undo'); }
  finally { busy = false; emit(); }
}

export async function redo() {
  if (busy) return;
  const a = undone.pop();
  if (!a) { toast('Nothing to redo.', '', 2000, null, 'undo'); return; }
  busy = true;
  try { await a.redo(); done.push(a); toast(`Redone: ${a.label}`, '', null, { label: 'Undo', kbd: UNDO_KEY, fn: undo }, 'undo'); }
  catch (e) { console.error(e); toast(`Could not redo “${a.label}”: ${e.message}`, 'bad', null, null, 'undo'); }
  finally { busy = false; emit(); }
}

/** Forget the history (called on every page change, because entries refer to the page that made them). */
export function clearUndo() {
  done.length = 0; undone.length = 0; emit();
  document.querySelectorAll('#toasts .toast[data-group="undo"]').forEach((t) => t.remove());   // their Undo button would do nothing now
}

/** Deep copy for snapshots of plain data (plans, configs). Blobs/Files are kept by reference. */
export function snapshot(x) {
  if (x == null || typeof x !== 'object') return x;
  if (x instanceof Blob) return x;
  if (Array.isArray(x)) return x.map(snapshot);
  return Object.fromEntries(Object.entries(x).map(([k, v]) => [k, snapshot(v)]));
}
