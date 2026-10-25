// Small shared UI pieces: toast notifications and a promise-based confirmation dialog (native <dialog>).
import { h } from './util.js';

export function fmtBytes(n) {
  if (n == null || Number.isNaN(n)) return '–';
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1073741824) return `${(n / 1048576).toFixed(1)} MB`;
  return `${(n / 1073741824).toFixed(2)} GB`;
}

let toastHost = null;
/** A message that fades after a few seconds. kind: 'good' | 'warn' | 'bad' | ''. Errors stay longer. */
export function toast(message, kind = '', ms = null) {
  if (!toastHost) { toastHost = h('div', { id: 'toasts', role: 'status', 'aria-live': 'polite' }); document.body.append(toastHost); }
  const el = h('div', { class: `toast ${kind}` }, message, h('button', { class: 'toast-x', 'aria-label': 'Dismiss', onclick: () => el.remove() }, '×'));
  toastHost.append(el);
  setTimeout(() => el.remove(), ms ?? (kind === 'bad' || kind === 'warn' ? 12000 : 4500));
  return el;
}

/**
 * Ask before doing something irreversible.
 *   title, body (node or string), confirmLabel, danger (red button),
 *   options: [{ key, label, checked, disabled, hint, danger }]   extra checkboxes, returned as { key: bool }
 *   typeToConfirm: a word the operator must type before the button enables (or a function of the options chosen)
 * Resolves to null when cancelled, else { options: { key: bool } }.
 */
export function confirmDialog({ title, body, confirmLabel = 'OK', danger = false, options = [], typeToConfirm = null }) {
  return new Promise((resolve) => {
    const boxes = {};
    const optEls = options.map((o) => {
      const cb = h('input', { type: 'checkbox', checked: !!o.checked, disabled: !!o.disabled, onchange: sync });
      boxes[o.key] = cb;
      return h('label', { class: `dlg-opt${o.danger ? ' danger' : ''}${o.disabled ? ' off' : ''}` }, cb,
        h('span', {}, h('b', {}, o.label), o.hint ? h('small', {}, o.hint) : null));
    });
    const typed = h('input', { type: 'text', class: 'dlg-type', placeholder: '', autocomplete: 'off', spellcheck: 'false', oninput: sync });
    const typeRow = h('div', { class: 'dlg-typerow', style: { display: 'none' } }, h('small', {}), typed);
    const ok = h('button', { class: `btn ${danger ? 'danger-solid' : 'primary'}`, type: 'button', onclick: () => close({ options: values() }) }, confirmLabel);
    const cancel = h('button', { class: 'btn', type: 'button', onclick: () => close(null) }, 'Cancel');
    const dlg = h('dialog', { class: 'modal' },
      h('form', { method: 'dialog', onsubmit: (e) => e.preventDefault() },
        h('h2', {}, title), h('div', { class: 'dlg-body' }, body), ...optEls, typeRow, h('div', { class: 'dlg-actions' }, cancel, ok)));
    let done = false;
    const values = () => Object.fromEntries(Object.entries(boxes).map(([k, cb]) => [k, cb.checked]));
    function need() { return typeof typeToConfirm === 'function' ? typeToConfirm(values()) : typeToConfirm; }
    function sync() {
      const word = need();
      typeRow.style.display = word ? '' : 'none';
      if (word) typeRow.firstChild.textContent = `Type “${word}” to confirm`;
      ok.disabled = !!word && typed.value.trim().toLowerCase() !== String(word).toLowerCase();
    }
    function close(result) { if (done) return; done = true; dlg.close(); dlg.remove(); resolve(result); }
    dlg.addEventListener('cancel', (e) => { e.preventDefault(); close(null); });
    document.body.append(dlg);
    sync();
    dlg.showModal();
    (danger ? cancel : ok).focus();
  });
}

/** The two pages under the Experiment tab. */
export function subTabs(active) {
  return h('div', { class: 'subtabs' },
    h('a', { href: '#/experiment', class: active === 'configure' ? 'on' : '' }, 'Configure & run'),
    h('a', { href: '#/sessions', class: active === 'sessions' ? 'on' : '' }, 'Recorded sessions'));
}
