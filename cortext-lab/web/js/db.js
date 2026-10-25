// IndexedDB wrapper. Three stores: sessions, trials, markers. Trials are never deleted, only flagged invalid.
// Whole sessions can be deleted on purpose with deleteSession() (the Sessions page asks first).

const DB_NAME = 'cortext-lab';
const VERSION = 2; // v2 adds 'stimulus_sets' (uploaded stimulus files); v1 data is kept as is
let dbp = null;

export function openDb() {
  if (dbp) return dbp;
  dbp = new Promise((resolve, reject) => {
    if (!('indexedDB' in window)) return reject(new Error('IndexedDB not available (private window?)'));
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('sessions')) db.createObjectStore('sessions', { keyPath: 'session_id' });
      if (!db.objectStoreNames.contains('trials')) {
        const t = db.createObjectStore('trials', { keyPath: 'key' });
        t.createIndex('session_id', 'session_id');
      }
      if (!db.objectStoreNames.contains('markers')) {
        const m = db.createObjectStore('markers', { keyPath: 'key' });
        m.createIndex('session_id', 'session_id');
      }
      if (!db.objectStoreNames.contains('stimulus_sets')) db.createObjectStore('stimulus_sets', { keyPath: 'name' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});
  return dbp;
}

function run(store, mode, fn) {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const tx = db.transaction(store, mode);
        const os = tx.objectStore(store);
        let out;
        try { out = fn(os); } catch (e) { reject(e); return; }
        tx.oncomplete = () => resolve(out && 'result' in out ? out.result : out);
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error);
      })
  );
}

export const db = {
  put: (store, obj) => run(store, 'readwrite', (os) => os.put(obj)),
  bulkPut: (store, arr) => run(store, 'readwrite', (os) => { arr.forEach((o) => os.put(o)); }),
  get: (store, key) => run(store, 'readonly', (os) => os.get(key)),
  delete: (store, key) => run(store, 'readwrite', (os) => os.delete(key)),
  getAll: (store) => run(store, 'readonly', (os) => os.getAll()),
  getAllBySession: (store, sessionId) =>
    run(store, 'readonly', (os) => os.index('session_id').getAll(IDBKeyRange.only(sessionId))),
};

/** Really delete one session: its record, every trial and every marker. */
export function deleteSession(sessionId) {
  return openDb().then((d) => new Promise((resolve, reject) => {
    const tx = d.transaction(['sessions', 'trials', 'markers'], 'readwrite');
    tx.objectStore('sessions').delete(sessionId);
    for (const name of ['trials', 'markers']) {
      const store = tx.objectStore(name);
      const req = store.index('session_id').openKeyCursor(IDBKeyRange.only(sessionId));
      req.onsuccess = () => { const c = req.result; if (c) { store.delete(c.primaryKey); c.continue(); } };
    }
    tx.oncomplete = () => resolve();
    tx.onerror = tx.onabort = () => reject(tx.error);
  }));
}

db.deleteSession = deleteSession;

export const trialKey = (sessionId, trialId) => `${sessionId}:${trialId}`;
export const markerKey = (sessionId, seq) => `${sessionId}:${String(seq).padStart(9, '0')}`;
