// IndexedDB wrapper. Three stores: sessions, trials, markers. There is deliberately no delete API:
// data is only ever marked invalid, never removed.

const DB_NAME = 'cortext-lab';
const VERSION = 1;
let dbp = null;

export function openDb() {
  if (dbp) return dbp;
  dbp = new Promise((resolve, reject) => {
    if (!('indexedDB' in window)) return reject(new Error('IndexedDB not available (private window?)'));
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      db.createObjectStore('sessions', { keyPath: 'session_id' });
      const t = db.createObjectStore('trials', { keyPath: 'key' });
      t.createIndex('session_id', 'session_id');
      const m = db.createObjectStore('markers', { keyPath: 'key' });
      m.createIndex('session_id', 'session_id');
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
  getAll: (store) => run(store, 'readonly', (os) => os.getAll()),
  getAllBySession: (store, sessionId) =>
    run(store, 'readonly', (os) => os.index('session_id').getAll(IDBKeyRange.only(sessionId))),
};

export const trialKey = (sessionId, trialId) => `${sessionId}:${trialId}`;
export const markerKey = (sessionId, seq) => `${sessionId}:${String(seq).padStart(9, '0')}`;
