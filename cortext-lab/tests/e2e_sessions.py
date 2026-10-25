#!/usr/bin/env python3
"""E2E for the Sessions page: copies badges, buckets, real deletion (browser + bridge files), refusal while running,
cross-site POST refusal. Uses a temporary data folder. Run: python tests/e2e_sessions.py (needs playwright + chromium)"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_helpers import set_num, set_participant  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8800 + os.getpid() % 90
BASE = f"http://127.0.0.1:{PORT}"
LOGS = os.path.join(ROOT, "bridge", "logs")


def api(path, body=None, headers=None):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json", **(headers or {})})
    try:
        return json.load(urllib.request.urlopen(req, timeout=10))
    except urllib.error.HTTPError as e:
        return {"http": e.code, **json.loads(e.read() or b"{}")}


def seed_dataset(data, sid):
    d = os.path.join(data, "sessions", sid)
    os.makedirs(d)
    json.dump({"session_id": sid, "participant_id": "DEL", "device": "Other", "created_at": "2026-10-25T09:00:00", "quality": {}}, open(os.path.join(d, "meta.json"), "w"))
    np.savez_compressed(os.path.join(d, "eeg_raw.npz"), data=np.zeros((2, 4), np.float32))
    return d


def main():
    data = tempfile.mkdtemp(prefix="cortext-sess-")
    before = set(os.listdir(LOGS)) if os.path.isdir(LOGS) else set()
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-lsl"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            env={**os.environ, "CORTEXT_DATA_DIR": data})
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE + "/api/status", timeout=1).read(); break
            except Exception:
                time.sleep(0.1)
        with sync_playwright() as p:
            page = p.chromium.launch().new_page(viewport={"width": 1500, "height": 900})
            errs = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            set_participant(page, BASE, "DEL")
            page.wait_for_selector("text=11 items in 3 categories")
            for l, v in {"Questions": "1", "Repeats per option": "1", "Flash on (ms)": "150", "Gap min (ms)": "80", "Gap max (ms)": "100", "Prompt shown first (ms)": "200"}.items():
                set_num(page, l, v)
            page.click("text=Start session")
            page.wait_for_function("window.cortext.runner && window.cortext.runner.phase === 'ready'")
            sid = page.evaluate("window.cortext.runner.session.session_id")
            page.keyboard.press("Space")
            page.wait_for_function("window.cortext.runner.phase === 'flash'")
            ddir = seed_dataset(data, sid)

            # a running session cannot be deleted: the bridge refuses, and so does the page
            r = api("/api/session/delete", {"session_id": sid})
            assert r.get("http") == 409 and not r["ok"], r
            assert os.path.isdir(ddir)
            print("delete while running: refused by the bridge: OK")

            # cross-site POSTs are refused (a random web page must not be able to delete data)
            r = api("/api/session/delete", {"session_id": sid}, {"Origin": "http://evil.example"})
            assert r.get("http") == 403, r
            print("cross-site POST: refused: OK")

            page.wait_for_function("window.cortext.runner.phase === 'response'", timeout=30000)
            page.keyboard.press("1")
            page.wait_for_function("window.cortext.runner.phase === 'post'")
            page.keyboard.press("Space")
            page.wait_for_function("window.cortext.runner.phase === 'finished'", timeout=15000)
            time.sleep(0.5)
            logs = [f for f in os.listdir(LOGS) if sid in f]
            assert logs, "bridge should have written a marker log"

            # ---- Sessions page: badges + bucket ----
            page.click("#topbar nav a[data-nav=experiment]")
            page.click(".subtabs a:text-is('Recorded sessions')")
            page.wait_for_selector(f"td:has-text('{sid}')")
            row = page.locator("tr", has=page.locator(f"td:text-is('{sid}')"))
            on = row.locator(".loc.on").all_inner_texts()
            assert set(on) >= {"browser", "logs", "dataset"} and "hub" not in on, on
            row.locator("input.bucket-in").fill("test"); row.locator("input.bucket-in").press("Enter")
            page.wait_for_selector(".toast:has-text('→ test')")
            assert json.load(open(os.path.join(ddir, "meta.json")))["bucket"] == "test"
            page.wait_for_selector("select option[value=test]", state="attached")
            print("sessions page: copies badges + bucket label written to meta.json: OK")

            # ---- delete for real ----
            row = page.locator("tr", has=page.locator(f"td:text-is('{sid}')"))
            row.locator("button:text-is('Delete')").click()
            page.wait_for_selector("dialog.modal[open]")
            assert "delete this session?" in page.inner_text("dialog.modal").lower()
            page.click("dialog.modal button:text-is('Cancel')")
            assert os.path.isdir(ddir) and page.evaluate(f"window.indexedDB.databases().then(d => d.length)") >= 1
            row.locator("button:text-is('Delete')").click()
            page.click("dialog.modal button:text-is('Delete permanently')")
            try:
                page.wait_for_selector(".toast:has-text('Deleted')", timeout=8000)
            except Exception:
                print("TOASTS:", page.evaluate("[...document.querySelectorAll('.toast')].map(t => t.textContent)"), "ERRS:", errs)
                raise
            assert not os.path.exists(ddir), "dataset folder must be gone"
            assert not [f for f in os.listdir(LOGS) if sid in f], "bridge logs must be gone"
            left = page.evaluate("""async () => { const { db } = await import('/js/db.js');
              return [(await db.getAll('sessions')).length, (await db.getAll('trials')).length, (await db.getAll('markers')).length]; }""")
            assert left == [0, 0, 0], left
            assert not [x for x in api("/api/sessions")["sessions"] if x.get("session_id") == sid], "deleted session should be gone from the inventory"
            page.wait_for_selector("text=No sessions yet")
            print("delete: IndexedDB records, dataset folder and raw logs all really gone; Cancel keeps everything: OK")
            assert not errs, errs
        print("E2E sessions OK")
    finally:
        proc.terminate()
        for f in set(os.listdir(LOGS)) - before:
            try:
                os.remove(os.path.join(LOGS, f))
            except OSError:
                pass


if __name__ == "__main__":
    main()
