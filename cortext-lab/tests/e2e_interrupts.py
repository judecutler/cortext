#!/usr/bin/env python3
"""Failure-path E2E: pause mid-flash, hidden tab, bridge killed/restarted mid-session, reload keeps data.
Run: python tests/e2e_interrupts.py   (needs playwright + chromium)"""
import os
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_helpers import set_num, set_participant  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8900 + os.getpid() % 90
BASE = f"http://127.0.0.1:{PORT}"


def start_bridge():
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-lsl"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(60):
        try:
            urllib.request.urlopen(BASE + "/api/status", timeout=1).read()
            return p
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("bridge did not start")


def phase(page, ph, timeout=30000):
    page.wait_for_function(f"window.cortext.runner && window.cortext.runner.phase === '{ph}'", timeout=timeout)


def main():
    logdir = os.path.join(ROOT, "bridge", "logs")
    before = set(os.listdir(logdir)) if os.path.isdir(logdir) else set()
    bridge = start_bridge()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 850})
            errs = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            set_participant(page, BASE, "INT")
            page.wait_for_selector("text=11 items in 3 categories")
            for label, val in {"Questions": "2", "Options per question": "4", "Repeats per option": "6", "Flash on (ms)": "300",
                               "Gap min (ms)": "100", "Gap max (ms)": "150", "Prompt shown first (ms)": "300"}.items():
                set_num(page, label, val)
            page.click("text=Start session")
            phase(page, "ready")
            sid = page.evaluate("window.cortext.runner.session.session_id")

            # ---- pause mid-flash ----
            page.keyboard.press("Space"); phase(page, "flash")
            page.keyboard.press("p")
            phase(page, "paused")
            tr = page.evaluate("window.cortext.runner.trials.map(t => [t.trial_id, t.status, t.valid, t.invalid_reason])")
            assert tr == [[0, "aborted", False, "paused"]], tr
            assert page.evaluate("document.querySelectorAll('.tile.flash').length") == 0
            page.keyboard.press("p")
            phase(page, "ready")                          # waits for the operator after resuming
            assert page.evaluate("window.cortext.runner.current.trial_index") == 0   # same question comes back
            print("pause mid-flash: trial aborted+kept, requeued, resume waits for Start: OK")

            # ---- hidden tab mid-flash ----
            page.keyboard.press("Space"); phase(page, "flash")
            page.evaluate("Object.defineProperty(document, 'hidden', {configurable: true, get: () => true}); document.dispatchEvent(new Event('visibilitychange'))")
            phase(page, "ready")
            page.evaluate("delete document.hidden")
            tr = page.evaluate("window.cortext.runner.trials.map(t => [t.trial_id, t.status, t.invalid_reason])")
            assert tr[1] == [1, "aborted", "tab_hidden"], tr
            assert page.evaluate("document.querySelectorAll('.tile.flash').length") == 0
            print("hidden tab mid-flash: trial aborted as tab_hidden, flash off, waits for Start: OK")

            # ---- bridge dies mid-session; session continues; bridge returns ----
            page.keyboard.press("Space"); phase(page, "prompt")
            bridge.terminate(); bridge.wait(5)
            page.wait_for_function("window.cortext.bridge.connected === false", timeout=10000)
            phase(page, "response", timeout=40000)        # flashing completed with the bridge down
            page.keyboard.press("1"); phase(page, "post")
            unsent = page.evaluate("window.cortext.markers.recent.filter(m => m.event === 'flash_on' && !m.bridge_sent).length")
            assert unsent == 24, unsent                   # 4 options x 6 repeats, all kept locally but flagged
            print(f"bridge down: session continued, {unsent} flash markers flagged bridge_sent=false (kept in IndexedDB): OK")
            bridge = start_bridge()
            page.wait_for_function("window.cortext.bridge.connected === true", timeout=15000)
            print("bridge restarted: page reconnected automatically: OK")
            page.keyboard.press("Space")                  # Next: the following question starts immediately
            phase(page, "response", timeout=40000)
            page.keyboard.press("2"); phase(page, "post"); page.keyboard.press("Space"); phase(page, "finished")

            # ---- reload: everything still there ----
            time.sleep(0.6)
            page.goto(BASE + f"/#/summary/{sid}")
            page.reload()
            page.wait_for_selector("text=Session summary")
            txt = page.inner_text("#view")
            assert "Attempts (incl. redos)" in txt
            n_attempts = page.evaluate("""async (sid) => { const o = indexedDB.open('cortext-lab'); const db = await new Promise(r => o.onsuccess = () => r(o.result));
              return await new Promise(r => { const q = db.transaction('trials').objectStore('trials').index('session_id').getAll(sid); q.onsuccess = () => r(q.result.length); }); }""", sid)
            assert n_attempts == 4, n_attempts            # paused(aborted), tab_hidden(aborted), 2 completed
            print("reload: all 4 attempts (2 aborted, 2 completed) still in IndexedDB: OK")
            assert not errs, errs
            browser.close()
        print("E2E interrupts OK")
    finally:
        bridge.terminate()
        try:
            bridge.wait(3)
        except Exception:
            bridge.kill()
        if os.path.isdir(logdir):
            for f in set(os.listdir(logdir)) - before:
                os.remove(os.path.join(logdir, f))


if __name__ == "__main__":
    main()
