#!/usr/bin/env python3
"""End-to-end test in headless Chromium (needs: pip install playwright && playwright install chromium).

Runs a short real session through the UI against the real bridge and checks IndexedDB, the bridge CSV, hotkeys,
discard / redo, early clicks, export and review. Screenshots go to tests/screenshots/ (git-ignored).
Run: python tests/e2e.py
"""
import csv
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_helpers import set_num, set_select, set_participant  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8300 + os.getpid() % 500
BASE = f"http://127.0.0.1:{PORT}"
SHOTS = os.path.join(ROOT, "tests", "screenshots")
os.makedirs(SHOTS, exist_ok=True)


def wait_phase(page, phase, timeout=30000):
    page.wait_for_function(f"window.cortext.runner && window.cortext.runner.phase === '{phase}'", timeout=timeout)


def main():
    logdir = os.path.join(ROOT, "bridge", "logs")
    before = set(os.listdir(logdir)) if os.path.isdir(logdir) else set()
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    errors = []
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE + "/api/status", timeout=1).read()
                break
            except Exception:
                time.sleep(0.1)
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
            ctx = browser.new_context(viewport={"width": 1500, "height": 900})
            page = ctx.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" and "ERR_TUNNEL" not in m.text and "ERR_NAME" not in m.text else None)

            set_participant(page, BASE, "E2E")
            page.wait_for_selector("text=11 items in 3 categories")
            page.screenshot(path=os.path.join(SHOTS, "01_experiment.png"))
            # small, fast session
            for label, val in {"Questions": "3", "Repeats per option": "2", "Flash on (ms)": "200", "Gap min (ms)": "100",
                               "Gap max (ms)": "150", "Prompt shown first (ms)": "400"}.items():
                set_num(page, label, val)
            page.click("text=Start session")
            page.wait_for_selector(".stage")
            wait_phase(page, "ready")
            page.wait_for_function("document.querySelector('.dot.green') !== null")  # bridge connected indicator
            page.screenshot(path=os.path.join(SHOTS, "02_ready.png"))

            # ---- trial 1: early click during flashing, then answer correctly ----
            page.keyboard.press("Space")
            wait_phase(page, "flash")
            tile = page.locator(".tile").first
            tile.dispatch_event("pointerdown")            # early click -> recorded, not accepted as an answer
            page.wait_for_selector(".tile.flash")
            assert page.evaluate("document.querySelectorAll('.tile.flash').length") == 1, "exactly one tile lit at a time"
            page.screenshot(path=os.path.join(SHOTS, "03_flash.png"))
            wait_phase(page, "response")
            true_id = page.evaluate("window.cortext.runner.current.true_answer_id")
            page.locator(f".tile[data-id='{true_id}']").dispatch_event("pointerdown")
            wait_phase(page, "post")
            page.screenshot(path=os.path.join(SHOTS, "04_post.png"))
            assert "Correct" in page.inner_text(".post .result"), page.inner_text(".post .result")
            assert "EEG indicates" in page.inner_text(".post .pred")
            assert "MOCK" in page.inner_text(".post .pred")
            assert "null" not in page.inner_text(".post"), page.inner_text(".post")
            page.keyboard.press("m")                       # Researcher -> Demo, mid-session
            assert page.evaluate("document.body.dataset.mode") == "demo"
            assert not page.locator(".dev").is_visible()
            page.screenshot(path=os.path.join(SHOTS, "04b_demo_post.png"))
            page.keyboard.press("m")
            assert page.locator(".dev").is_visible()
            page.keyboard.press("Space")

            # ---- trial 2: wrong click via keyboard (letter), then Backspace discard, pick a reason ----
            wait_phase(page, "response")
            page.keyboard.press("a")
            wait_phase(page, "post")
            page.keyboard.press("Backspace")
            page.wait_for_selector("text=Trial marked INVALID")
            page.click("button:has-text('Distracted')")
            page.wait_for_selector("text=(distracted)")
            page.screenshot(path=os.path.join(SHOTS, "05_discarded.png"))
            page.keyboard.press("Space")

            # ---- trial 3: Redo mid-flash (aborts, re-queues), then complete the retry ----
            wait_phase(page, "flash")
            page.keyboard.press("r")
            page.wait_for_function("window.cortext.runner.trials.length >= 4")
            wait_phase(page, "flash")
            wait_phase(page, "response")
            page.keyboard.press("2")
            wait_phase(page, "post")
            page.keyboard.press("Space")
            wait_phase(page, "finished")
            page.screenshot(path=os.path.join(SHOTS, "06_finished.png"))

            session_id = page.evaluate("window.cortext.runner.session.session_id")
            time.sleep(0.6)  # marker flush interval
            data = page.evaluate("""async (sid) => {
              const open = indexedDB.open('cortext-lab');
              const db = await new Promise(r => open.onsuccess = () => r(open.result));
              const all = (s) => new Promise(r => { const q = db.transaction(s).objectStore(s).index('session_id').getAll(sid); q.onsuccess = () => r(q.result); });
              return { trials: await all('trials'), markers: await all('markers'),
                       session: await new Promise(r => { const q = db.transaction('sessions').objectStore('sessions').get(sid); q.onsuccess = () => r(q.result); }) };
            }""", session_id)
            trials, markers = data["trials"], sorted(data["markers"], key=lambda m: m["seq"])
            print(f"trials stored: {len(trials)}, markers stored: {len(markers)}, session status: {data['session']['status']}")

            # trial bookkeeping: t0 complete, t1 discarded (kept), t2 aborted by redo (kept), t3 retry complete
            by_id = {t["trial_id"]: t for t in trials}
            assert len(trials) == 4, [t["status"] for t in trials]
            assert by_id[0]["valid"] and by_id[0]["correct"] is True and len(by_id[0]["early_clicks"]) == 1
            assert not by_id[1]["valid"] and by_id[1]["invalid_reason"] == "distracted" and by_id[1]["clicked_answer_id"] is not None
            assert len(by_id[1]["validity_log"]) == 2   # unspecified, then distracted
            assert not by_id[2]["valid"] and by_id[2]["invalid_reason"] == "redo" and by_id[2]["status"] == "aborted"
            assert by_id[3]["valid"] and by_id[3]["attempt"] == 1 and by_id[3]["trial_index"] == by_id[2]["trial_index"]
            assert by_id[3]["clicked_answer_id"] is not None   # via key '2' = second position
            for t in (by_id[0], by_id[1], by_id[3]):
                assert len(t["flash_order"]) == 8, len(t["flash_order"])  # 4 options x 2 repeats
                ids = [f["option_id"] for f in t["flash_order"]]
                assert all(a != b for a, b in zip(ids, ids[1:]))
                assert all(abs(f["onset_err_ms"]) < 40 for f in t["flash_order"]), [f["onset_err_ms"] for f in t["flash_order"]]
            assert by_id[3]["options"][1]["id"] == by_id[3]["clicked_answer_id"]

            # marker contract
            fl = [m for m in markers if m["event"] == "flash_on"]
            assert len(fl) == 8 * 3 + len([m for m in markers if m["event"] == "flash_on" and m["trial_id"] == 2]), "flash_on count"
            for m in fl:
                for k in ("t_ms", "session_id", "trial_id", "option_id", "category", "is_target", "repeat_index"):
                    assert m[k] is not None, (k, m)
                assert m["session_id"] == session_id
            for tid in (0, 1, 3):
                tm = [m for m in fl if m["trial_id"] == tid]
                assert sum(1 for m in tm if m["is_target"]) == 2 and len(tm) == 8
                assert all(m["is_target"] == (m["option_id"] == by_id[tid]["true_answer_id"]) for m in tm)
            assert all(m["bridge_sent"] for m in fl), "some flash markers were not delivered to the bridge"
            assert all(markers[i]["seq"] == i for i in range(len(markers))), "marker seq has gaps"
            events = {m["event"] for m in markers}
            for need in ("session_start", "prompt_on", "flash_phase_start", "flash_on", "flash_off", "flash_phase_end", "response_cue", "click", "early_click", "trial_aborted", "trial_redo", "session_end"):
                assert need in events, f"missing marker event {need}: {sorted(events)}"

            # bridge CSV has every marker, with transit estimates
            time.sleep(0.5)
            new = [f for f in os.listdir(logdir) if f not in before and f.startswith("markers_E2E")]
            assert new, os.listdir(logdir)
            rows = list(csv.DictReader(open(os.path.join(logdir, new[0]))))
            assert len(rows) == len(markers), (len(rows), len(markers))
            transit = [float(r["est_transit_ms"]) for r in rows if r["event"] == "flash_on"]
            print(f"bridge CSV rows: {len(rows)}; flash_on transit ms: min {min(transit):.2f} max {max(transit):.2f}")
            assert max(transit) < 50 and min(transit) > -5

            # ---- summary / review / sessions / export ----
            page.goto(BASE + f"/#/summary/{session_id}")
            page.wait_for_selector("text=Session summary")
            txt = page.inner_text("#view")
            assert "Attempts (incl. redos)" in txt and "Discard rate" in txt
            page.screenshot(path=os.path.join(SHOTS, "07_summary.png"))
            page.goto(BASE + f"/#/review/{session_id}")
            page.wait_for_selector("text=Review trials")
            page.select_option("tbody tr >> nth=0 >> select", "misclick")   # mark a previously valid trial invalid
            page.wait_for_function("document.querySelectorAll('tbody tr.invalid').length >= 3")
            page.screenshot(path=os.path.join(SHOTS, "08_review.png"))
            with page.expect_download() as dl:
                page.click("text=Export session (zip)")
            zpath = os.path.join(SHOTS, "export.zip")
            dl.value.save_as(zpath)
            import zipfile
            z = zipfile.ZipFile(zpath)
            assert z.testzip() is None and sorted(z.namelist()) == ["markers.csv", "model_feedback.csv", "session.json", "trials.csv"], z.namelist()
            tr = list(csv.DictReader(z.read("trials.csv").decode().splitlines()))
            assert len(tr) == 4 and tr[0]["valid"] == "false" and tr[0]["invalid_reason"] == "misclick"
            mk = list(csv.DictReader(z.read("markers.csv").decode().splitlines()))
            assert len(mk) == len(markers)
            sj = json.loads(z.read("session.json"))
            assert sj["session"]["participant_id"] == "E2E" and len(sj["trials"]) == 4
            page.goto(BASE + "/#/sessions")
            page.wait_for_selector("text=Export ALL sessions")

            # ---- timing test page ----
            tp = ctx.new_page()
            tp.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            tp.goto(BASE + "/#/timing")
            tp.wait_for_selector("#conn .dot.green")
            tp.fill("#interval", "300"); tp.fill("#ondur", "100"); tp.fill("#count", "20")
            tp.click("#go")
            tp.wait_for_selector("text=Verdicts", timeout=30000)
            tp.screenshot(path=os.path.join(SHOTS, "10_timing.png"), full_page=True)
            ttxt = tp.inner_text("#results")
            print("timing page:", " | ".join(l for l in ttxt.splitlines() if "PASS" in l or "CHECK" in l)[:300])
            assert "markers sent 20/20" in ttxt, ttxt
            time.sleep(0.5)
            tcsv = [f for f in os.listdir(logdir) if f not in before and f.startswith("markers_timing-test")]
            assert tcsv, "no timing CSV on the bridge"
            trows = [r for r in csv.DictReader(open(os.path.join(logdir, tcsv[0]))) if r["event"] == "timing_flash"]
            assert len(trows) == 20, len(trows)
            tp.close()

            # ---- demo mode ----
            page.goto(BASE + "/#/setup")
            page.wait_for_selector("h1:text-is('Setup')")
            page.keyboard.press("m")
            assert page.evaluate("document.body.dataset.mode") == "demo"
            page.screenshot(path=os.path.join(SHOTS, "09_demo_setup.png"))
            page.keyboard.press("m")
            browser.close()
        bad = [e for e in errors if "favicon" not in e and "WebSocket" not in e]
        assert not bad, bad
        print("E2E OK")
    finally:
        proc.terminate()
        try:
            proc.wait(3)
        except Exception:
            proc.kill()
        if os.path.isdir(logdir):
            for f in set(os.listdir(logdir)) - before:
                os.remove(os.path.join(logdir, f))


if __name__ == "__main__":
    main()
