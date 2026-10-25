#!/usr/bin/env python3
"""E2E: stimulus upload (folder, and config + images) and the one-at-a-time ("sequential") mode with model probabilities.
Run: python tests/e2e_sequential.py   (needs playwright + chromium; screenshots in tests/screenshots/)"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_helpers import set_num, set_select, set_participant  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9100 + os.getpid() % 90
BASE = f"http://127.0.0.1:{PORT}"
SHOTS = os.path.join(ROOT, "tests", "screenshots")
os.makedirs(SHOTS, exist_ok=True)
STIM = os.path.join(ROOT, "web", "stimuli")


def phase(page, ph, timeout=30000):
    page.wait_for_function(f"window.cortext.runner && window.cortext.runner.phase === '{ph}'", timeout=timeout)


def main():
    logdir = os.path.join(ROOT, "bridge", "logs")
    before = set(os.listdir(logdir)) if os.path.isdir(logdir) else set()
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-lsl"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tmp = tempfile.mkdtemp(prefix="stimup-")
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE + "/api/status", timeout=1).read()
                break
            except Exception:
                time.sleep(0.1)

        # ---- files to upload: (a) a folder: pets/animals/*.svg + pets/objects/*.svg ; (b) config.json + images ----
        folder = os.path.join(tmp, "pets")
        for cat, names in {"animals": ["dog", "cat", "bird"], "objects": ["cup", "house"]}.items():
            os.makedirs(os.path.join(folder, cat))
            for n in names:
                shutil.copy(os.path.join(STIM, f"{n}.svg"), os.path.join(folder, cat, f"{n}.svg"))
        cfg_dir = os.path.join(tmp, "cfg")
        os.makedirs(cfg_dir)
        conf = {"items": [
            {"id": "fish", "category": "animal", "label": "fish", "text": "Fish", "image": "somewhere/fish.svg"},
            {"id": "frog", "category": "animal", "label": "frog", "text": "Frog", "image": "frog.svg"},
            {"id": "circle", "category": "shape", "label": "circle", "text": "Circle", "image": "circle.svg"},
            {"id": "square", "category": "shape", "label": "square", "text": "Square", "image": "missing_square.svg"}]}
        json.dump(conf, open(os.path.join(cfg_dir, "my_stimuli.json"), "w"))
        for n in ("fish", "frog", "circle"):
            shutil.copy(os.path.join(STIM, f"{n}.svg"), os.path.join(cfg_dir, f"{n}.svg"))

        errors = []
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1500, "height": 900})
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" and "ERR_TUNNEL" not in m.text and "ERR_NAME" not in m.text else None)
            set_participant(page, BASE, "SEQ")
            page.wait_for_selector("text=12 items in 3 categories")

            # ---- upload a folder ----
            page.locator("input[webkitdirectory]").set_input_files(folder)
            page.wait_for_selector("text=Saved \"pets\"")
            page.wait_for_selector("text=5 items in 2 categories (animals, objects)")
            assert page.locator(".field:has(label:text-is('Use')) select").evaluate("s => s.value") == "set:pets"
            # ---- upload config + images (one image deliberately missing) ----
            page.fill("input[placeholder='name for this set (optional)']", "cfgset")
            page.locator(".card:has(h2:text-is('Stimulus set')) input[type=file]:not([webkitdirectory])").set_input_files(
                [os.path.join(cfg_dir, f) for f in os.listdir(cfg_dir)])
            page.wait_for_selector("text=Saved \"cfgset\"")
            warn = page.inner_text(".card:has(h2:text-is('Stimulus set'))")
            assert "missing_square.svg" in warn and "was not uploaded" in warn, warn
            page.screenshot(path=os.path.join(SHOTS, "11_upload.png"), full_page=True)

            # ---- reload: uploaded sets persist and can be re-selected ----
            page.reload()
            page.wait_for_selector("h1:text-is('Experiment')")
            opts = page.locator(".field:has(label:text-is('Use')) select option").all_inner_texts()
            assert any("Uploaded: pets (5 items)" in o for o in opts) and any("Uploaded: cfgset (4 items)" in o for o in opts), opts
            page.locator(".field:has(label:text-is('Use')) select").select_option("set:pets")
            page.wait_for_selector("text=5 items in 2 categories")

            # ---- configure the sequential mode ----
            set_select(page, "Presentation mode", "mcq_sequential")
            assert page.locator("text=Flash timing (flash mode)").is_hidden() and page.locator("text=One-at-a-time timing").is_visible()
            for label, val in {"Questions": "2", "Options per question": "4", "Prompt shown first (ms)": "500",
                               "Each option on screen (ms)": "300", "Gap between options (ms)": "150", "Times each option is shown": "2"}.items():
                set_num(page, label, val)
            page.screenshot(path=os.path.join(SHOTS, "12_setup_sequential.png"), full_page=True)
            page.click("text=Start session")
            phase(page, "ready")
            sid = page.evaluate("window.cortext.runner.session.session_id")

            # ---- trial 1: question alone, then one option at a time, tiles hidden, then probabilities ----
            page.keyboard.press("Space")
            phase(page, "prompt")
            page.wait_for_selector(".prompt.show")
            assert page.evaluate("getComputedStyle(document.querySelector('.tiles')).visibility") == "hidden", "tiles must be hidden before the reveal"
            q = page.inner_text(".prompt")
            page.wait_for_selector(".pstim.on")
            assert page.evaluate("document.querySelectorAll('.pstim.on').length") == 1
            assert page.inner_text(".prompt") == q and page.locator(".prompt.show").is_visible(), "question must stay on screen"
            assert page.evaluate("getComputedStyle(document.querySelector('.tiles')).visibility") == "hidden"
            page.screenshot(path=os.path.join(SHOTS, "13_seq_stimulus.png"))
            seen = set()
            t0 = time.time()
            while page.evaluate("window.cortext.runner.phase") == "flash" and time.time() - t0 < 20:
                n_on = page.evaluate("document.querySelectorAll('.pstim.on').length")
                assert n_on <= 1, "never more than one option on screen at a time"
                txt = page.evaluate("(document.querySelector('.pstim.on .ptext')||{}).textContent || ''")
                if txt:
                    seen.add(txt)
                time.sleep(0.02)
            assert len(seen) == 4, f"all 4 options should have been shown, saw {seen}"
            phase(page, "response")
            assert page.evaluate("getComputedStyle(document.querySelector('.tiles')).visibility") == "visible"
            assert page.evaluate("getComputedStyle(document.querySelector('.presenter')).display") == "none"
            pct = [int(x.rstrip("%")) for x in page.locator(".tile .probtext").all_inner_texts()]
            assert len(pct) == 4 and 96 <= sum(pct) <= 104, pct
            assert "MOCK" in page.inner_text(".subprompt"), page.inner_text(".subprompt")
            page.screenshot(path=os.path.join(SHOTS, "14_seq_choose.png"))
            true_id = page.evaluate("window.cortext.runner.current.true_answer_id")
            page.locator(f".tile[data-id='{true_id}']").dispatch_event("pointerdown")
            phase(page, "post")
            page.screenshot(path=os.path.join(SHOTS, "15_seq_post.png"))
            assert "Correct" in page.inner_text(".post .result")
            page.keyboard.press("Space")

            # ---- trial 2: wrong click by keyboard (position A) ----
            phase(page, "response", timeout=40000)
            page.keyboard.press("a")
            phase(page, "post")
            page.keyboard.press("Space")
            phase(page, "finished")
            time.sleep(0.6)

            data = page.evaluate("""async (sid) => {
              const open = indexedDB.open('cortext-lab'); const db = await new Promise(r => open.onsuccess = () => r(open.result));
              const all = (s) => new Promise(r => { const q = db.transaction(s).objectStore(s).index('session_id').getAll(sid); q.onsuccess = () => r(q.result); });
              return { trials: await all('trials'), markers: await all('markers'),
                       session: await new Promise(r => { const q = db.transaction('sessions').objectStore('sessions').get(sid); q.onsuccess = () => r(q.result); }) };
            }""", sid)
            trials = sorted(data["trials"], key=lambda t: t["trial_id"])
            markers = sorted(data["markers"], key=lambda m: m["seq"])
            sess = data["session"]
            assert sess["stimuli_source"] == {"type": "upload", "name": "pets"}, sess["stimuli_source"]
            snap = sess["stimuli_snapshot"]
            assert sorted(i["id"] for i in snap["items"]) == ["bird", "cat", "cup", "dog", "house"]
            assert not any("blob:" in (i["image"] or "") for i in snap["items"]), "snapshot must hold file names, not blob URLs"
            assert len(trials) == 2 and all(t["valid"] and t["type"] == "mcq_sequential" for t in trials)
            for t in trials:
                assert len(t["flash_order"]) == 8, len(t["flash_order"])        # 4 options x 2 repeats
                ids = [f["option_id"] for f in t["flash_order"]]
                assert all(a != b for a, b in zip(ids, ids[1:])), ids
                assert t["flash_params"] == {"on_ms": 300, "off_min_ms": 150, "off_max_ms": 150, "repeats": 2}, t["flash_params"]
                md = t["model_display"]
                assert md and md["source"] == "mock" and abs(sum(md["probabilities"].values()) - 1) < 1e-6
                assert all(abs(f["onset_err_ms"]) < 40 and abs(f["dur_err_ms"]) < 40 for f in t["flash_order"])
            assert trials[0]["correct"] is True and trials[1]["options"][0]["id"] == trials[1]["clicked_answer_id"]
            fl = [m for m in markers if m["event"] == "flash_on"]
            assert len(fl) == 16 and all((m["extra"] or {}).get("mode") == "sequential" for m in fl)
            for m in fl:
                for k in ("session_id", "trial_id", "option_id", "category", "is_target", "repeat_index", "t_ms"):
                    assert m[k] is not None, (k, m)
            cues = [m for m in markers if m["event"] == "response_cue"]
            assert len(cues) == 2 and all(c["extra"]["model_probs_shown"] for c in cues)
            clicks = [m for m in markers if m["event"] == "click"]
            assert len(clicks) == 2 and all(c["extra"]["model_shown_before_click"] for c in clicks)
            # the model's numbers are shown only AFTER the last stimulus offset (EEG window is not contaminated)
            for t in trials:
                last_off = max(f["frame_off_ms"] for f in t["flash_order"])
                shown_wall = t["model_display"]["shown_wall_ms"]
                end_wall = last_off + sess["time_origin_wall_ms"]
                assert shown_wall >= end_wall - 5, (shown_wall - end_wall)
            assert all(m["bridge_sent"] for m in fl)

            # ---- export: model_feedback.csv ----
            page.goto(BASE + f"/#/summary/{sid}")
            page.wait_for_selector("text=Session summary")
            assert "anchored on the model" in page.inner_text("#view")
            with page.expect_download() as dl:
                page.click("text=Export session (zip)")
            zpath = os.path.join(tmp, "x.zip")
            dl.value.save_as(zpath)
            z = zipfile.ZipFile(zpath)
            assert sorted(z.namelist()) == ["markers.csv", "model_feedback.csv", "session.json", "trials.csv"], z.namelist()
            fb = list(csv.DictReader(z.read("model_feedback.csv").decode().splitlines()))
            assert len(fb) == 8, len(fb)                                          # 2 trials x 4 options
            for tid in {r["trial_id"] for r in fb}:
                rs = [r for r in fb if r["trial_id"] == tid]
                assert abs(sum(float(r["model_probability"]) for r in rs) - 1) < 1e-6
                assert sum(r["is_true_answer"] == "true" for r in rs) == 1 and sum(r["is_clicked"] == "true" for r in rs) == 1
                assert all(r["model_shown_before_click"] == "true" and r["presentation_mode"] == "mcq_sequential" for r in rs)
            tr = list(csv.DictReader(z.read("trials.csv").decode().splitlines()))
            assert all(r["model_shown_before_click"] == "true" and r["model_probabilities"].startswith("{") for r in tr)
            browser.close()
        bad = [e for e in errors if "favicon" not in e and "WebSocket" not in e]
        assert not bad, bad
        print("sequential trials: question stays, one option at a time, tiles revealed afterwards with probabilities: OK")
        print("upload (folder + config/images + warnings + persistence + snapshot) and model_feedback.csv: OK")
        print("E2E sequential OK")
    finally:
        bridge.terminate()
        try:
            bridge.wait(3)
        except Exception:
            bridge.kill()
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.isdir(logdir):
            for f in set(os.listdir(logdir)) - before:
                os.remove(os.path.join(logdir, f))


if __name__ == "__main__":
    main()
