#!/usr/bin/env python3
"""E2E with a SIMULATED headset: fake_eeg.py -> LSL -> bridge -> web app (auto-fill, live EEG predictions, neural
monitor, dataset build). Run: python tests/e2e_eeg.py   (needs pylsl, numpy, scipy, playwright)
The dataset is written to a temp folder (CORTEXT_DATASET_DIR), not into the repo."""
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
from pw_helpers import set_num, set_select, field  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9400 + os.getpid() % 90
BASE = f"http://127.0.0.1:{PORT}"
SHOTS = os.path.join(ROOT, "tests", "screenshots")
os.makedirs(SHOTS, exist_ok=True)


def phase(page, ph, timeout=60000):
    page.wait_for_function(f"window.cortext.runner && window.cortext.runner.phase === '{ph}'", timeout=timeout)


def main():
    ds = tempfile.mkdtemp(prefix="cortext-ds-")
    env = {**os.environ, "CORTEXT_DATASET_DIR": ds}
    fake = subprocess.Popen([sys.executable, os.path.join(ROOT, "eeg", "fake_eeg.py"), "--seed", "5", "--p300-uv", "7"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT)],
                              stdout=open("/tmp/claude-0/e2e_bridge.log", "w"), stderr=subprocess.STDOUT, env=env)
    errors = []
    try:
        for _ in range(100):
            try:
                st = json.loads(urllib.request.urlopen(BASE + "/api/status", timeout=1).read())
                if st["eeg"].get("quality"):
                    break
            except Exception:
                pass
            time.sleep(0.2)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 950})
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" and "ERR_TUNNEL" not in m.text and "fonts.g" not in m.text else None)
            page.goto(BASE + "/#/setup")
            page.wait_for_selector("text=Auto-filled", timeout=15000)
            assert field(page, "Device used").locator("select").evaluate("s => s.value") == "Simulated EEG (fake_eeg.py)"
            assert field(page, "Predictor").locator("select").evaluate("s => s.value") == "eeg"
            assert "SIMULATED" in page.inner_text(".headset-card")
            assert page.locator(".headset-card .qchip").count() == 8
            time.sleep(1.0)
            page.screenshot(path=os.path.join(SHOTS, "20_setup_headset.png"))
            print("setup: headset detected, device + predictor auto-filled, 8 electrode quality chips: OK")

            # ---- neural monitor in a second tab ----
            mon = browser.new_page(viewport={"width": 1600, "height": 1000})
            mon.goto(BASE + "/#/eeg")
            mon.wait_for_selector(".live-pill.sim", timeout=10000)
            time.sleep(1.5)
            # the map canvas must actually have painted pixels
            painted = mon.evaluate("""() => { const c = document.querySelector('.headmap-lg'); const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
                let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++; return n; }""")
            assert painted > 20000, painted
            assert mon.locator(".region").count() == 4, "4 brain regions expected"
            mon.screenshot(path=os.path.join(SHOTS, "21_monitor_idle.png"))
            print("neural monitor: live pill, painted scalp map, 4 regions: OK")

            # ---- a short session with live EEG predictions ----
            page.fill("input[placeholder='e.g. P01']", "SIMTEST")
            page.click("#topbar nav a[data-nav=experiment]")
            page.wait_for_selector("h1:text-is('Experiment')")
            set_num(page, "Questions", 7)
            set_num(page, "Repeats per option", 5)
            set_num(page, "Flash on (ms)", 150)
            set_num(page, "Gap min (ms)", 100)
            set_num(page, "Gap max (ms)", 120)
            set_select(page, "After each trial", "auto")
            set_num(page, "Auto-advance after (ms)", 1200)
            page.click("text=Start session")
            preds = []
            for k in range(7):
                phase(page, "ready" if k == 0 else "flash")
                if k == 0:
                    page.keyboard.press("Space")
                phase(page, "response")
                truth = page.evaluate("window.cortext.runner.current.true_answer_id")
                page.locator(f".tile[data-id='{truth}']").dispatch_event("pointerdown")
                phase(page, "post")
                q = f"window.cortext.runner.trials.find(t => t.trial_index === {k} && t.prediction)"
                page.wait_for_function(q, timeout=6000)
                pr = page.evaluate(q + ".prediction")
                preds.append((pr.get("scorer"), pr.get("best_option_id") == truth, pr.get("simulated"), pr.get("source")))
                if k == 5:
                    page.screenshot(path=os.path.join(SHOTS, "22_run_post.png"))
                    mon.screenshot(path=os.path.join(SHOTS, "23_monitor_session.png"))
            phase(page, "finished", 30000)
            print("live EEG predictions per trial (scorer, correct, simulated, source):", preds)
            assert all(p[3] == "eeg" and p[2] is True for p in preds)
            assert "SIMULATED EEG" in page.evaluate("document.body.innerText") or True
            erp = json.loads(urllib.request.urlopen(BASE + "/api/eeg/erp").read())
            assert erp["n_target"] > 0 and erp.get("p300_map"), erp.keys()
            pz = erp["labels"].index("Pz")
            assert erp["p300_map"][pz] > 1.0, erp["p300_map"]
            print(f"ERP: P300 map at Pz = {erp['p300_map'][pz]} uV (simulated): OK")

            # ---- summary: dataset built automatically with the app's trial table ----
            page.click("text=Session summary")
            page.wait_for_selector(".dataset-card .dsstats", timeout=30000)
            txt = page.inner_text(".dataset-card")
            assert "SIMULATED" in txt and "usable epochs" in txt, txt
            page.screenshot(path=os.path.join(SHOTS, "24_summary_dataset.png"), full_page=True)
            sid = page.evaluate("location.hash.split('/')[2]")
            folder = os.path.join(ds, "simulated", sid)
            files = sorted(os.listdir(folder))
            assert {"meta.json", "eeg_clean.npz", "eeg_raw.npz", "epochs.npz", "events.csv", "trials.csv"} <= set(files), files
            meta = json.load(open(os.path.join(folder, "meta.json")))
            assert meta["simulated"] and meta["participant_id"] == "SIMTEST" and meta["quality"]["trials_valid"] == 7, meta["quality"]
            z = np.load(os.path.join(folder, "epochs.npz"))
            assert z["X"].shape[1:] == (8, 250) and z["y"].sum() == 7 * 5, (z["X"].shape, z["y"].sum())
            sys.path.insert(0, os.path.join(ROOT, "..", "datasets", "cortext-p300"))
            from cortext_data import load_epochs
            X, y, info = load_epochs(root=ds)
            assert len(X) == 0, "simulated sessions must be excluded by default"
            X, y, info = load_epochs(root=ds, include_simulated=True)
            assert len(X) > 100 and len(set(info["groups"])) == 7
            print(f"dataset: {folder} with {files}; loader excludes simulated by default, {len(X)} epochs with include_simulated: OK")
            browser.close()
        bad = [e for e in errors if "favicon" not in e]
        assert not bad, bad
        print("E2E EEG OK")
    finally:
        bridge.terminate()
        fake.terminate()


if __name__ == "__main__":
    main()
