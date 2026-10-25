#!/usr/bin/env python3
"""E2E: Setup → "P300 model" picker in headless Chrome, against the real bridge + simulated headset (fake_eeg.py).
  * lists the team library with participant, data, held-out accuracy, headset fit (wrong-channel model flagged)
  * "Use this" switches the running model live (no restart), Undo switches back, active/ follows
  * warns when the model's participant differs from the Setup participant, and when it was trained on simulated EEG
  * "Add a model file…" uploads into the library
  * a session then really predicts with the picked model (prediction.scorer == "model")
Run: python tests/e2e_models.py (needs playwright + chromium + pylsl)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_helpers import set_num, set_select, field  # noqa: E402
from test_models import make_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9700 + os.getpid() % 90
BASE = f"http://127.0.0.1:{PORT}"
SHOTS = os.path.join(ROOT, "tests", "screenshots")
os.makedirs(SHOTS, exist_ok=True)
FAKE = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"]


def phase(page, ph, timeout=40000):
    page.wait_for_function(f"window.cortext.runner && window.cortext.runner.phase === '{ph}'", timeout=timeout)


def running_text(page):
    return page.inner_text(".model-card .model-running")


def main():
    tmp = tempfile.mkdtemp(prefix="cortext-e2e-models-")
    lib, act = os.path.join(tmp, "library"), os.path.join(tmp, "active")
    os.makedirs(lib)
    os.makedirs(act)
    make_model(os.path.join(lib, "P01_20261025-0900_SIM.npz"), ["P01"], simulated=True, labels=FAKE)
    make_model(os.path.join(lib, "P02_20261025-0930.npz"), ["P02"], labels=FAKE)
    make_model(os.path.join(lib, "P03_32ch.npz"), ["P03"], labels=["T7", "T8", "Cz", "Oz"])
    upload_src = make_model(os.path.join(tmp, "P04_from_a_friend.npz"), ["P04"], labels=FAKE)
    env = {**os.environ, "CORTEXT_MODELS_DIR": tmp, "CORTEXT_DATASET_DIR": os.path.join(tmp, "ds")}
    fake = subprocess.Popen([sys.executable, os.path.join(ROOT, "eeg", "fake_eeg.py"), "--seed", "3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    errors = []
    try:
        for _ in range(150):
            try:
                if json.loads(urllib.request.urlopen(BASE + "/api/status", timeout=1).read())["eeg"].get("found"):
                    break
            except Exception:
                pass
            time.sleep(0.2)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1500, "height": 1000})
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" and "ERR_" not in m.text and "Failed to load resource" not in m.text else None)

            page.goto(BASE + "/#/setup")
            page.wait_for_selector(".model-card tr[data-id='library/P02_20261025-0930.npz']", timeout=20000)
            page.fill("input[placeholder='e.g. P01']", "P01")
            page.wait_for_function("document.querySelector('.model-card tr[data-id=\"library/P03_32ch.npz\"]')?.textContent.includes('wrong channels')", timeout=15000)
            assert "No trained model" in running_text(page), running_text(page)
            assert page.locator(".model-card tr[data-id='library/P03_32ch.npz'] button:text('Use this')").is_disabled()
            row = page.locator(".model-card tr[data-id='library/P02_20261025-0930.npz']")
            assert "P02" in row.inner_text() and "70% vs 25% chance" in row.inner_text() and "fits" in row.inner_text(), row.inner_text()

            # switch to P02 live: running line, mismatch warning, active/ holds it
            row.locator("text=Use this").click()
            page.wait_for_function("document.querySelector('.model-card .model-running').textContent.includes('P02_20261025-0930.npz')")
            page.wait_for_selector("text=but the participant above is P01")
            assert os.listdir(act) == ["P02_20261025-0930.npz"], os.listdir(act)
            st = json.loads(urllib.request.urlopen(BASE + "/api/status").read())["eeg"]["scorer"]
            assert st["active"] == "model" and st["model_file"].endswith("P02_20261025-0930.npz"), st

            # Undo goes back to "no trained model"
            page.locator(".toast .toast-act").click()
            page.wait_for_function("document.querySelector('.model-card .model-running').textContent.includes('No trained model')")
            assert [n for n in os.listdir(act) if n.endswith(".npz")] == []

            # pick P01 (the participant in the chair): no mismatch warning, but the SIMULATED warning
            page.locator(".model-card tr[data-id='library/P01_20261025-0900_SIM.npz'] >> text=Use this").click()
            page.wait_for_function("document.querySelector('.model-card .model-running').textContent.includes('P01_20261025-0900_SIM.npz')")
            page.wait_for_selector("text=trained on SIMULATED EEG")
            assert page.locator("text=but the participant above is").count() == 0
            page.locator(".model-card").screenshot(path=os.path.join(SHOTS, "40_model_picker.png"))

            # upload a friend's model file into the library
            page.locator(".model-card input[type=file]").set_input_files(upload_src)
            page.wait_for_selector(".model-card tr[data-id='library/P04_from_a_friend.npz']", timeout=15000)
            assert os.path.exists(os.path.join(lib, "P04_from_a_friend.npz"))

            # a session now predicts with the picked model
            page.click("#topbar nav a[data-nav=experiment]")
            page.wait_for_selector("text=12 items in 3 categories")
            set_select(page, "Presentation mode", "mcq_sequential")
            for label, val in {"Questions": "2", "Options per question": "3", "Prompt shown first (ms)": "300",
                               "Each option on screen (ms)": "500", "Gap between options (ms)": "200", "Times each option is shown": "1"}.items():
                set_num(page, label, val)
            page.click("text=Start session")
            phase(page, "ready")
            page.keyboard.press("Space")
            for _ in range(2):
                phase(page, "response")
                page.wait_for_function("window.cortext.runner.current && window.cortext.runner.current.prediction", timeout=20000)
                tr = page.evaluate("window.cortext.runner.current")
                pred = tr["prediction"]
                assert pred.get("scorer") == "model", pred
                page.locator(f".tile[data-id='{tr['true_answer_id']}']").dispatch_event("pointerdown")
                phase(page, "post")
                page.keyboard.press("Space")
            phase(page, "finished", timeout=30000)
            browser.close()
        bad = [e for e in errors if "favicon" not in e]
        assert not bad, bad
        print("models e2e OK: library listed (fit, accuracy, participant), live switch + Undo, participant/simulated warnings, "
              "upload, session predicted with the picked model. Screenshot: tests/screenshots/40_model_picker.png")
    finally:
        bridge.terminate()
        fake.terminate()
        bridge.wait(5)
        fake.wait(5)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
