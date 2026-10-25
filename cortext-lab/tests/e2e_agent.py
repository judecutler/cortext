#!/usr/bin/env python3
"""E2E: the Agent tab's system map in headless Chrome, against the real bridge + simulated headset (fake_eeg.py).
  * the map shows input → filter layers → epoching → model → output → result, each filter block with the real signal
  * + adds a layer to the DRAFT (nothing live changes), the inspector shows before/after, Ctrl+Z / redo work
  * Apply to live switches the bridge's chain (filters/active follows); a default-chain model is then paused
    (broken wire, mismatch warning) and "Use this" on it switches the filters back to its chain
  * bypassing a layer, the filter library (save draft, delete + Undo), and the Experiment tab's Agent summary
  * old #/setup links land on the Experiment tab, which now holds participant + headset + storage
Run: python tests/e2e_agent.py (needs playwright + chromium + pylsl)."""
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
from test_models import make_model  # noqa: E402
import pw_helpers  # noqa: E402,F401  (old test defaults: flash mode + mock predictor)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9600 + os.getpid() % 90
BASE = f"http://127.0.0.1:{PORT}"
SHOTS = os.path.join(ROOT, "tests", "screenshots")
os.makedirs(SHOTS, exist_ok=True)
FAKE = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"]


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=5).read())


def wait_for(fn, timeout=15, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if fn():
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


def main():
    tmp = tempfile.mkdtemp(prefix="cortext-e2e-agent-")
    mdir, fdir = os.path.join(tmp, "models"), os.path.join(tmp, "filters")
    os.makedirs(os.path.join(mdir, "library"))
    os.makedirs(os.path.join(mdir, "active"))
    make_model(os.path.join(mdir, "library", "P01_default.npz"), ["P01"], labels=FAKE)
    env = {**os.environ, "CORTEXT_MODELS_DIR": mdir, "CORTEXT_FILTERS_DIR": fdir, "CORTEXT_DATASET_DIR": os.path.join(tmp, "ds")}
    fake = subprocess.Popen([sys.executable, os.path.join(ROOT, "eeg", "fake_eeg.py"), "--seed", "5"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-csv"], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    errors = []
    try:
        wait_for(lambda: get("/api/status")["eeg"].get("found"), 30, "the simulated headset")
        time.sleep(3)          # a few seconds of EEG in the ring buffer for the preview
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1500, "height": 1000})
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" and "ERR_" not in m.text and "Failed to load resource" not in m.text else None)

            # ---------- merged Experiment tab ----------
            page.goto(BASE + "/#/setup")
            page.wait_for_selector("h1:text-is('Experiment')")
            assert page.url.endswith("#/experiment"), page.url
            for sel in ("#participant", "#headset", "#agentSummary", "#storage", "text=Display hardware", "text=Presets", "text=Questions & image order"):
                assert page.locator(sel).count(), sel
            page.fill("input[placeholder='e.g. P01']", "P01")
            page.wait_for_function("document.querySelector('#agentSummary').textContent.includes('Band-pass 0.5-20 Hz')", timeout=10000)
            assert "null" not in page.inner_text("#agentSummary")

            # ---------- Agent tab: the map ----------
            page.click("#topbar nav a[data-nav=agent]")
            page.wait_for_selector("h1:text-is('Agent')")
            page.wait_for_selector(".lane .node[data-key='layer:0']")
            keys = page.eval_on_selector_all(".lane:not(.lane-off) .node", "ns => ns.map(n => n.dataset.key)")
            assert keys == ["input", "layer:0", "epoch", "model", "output", "result"], keys
            page.wait_for_function("document.querySelector(\".node[data-key='layer:0'] .node-std\").textContent.includes('µV')", timeout=15000)
            assert page.locator(".lane-off .node").count() == 4
            assert "Running:" in page.inner_text(".agent-banner") and "Draft" not in page.inner_text(".agent-banner")

            # + adds a notch in front of the band-pass, as a draft only
            page.locator(".ins[data-at='0'] .ins-btn").click()
            page.locator(".ins[data-at='0'] .ins-menu button", has_text="Mains notch").click()
            page.wait_for_selector(".node[data-key='layer:0'] >> text=Mains notch")
            page.wait_for_selector(".agent-banner >> text=Apply to live")
            assert get("/api/filters")["running"]["summary"] == ["Band-pass 0.5-20 Hz (order 2)"], "a draft must not change the live chain"
            page.click(".node[data-key='layer:0']")
            page.wait_for_selector(".inspector h2:text-is('Mains notch')")
            page.wait_for_function("document.querySelector('.cmp-legend').textContent.includes('after')", timeout=15000)
            assert "will pause" not in page.inner_text(".agent-banner")      # no model picked yet

            # undo / redo the draft edit
            page.keyboard.press("Control+z")
            page.wait_for_function("document.querySelectorAll('.lane .node.layer').length === 1")
            page.keyboard.press("Control+y")
            page.wait_for_function("document.querySelectorAll('.lane .node.layer').length === 2")

            # pick the default-chain model: it runs
            page.click(".node[data-key='model']")
            page.wait_for_selector(".model-card tr[data-id='library/P01_default.npz']")
            page.locator(".model-card tr[data-id='library/P01_default.npz'] button:text('Use this')").click()
            wait_for(lambda: get("/api/status")["eeg"]["scorer"]["active"] == "model", 10, "the model to run")
            page.wait_for_selector(".agent-banner >> text=Applying pauses P01_default.npz")
            assert page.locator(".wire.broken").count() == 1, "the draft differs from the model's chain: broken wire"

            # apply the draft: chain switches live, the model is paused
            page.locator(".agent-banner button", has_text="Apply to live").click()
            wait_for(lambda: get("/api/filters")["running"]["summary"][0] == "Notch 60 Hz", 10, "the notch chain to run")
            st = get("/api/status")["eeg"]["scorer"]
            assert st["active"] != "model" and "different filters" in st["model_filter_mismatch"], st
            assert os.listdir(os.path.join(fdir, "active")), "the running chain is saved to filters/active"
            page.wait_for_function("!document.querySelector('.agent-banner').textContent.includes('Apply to live')", timeout=10000)
            page.wait_for_selector(".model-card >> text=paused: it was trained with different filters")
            page.locator(".workzone").screenshot(path=os.path.join(SHOTS, "50_agent_map.png"))

            # Undo the apply: back to the default chain, model runs again
            page.locator(".toast[data-group='undo'] .toast-act").click()
            wait_for(lambda: get("/api/filters")["running"]["summary"] == ["Band-pass 0.5-20 Hz (order 2)"], 10, "undo of apply")
            wait_for(lambda: get("/api/status")["eeg"]["scorer"]["active"] == "model", 10, "model back after undo")

            # re-apply the notch, then "Use this" on the model brings its filters back
            page.evaluate("fetch('/api/filters/apply', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({spec: {name: 'Notch', layers: [{type: 'notch', freq: 60}, {type: 'bandpass', low: 0.5, high: 20, order: 2}]}})})")
            wait_for(lambda: get("/api/status")["eeg"]["scorer"]["model_filter_mismatch"], 10, "mismatch after re-apply")
            page.click(".node[data-key='model']")
            page.locator(".model-card button", has_text="No trained model").click()
            wait_for(lambda: not get("/api/status")["eeg"]["scorer"]["model_file"], 10, "no model")
            page.locator(".model-card tr[data-id='library/P01_default.npz'] button:text('Use this')").click()
            wait_for(lambda: get("/api/filters")["running"]["summary"] == ["Band-pass 0.5-20 Hz (order 2)"], 10, "model's chain to follow")
            wait_for(lambda: get("/api/status")["eeg"]["scorer"]["active"] == "model", 10, "model running with its chain")

            # bypass: a bypassed layer drops out of the signature
            page.wait_for_function("document.querySelectorAll('.lane .node.layer').length === 1", timeout=10000)
            page.locator(".ins[data-at='0'] .ins-btn").click()
            page.locator(".ins[data-at='0'] .ins-menu button", has_text="Mains notch").click()
            page.wait_for_selector(".agent-banner >> text=Apply to live")
            page.locator(".node[data-key='layer:0'] .chip-btn[aria-label='Bypass']").click()
            page.wait_for_selector(".node[data-key='layer:0'].bypassed")
            assert page.locator(".wire.broken").count() == 0, "with the notch bypassed the draft matches the model's chain again"

            # library: save the draft, delete it, Undo brings it back
            page.locator(".node[data-key='layer:0'] .chip-btn[aria-label='Switch on']").click()
            page.fill("#saveDraftName", "E2E notch")
            page.locator(".lib-save button", has_text="Save draft to library").click()
            wait_for(lambda: os.path.exists(os.path.join(fdir, "library", "e2e-notch.json")), 10, "saved chain file")
            saved = json.load(open(os.path.join(fdir, "library", "e2e-notch.json")))
            assert saved["description"] == "" and [l["type"] for l in saved["layers"]] == ["notch", "bandpass"], saved
            page.wait_for_selector(".lib-card tr >> text=E2E notch")
            page.locator(".lib-card tr", has_text="E2E notch").locator("button", has_text="Delete").click()
            wait_for(lambda: not os.path.exists(os.path.join(fdir, "library", "e2e-notch.json")), 10, "deleted chain file")
            page.locator(".toast[data-group='undo'] .toast-act").click()
            wait_for(lambda: os.path.exists(os.path.join(fdir, "library", "e2e-notch.json")), 10, "undo of delete")

            # other blocks open their panels
            for key, want in (("input", "Raw signal"), ("epoch", "Artifact limit"), ("output", "NOT calibrated"), ("result", "Voice & narrator"), ("off:clean", "zero-phase")):
                page.click(f".node[data-key='{key}']")
                page.wait_for_selector(f".inspector >> text={want}")
            page.screenshot(path=os.path.join(SHOTS, "51_agent_page.png"), full_page=True)

            # narrow screen: the map stacks vertically, no horizontal page scroll
            page.set_viewport_size({"width": 390, "height": 900})
            page.goto(BASE + "/#/agent")
            page.wait_for_selector(".lane .node[data-key='layer:0']")
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "horizontal scroll at phone width"
            browser.close()
        bad = [e for e in errors if "favicon" not in e]
        assert not bad, bad
        print("agent e2e OK: merged Experiment tab, system map with live per-layer signal, draft add/undo/redo, apply + undo, "
              "model paused on a filter mismatch and its chain restored by Use this, bypass, library save/delete/undo, "
              "phone width. Screenshots: tests/screenshots/50_agent_map.png, 51_agent_page.png")
    finally:
        bridge.terminate()
        fake.terminate()
        bridge.wait(5)
        fake.wait(5)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
