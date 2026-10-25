#!/usr/bin/env python3
"""E2E: the hackathon demo exactly as it ships, from a FRESH browser (no saved settings) and the repo's models/active/.

    python tests/e2e_showtime.py              # simulated headset with a 5 uV P300 (default)
    python tests/e2e_showtime.py --p300-uv 0  # control: no P300 in the signal (should sit near chance, modest %)
    python tests/e2e_showtime.py --muse       # the same demo on a simulated Muse (4 ch + AUX, 256 Hz): the bridge
                                              # must swap in the Muse p3000 function and its filter chain by itself
    python tests/e2e_showtime.py --p300-uv 0 --sim-p300 replace
                                              # "broken headset" (no P300 in its signal): the decoder scores the
                                              # simulated P300 instead; also flips the Headset card's Decoder input

Checks the defaults are the demo flow (one at a time x3, 5 s question read aloud, Live EEG, narrator before the click),
that the running model is the p3000 function from the first trial (no calibration), that each trial goes question ->
pictures -> narrated guess with the model's own percent and the participant's display name -> click, and that no page
errors happen. Prints the hit rate, stated confidence and seconds per question. SIMULATED EEG: this proves the wiring
and the timing, not that it works on a brain. Needs pylsl, numpy, scipy, playwright + chromium.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

os.environ["CORTEXT_E2E_FRESH_DEFAULTS"] = "1"   # test the real defaults, not the old ones pw_helpers seeds
from playwright.sync_api import sync_playwright  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_helpers import set_num  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9700 + os.getpid() % 200
BASE = f"http://127.0.0.1:{PORT}"
SHOTS = os.path.join(ROOT, "tests", "screenshots")
os.makedirs(SHOTS, exist_ok=True)


def phase(page, ph, timeout=60000):
    page.wait_for_function(f"window.cortext.runner && window.cortext.runner.phase === '{ph}'", timeout=timeout)


def report(rows, a):
    hits = sum(r["correct"] for r in rows)
    confs = [r["conf"] for r in rows]
    print(f"SIMULATED {a.p300_uv} uV P300{f' + decoder sim-p300 {a.sim_p300}' if a.sim_p300 else ''}, p3000 function, {a.trials} questions: {hits}/{a.trials} correct (chance {a.trials / 4:.0f}), "
          f"stated confidence mean {sum(confs) / len(confs):.0%}, max {max(confs):.0%}")
    print("seconds per question (prompt + read-aloud + 12 pictures + scoring + narration, operator click instant):",
          [r["s"] for r in rows], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p300-uv", type=float, default=5.0)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--muse", action="store_true", help="simulate a Muse headband instead of the 8-channel Unicorn")
    ap.add_argument("--sim-p300", choices=["off", "overlay", "replace"], default=None, help="bridge --sim-p300")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="cortext-show-")
    env = {**os.environ, "CORTEXT_DATASET_DIR": os.path.join(tmp, "ds"), "CORTEXT_DATA_DIR": os.path.join(tmp, "data"),
           "CORTEXT_BRIDGE_LOG_DIR": os.path.join(tmp, "logs")}
    fake = subprocess.Popen([sys.executable, os.path.join(ROOT, "eeg", "fake_eeg.py"), "--seed", "11", "--p300-uv", str(a.p300_uv)]
                            + (["--muse"] if a.muse else []),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT)]
                              + (["--sim-p300", a.sim_p300] if a.sim_p300 else []), env=env,
                              stdout=open(os.path.join(tmp, "bridge.log"), "w"), stderr=subprocess.STDOUT)
    errors = []
    try:
        for _ in range(150):
            try:
                st = json.loads(urllib.request.urlopen(BASE + "/api/status", timeout=1).read())
                if st["eeg"].get("quality"):
                    break
            except Exception:
                pass
            time.sleep(0.2)
        sc = st["eeg"]["scorer"]
        model_name = "p3000_function_muse.npz" if a.muse else "p3000_function.npz"
        assert sc["active"] == "model" and sc["model_file"].endswith(model_name), sc
        assert not sc["model_error"] and not sc["model_filter_mismatch"], sc
        if a.muse:
            assert st["eeg"]["labels"] == ["TP9", "AF7", "AF8", "TP10"] and st["eeg"]["dropped_channels"] == ["Right AUX"], st["eeg"]
        print(f"bridge: running model {os.path.basename(sc['model_file'])} from models/active/ before any trial: OK")

        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
            page = browser.new_page(viewport={"width": 1600, "height": 950})
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            # "Failed to load resource" console lines carry no URL, so failed requests are checked by URL instead.
            # /api/tts may fail where the bridge has no working speech engine; the page then uses the browser's voice.
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                    if m.type == "error" and not re.search(r"ERR_TUNNEL|fonts\.g|ERR_NAME|net::|Failed to load resource", m.text) else None)
            page.on("response", lambda r: errors.append(f"HTTP {r.status} {r.url}")
                    if r.status >= 400 and "/api/tts" not in r.url else None)
            page.goto(BASE + "/#/experiment")
            page.wait_for_selector("h1:text-is('Experiment')")
            s = page.evaluate("import('./js/settings.js').then(m => m.loadSettings())")
            want = {"trial_type": "mcq_sequential", "prompt_ms": 5000, "seq_repeats": 3, "seq_on_ms": 1000, "predictor_mode": "eeg",
                    "speak_question": True, "narrate": "before_click", "show_model_probs": True, "order_mode": "random"}
            assert {k: s[k] for k in want} == want, {k: s[k] for k in want}
            print("fresh browser defaults = the demo flow:", want)

            if a.sim_p300:      # the operator control on the Headset card mirrors the bridge, and switches it
                sel = page.locator("#headset select.sim-mode")
                page.wait_for_function(f"document.querySelector('#headset select.sim-mode')?.value === '{a.sim_p300}'", timeout=15000)
                sel.select_option("overlay")
                page.wait_for_function("fetch('/api/status').then(r => r.json()).then(s => s.eeg.sim_p300.mode === 'overlay')", timeout=5000)
                sel.select_option(a.sim_p300)
                page.wait_for_function(f"fetch('/api/status').then(r => r.json()).then(s => s.eeg.sim_p300.mode === '{a.sim_p300}')", timeout=5000)
                print(f"Headset card Decoder input shows and switches the bridge ({a.sim_p300} -> overlay -> {a.sim_p300}): OK")
            page.fill("input[placeholder='e.g. P01']", "P01")
            page.fill("input[placeholder='e.g. Jude (optional)']", "Jude")
            page.wait_for_selector("#agentSummary >> text=Live EEG")
            page.wait_for_selector(f"#agentSummary >> text={model_name}", timeout=15000)   # once the bridge status arrives
            set_num(page, "Questions", a.trials)
            page.click("text=Start session")
            phase(page, "ready")
            page.keyboard.press("m")                     # demo view
            page.keyboard.press("Space")
            rows = []
            for k in range(a.trials):
                t0 = time.time()
                phase(page, "response", 90000)
                tr = page.evaluate("window.cortext.runner.current")
                pred = tr.get("prediction") or {}
                truth = tr["true_answer_id"]
                # the narrator speaks before the click: wait for its caption (template line, offline)
                page.wait_for_function("window.cortext.runner.current.narration", timeout=20000)
                narr = page.evaluate("window.cortext.runner.current.narration")
                cap = page.inner_text(".narrator")
                probs = pred.get("probabilities") or {}
                best = pred.get("best_option_id")
                pct = round(100 * probs.get(best, 0))
                assert pred.get("source") == "eeg" and pred.get("scorer") == "model", pred
                assert pred.get("sim_p300") == (a.sim_p300 or "off"), pred
                assert f"{pct}%" in cap and "Simulated EEG" in cap and "mock" not in cap, (cap, pct)   # real headset: "Based on Jude's brain waves..." (voice.test.mjs)
                assert narr.get("simulated") is True
                assert "Jude" not in narr["text"], narr    # the display name is never stored with the trial
                stray = re.findall(r".{0,30}\b(?:null|undefined|NaN)\b.{0,30}", page.inner_text("body"))
                assert not stray, stray
                if k == 0:
                    page.screenshot(path=os.path.join(SHOTS, "40_showtime_reveal.png"))
                page.locator(f".tile[data-id='{truth}']").dispatch_event("pointerdown")
                phase(page, "post")
                rows.append({"correct": best == truth, "conf": probs.get(best, 0), "s": round(time.time() - t0, 1)})
                page.keyboard.press("Space")   # operator: next question (after the last one: end of session)
            phase(page, "finished", 30000)
            report(rows, a)
            page.click("text=Session summary")
            page.wait_for_selector(".dataset-card .dsstats", timeout=60000)   # dataset built from the session
            stray = re.findall(r"(?s).{0,60}\b(?:null|undefined|NaN)\b.{0,60}", page.inner_text("body"))
            assert not stray, stray
            page.screenshot(path=os.path.join(SHOTS, "41_showtime_summary.png"), full_page=True)
            page.click("text=Review / clean trials")
            page.wait_for_selector("h1:text-is('Review trials')")
            stray = re.findall(r"(?s).{0,60}\b(?:null|undefined|NaN)\b.{0,60}", page.inner_text("body"))
            assert not stray, stray
            browser.close()

        assert not errors, errors
        assert max(r["conf"] for r in rows) <= 0.9 + 1e-9, "no live percentage may exceed 90%"
        if a.p300_uv >= 4:
            assert sum(r["correct"] for r in rows) > a.trials / 4, rows   # better than chance (p3000 is weak: ~40% here)
        print("showtime OK")
    finally:
        for proc in (bridge, fake):
            proc.terminate()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
