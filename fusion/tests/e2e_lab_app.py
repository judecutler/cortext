#!/usr/bin/env python3
"""End-to-end: the real CorText Lab app (headless Chromium) <-> the fusion servers.

Needs: pip install playwright && playwright install chromium. Not part of `pytest -q`.
Run from the repo root:  python fusion/tests/e2e_lab_app.py

What it checks
1. The app, set to predictor mode "ws", connects to `cortext-fusion`'s WebSocket server.
2. A fake P300 process sees each flash (from the fusion server's live trial table, standing in
   for LSL) and POSTs one score per flash_index over HTTP. It always favours the option in
   screen position 2, so the expected answer is known without labels.
3. The app shows "EEG indicates <that option>" (not MOCK, not "no answer") and stores it.
4. The app's own exporter writes trials.csv / markers.csv / session.json, and
   `load_lab_export` reads them back with the right trials, flashes, options and clicks.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve()
FUSION = HERE.parents[1]
LAB = FUSION.parent / "cortext-lab"
sys.path.insert(0, str(FUSION))

from cortext_fusion.config import config_from_dict  # noqa: E402
from cortext_fusion.server import Predictor, serve, serve_ws  # noqa: E402
from cortext_fusion.stimulus_log import load_lab_export  # noqa: E402

PORT = 8300 + os.getpid() % 400
BASE = f"http://127.0.0.1:{PORT}"
FAVOURED_POSITION = 1  # 0-based: the second tile


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def fake_p300(pred: Predictor, http: str, stop: threading.Event, posted: list):
    done = set()
    while not stop.is_set():
        with pred.lock:
            items = [(lt.raw_trial_id, lt.meta.session_id, dict(lt.flash_options), list(lt.meta.option_ids))
                     for lt in pred.live.values()]
        for raw_tid, sess, flashes, opts in items:
            for fi, oid in flashes.items():
                if (sess, raw_tid, fi) in done:
                    continue
                v = 0.9 if oid == opts[FAVOURED_POSITION] else 0.1
                post(http + "/sequential/flash", {"trial_id": raw_tid, "session_id": sess, "flash_index": fi, "value": v})
                done.add((sess, raw_tid, fi))
                posted.append(fi)
        time.sleep(0.03)


def main():
    cfg = config_from_dict({"models": [{"name": "p300", "adapter": "per_flash", "weight": 1.0}],
                            "sequential": {"stop_confidence": 0.99, "min_repeats": 2, "final_wait_s": 1.0}})
    pred = Predictor.from_config(cfg)
    http_srv, ws_srv = serve(pred, port=0), serve_ws(pred, port=0)
    for s in (http_srv, ws_srv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    http = f"http://127.0.0.1:{http_srv.server_address[1]}"
    ws_url = f"ws://127.0.0.1:{ws_srv.server_address[1]}"
    stop, posted = threading.Event(), []
    threading.Thread(target=fake_p300, args=(pred, http, stop, posted), daemon=True).start()

    bridge = subprocess.Popen([sys.executable, str(LAB / "bridge" / "bridge.py"), "--port", str(PORT)],
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
            browser = p.chromium.launch()
            ctx = browser.new_context(viewport={"width": 1400, "height": 900})
            ctx.add_init_script(f"localStorage.setItem('cortext-lab.settings.v1', JSON.stringify("
                                f"{{predictor_mode: 'ws', predictor_url: '{ws_url}'}}))")
            page = ctx.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
            page.goto(BASE + "/")
            page.wait_for_selector("text=New session")
            page.fill("input[placeholder='e.g. P01']", "FUSIONE2E")
            vals = {0: "2", 2: "3", 3: "150", 4: "80", 5: "120", 6: "300"}  # n_trials, repeats, on, off min/max, prompt
            for idx, v in vals.items():
                page.locator(".card input[type=number]").nth(idx).fill(v)
            page.click("text=Start session")
            page.wait_for_selector(".stage")

            def phase(ph, timeout=30000):
                page.wait_for_function(f"window.cortext.runner && window.cortext.runner.phase === '{ph}'", timeout=timeout)

            phase("ready")
            shown = []
            for t in range(2):
                page.keyboard.press("Space")
                phase("response")
                cur = page.evaluate("window.cortext.runner.current")
                opts = sorted(cur["options"], key=lambda o: o["position"])
                expected = opts[FAVOURED_POSITION]
                page.locator(f".tile[data-id='{cur['true_answer_id']}']").dispatch_event("pointerdown")
                phase("post")
                page.wait_for_function("document.querySelector('.post .pred') && "
                                       "!document.querySelector('.post .pred').innerText.includes('waiting')", timeout=6000)
                txt = page.inner_text(".post .pred")
                shown.append(txt)
                assert "EEG indicates" in txt and "MOCK" not in txt and "no answer" not in txt, txt
                assert expected["label"] in txt or expected["id"] in txt, (txt, expected)
                page.keyboard.press("Space")
            phase("finished")
            sid = page.evaluate("window.cortext.runner.session.session_id")
            time.sleep(0.6)
            files = page.evaluate("""async (sid) => { const m = await import('/js/exporter.js');
                                     return await m.sessionFiles(sid); }""", sid)
            stored = page.evaluate("window.cortext.runner.trials.map(t => t.prediction)")
            browser.close()
        out = Path(tempfile.mkdtemp(prefix="lab_export_")) / sid
        out.mkdir(parents=True)
        for f in files:
            (out / f["name"]).write_text(f["data"])
        metas = load_lab_export(out)
        run = [m for m in metas.values() if m.flashes]
        assert len(run) == 2, list(metas)
        for m in run:
            assert len(m.flashes) == len(m.option_ids) * 3 and m.n_repeats == 3
            assert m.clicked_option == m.target_option and m.valid and m.participant_id == "FUSIONE2E"
        assert all(p and p["source"] == "ws" and p["best_option_id"] for p in stored), stored
        assert not [e for e in errors if "ws" not in e.lower()], errors
        print("App displayed:", *shown, sep="\n  ")
        print(f"Scores posted by the fake P300 process: {len(posted)}; exported trials loaded: {len(run)} "
              f"({out})")
        print("E2E OK")
        return 0
    finally:
        stop.set()
        bridge.terminate()
        http_srv.shutdown()
        ws_srv.shutdown()


if __name__ == "__main__":
    sys.exit(main())
