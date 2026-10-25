#!/usr/bin/env python3
"""E2E: voice features in headless Chrome against the real bridge and a FAKE OpenAI-compatible LLM.
  * Setup: narrator name, LLM provider, mood + editable context, Test narrator
  * Session 1 (one at a time): question read aloud, AUDIO-only stimuli (buffers, latency in markers), narrator after click
  * Session 2 (flash): audio + visual stimuli, narrator BEFORE the click with the built-in lines (flagged as shown)
Run: python tests/e2e_voice.py  (needs playwright + chromium; the bridge needs a TTS engine for buffer timing, e.g.
pyttsx3 + espeak-ng, or edge-tts with internet; without one the test checks the browser-speech fallback flag)."""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_helpers import set_num, set_select, set_participant, field  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9400 + os.getpid() % 90
LLM_PORT = PORT + 200
BASE = f"http://127.0.0.1:{PORT}"
SHOTS = os.path.join(ROOT, "tests", "screenshots")
os.makedirs(SHOTS, exist_ok=True)
llm_calls = []


class FakeLLM(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
        llm_calls.append(body)
        facts = json.loads(body["messages"][1]["content"].split("(JSON): ", 1)[1])
        text = (f"And it's {facts['name']}! The brain goes for option {facts['guess_letter']}, the {facts['guess_label']}, "
                f"at {facts['confidence_percent']} percent{' in this simulation' if facts.get('simulated') else ''}!")
        text = text.replace(" percent", "%")
        out = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def phase(page, ph, timeout=40000):
    page.wait_for_function(f"window.cortext.runner && window.cortext.runner.phase === '{ph}'", timeout=timeout)


def markers(page, event):
    return page.evaluate(f"window.cortext.markers.recent.filter(m => m.event === '{event}')")


def main():
    llm = ThreadingHTTPServer(("127.0.0.1", LLM_PORT), FakeLLM)
    threading.Thread(target=llm.serve_forever, daemon=True).start()
    env = {**os.environ, "OPENAI_BASE_URL": f"http://127.0.0.1:{LLM_PORT}/v1", "OPENAI_API_KEY": "k", "OPENAI_MODEL": "fake"}
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-lsl"],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE + "/api/status", timeout=1).read()
                break
            except Exception:
                time.sleep(0.1)
        status = json.loads(urllib.request.urlopen(BASE + "/api/voice/status").read())
        has_tts = bool(status["tts_engines"])
        if has_tts:   # make sure the bridge can really speak here (edge-tts needs internet)
            try:
                urllib.request.urlopen(BASE + "/api/tts?text=test", timeout=30).read()
            except Exception:
                has_tts = False
        errors = []
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
            page = browser.new_page(viewport={"width": 1500, "height": 950})
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" and "ERR_TUNNEL" not in m.text and "ERR_NAME" not in m.text and "Failed to load resource" not in m.text else None)

            # ---------- Setup: narrator ----------
            page.goto(BASE + "/#/setup")
            page.wait_for_selector("h2:text-is('Voice & narrator (demo)')")
            page.fill("input[placeholder='e.g. Jude (optional)']", "Jude")
            set_select(page, "Narrator writes with", "openai")
            set_select(page, "Mood", "sports")
            ctx = field(page, "Narrator context").locator("textarea")
            assert "sports commentator" in ctx.input_value()
            ctx.fill(ctx.input_value() + " This run uses spoken words instead of pictures.")
            assert field(page, "Mood").locator("select").input_value() == "custom"
            page.wait_for_selector("text=fake @ http://127.0.0.1")
            page.click("text=▶ Test narrator")
            page.wait_for_function("document.querySelector('.voice-test + div').textContent.includes('78%')", timeout=20000)
            msg = page.inner_text(".voice-test + div")
            assert "openai" in msg and "Jude" in msg, msg
            assert "spoken words instead of pictures" in llm_calls[-1]["messages"][0]["content"], llm_calls[-1]
            assert "simulation" in llm_calls[-1]["messages"][0]["content"]          # fixed rules appended
            page.screenshot(path=os.path.join(SHOTS, "30_voice_setup.png"), full_page=True)
            saved = page.evaluate("JSON.parse(localStorage.getItem('cortext-lab.settings.v1'))")
            assert saved["narrator_llm"] == "openai" and saved["narrator_mood"] == "custom" and "spoken words" in saved["narrator_context"], saved

            # ---------- Session 1: sequential, question aloud, audio only, narrator after click ----------
            set_participant(page, BASE, "VOX")
            page.wait_for_selector("text=11 items in 3 categories")
            set_select(page, "Presentation mode", "mcq_sequential")
            for label, val in {"Questions": "2", "Options per question": "3", "Prompt shown first (ms)": "300",
                               "Each option on screen (ms)": "700", "Gap between options (ms)": "200", "Times each option is shown": "1"}.items():
                set_num(page, label, val)
            field(page, "Read the question aloud").locator("input").check()
            set_select(page, "Stimulus modality", "audio")
            set_select(page, "Narrator", "after_click")
            page.click("text=Start session")
            phase(page, "ready")
            sid = page.evaluate("window.cortext.runner.session.session_id")
            page.keyboard.press("Space")
            phase(page, "prompt")
            page.wait_for_function("window.cortext.markers.recent.some(m => m.event === 'question_audio_end')", timeout=30000)
            phase(page, "flash")
            assert page.locator(".listen").is_visible(), "audio-only shows the static Listen screen"
            t0 = time.time()
            while page.evaluate("window.cortext.runner.phase") == "flash" and time.time() - t0 < 20:
                assert page.evaluate("document.querySelectorAll('.pstim.on').length") == 0, "audio-only: nothing visual per option"
                time.sleep(0.03)
            page.screenshot(path=os.path.join(SHOTS, "31_audio_only.png"))
            on = [m for m in markers(page, "flash_on") if m["session_id"] == sid]
            assert len(on) == 3, on
            used = on[0]["extra"].get("audio_engine")
            for m in on:
                x = m["extra"]
                assert x["modality"] == "audio", x
                if has_tts:
                    assert x["audio_timing"] == "buffer" and x["audio_engine"] in ("edge", "pyttsx3"), x
                    assert x["audio_ms"] > 100 and "audio_offset_ms" in x, x
                else:
                    assert x["audio_timing"] == "approximate", x
            qs, qe = markers(page, "question_audio_start")[-1], markers(page, "question_audio_end")[-1]
            assert qs["t_ms"] < qe["t_ms"] <= on[0]["t_ms"] - 300, "the stimuli start after the question has been spoken"
            phase(page, "response")
            true_id = page.evaluate("window.cortext.runner.current.true_answer_id")
            page.locator(f".tile[data-id='{true_id}']").dispatch_event("pointerdown")
            phase(page, "post")
            page.wait_for_function("document.querySelector('.narrator.show:not(.thinking) .narr-text')?.textContent.includes('%')", timeout=20000)
            narr = page.inner_text(".narrator")
            assert "Jude" in narr and "simulation" in narr.lower(), narr          # mock predictor -> said to be simulated
            page.screenshot(path=os.path.join(SHOTS, "32_narrator_after.png"))
            assert page.locator(".narr-orb").evaluate("e => getComputedStyle(e).animationName") == "narr-pulse"
            tr = page.evaluate("window.cortext.runner.current")
            n = tr["narration"]
            assert n["when"] == "after_click" and n["provider"] == "openai" and "{name}" in n["text"] and "Jude" not in json.dumps(tr), n
            assert not (tr["model_display"] or {}).get("narrated"), "after-click narration does not flag the click"
            page.wait_for_function("window.cortext.markers.recent.some(m => m.event === 'narration_end')", timeout=30000)
            page.keyboard.press("Space")
            phase(page, "prompt")
            page.click("text=End session")
            phase(page, "finished")
            assert "Jude" not in json.dumps(page.evaluate("window.cortext.runner.session")), "display name never in the session record"

            # ---------- Session 2: flash mode, audio + visual, narrator before the click, built-in lines ----------
            page.goto(BASE + "/#/setup")
            page.wait_for_selector("h2:text-is('Voice & narrator (demo)')")
            set_select(page, "Narrator writes with", "template")
            set_select(page, "Mood", "default")
            page.click("#topbar nav a[data-nav=experiment]")
            page.wait_for_selector("text=11 items in 3 categories")
            set_select(page, "Presentation mode", "mcq_flash")
            for label, val in {"Questions": "1", "Options per question": "3", "Prompt shown first (ms)": "300",
                               "Repeats per option": "2", "Flash on (ms)": "300", "Gap min (ms)": "300", "Gap max (ms)": "350"}.items():
                set_num(page, label, val)
            field(page, "Read the question aloud").locator("input").uncheck()
            set_select(page, "Stimulus modality", "audiovisual")
            set_select(page, "Narrator", "before_click")
            page.click("text=Start session")
            phase(page, "ready")
            sid2 = page.evaluate("window.cortext.runner.session.session_id")
            page.keyboard.press("Space")
            phase(page, "flash")
            page.wait_for_selector(".tile.flash", timeout=10000)
            phase(page, "response")
            page.wait_for_function("document.querySelector('.narrator.show:not(.thinking) .narr-text')?.textContent.includes('%')", timeout=20000)
            page.screenshot(path=os.path.join(SHOTS, "33_narrator_before.png"))
            tr = page.evaluate("window.cortext.runner.current")
            assert tr["narration"]["when"] == "before_click" and tr["narration"]["provider"] == "template", tr["narration"]
            assert tr["model_display"] and tr["model_display"]["narrated"], "before-click narration flags model_shown_before_click"
            on2 = [m for m in markers(page, "flash_on") if m["session_id"] == sid2]
            assert len(on2) == 6 and all(m["extra"]["modality"] == "audiovisual" for m in on2), on2
            page.locator(f".tile[data-id='{tr['true_answer_id']}']").dispatch_event("pointerdown")
            phase(page, "post")
            clicks = [m for m in markers(page, "click") if m["session_id"] == sid2]
            assert clicks and clicks[-1]["extra"]["model_shown_before_click"] is True, clicks
            page.keyboard.press("Space")
            phase(page, "finished")
            browser.close()
        bad = [e for e in errors if "favicon" not in e]
        assert not bad, bad
        print(f"voice e2e OK: setup narrator (custom context reaches the LLM), question aloud before stimuli, audio-only "
              f"stimuli ({'buffer timing via ' + str(used) if has_tts else 'browser-speech fallback'}), "
              f"narrator after/before click, name kept out of the data. Screenshots: tests/screenshots/3*.png")
    finally:
        bridge.terminate()
        bridge.wait(5)
        llm.shutdown()


if __name__ == "__main__":
    main()
