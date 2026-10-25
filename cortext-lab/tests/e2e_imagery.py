#!/usr/bin/env python3
"""E2E: the optional imagery window (Experiment → Design → Imagery window), in both presentation modes.

Checks: the field is there and defaults to 0 (off, no imagery markers); with it on, each trial goes
prompt → imagery (dark stage, only the fixation cross visible) → flashes, with frame-locked imagery_on / imagery_off
markers labelled with the correct answer, the window lasting about imagery_ms, and the first flash at least
1 s after imagery_off. Run: python tests/e2e_imagery.py  (needs playwright + chromium)
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_helpers import field, set_num, set_select, set_participant  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9400 + os.getpid() % 90
BASE = f"http://127.0.0.1:{PORT}"
SHOTS = os.path.join(ROOT, "tests", "screenshots")
os.makedirs(SHOTS, exist_ok=True)
IMAG_MS = 1500


def phase(page, ph, timeout=30000):
    page.wait_for_function(f"window.cortext && window.cortext.runner && window.cortext.runner.phase === '{ph}'", timeout=timeout)


def run_one(page, mode, imagery_ms, shot=None):
    """Configure, run ONE question, click the right answer, return that trial's markers."""
    set_participant(page, BASE, f"IMG-{mode}")
    page.wait_for_selector("text=11 items in 3 categories")
    assert field(page, "Imagery window (ms)").locator("input").count() == 1
    set_select(page, "Presentation mode", mode)
    set_num(page, "Questions", 1)
    set_num(page, "Prompt shown first (ms)", 300)
    set_num(page, "Imagery window (ms)", imagery_ms)
    if mode == "mcq_flash":
        set_num(page, "Repeats per option", 1)
        set_num(page, "Flash on (ms)", 100)
    else:
        set_num(page, "Each option on screen (ms)", 200)
        set_num(page, "Times each option is shown", 1)
    page.click("text=Start session")
    phase(page, "ready")
    page.keyboard.press("Space")
    if imagery_ms > 0:
        phase(page, "imagery", timeout=10000)
        page.wait_for_function("document.querySelector('.stage').dataset.phase === 'imagery'")
        vis = page.evaluate("""() => {
          const v = (s) => { const e = document.querySelector(s); return !!e && getComputedStyle(e).visibility !== 'hidden' && getComputedStyle(e).display !== 'none'; };
          return { fix: v('.stage .fixation'), prompt: v('.stage .prompt'), sub: v('.stage .subprompt'), tiles: v('.stage .tiles') };
        }""")
        assert vis == {"fix": True, "prompt": False, "sub": False, "tiles": False}, vis
        if shot:
            page.screenshot(path=os.path.join(SHOTS, shot))
    phase(page, "response", timeout=30000)
    if mode == "mcq_flash":
        assert page.evaluate("getComputedStyle(document.querySelector('.stage .fixation')).display") == "none"
    t = page.evaluate("window.cortext.runner.current")
    page.locator(f".tile[data-id='{t['true_answer_id']}']").click()
    phase(page, "post")
    out = page.evaluate(f"window.cortext.runner.markers.recent.filter(m => m.trial_id === {json.dumps(t['trial_id'])})")
    page.evaluate("window.cortext.runner.stop()")
    phase(page, "finished")
    return t, out


def main():
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-lsl"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    errors = []
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE + "/api/status", timeout=1).read()
                break
            except Exception:
                time.sleep(0.25)
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(viewport={"width": 1300, "height": 900})
            page.on("pageerror", lambda e: errors.append(str(e)))

            # default is off: no imagery markers, trial unchanged
            set_participant(page, BASE, "IMG-default")
            page.wait_for_selector("text=11 items in 3 categories")
            assert field(page, "Imagery window (ms)").locator("input").input_value() == "0"
            _, ms = run_one(page, "mcq_flash", 0)
            assert not [m for m in ms if m["event"].startswith("imagery")], "imagery markers with the window off"
            assert any(m["event"] == "flash_on" for m in ms)

            # a leftover imagery value must not leak into a loaded preset that predates the setting
            set_participant(page, BASE, "IMG-preset")
            page.wait_for_selector("text=11 items in 3 categories")
            set_num(page, "Imagery window (ms)", 2500)
            page.locator("select[aria-label=Preset]").select_option(index=0)
            page.click("button:has-text('Load')")
            page.wait_for_selector("text=Loaded")
            assert field(page, "Imagery window (ms)").locator("input").input_value() == "0", "preset load left imagery on"

            for mode in ("mcq_flash", "mcq_sequential"):
                t, ms = run_one(page, mode, IMAG_MS, shot=f"20_imagery_{mode}.png")
                ev = [m["event"] for m in ms]
                on = next(m for m in ms if m["event"] == "imagery_on")
                off = next(m for m in ms if m["event"] == "imagery_off")
                first = next(m for m in ms if m["event"] == "flash_on")
                assert ev.index("prompt_on") < ev.index("imagery_on") < ev.index("imagery_off") < ev.index("flash_on"), ev
                assert on["option_id"] == t["true_answer_id"] and on["is_target"] is True, on
                assert on["extra"]["imagery_ms"] == IMAG_MS, on
                dur = off["t_ms"] - on["t_ms"]
                assert IMAG_MS - 20 <= dur <= IMAG_MS + 150, dur
                gap = first["t_ms"] - off["t_ms"]
                assert gap >= 1000 - 20, gap
                print(f"{mode}: imagery window {dur:.0f} ms (target {IMAG_MS}), first flash {gap:.0f} ms after it")
            b.close()
        assert not errors, errors
        print("E2E imagery OK")
    finally:
        bridge.terminate()


if __name__ == "__main__":
    main()
