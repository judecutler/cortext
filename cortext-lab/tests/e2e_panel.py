#!/usr/bin/env python3
"""E2E: the run view's info panel can be hidden/shown (button + I key), nav stays visible, choice is remembered."""
import os, subprocess, sys, time, urllib.request
from playwright.sync_api import sync_playwright
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_helpers import set_participant  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9200 + os.getpid() % 90
BASE = f"http://127.0.0.1:{PORT}"

def main():
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-lsl"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(60):
            try: urllib.request.urlopen(BASE + "/api/status", timeout=1); break
            except Exception: time.sleep(0.25)
        with sync_playwright() as p:
            b = p.chromium.launch(); page = b.new_page(viewport={"width": 1300, "height": 800})
            set_participant(page, BASE, "P99")
            page.wait_for_selector("text=11 items in 3 categories")
            page.click("text=Start session")
            page.wait_for_selector(".run")
            assert page.is_visible("aside.dev")
            page.click("text=Hide panel")
            assert not page.is_visible("aside.dev") and page.is_visible(".panel-reveal") and page.is_visible("#topbar nav")
            page.keyboard.press("i")
            assert page.is_visible("aside.dev") and not page.is_visible(".panel-reveal")
            page.keyboard.press("i")
            assert page.evaluate("localStorage.getItem('cortext-lab.run.panel')") == "hidden"
            page.click(".panel-reveal"); assert page.is_visible("aside.dev")
            assert page.evaluate("localStorage.getItem('cortext-lab.run.panel')") == "shown"
            b.close()
        print("E2E panel OK")
    finally:
        bridge.terminate()

if __name__ == "__main__":
    main()
