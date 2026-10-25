#!/usr/bin/env python3
"""E2E: question list, shuffle buttons, presets (built-in, save/load/export/import/delete) and running a session from a list."""
import os, subprocess, sys, time, urllib.request
from playwright.sync_api import sync_playwright
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_helpers import set_participant  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9300 + os.getpid() % 90
BASE = f"http://127.0.0.1:{PORT}"

def main():
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-lsl"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    errors = []
    try:
        for _ in range(60):
            try: urllib.request.urlopen(BASE + "/api/status", timeout=1); break
            except Exception: time.sleep(0.25)
        with sync_playwright() as p:
            b = p.chromium.launch(); page = b.new_page(viewport={"width": 1300, "height": 900}, accept_downloads=True)
            page.on("pageerror", lambda e: errors.append(str(e)))
            set_participant(page, BASE, "PRE")
            page.wait_for_selector("text=11 items in 3 categories")
            # generate a list
            page.click("text=Generate from stimulus set")
            n = page.locator(".planrow").count(); assert n == 12, n
            prompts = lambda: page.eval_on_selector_all(".plan-prompt", "els => els.map(e => e.value)")
            chips = lambda: page.eval_on_selector_all(".planrow:first-child .chip-name", "els => els.map(e => e.textContent)")
            # wording
            page.fill("input[placeholder='Which one is the {label}?']", "Pick the {label}")
            page.click("text=Apply wording to all")
            ps = prompts(); assert all(x.startswith("Pick the ") for x in ps), ps
            # shuffles: some change with overwhelming probability
            before_q = prompts(); before_c = chips()
            page.click("text=Shuffle questions"); assert prompts() != before_q
            seen = False
            for _ in range(6):
                page.click("text=Shuffle images")
                if chips() != before_c: seen = True; break
            assert seen, "image shuffle never changed order"
            # move first chip later with the arrow
            first = chips(); page.locator(".planrow:first-child .chip-btn[aria-label='Move later']").first.click()
            assert chips()[0] == first[1] and chips()[1] == first[0]
            # save / reload / export
            page.fill("input[placeholder='name for a new preset']", "My list"); page.click("text=Save as new")
            page.locator("select[aria-label=Preset]").select_option("My list")
            with page.expect_download() as dl: page.click("button:has-text('Export')")
            path = dl.value.path(); assert os.path.getsize(path) > 200
            # clearing is instant and undoable (no "are you sure?" dialog)
            page.click("text=Clear list")
            assert page.locator(".planrow").count() == 0
            page.evaluate("document.activeElement.blur()"); page.keyboard.press("Control+z")
            page.wait_for_function("document.querySelectorAll('.planrow').length === 12")
            page.keyboard.press("Control+Shift+z")
            page.wait_for_function("document.querySelectorAll('.planrow').length === 0")
            page.click("button:has-text('Load')"); assert page.locator(".planrow").count() == 12
            page.locator("input[type=file][accept='.json']").set_input_files(path)
            page.wait_for_selector("select[aria-label=Preset] option:has-text('My list (2)')", state="attached")
            # built-in is protected
            page.locator("select[aria-label=Preset]").select_option(index=0)
            page.click("button:has-text('Update')"); page.wait_for_selector("text=Built-in presets cannot be changed")
            # run from the list, fixed order, 2 questions
            page.locator("select[aria-label=Preset]").select_option("My list")
            page.click("button:has-text('Load')")
            page.locator(".planrow .btn.danger").nth(0).click()
            while page.locator(".planrow").count() > 2: page.locator(".planrow .btn.danger").nth(0).click()
            page.check("text=Use this list for the session >> xpath=../..//input[@type=checkbox]") if False else None
            page.evaluate("document.querySelector('.planlist').closest('.card').querySelector('input[type=checkbox]').click()")
            page.locator("select").filter(has_text="Exactly as listed").select_option("fixed")
            expected = [page.eval_on_selector_all(f".planrow:nth-child({i}) .chip-name", "e => e.map(x => x.textContent)") for i in (1, 2)]
            page.click("text=Start session")
            page.wait_for_function("window.cortext && window.cortext.runner && window.cortext.runner.phase === 'ready'", timeout=30000)
            assert page.evaluate("window.cortext.runner.session.planned_trials") == 2
            page.keyboard.press("Space")
            page.wait_for_function("window.cortext.runner.current", timeout=10000)
            shown = page.evaluate("window.cortext.runner.current.options.sort((a,b)=>a.position-b.position).map(o => o.label)")
            assert shown == expected[0], (shown, expected[0])
            b.close()
        assert not errors, errors
        print("E2E presets OK")
    finally:
        bridge.terminate()

if __name__ == "__main__":
    main()
