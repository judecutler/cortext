#!/usr/bin/env python3
"""E2E for the workflow features: Images & categories board (drag, keyboard moves, add/rename/delete categories,
add/edit images, editable copy of the built-in set), undo/redo, Next session, Restart, Redo session + superseded mark.
Run: python tests/e2e_ux.py (needs playwright + chromium)"""
import base64
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_helpers import set_num, set_participant  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "bridge", "logs")
PORT = 9500 + os.getpid() % 90
BASE = f"http://127.0.0.1:{PORT}"
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

IDB = """async ([store, key]) => {
  const db = await new Promise((res, rej) => { const r = indexedDB.open('cortext-lab'); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
  return await new Promise((res) => { const q = db.transaction(store).objectStore(store).get(key); q.onsuccess = () => {
    const v = q.result; if (v && v.files) v.files = Object.keys(v.files); res(v); }; });
}"""


def col(page, name):
    return page.locator(f".catbody[data-cat='{name}']")


def ids_in(page, name):
    return page.eval_on_selector_all(f".catbody[data-cat='{name}'] .icard", "els => els.map(e => e.dataset.id)")


def undo(page, redo=False):
    page.evaluate("document.activeElement && document.activeElement.blur()")
    page.keyboard.press("Control+Shift+z" if redo else "Control+z")


def phase(page, ph, timeout=20000):
    page.wait_for_function(f"window.cortext.runner && window.cortext.runner.phase === '{ph}'", timeout=timeout)


def answer_one_trial(page):
    page.keyboard.press("Space")
    phase(page, "response")
    page.keyboard.press("1")
    phase(page, "post")


def main():
    data = tempfile.mkdtemp(prefix="cortext-ux-")
    before = set(os.listdir(LOGS)) if os.path.isdir(LOGS) else set()
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-lsl"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env={**os.environ, "CORTEXT_DATA_DIR": data})
    errors = []
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE + "/api/status", timeout=1).read(); break
            except Exception:
                time.sleep(0.1)
        with sync_playwright() as p:
            page = p.chromium.launch().new_page(viewport={"width": 1500, "height": 950})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("dialog", lambda d: d.accept())   # the native "leave a running session?" prompt
            set_participant(page, BASE, "UX1")

            # ================= Images & categories =================
            page.goto(BASE + "/#/library")
            page.wait_for_selector("h1:text-is('Images & categories')")
            assert "built-in sample set" in page.inner_text(".note-banner")
            assert ids_in(page, "animal")[:2] == ["dog", "cat"]

            # drag dog onto the shapes: the built-in set turns into an editable copy that the experiment now uses
            page.drag_and_drop(".icard[data-id='dog']", ".catbody[data-cat='shape']")
            page.wait_for_function("document.querySelector(\".catbody[data-cat='shape'] .icard[data-id='dog']\")")
            src = page.evaluate("JSON.parse(localStorage.getItem('cortext-lab.settings.v1')).stimuli_source")
            assert src == "set:Sample set (my copy)", src
            stored = page.evaluate(IDB, ["stimulus_sets", "Sample set (my copy)"])
            assert len(stored["files"]) == 11 and next(i for i in stored["config"]["items"] if i["id"] == "dog")["category"] == "shape"
            assert page.locator(".note-banner").count() == 0, "no read-only banner on the copy"

            # undo / redo
            undo(page)
            page.wait_for_function("document.querySelector(\".catbody[data-cat='animal'] .icard[data-id='dog']\")")
            undo(page, redo=True)
            page.wait_for_function("document.querySelector(\".catbody[data-cat='shape'] .icard[data-id='dog']\")")
            undo(page)   # dog back with the animals for the rest of the test
            page.wait_for_function("document.querySelector(\".catbody[data-cat='animal'] .icard[data-id='dog']\")")

            # keyboard: Alt+→ moves to the next category, Alt+↓ reorders
            page.focus(".icard[data-id='cat']"); page.keyboard.press("Alt+ArrowRight")
            page.wait_for_function("document.querySelector(\".catbody[data-cat='object'] .icard[data-id='cat']\")")
            assert page.evaluate("document.activeElement.dataset.id") == "cat", "focus follows the moved card"
            page.keyboard.press("Alt+ArrowDown")
            page.wait_for_function("(() => { const ids = [...document.querySelectorAll(\".catbody[data-cat='object'] .icard\")].map(e => e.dataset.id); return ids.indexOf('cat') === ids.length - 1; })()")

            # new category, rename it, add an image to it
            page.fill("input[aria-label='New category name']", "tools"); page.keyboard.press("Enter")
            page.wait_for_selector(".catbody[data-cat='tools']")
            page.click(".catname:text-is('tools')")
            page.fill(".catname-in", "Tools & things"); page.keyboard.press("Enter")
            page.wait_for_selector(".catbody[data-cat='Tools & things']")
            png = os.path.join(data, "hammer.png"); open(png, "wb").write(PNG)
            with page.expect_file_chooser() as fc:
                page.locator(".catcol", has=page.locator(".catbody[data-cat='Tools & things']")).locator("button:has-text('Add images')").click()
            fc.value.set_files(png)
            page.wait_for_function("document.querySelector(\".catbody[data-cat='Tools & things'] .icard[data-id='hammer']\")")

            # edit: new on-screen text and category; the id stays
            page.dblclick(".icard[data-id='cup']")
            page.wait_for_selector("dialog.modal[open]")
            page.locator("dialog.modal .field", has_text="Text on screen").locator("input").fill("Mug")
            page.locator("dialog.modal select").select_option("Tools & things")
            page.click("dialog.modal button:text-is('Save')")
            page.wait_for_function("document.querySelector(\".catbody[data-cat='Tools & things'] .icard[data-id='cup']\")")
            assert "Mug" in page.inner_text(".icard[data-id='cup']")

            # delete a category: its images move (default) and it comes back with undo
            page.click("button[aria-label='Delete shape']")
            page.click("dialog.modal button:text-is('Delete category')")
            page.wait_for_function("!document.querySelector(\".catbody[data-cat='shape']\")")
            assert "circle" in ids_in(page, "animal")
            undo(page)
            page.wait_for_selector(".catbody[data-cat='shape']")
            assert ids_in(page, "shape") == ["circle", "square", "pentagram"]

            # the experiment page sees the edited set and its categories
            page.click("text=Configure & run")
            page.wait_for_selector("text=12 items in 4 categories")
            assert "Tools & things" in page.inner_text(".catchips")

            # ================= sessions: undo a discard, next, restart, redo =================
            for label, v in {"Questions": "1", "Repeats per option": "1", "Flash on (ms)": "150", "Gap min (ms)": "80",
                             "Gap max (ms)": "100", "Prompt shown first (ms)": "200"}.items():
                set_num(page, label, v)
            page.click("text=Start session")
            phase(page, "ready")
            sid1 = page.evaluate("window.cortext.runner.session.session_id")
            answer_one_trial(page)
            page.keyboard.press("Backspace")
            page.wait_for_function("window.cortext.runner.trials[0].valid === false")
            undo(page)
            page.wait_for_function("window.cortext.runner.trials[0].valid === true")
            page.keyboard.press("Space")
            phase(page, "finished")
            page.wait_for_selector(".overlay.show >> text=Session complete")
            assert "1 valid of 1 attempt" in page.inner_text(".overlay")

            # Next session for a new participant (Enter in the participant field)
            page.fill(".next-pid", "UX2"); page.press(".next-pid", "Enter")
            page.wait_for_function(f"window.cortext.runner && window.cortext.runner.session.session_id !== '{sid1}' && window.cortext.runner.phase === 'ready'")
            s2 = page.evaluate("window.cortext.runner.session")
            assert s2["participant_id"] == "UX2" and s2["previous_session_id"] == sid1, s2
            assert s2["settings"]["seed"] != page.evaluate(IDB, ["sessions", sid1])["settings"]["seed"], "next session gets a new seed"
            sid2 = s2["session_id"]

            # Restart mid-session: same questions again, the first attempt kept and marked superseded
            page.click("button:has-text('Restart session')")
            page.click("dialog.modal button:text-is('Restart session')")
            page.wait_for_function(f"window.cortext.runner.session.session_id !== '{sid2}' && window.cortext.runner.phase === 'ready'")
            s3 = page.evaluate("window.cortext.runner.session")
            sid3 = s3["session_id"]
            assert s3["redo_of"] == sid2 and s3["participant_id"] == "UX2" and s3["settings"]["seed"] == s2["settings"]["seed"], s3
            assert page.evaluate(IDB, ["sessions", sid2])["superseded_by"] == sid3

            # finish it, then Redo this session from the end screen
            answer_one_trial(page); page.keyboard.press("Space"); phase(page, "finished")
            page.click(".overlay button:has-text('Redo this session')")
            page.click("dialog.modal button:text-is('Redo session')")
            page.wait_for_function(f"window.cortext.runner.session.session_id !== '{sid3}' && window.cortext.runner.phase === 'ready'")
            sid4 = page.evaluate("window.cortext.runner.session.session_id")
            assert page.evaluate(IDB, ["sessions", sid3])["superseded_by"] == sid4

            # summary of the superseded one: banner, "use it after all", and undo of that
            page.goto(BASE + f"/#/summary/{sid3}")
            page.wait_for_selector(".warn-banner:has-text('Superseded')")
            page.click(".warn-banner button:has-text('Use it after all')")
            page.wait_for_function("!document.querySelector('.warn-banner')")
            assert not page.evaluate(IDB, ["sessions", sid3]).get("superseded_by")
            undo(page)
            page.wait_for_selector(".warn-banner:has-text('Superseded')")

            # the sessions list shows which is which
            page.goto(BASE + "/#/sessions")
            page.wait_for_selector(f"tr:has-text('{sid3}')")
            assert "superseded" in page.inner_text(f"tr:has-text('{sid3}')")
            assert "redo" in page.inner_text(f"tr:has-text('{sid4}')")
        assert not errors, errors
        print("E2E UX (library, undo, next/restart/redo session) OK")
    finally:
        proc.terminate()
        for f in (set(os.listdir(LOGS)) - before) if os.path.isdir(LOGS) else []:
            os.remove(os.path.join(LOGS, f))


if __name__ == "__main__":
    main()
