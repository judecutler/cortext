"""Small Playwright helpers shared by the e2e scripts."""
import json
import re

# The suites below were written when the app's defaults were flash mode + mock predictor. Since 2026-10-25 the defaults
# are the demo flow (one at a time x3, Live EEG, question read aloud, narrator before the click; see web/js/settings.js
# DEMO_DEFAULTS), which needs a headset. Importing this module makes every new page start from the OLD defaults as
# saved settings, unless the page already has saved settings (so tests that change and reload keep their changes).
# e2e_showtime.py sets CORTEXT_E2E_FRESH_DEFAULTS=1 to test the real defaults instead.
LEGACY_TEST_SETTINGS = {
    "trial_type": "mcq_flash", "prompt_ms": 2000, "seq_repeats": 1, "predictor_mode": "mock", "speak_question": False,
    "narrate": "off", "_defaults_rev": 3,
}
_INIT = ("try { const k = 'cortext-lab.settings.v1'; if (!localStorage.getItem(k)) localStorage.setItem(k, %s); } catch (e) {}"
         % json.dumps(json.dumps(LEGACY_TEST_SETTINGS)))


def _install_legacy_defaults():
    import os
    if os.environ.get("CORTEXT_E2E_FRESH_DEFAULTS") == "1":
        return
    try:
        from playwright.sync_api import Browser, BrowserContext
    except ImportError:
        return
    if getattr(Browser, "_cortext_patched", False):
        return
    orig_ctx, orig_page = Browser.new_context, Browser.new_page

    def new_context(self, *a, **kw):
        ctx = orig_ctx(self, *a, **kw)
        ctx.add_init_script(_INIT)
        return ctx

    def new_page(self, *a, **kw):
        page = orig_page(self, *a, **kw)
        page.context.add_init_script(_INIT)
        return page

    Browser.new_context, Browser.new_page, Browser._cortext_patched = new_context, new_page, True
    del BrowserContext


_install_legacy_defaults()


def field(page, label):
    """The .field row whose <label> text is exactly `label` (setup page)."""
    return page.locator(".field").filter(has=page.locator("label", has_text=re.compile(rf"^{re.escape(label)}$")))


def set_num(page, label, value):
    field(page, label).locator("input").fill(str(value))


def set_select(page, label, value):
    field(page, label).locator("select").select_option(value)


def set_participant(page, base, participant, device="Other"):
    """Experiment tab, section 1: who is being recorded (the session is configured and started on the same page)."""
    page.goto(base + "/#/experiment")
    page.wait_for_selector("h1:text-is('Experiment')")
    page.fill("input[placeholder='e.g. P01']", participant)
    field(page, "Device used").locator("select").select_option(device)
