"""Small Playwright helpers shared by the e2e scripts."""
import re


def field(page, label):
    """The .field row whose <label> text is exactly `label` (setup page)."""
    return page.locator(".field").filter(has=page.locator("label", has_text=re.compile(rf"^{re.escape(label)}$")))


def set_num(page, label, value):
    field(page, label).locator("input").fill(str(value))


def set_select(page, label, value):
    field(page, label).locator("select").select_option(value)


def set_participant(page, base, participant, device="Other"):
    """Setup tab: who is being recorded. Then open the Experiment tab (where the session is configured and started)."""
    page.goto(base + "/#/setup")
    page.wait_for_selector("h1:text-is('Setup')")
    page.fill("input[placeholder='e.g. P01']", participant)
    field(page, "Device used").locator("select").select_option(device)
    page.click("#topbar nav a[data-nav=experiment]")
    page.wait_for_selector("h1:text-is('Experiment')")
