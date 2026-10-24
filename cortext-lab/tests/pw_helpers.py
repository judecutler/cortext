"""Small Playwright helpers shared by the e2e scripts."""
import re


def field(page, label):
    """The .field row whose <label> text is exactly `label` (setup page)."""
    return page.locator(".field").filter(has=page.locator("label", has_text=re.compile(rf"^{re.escape(label)}$")))


def set_num(page, label, value):
    field(page, label).locator("input").fill(str(value))


def set_select(page, label, value):
    field(page, label).locator("select").select_option(value)
