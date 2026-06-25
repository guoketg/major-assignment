"""Inspect the model select element structure"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto("http://localhost:3001", timeout=20000)
    page.wait_for_load_state("networkidle", timeout=15000)

    # Find all select elements
    selects = page.locator("select")
    print(f"Select elements: {selects.count()}")
    for i in range(selects.count()):
        sel = selects.nth(i)
        name = sel.get_attribute("name") or ""
        sel_id = sel.get_attribute("id") or ""
        print(f"  Select[{i}]: name='{name}' id='{sel_id}'")
        opts = sel.locator("option")
        for j in range(opts.count()):
            val = opts.nth(j).get_attribute("value") or ""
            txt = opts.nth(j).inner_text()
            print(f"    Option[{j}]: value='{val}' text='{txt}'")

    # Also find elements containing "DeepSeek" in the HTML
    print("\n=== Searching for DeepSeek in HTML ===")
    els = page.locator("*:has-text('DeepSeek')")
    print(f"Elements with DeepSeek text: {els.count()}")
    for i in range(min(els.count(), 5)):
        tag = els.nth(i).evaluate("el => el.tagName")
        text = els.nth(i).inner_text()[:100]
        print(f"  [{i}] tag={tag} text='{text}'")

    browser.close()
