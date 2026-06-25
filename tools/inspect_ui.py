"""Inspect the actual UI elements on the page"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto("http://localhost:3001", timeout=20000)
    page.wait_for_load_state("networkidle", timeout=15000)

    page.screenshot(path="tools/screenshot_debug.png", full_page=True)

    text = page.inner_text("body")
    print("=== Page visible text (first 2000 chars) ===")
    print(text[:2000])
    print()

    print("=== All buttons ===")
    btns = page.locator("button")
    for i in range(btns.count()):
        txt = btns.nth(i).inner_text()
        print(f"  Button {i}: \"{txt}\"")

    print()
    print("=== All text inputs ===")
    inputs = page.locator('input[type="text"]')
    for i in range(inputs.count()):
        ph = inputs.nth(i).get_attribute("placeholder")
        print(f"  Input {i}: placeholder=\"{ph}\"")

    browser.close()
