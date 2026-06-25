"""Inspect model buttons and other UI elements in detail"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto("http://localhost:3001", timeout=20000)
    page.wait_for_load_state("networkidle", timeout=15000)

    # Get all text elements that contain model names
    html = page.content()

    # Search for model-related elements
    print("=== Looking for model elements ===")
    # Find elements with model text
    for model_str in ["DeepSeek", "V4 Flash", "V4 Pro", "思考模式"]:
        els = page.locator(f'text="{model_str}"')
        print(f"\nElements with text '{model_str}': {els.count()}")
        for i in range(min(els.count(), 3)):
            try:
                tag = els.nth(i).evaluate("el => el.tagName")
                class_name = els.nth(i).evaluate("el => el.className")
                text = els.nth(i).inner_text()
                role = els.nth(i).get_attribute("role")
                print(f"  [{i}] tag={tag} class={class_name} role={role} text='{text[:80]}'")
            except:
                pass

    # Scroll down to see bottom area
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(500)

    # Get the bottom model area HTML
    print("\n=== Bottom area (last 2000 chars of body text) ===")
    text = page.inner_text("body")
    print(text[-2000:])

    browser.close()
