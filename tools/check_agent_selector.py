"""Verify agent selector and model position in frontend"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto("http://localhost:3000")
    page.wait_for_timeout(2000)

    text = page.evaluate("() => document.body.innerText")
    print("=== Check for agent chips ===")
    for key in ["自动", "对话", "调研", "创新", "实验"]:
        found = key in text
        print(f"  {'✅' if found else '❌'} Agent chip '{key}'")

    # Check model selector near input area
    input_bar = page.evaluate("""() => {
        const all = document.body.innerText;
        // Find the last occurrence of model options
        const lines = all.split('\\n');
        const inputRelated = lines.filter(l =>
            l.includes('Flash') || l.includes('V4') || l.includes('DeepSeek')
        );
        return inputRelated.join(' | ');
    }""")
    print(f"\n  Model text: {input_bar}")

    # Check header no longer has model selector
    header_text = page.evaluate("""() => {
        // Find the header-like divs
        const divs = document.querySelectorAll('div');
        for (const d of divs) {
            if (d.innerText.includes('清空对话') || d.innerText.includes('论文搜索')) {
                const txt = d.innerText;
                if (txt.length < 200) return txt;
            }
        }
        return 'no match';
    }""")
    print(f"  Header text: {header_text}")

    page.screenshot(path="tools/agent_selector.png", full_page=True)
    print("\n✅ Screenshot saved")
    browser.close()
