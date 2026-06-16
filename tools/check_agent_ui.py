"""Detailed check of agent pipeline including tool calls"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto("http://localhost:3000")
    page.wait_for_timeout(2000)

    page.click("button:has-text('新对话')")
    page.wait_for_timeout(500)

    page.fill("input[placeholder*='输入消息']", "帮我调研一下CLIP噪声标签学习的最新论文")
    page.click("button:has-text('发送')")

    # Wait 3 seconds to capture both Agent and Tool events
    page.wait_for_timeout(3000)

    # Get ALL visible text lines
    body_text = page.evaluate("() => document.body.innerText")
    lines = [l.strip() for l in body_text.split("\n") if l.strip()]
    print("=== All visible text lines ===")
    for line in lines:
        if any(w in line for w in ["Agent", "智能", "文献", "对话", "工具", "搜索", "解析", "论文", "完成", "进行中", "流水线", "思考", "收起", "展开", "🔧", "✅", "⏳"]):
            print(f"  {line}")

    browser.close()
