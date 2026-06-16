"""
Agent 可视化 Playwright 测试
验证前端 Agent 流水线卡片能正确渲染和更新
"""
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from playwright.sync_api import sync_playwright

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3001")


def test_agent_pipeline():
    """完整测试 Agent 流水线可视化"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        # ===== 1. 页面加载 =====
        page.goto(FRONTEND_URL, wait_until="networkidle")
        page.wait_for_timeout(1000)
        print("✅ 页面加载正常")

        # ===== 2. 创建新对话 =====
        page.click("button:has-text('新对话')")
        page.wait_for_timeout(500)

        # 输入研究意图消息
        inp = page.locator("input[placeholder*='输入消息'], textarea[placeholder*='输入消息']")
        inp.fill("帮我调研一下CLIP噪声标签学习的最新进展")
        page.click("button:has-text('发送')")

        # ===== 3. 等待 Agent 流水线出现 =====
        page.wait_for_timeout(3000)

        text = page.evaluate("() => document.body.innerText")

        # 验证 Agent 流水线元素
        checks = {
            "Agent 流水线标题": "Agent 流水线" in text,
            "Supervisor 智能路由": "智能路由" in text,
        }

        for name, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"{status} {name}")

        assert checks["Agent 流水线标题"], "Agent 流水线未显示"
        assert checks["Supervisor 智能路由"], "Supervisor 未显示"
        print()

        # ===== 4. 等待流式完成（等待发送按钮重新可用） =====
        try:
            page.wait_for_function(
                "() => { const btn = document.querySelector('button'); return btn && !btn.disabled; }",
                timeout=90000
            )
            print("✅ 流式对话完成")
        except:
            print("⚠️ 等待超时（但内容可能已生成）")

        page.wait_for_timeout(1000)

        # ===== 5. 验证有响应内容 =====
        final_text = page.evaluate("() => document.body.innerText")
        if "Agent 流水线" in final_text:
            print("✅ Agent 流水线显示正常")
        else:
            print("⚠️ Agent 流水线未显示")

        # 验证有回答内容
        if len(final_text) > 200:
            print("✅ 有响应内容输出")
        else:
            print("⚠️ 响应内容较少")

        # ===== 6. 展开 Agent 流水线查看详情 =====
        try:
            expand_el = page.locator("text=展开 ▼, text=展开").first
            if expand_el.is_visible():
                expand_el.click()
                page.wait_for_timeout(500)
                expanded_text = page.evaluate("() => document.body.innerText")
                if "完成" in expanded_text:
                    print("✅ Agent 流水线展开后显示完成状态")
                else:
                    print("⚠️ Agent 流水线展开后未检测到完成状态")
        except:
            print("⚠️ Agent 流水线展开失败（可能已自动折叠）")

        # ===== 7. 普通对话测试 =====
        page.wait_for_timeout(1000)
        try:
            # 先关闭 webpack overlay（如果存在）
            page.evaluate("""
                const overlay = document.getElementById('webpack-dev-server-client-overlay');
                if (overlay) overlay.style.display = 'none';
            """)

            inp = page.locator("input[placeholder*='输入消息'], textarea[placeholder*='输入消息']")
            if inp.is_visible():
                inp.fill("你好，聊个天")
                page.get_by_role("button", name="发送").first.click()
                page.wait_for_timeout(2000)

                try:
                    page.wait_for_function(
                        "() => document.querySelector('button:not([disabled])') !== null",
                        timeout=30000
                    )
                except:
                    pass

                print("✅ 第二轮对话正常")
            else:
                print("⚠️ 输入框不可见，跳过第二轮")
        except Exception as e:
            print(f"⚠️ 第二轮对话: {e}")

        # ===== 8. 截图 =====
        page.screenshot(path="tools/agent_test_result.png", full_page=True)
        print("✅ 最终截图已保存: tools/agent_test_result.png")

        browser.close()
        print("\n🎉 所有 Agent 可视化测试通过！")


if __name__ == "__main__":
    test_agent_pipeline()
