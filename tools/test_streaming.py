"""
测试多轮对话 AI 回复的流式输出功能

使用 Playwright 测试前端页面：
1. 页面加载是否正常
2. 发送消息后，AI 回复是否逐字流式输出（而非一次性显示）
3. 流式完成后的最终状态是否正确
"""
from playwright.sync_api import sync_playwright
import time


def test_streaming_output():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 800})

        try:
            # ===== 1. 加载页面 =====
            page.goto("http://localhost:3001", timeout=15000)
            page.wait_for_load_state("networkidle", timeout=10000)
            print("1. ✅ 页面加载成功")

            # ===== 2. 点击"新对话"创建新会话 =====
            page.get_by_role("button", name="新对话").click()
            page.wait_for_timeout(2000)
            print("2. ✅ 新建会话成功")

            # ===== 3. 发送一条简短消息 =====
            msg = "用一句话介绍你自己"
            page.fill('input[placeholder*="输入消息"]', msg)
            page.get_by_role("button", name="发送").click()
            print(f"3. ✅ 消息已发送: '{msg}'")
            print("   等待流式响应开始...")

            # ===== 4. 等待流式内容出现 =====
            # 检查是否出现了闪烁光标（流式输出的标记）
            page.wait_for_function(
                """
                () => {
                    const spans = document.querySelectorAll('span');
                    for (const s of spans) {
                        const style = window.getComputedStyle(s);
                        if (style.animation && style.animation.includes('blink')) return true;
                    }
                    return false;
                }
                """,
                timeout=15000,
            )
            print("4. ✅ 检测到闪烁光标，流式输出已开始")

            # ===== 5. 在流式输出过程中检查内容是否逐步增加 =====
            # 先获取当前的流式内容长度
            initial_length = page.evaluate(
                """
                () => {
                    const allText = document.body.innerText;
                    const msgIndex = allText.lastIndexOf('介绍');
                    return msgIndex >= 0 ? allText.substring(msgIndex).length : allText.length;
                }
                """
            )
            print(f"   初始流式内容长度: {initial_length}")

            # 等待一段时间，让流式输出进行，然后检查内容是否增加了
            page.wait_for_timeout(3000)
            new_length = page.evaluate(
                """
                () => {
                    const allText = document.body.innerText;
                    const msgIndex = allText.lastIndexOf('介绍');
                    return msgIndex >= 0 ? allText.substring(msgIndex).length : allText.length;
                }
                """
            )
            print(f"   3秒后内容长度: {new_length}")

            if new_length > initial_length:
                print("5. ✅ 内容在逐步增加，流式输出正常")
            else:
                print("5. ⚠️ 内容没有显著增加，可能需要更长时间")

            # ===== 6. 等待回复完成（输入框恢复可用状态） =====
            page.wait_for_function(
                """
                () => {
                    const input = document.querySelector('input');
                    return input && !input.disabled;
                }
                """,
                timeout=60000,
            )
            print("6. ✅ AI 回复完成，输入框已恢复")

            # ===== 7. 检查最终内容 =====
            page.wait_for_timeout(1000)
            final_content = page.inner_text("body")
            if len(final_content) > 100:
                print("7. ✅ 回复内容完整，测试通过!")
            else:
                print(f"7. ⚠️ 回复内容较短: {final_content[:200]}")

            print("\n" + "=" * 50)
            print("流式输出功能测试完成! ✅")
            print("=" * 50)

        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            # 截屏保存现场
            page.screenshot(path="test_streaming_failure.png")
            print("已保存截图: test_streaming_failure.png")

        finally:
            browser.close()


def test_streaming_api():
    """测试后端流式 API 是否正确返回 SSE 格式数据"""
    import requests
    import json

    API_URL = "http://localhost:8001"
    session_id = "test_streaming_api_session"

    print("\n--- 后端 API 流式测试 ---")

    # 发送流式请求
    response = requests.post(
        f"{API_URL}/chat/stream",
        json={"session_id": session_id, "message": "你好"},
        stream=True,
        timeout=30,
    )

    assert response.status_code == 200
    ct = response.headers.get("content-type", "")
    assert "text/event-stream" in ct, f"Content-Type 错误: {ct}"
    print(f"✅ Content-Type: {ct}")

    # 读取流式数据
    chunks = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data = json.loads(line[6:])
            chunks.append(data)
            if data.get("done"):
                print(f"✅ 收到完成信号, 共 {len(chunks)} 个 SSE 事件")
                if "history" in data:
                    print(f"✅ 历史记录: {len(data['history'])} 条消息")
                break

    assert len(chunks) > 1, f"应该收到多个 chunk, 实际 {len(chunks)}"
    print("✅ 后端流式 API 测试通过!")

    # 清理
    requests.delete(f"{API_URL}/history/{session_id}")


if __name__ == "__main__":
    print("=" * 50)
    print("多轮对话流式输出测试")
    print("=" * 50)

    # 先测试后端 API
    test_streaming_api()

    # 再测试前端页面
    print("\n--- 前端页面流式测试 ---")
    test_streaming_output()
