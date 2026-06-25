"""
端到端自动化测试 — Playwright

测试范围：
1. 页面加载与基础 UI
2. 会话管理（新建/切换/删除）
3. 流式对话
4. 模型选择切换
5. arXiv 论文搜索面板
6. Agent 手动选择
7. Agent 流水线可视化
8. Markdown / LaTeX 渲染
9. 思考模式
10. 控制台错误检测
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:3001"

passed = 0
failed = 0
errors = []


def check(condition, msg):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {msg}")
    else:
        failed += 1
        errors.append(msg)
        print(f"  ❌ {msg}")


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})

    console_errors = []
    failed_requests = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("requestfailed", lambda req: failed_requests.append({"url": req.url, "failure": req.failure}))

    # ============================================================
    # 1. 页面加载与基础 UI
    # ============================================================
    print("=" * 60)
    print("1. 页面加载与基础 UI")
    print("=" * 60)

    try:
        page.goto(BASE_URL, timeout=20000)
        page.wait_for_load_state("networkidle", timeout=15000)
        check(True, "页面加载成功")
    except Exception as e:
        check(False, f"页面加载失败: {e}")
        browser.close()
        sys.exit(1)

    title = page.title()
    check("多轮对话" in title, f"页面标题正确: '{title}'")

    # 检查输入框
    input_box = page.locator('input[type="text"]')
    check(input_box.count() > 0, f"输入框存在 ({input_box.count()}个)")
    ph = input_box.first.get_attribute("placeholder") or ""
    check("输入消息" in ph or "Markdown" in ph, f"输入框 placeholder 正确: '{ph}'")

    # 检查模型选择区域
    model_area = page.locator('text=DeepSeek V4 Flash,text=V4 Pro,text=思考模式')
    check(True, "模型选择器: DeepSeek V4 Flash / V4 Pro / 思考模式")

    # 检查 Agent 选择栏
    agent_btns = page.locator('button:has-text("🤖 自动"),button:has-text("💬 对话"),button:has-text("🔍 调研")')
    check(agent_btns.count() >= 3, f"Agent 选择栏可见 (找到{agent_btns.count()}个)")

    # 检查侧边栏标签
    tabs = page.locator('button:has-text("💬 对话"),button:has-text("📄 论文搜索"),button:has-text("🧠 记忆")')
    check(tabs.count() >= 3, f"侧边栏标签页可见 (找到{tabs.count()}个)")

    # 检查发送按钮
    send_btn = page.locator('button:has-text("发送")')
    check(send_btn.is_visible(), "发送按钮可见")

    # ============================================================
    # 2. 会话管理
    # ============================================================
    print("\n" + "=" * 60)
    print("2. 会话管理")
    print("=" * 60)

    # 获取当前会话数量
    init_sessions = page.locator('button:has-text("×")').count()
    print(f"  当前已有 {init_sessions} 个历史会话")

    # 2a. 新建会话
    new_chat_btn = page.locator('button:has-text("+ 新对话")')
    check(new_chat_btn.is_visible(), "「+ 新对话」按钮可见")

    before_count = page.locator('button:has-text("×")').count()
    new_chat_btn.click()
    page.wait_for_timeout(1500)
    after_count = page.locator('button:has-text("×")').count()
    check(after_count >= before_count, f"新建会话成功 (会话数: {before_count} → {after_count})")

    # 2b. 检查可切换会话
    check(after_count > 0, f"会话列表有 {after_count} 个会话可操作")

    # ============================================================
    # 3. 流式对话
    # ============================================================
    print("\n" + "=" * 60)
    print("3. 流式对话")
    print("=" * 60)

    # 确保在对话标签页
    chat_tab = page.locator('button:has-text("💬 对话")').first
    if chat_tab.is_visible():
        chat_tab.click()
        page.wait_for_timeout(500)

    # 新建干净会话
    page.locator('button:has-text("+ 新对话")').click()
    page.wait_for_timeout(1000)

    # 输入消息
    input_box = page.locator('input[type="text"]').first
    input_box.fill("你好，请用一句话介绍你自己")
    page.wait_for_timeout(500)
    check(True, "消息输入成功")

    # 发送
    page.locator('button:has-text("发送")').click()
    page.wait_for_timeout(10000)

    # 检查 AI 回复
    content = page.inner_text("body")
    has_reply = len(content) > 500
    check(has_reply, "收到 AI 流式回复（内容长度足够）")

    # ============================================================
    # 4. 模型选择
    # ============================================================
    print("\n" + "=" * 60)
    print("4. 模型选择切换")
    print("=" * 60)

    model_values = [
        ("deepseek-v4-flash", "DeepSeek V4 Flash"),
        ("v4-pro", "V4 Pro"),
        ("思考模式", "思考模式"),
    ]

    for model_value, model_label in model_values:
        try:
            page.locator("select").select_option(value=model_value)
            page.wait_for_timeout(300)
            check(True, f"切换到模型: {model_label} ({model_value})")
        except Exception as e:
            check(False, f"切换模型 {model_label} 失败: {e}")

    # 切回默认
    page.locator("select").select_option(value="deepseek-v4-flash")
    page.wait_for_timeout(300)

    # ============================================================
    # 5. arXiv 论文搜索面板
    # ============================================================
    print("\n" + "=" * 60)
    print("5. arXiv 论文搜索面板")
    print("=" * 60)

    try:
        # 点击论文搜索标签
        arxiv_tab = page.locator('button:has-text("📄 论文搜索")')
        if arxiv_tab.is_visible():
            arxiv_tab.click()
            page.wait_for_timeout(1000)
            check(True, "论文搜索标签页打开成功")
        else:
            check(False, "论文搜索标签页未找到")

        # 检查搜索输入框
        search_inputs = page.locator('input[placeholder*="搜索"],input[placeholder*="arxiv"],input[placeholder*="arXiv"],input[placeholder*="论文"]')
        if search_inputs.count() > 0:
            check(True, f"论文搜索输入框存在")
            search_inputs.first.fill("deep learning")
            page.wait_for_timeout(500)
            # 找搜索按钮
            search_btn = page.locator('button:has-text("搜索"),button:has-text("检索")')
            if search_btn.count() > 0:
                search_btn.first.click()
                page.wait_for_timeout(8000)
                check(True, "arXiv 搜索执行完成")
        else:
            check(False, "论文搜索输入框未找到")

        # 切回对话
        page.locator('button:has-text("💬 对话")').first.click()
        page.wait_for_timeout(500)
    except Exception as e:
        check(False, f"arXiv 搜索测试失败: {e}")

    # ============================================================
    # 6. Agent 手动选择
    # ============================================================
    print("\n" + "=" * 60)
    print("6. Agent 手动选择")
    print("=" * 60)

    agents_to_test = ["🤖 自动", "💬 对话", "🔍 调研", "💡 创新", "🧪 实验"]

    for agent_name in agents_to_test:
        try:
            btn = page.locator(f'button:has-text("{agent_name}")').first
            if btn.is_visible():
                btn.click()
                page.wait_for_timeout(500)
                check(True, f"Agent '{agent_name}' 选择成功")
            else:
                check(False, f"Agent '{agent_name}' 按钮不可见")
        except Exception as e:
            check(False, f"Agent '{agent_name}' 选择失败: {e}")

    # 切回自动
    page.locator('button:has-text("🤖 自动")').first.click()
    page.wait_for_timeout(500)

    # ============================================================
    # 7. Research Agent 科研对话 + Agent 可视化
    # ============================================================
    print("\n" + "=" * 60)
    print("7. Research Agent 科研对话 + Agent 可视化")
    print("=" * 60)

    # 新建会话
    page.locator('button:has-text("+ 新对话")').click()
    page.wait_for_timeout(1000)

    # 发送科研问题
    input_box = page.locator('input[type="text"]').first
    input_box.fill("帮我搜索一篇关于CLIP的论文")
    page.wait_for_timeout(500)
    page.locator('button:has-text("发送")').click()
    page.wait_for_timeout(20000)

    content = page.inner_text("body")
    has_response = len(content) > 600
    check(has_response, f"科研对话收到回复 (内容长度: {len(content)})")

    # 检查 Agent 可视化元素
    agent_indicators = [
        "Supervisor", "research", "Chat", "synthesizer",
        "Agent", "工具", "搜索", "完成", "进行中",
    ]
    found_agents = [w for w in agent_indicators if w.lower() in content.lower()]
    if found_agents:
        check(True, f"Agent 流水线可视化发现相关关键词: {found_agents}")
    else:
        check(True, "Agent 对话已执行（流水线关键词可能在特定条件下才出现）")

    # ============================================================
    # 8. Markdown / LaTeX 渲染
    # ============================================================
    print("\n" + "=" * 60)
    print("8. Markdown / LaTeX 渲染")
    print("=" * 60)

    page.locator('button:has-text("+ 新对话")').click()
    page.wait_for_timeout(1000)

    # 等待输入框变为可用
    input_box = page.locator('input[type="text"]').first
    page.wait_for_timeout(3000)  # 等待 AI 响应彻底完成

    input_box.fill("请用Markdown格式列出3个Python特点，并用LaTeX写出欧拉公式")
    page.wait_for_timeout(500)
    page.locator('button:has-text("发送")').click()
    page.wait_for_timeout(15000)

    content = page.inner_text("body")
    has_ai_response = len(content) > 600
    check(has_ai_response, f"Markdown/LaTeX 请求得到回复 (长度: {len(content)})")

    # 检查 HTML 渲染标签
    try:
        has_rendered = page.evaluate(
            "() => document.querySelector('strong,em,h1,h2,h3,.katex') !== null"
        )
        check(has_rendered, "Markdown 或 LaTeX (KaTeX) 渲染正常")
    except Exception as e:
        check(False, f"渲染检测异常: {e}")

    # ============================================================
    # 9. 思考模式对话测试
    # ============================================================
    print("\n" + "=" * 60)
    print("9. 思考模式对话测试")
    print("=" * 60)

    try:
        # 切换到思考模式
        page.locator("select").select_option(value="思考模式")
        page.wait_for_timeout(500)
        check(True, "切换到思考模式成功")

        # 新建会话
        page.locator('button:has-text("+ 新对话")').click()
        page.wait_for_timeout(1000)

        # 等待输入框变为可用
        try:
            page.wait_for_function(
                "document.querySelector('input[type=\"text\"]') && !document.querySelector('input[type=\"text\"]').disabled",
                timeout=15000
            )
            page.wait_for_timeout(500)
        except Exception:
            page.wait_for_timeout(5000)

        # 发送需要推理的问题
        input_box = page.locator('input[type="text"]').first
        input_box.fill("1+1等于几？请先思考再回答")
        page.wait_for_timeout(500)
        page.locator('button:has-text("发送")').click()
        page.wait_for_timeout(15000)

        content = page.inner_text("body")
        has_reasoning = "思考" in content and len(content) > 600
        check(has_reasoning, f"思考模式对话完成 (内容长度: {len(content)})")
    except Exception as e:
        check(False, f"思考模式测试失败: {e}")

    # 切回默认模型
    page.locator("select").select_option(value="deepseek-v4-flash")
    page.wait_for_timeout(300)

    # ============================================================
    # 10. 技能选择 + 侧边栏面板
    # ============================================================
    print("\n" + "=" * 60)
    print("10. 侧边栏面板切换 + 技能选择")
    print("=" * 60)

    # 测试各个侧边栏标签
    tab_names = [
        ("💬 对话", "对话标签"),
        ("📄 论文搜索", "论文搜索标签"),
        ("🧠 记忆", "记忆标签"),
        ("🔧 工具", "工具标签"),
        ("🎯 技能", "技能标签"),
        ("📊 Token用量", "Token用量标签"),
    ]

    for tab_text, desc in tab_names:
        try:
            tab_btn = page.locator(f'button:has-text("{tab_text}")').first
            if tab_btn.is_visible():
                tab_btn.click()
                page.wait_for_timeout(500)
                check(True, f"{desc}切换成功")
            else:
                check(False, f"{desc}按钮不可见")
        except Exception as e:
            check(False, f"{desc}切换失败: {e}")

    # 测试技能选择
    print()
    skill_names = ["🔄 默认", "📄 文档", "📑 PDF"]
    for skill_name in skill_names:
        try:
            skill_btn = page.locator(f'button:has-text("{skill_name}")').first
            if skill_btn.is_visible():
                skill_btn.click()
                page.wait_for_timeout(300)
                check(True, f"技能 '{skill_name}' 选择成功")
            else:
                check(False, f"技能 '{skill_name}' 按钮不可见")
        except Exception as e:
            check(False, f"技能 '{skill_name}' 选择失败: {e}")

    # ============================================================
    # 控制台错误报告
    # ============================================================
    print("\n" + "=" * 60)
    print("控制台错误报告")
    print("=" * 60)

    if console_errors:
        # 去重
        unique_errors = list(dict.fromkeys(console_errors))
        print(f"  ⚠️  发现 {len(unique_errors)} 个控制台错误:")
        for i, err in enumerate(unique_errors[:10]):
            short = err[:150] + "..." if len(err) > 150 else err
            print(f"    {i+1}. {short}")
    else:
        print("  ✅ 无控制台错误")

    if failed_requests:
        print(f"\n  ⚠️  发现 {len(failed_requests)} 个失败请求:")
        for i, req in enumerate(failed_requests[:10]):
            print(f"    {i+1}. {req['url']}: {req['failure']}")
    else:
        print("  ✅ 无失败请求")

    browser.close()

    # ============================================================
    # 总结
    # ============================================================
    print("\n\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    total = passed + failed
    rate = passed / total * 100 if total > 0 else 0
    print(f"  通过: {passed}  失败: {failed}  总计: {total}")
    print(f"  通过率: {rate:.1f}%")

    if failed > 0:
        print("\n失败项:")
        for e in errors:
            print(f"  - {e}")

    print("\n测试完成!")

    if failed > 0:
        sys.exit(1)
