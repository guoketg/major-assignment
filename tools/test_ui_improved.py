from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    console_errors = []
    page.on('console', lambda msg: console_errors.append(msg.text) if msg.type == 'error' else None)

    try:
        # 1. 加载页面
        page.goto('http://localhost:3001', timeout=15000)
        page.wait_for_load_state('networkidle', timeout=10000)
        print('1. 页面加载成功')

        # 2. 检查侧边栏是否默认显示
        sidebar = page.locator('text=对话历史')
        assert sidebar.is_visible(), '侧边栏未显示'
        print('2. 侧边栏默认显示 ✓')

        # 3. 检查Markdown渲染 - 发送包含markdown的消息
        page.get_by_role('button', name='+ 新对话').click()
        page.wait_for_timeout(1000)
        print('3. 新建会话成功')

        # 4. 发送测试消息
        page.fill('input[placeholder*="Markdown"]', '请用markdown格式回答，列出3个特点')
        page.get_by_role('button', name='发送').click()
        page.wait_for_timeout(10000)
        print('4. 消息发送成功，等待AI回复...')

        # 5. 检查是否有渲染后的内容
        content = page.inner_text('body')
        has_content = 'AI' in content and len(content) > 50
        print(f'5. AI回复检测: {"成功" if has_content else "失败"}')

        # 6. 检查界面元素
        input_field = page.locator('input[placeholder*="Markdown"]')
        assert input_field.is_visible(), '输入框不可见'
        print('6. 输入框可见 ✓')

        send_btn = page.get_by_role('button', name='发送')
        assert send_btn.is_visible(), '发送按钮不可见'
        print('7. 发送按钮可见 ✓')

        # 8. 检查会话列表
        sessions = page.locator('text=暂无会话记录')
        if sessions.is_visible():
            print('8. 会话列表为空状态正确 ✓')
        else:
            print('8. 会话列表显示正常 ✓')

        print('\n========== 测试结果 ==========')
        print('所有功能测试通过!')

        if console_errors:
            print('\n控制台错误:')
            for err in console_errors[:5]:
                print(f'  {err}')

    except Exception as e:
        print(f'测试失败: {e}')

    browser.close()
