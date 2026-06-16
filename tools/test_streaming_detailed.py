from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    console_errors = []
    page.on('console', lambda msg: console_errors.append(f"[{msg.type}] {msg.text}") if msg.type == 'error' else None)
    failed_requests = []
    page.on('requestfailed', lambda req: failed_requests.append(f"{req.url}: {req.failure}"))

    try:
        # 1. 加载页面
        print('1. 正在加载页面...')
        page.goto('http://localhost:3001', timeout=30000)
        page.wait_for_load_state('domcontentloaded', timeout=15000)
        print('   页面 DOM 加载成功')

        # 等待侧边栏出现
        page.wait_for_selector('text=对话历史', timeout=10000)
        print('2. 侧边栏加载成功')

        # 检查会话列表
        session_list = page.locator('text=暂无会话记录')
        if session_list.is_visible():
            print('3. 会话列表为空状态正确')

        # 新建会话
        page.get_by_role('button', name='+ 新对话').click()
        page.wait_for_timeout(2000)
        print('4. 新建会话成功')

        # 输入消息
        page.fill('input[placeholder*="Markdown"]', '测试')
        page.get_by_role('button', name='发送').click()
        print('5. 消息已发送，等待响应...')

        # 等待 AI 回复（最多60秒）
        page.wait_for_timeout(60000)

        # 检查页面内容
        content = page.inner_text('body')
        print(f'6. 页面内容长度: {len(content)} 字符')
        print(f'   内容预览: {content[:300]}...')

        if 'AI' in content or '你好' in content:
            print('7. 检测到 AI 回复')
        else:
            print('7. 未检测到 AI 回复')

        if console_errors:
            print('\n控制台错误:')
            for err in console_errors[:10]:
                print(f'   {err}')

        if failed_requests:
            print('\n失败的请求:')
            for req in failed_requests[:10]:
                print(f'   {req}')

    except Exception as e:
        print(f'测试失败: {e}')
        if console_errors:
            print('\n控制台错误:')
            for err in console_errors[:10]:
                print(f'   {err}')
        if failed_requests:
            print('\n失败的请求:')
            for req in failed_requests[:10]:
                print(f'   {req}')

    browser.close()
