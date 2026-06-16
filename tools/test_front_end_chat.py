from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    console_errors = []
    page.on('console', lambda msg: console_errors.append(msg.text) if msg.type == 'error' else None)
    failed_requests = []
    page.on('requestfailed', lambda req: failed_requests.append({'url': req.url, 'failure': req.failure}))

    try:
        # 1. 加载页面
        page.goto('http://localhost:3001', timeout=15000)
        page.wait_for_load_state('networkidle', timeout=10000)
        print('1. 页面加载成功')

        # 2. 点击"新建会话"按钮
        page.get_by_role('button', name='新建会话').click()
        page.wait_for_timeout(1000)
        print('2. 点击新建会话成功')

        # 3. 检查当前是否有 session_id
        current_text = page.inner_text('body')
        if 'session_' in current_text:
            print('3. 会话创建成功，当前有 session_id')
        else:
            print('3. 当前文本:', current_text[:200])

        # 4. 输入消息
        page.fill('input[type="text"]', '你好，你叫什么名字？')
        page.wait_for_timeout(500)
        print('4. 输入消息成功')

        # 5. 点击发送
        page.get_by_text('发送').click()
        page.wait_for_timeout(5000)  # 等待 AI 回复

        # 6. 检查是否有 AI 回复
        content = page.inner_text('body')
        if 'AI' in content or '你好' in content:
            print('5. 收到 AI 回复')
        else:
            print('5. 未收到 AI 回复，页面内容:', content[:300])

        print('6. 测试完成!')

    except Exception as e:
        print(f'测试失败: {e}')

    if console_errors:
        print('\n控制台错误:')
        for err in console_errors:
            print(f'  {err}')

    if failed_requests:
        print('\n失败的请求:')
        for req in failed_requests:
            print(f'  {req["url"]}: {req["failure"]}')

    browser.close()
