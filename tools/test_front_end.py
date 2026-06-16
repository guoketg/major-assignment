from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    console_messages = []
    page.on('console', lambda msg: console_messages.append({'type': msg.type, 'text': msg.text}))

    failed_requests = []
    page.on('requestfailed', lambda req: failed_requests.append({'url': req.url, 'failure': req.failure}))

    try:
        page.goto('http://localhost:3001', timeout=15000)
        page.wait_for_load_state('networkidle', timeout=10000)
        print('页面加载成功')
        print('页面标题:', page.title())

        content = page.inner_text('body')
        print('页面内容前300字:', content[:300])

    except Exception as e:
        print(f'页面加载失败: {e}')

    if console_messages:
        print('\n控制台消息:')
        for msg in console_messages:
            if msg['type'] in ['error', 'warning']:
                print(f"  [{msg['type']}] {msg['text']}")

    if failed_requests:
        print('\n失败的请求:')
        for req in failed_requests:
            print(f"  {req['url']}: {req['failure']}")

    browser.close()
