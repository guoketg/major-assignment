import requests
import json

# 测试流式 API
response = requests.post(
    'http://localhost:8001/chat/stream',
    json={'session_id': 'test_stream', 'message': '你好，请回复短一点'},
    stream=True
)

print(f"状态码: {response.status_code}")
print(f"Content-Type: {response.headers.get('Content-Type')}")
print("\n流式响应内容:")

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            if not data.get('done'):
                print(f"片段: {data.get('content')}", end='', flush=True)
            else:
                print(f"\n\n完成! 历史消息数: {len(data.get('history', []))}")
