"""
列出 DashScope MCP 服务器上可用的工具
"""
import os
import sys
import json

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MCP_URL = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
api_key = os.getenv("DASHSCOPE_API_KEY", "")

if not api_key:
    print("[ERROR] DASHSCOPE_API_KEY not set")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
}

with httpx.Client(timeout=30, headers=headers) as client:
    # 1. Initialize
    print("1. 初始化会话...")
    resp = client.post(MCP_URL, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "0.1.0",
            "capabilities": {},
            "clientInfo": {"name": "test-agent", "version": "1.0.0"},
        },
    })
    print(f"   状态: {resp.status_code}")
    print(f"   响应: {resp.text[:500]}")

    # 2. List tools
    print("\n2. 列出可用工具...")
    resp = client.post(MCP_URL, json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        "params": {},
    })
    print(f"   状态: {resp.status_code}")
    try:
        data = resp.json()
        print(f"   响应: {json.dumps(data, ensure_ascii=False, indent=2)[:2000]}")
    except:
        print(f"   原始响应: {resp.text[:1000]}")

    # 3. Try to call tool with correct name
    print("\n3. 尝试搜索...")
    resp = client.post(MCP_URL, json={
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {
            "name": "web_search",
            "arguments": {"query": "CLIP模型最新进展", "max_results": 5},
        },
    })
    print(f"   状态: {resp.status_code}")
    try:
        data = resp.json()
        print(f"   响应: {json.dumps(data, ensure_ascii=False, indent=2)[:2000]}")
    except:
        print(f"   原始响应: {resp.text[:1000]}")
