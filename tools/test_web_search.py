"""
测试 DashScope MCP 联网搜索工具
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.tools.web_search_tool import web_search

print("=" * 60)
print("测试 DashScope MCP 联网搜索")
print("=" * 60)

api_key = os.getenv("DASHSCOPE_API_KEY", "")
if not api_key:
    print("[WARN] DASHSCOPE_API_KEY 未设置")
else:
    print(f"[OK] DASHSCOPE_API_KEY 已配置: {api_key[:10]}...")

print("\n搜索: CLIP模型 最新进展 2025")
print("-" * 40)
result = web_search.invoke({"query": "CLIP模型 最新进展 2025", "max_results": 5})
print(f"\n结果长度: {len(result)}")
print(f"结果前500字:\n{result[:500]}")
print("..." if len(result) > 500 else "")
print(f"结果后200字:\n{result[-200:]}" if len(result) > 200 else result)

print("\n" + "=" * 60)
print("测试完成")
