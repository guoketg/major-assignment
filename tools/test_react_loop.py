"""
测试 ReAct 循环的轮次行为
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 启用详细日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

from backend.agent.research_agent import research_agent_node
from backend.agent.state import create_initial_state

state = create_initial_state(
    session_id="test_session",
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "帮我调研一下CLIP模型的最新进展"},
    ],
)

print("\n" + "=" * 60)
print("调用 research_agent_node (带详细日志)")
print("=" * 60)

result = research_agent_node(state)

print("\n" + "=" * 60)
print("结果")
print("=" * 60)
output = result.get("output_text", "")
print(f"output_text 长度: {len(output)}")
print(f"完整 output_text:")
print(output)
print(f"\ncurrent_agent: {result.get('current_agent')}")
msgs = result.get("messages", [])
if msgs:
    last = msgs[-1]
    print(f"messages 最后一条: role={last.get('role')}, content_len={len(last.get('content',''))}")
print("=" * 60)
