"""
详细测试 Research Agent 的 LLM 调用逻辑
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.research_agent import research_agent_node, _to_langchain_msg, RESEARCH_SYSTEM_PROMPT
from backend.agent.llm import get_llm
from backend.tools.arxiv_tool import search_arxiv
from backend.agent.state import create_initial_state
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# ====== 模拟 research_agent_node 内部流程 ======
print("=" * 60)
print("逐步追踪 research_agent_node 内部流程")
print("=" * 60)

model = "deepseek-chat"
state = create_initial_state(
    session_id="test_session",
    model=model,
    messages=[
        {"role": "user", "content": "帮我调研一下CLIP模型的最新进展"},
    ],
)

# 重建 research_agent_node 的内部逻辑
messages = state.get("messages", [])
memory = state.get("memory", {})
context = ""  # get_context_for_agent("research_agent", memory)
system_prompt = RESEARCH_SYSTEM_PROMPT
if context:
    system_prompt += f"\n\n当前上下文:\n{context}"

llm = get_llm(model).bind_tools([search_arxiv])

llm_messages = [
    SystemMessage(content=system_prompt),
    *[_to_langchain_msg(m) for m in messages],
]

print(f"用户消息: {messages[-1]['content']}")

# === 第1轮 LLM 调用 ===
print("\n--- 第1轮 LLM 调用 ---")
response = llm.invoke(llm_messages)
first_reply = response.content or ""
print(f"content长度: {len(first_reply)}")
print(f"content前100字: {first_reply[:100]}")
print(f"tool_calls数量: {len(response.tool_calls)}")
for tc in response.tool_calls:
    print(f"  工具调用: {tc['name']}({json.dumps(tc['args'])})")

# === 执行工具 ===
if response.tool_calls:
    print("\n--- 执行工具 ---")
    llm_messages.append(response)

    for i, tc in enumerate(response.tool_calls):
        if tc["name"] == "search_arxiv":
            print(f"\n[工具 {i+1}] search_arxiv({tc['args']})")
            try:
                tool_result = search_arxiv.invoke(tc["args"])
                print(f"  [结果] 长度={len(str(tool_result))}")
                print(f"  [结果] 前100字: {str(tool_result)[:100]}")
            except Exception as e:
                print(f"  [ERROR] {e}")
                tool_result = f"工具调用失败: {str(e)}"

            llm_messages.append(ToolMessage(
                content=str(tool_result)[:4000],
                tool_call_id=tc["id"],
                name=tc["name"],
            ))

    # === 第2轮 LLM 调用 ===
    print("\n--- 第2轮 LLM 调用 ---")
    try:
        response2 = llm.invoke(llm_messages)
        second_reply = response2.content or ""
        print(f"second_reply长度: {len(second_reply)}")
        print(f"second_reply前200字: {second_reply[:200]}")
        print(f"second_reply后200字: {second_reply[-200:]}")
        print(f"second_reply是否有tool_calls: {bool(response2.tool_calls)}")

        # 合并逻辑
        if second_reply and len(second_reply) > 20:
            reply = second_reply
            print(f"\n[决策] 使用 second_reply (长度={len(second_reply)})")
        elif first_reply:
            reply = first_reply
            print(f"\n[决策] 使用 first_reply (长度={len(first_reply)})")
        else:
            reply = second_reply or first_reply or "调研完成，但未生成具体内容。"
            print(f"\n[决策] 使用 fallback")

        print(f"\n最终 reply 长度: {len(reply)}")
        print(f"最终 reply 前300字: {reply[:300]}")

    except Exception as e:
        print(f"[ERROR] 第2轮调用失败: {e}")
        import traceback
        traceback.print_exc()
        reply = first_reply or "调研失败"

print("\n" + "=" * 60)
print("测试完成")
