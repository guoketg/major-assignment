"""
测试 Research Agent 的 LLM 调用逻辑

检查：
1. LLM 是否真正调用了 search_arxiv 工具
2. 工具调用前后的内容是什么
3. 最终 reply 包含了什么
"""
import sys
import os
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.research_agent import research_agent_node, _to_langchain_msg
from backend.agent.llm import get_llm
from backend.tools.arxiv_tool import search_arxiv

# ====== 测试 1: 检查 LLM + 工具绑定 ======
print("=" * 60)
print("测试 1: 检查 LLM + 工具绑定")
print("=" * 60)

model = "deepseek-chat"
try:
    llm = get_llm(model).bind_tools([search_arxiv])
    print(f"[OK] LLM 创建成功: {llm}")

    # 检查绑定的工具
    if hasattr(llm, 'bound') and llm.bound:
        print(f"[OK] 工具已绑定")
    else:
        print(f"[WARN] 可能未绑定工具")

    # 检查工具定义
    if hasattr(llm, 'kwargs'):
        tools = llm.kwargs.get('tools', [])
        if tools:
            print(f"[OK] 工具定义: {json.dumps(tools, ensure_ascii=False, indent=2)[:500]}")
        else:
            print(f"[WARN] 未找到工具定义")
except Exception as e:
    print(f"[ERROR] {e}")

# ====== 测试 2: 直接调用 LLM 看是否返回 tool_calls ======
print("\n" + "=" * 60)
print("测试 2: LLM 是否返回 tool_calls")
print("=" * 60)

try:
    from langchain_core.messages import SystemMessage, HumanMessage

    system_prompt = """你是一个深度学习文献调研专家。

你的工作流程：
1. 分析用户的研究问题
2. 如需检索论文，使用 search_arxiv 工具（可多次搜索不同关键词）
3. 阅读并分析每篇论文的核心方法、创新点和局限性
4. 给出详细的结构化调研结果

重要规则：
- 最终回复必须包含：该领域简介、核心方法分类、各方法优缺点、未来方向
- 每次调研都要给出完整全面的分析，不少于 300 字
- 使用中文回答

可用工具：
- search_arxiv(query, max_results): 搜索 arXiv 论文

注意：如果已有足够知识可直接回答，无需调用工具。
搜索时使用具体的关键词，一次搜索不够可以多次。"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="帮我调研一下CLIP模型的最新进展"),
    ]

    print("正在调用 LLM...")
    response = llm.invoke(messages)

    print(f"\n[DEBUG] response.type = {type(response)}")
    print(f"[DEBUG] response.content = {response.content[:200] if response.content else '(空)'}")
    print(f"[DEBUG] response.tool_calls = {response.tool_calls}")
    print(f"[DEBUG] response.additional_kwargs keys = {list(response.additional_kwargs.keys())}")

    if response.tool_calls:
        print(f"\n[OK] 模型调用了工具！")
        for tc in response.tool_calls:
            print(f"  工具名: {tc['name']}")
            print(f"  参数: {tc['args']}")
    else:
        print(f"\n[WARN] 模型没有调用工具！")
        print(f"  完整内容前500字: {response.content[:500]}")

except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

# ====== 测试 3: 模拟完整 research_agent_node 调用 ======
print("\n" + "=" * 60)
print("测试 3: 模拟 research_agent_node 调用")
print("=" * 60)

try:
    from backend.agent.state import create_initial_state

    state = create_initial_state(
        session_id="test_session",
        model=model,
        messages=[
            {"role": "user", "content": "帮我调研一下CLIP模型的最新进展"},
        ],
    )

    print("正在执行 research_agent_node...")
    result = research_agent_node(state)

    print(f"\n[DEBUG] current_agent = {result.get('current_agent')}")
    output = result.get("output_text", "")
    print(f"[DEBUG] output_text 长度 = {len(output)}")
    print(f"[DEBUG] output_text 前200字 = {output[:200]}")
    print(f"[DEBUG] output_text 后200字 = {output[-200:]}")

    msgs = result.get("messages", [])
    if msgs:
        last = msgs[-1]
        print(f"\n[DEBUG] messages 最后一条 role={last.get('role')}, content_len={len(last.get('content',''))}")

    memory = result.get("memory", {})
    papers = memory.get("papers_archive", [])
    print(f"[DEBUG] 归档论文数: {len(papers)}")
    for p in papers:
        print(f"  - {p.get('title', 'N/A')}")

except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
