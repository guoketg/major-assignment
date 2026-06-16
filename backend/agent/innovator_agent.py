"""
Innovator Agent（创新构思 Agent）

科研方法论专家，基于调研结果构思创新方案。
可使用 web_search 查找最新研究趋势，使用 docx 工具保存创新方案。

流程：
1. 读取 memory.papers_archive
2. 分析现有方法的局限性
3. 构思 ≥3 种创新路径
4. 多维对比（新颖性、复杂度、效果潜力）
5. 更新 memory.innovation_candidates
6. 可选：保存到 Word 文档
"""
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from backend.agent.state import AgentState
from backend.agent.llm import get_llm
from backend.tools.web_search_tool import web_search
from backend.tools.docx_tool import create_docx, add_section, add_table
from backend.memory.manager import get_context_for_agent

INNOVATOR_SYSTEM_PROMPT = """你是一个深度学习创新方法论专家。

你的任务：基于已有的文献调研结果，构思创新的研究方向。

工作流程：
1. 分析现有方法的局限性
2. 如需查找最新研究趋势，使用 web_search 搜索
3. 如需保存方案到 Word 文档，使用 create_docx/add_section/add_table 工具

对每种创新方案，请从以下维度评估：
- 新颖性（high/medium/low）
- 实现复杂度（high/medium/low）
- 预期效果提升（high/medium/low）
- 计算开销（high/medium/low）
- 与基线方法的兼容性

用表格对比不同方案。

可用工具：
- web_search(query, count): 联网搜索最新研究趋势
- create_docx(title): 创建 Word 文档
- add_section(filepath, heading, content): 添加章节
- add_table(filepath, headers, rows): 添加表格

如果已有调研结果（papers_archive），请基于它们提出创新。
如果没有，请基于你的知识提出有前景的研究方向。

回复请用中文。
"""


def innovator_agent_node(state: AgentState) -> Dict[str, Any]:
    """Innovator Agent 节点 — 创新构思"""
    import logging
    logger = logging.getLogger(__name__)

    messages = state.get("messages", [])
    model = state.get("model", "deepseek-chat")
    memory = state.get("memory", {})

    if not messages:
        return {"current_agent": "synthesize"}

    context = get_context_for_agent("innovator_agent", memory)
    system_prompt = INNOVATOR_SYSTEM_PROMPT
    if context:
        system_prompt += f"\n\n上下文:\n{context}"

    # 添加上下文：已有的论文列表
    papers = memory.get("papers_archive", [])
    if papers:
        paper_summary = "已有文献:\n" + "\n".join(
            f"- {p.get('title', '')}" for p in papers[-10:]
        )
        system_prompt += f"\n\n{paper_summary}"

    try:
        # 绑定所有工具的 LLM
        tools_list = [web_search, create_docx, add_section, add_table]
        llm = get_llm(model).bind_tools(tools_list)

        llm_messages = [
            SystemMessage(content=system_prompt),
            *[_to_langchain_msg(m) for m in messages],
        ]

        response = llm.invoke(llm_messages)
        reply = response.content or ""

        # 如果有工具调用，执行并再次调用 LLM
        if response.tool_calls:
            llm_messages.append(response)
            for tc in response.tool_calls:
                try:
                    if tc["name"] == "web_search":
                        tool_result = web_search.invoke(tc["args"])
                    elif tc["name"] == "create_docx":
                        tool_result = create_docx.invoke(tc["args"])
                    elif tc["name"] == "add_section":
                        tool_result = add_section.invoke(tc["args"])
                    elif tc["name"] == "add_table":
                        tool_result = add_table.invoke(tc["args"])
                    else:
                        tool_result = f"未知工具: {tc['name']}"
                except Exception as e:
                    tool_result = f"工具调用失败: {e}"
                llm_messages.append(ToolMessage(
                    content=str(tool_result)[:4000],
                    tool_call_id=tc["id"],
                    name=tc["name"],
                ))
            # 第二次调用生成最终回复
            response2 = llm.invoke(llm_messages)
            reply = response2.content or reply

        # 更新创新方案记忆
        import re
        updated_memory = dict(memory)
        candidates = updated_memory.get("innovation_candidates", [])

        plan_patterns = re.findall(
            r'(?:方案|方法|思路)[：:]\s*(.+?)(?:\n|$)',
            reply,
        )
        for plan in plan_patterns[:5]:
            plan = plan.strip()
            if plan and not any(c.get("name") == plan for c in candidates):
                candidates.append({
                    "name": plan,
                    "novelty": "medium",
                    "potential": "medium",
                    "difficulty": "medium",
                    "status": "proposed",
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                })

        updated_memory["innovation_candidates"] = candidates

        # 追加到消息历史
        timestamp = __import__("datetime").datetime.now().isoformat()
        new_messages = messages + [
            {"role": "assistant", "content": reply, "timestamp": timestamp}
        ]

        return {
            "messages": new_messages,
            "output_text": reply,
            "memory": updated_memory,
            "current_agent": "synthesize",
            "sub_task_queue": state.get("sub_task_queue", []),
        }

    except Exception as e:
        logger.error(f"[innovator] Error: {e}")
        return {
            "output_text": f"创新构思失败: {str(e)}",
            "current_agent": "synthesize",
        }


def _to_langchain_msg(msg: Dict[str, Any]):
    """将原始消息字典转为 LangChain 消息对象"""
    role = msg.get("role", "")
    content = msg.get("content", "")
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        return HumanMessage(content=content)


def innovator_router(state: AgentState) -> str:
    """innovator_router"""
    import logging
    _log = logging.getLogger(__name__)
    session_id = state.get("session_id", "")
    plan_tasks = []
    try:
        from backend.agent.supervisor import _SUBTASK_CACHE
        cached = _SUBTASK_CACHE.get(session_id, {})
        plan_tasks = cached.get("plan", [])
    except ImportError:
        pass
    if not plan_tasks:
        plan_tasks = state.get("sub_task_queue", [])
    pending = [t for t in plan_tasks if isinstance(t, dict) and t.get("status") == "pending"]
    _log.info(f"[innovator_router] tasks={len(plan_tasks)}, pending={len(pending)}")
    if pending:
        return "next_agent"
    return "synthesize"