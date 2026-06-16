"""
Experiment Agent（实验设计与分析 Agent）

负责设计消融实验方案、分析用户实验结果、迭代优化。
可使用 web_search 查找实验方法，docx 工具保存实验日志。

流程：
1. 读取 memory.innovation_candidates
2. 设计消融实验 → 拆分组件
3. 分析用户反馈 → 定位瓶颈
4. 更新 memory.experiment_log
5. 可选：保存到 Word 文档
"""
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from backend.agent.state import AgentState
from backend.agent.llm import get_llm
from backend.tools.web_search_tool import web_search
from backend.tools.docx_tool import create_docx, add_section, add_table
from backend.memory.manager import get_context_for_agent

EXPERIMENT_SYSTEM_PROMPT = """你是一个深度学习实验设计专家。

你的任务：设计消融实验方案，分析实验结果。

工作流程：
1. 分析需要验证的创新方案
2. 设计消融实验（逐步拆解组件）
3. 如需查找实验方法，使用 web_search 搜索
4. 如需保存实验日志到 Word，使用 create_docx/add_section/add_table
5. 分析实验结果、定位性能瓶颈、提出改进建议

实验设计原则：
- 每次只改变一个变量
- 设置合理的基线
- 包含必要的控制实验
- 结果分析要量化

可用工具：
- web_search(query, count): 联网搜索实验方法和基线结果
- create_docx(title): 创建实验日志文档
- add_section(filepath, heading, content): 添加章节
- add_table(filepath, headers, rows): 添加实验数据表格

回复请用中文。
"""


def experiment_agent_node(state: AgentState) -> Dict[str, Any]:
    """Experiment Agent 节点 — 实验分析"""
    import logging
    logger = logging.getLogger(__name__)

    messages = state.get("messages", [])
    model = state.get("model", "deepseek-chat")
    memory = state.get("memory", {})

    if not messages:
        return {"current_agent": "synthesize"}

    context = get_context_for_agent("experiment_agent", memory)
    system_prompt = EXPERIMENT_SYSTEM_PROMPT
    if context:
        system_prompt += f"\n\n上下文:\n{context}"

    # 添加上下文：已有的创新方案
    candidates = memory.get("innovation_candidates", [])
    if candidates:
        cand_summary = "待验证的创新方案:\n" + "\n".join(
            f"- {c.get('name', '')} (难度:{c.get('difficulty', '?')}, 潜力:{c.get('potential', '?')})"
            for c in candidates[-5:]
        )
        system_prompt += f"\n\n{cand_summary}"

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

        # 记录实验日志
        updated_memory = dict(memory)
        logs = updated_memory.get("experiment_log", [])
        logs.append({
            "step": len(logs) + 1,
            "analysis": reply[:200],
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        })
        updated_memory["experiment_log"] = logs

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
        logger.error(f"[experiment] Error: {e}")
        return {
            "output_text": f"实验分析失败: {str(e)}",
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


def experiment_router(state: AgentState) -> str:
    """experiment_router"""
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
    _log.info(f"[experiment_router] tasks={len(plan_tasks)}, pending={len(pending)}")
    if pending:
        return "next_agent"
    return "synthesize"