"""
Chat Agent（对话助手）

处理普通对话、闲聊、知识问答等不需要专门 Agent 的场景。
直接调用 LLM 生成回复，可调用 web_search 获取实时信息。
"""
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from backend.agent.state import AgentState
from backend.agent.llm import get_llm
from backend.tools.web_search_tool import web_search

CHAT_SYSTEM_PROMPT = """你是一个深度学习研究助手，友好、专业。

你可以：
1. 回答深度学习、机器学习相关的知识问题
2. 帮助用户理解论文中的概念
3. 讨论技术细节和实现方法
4. 进行日常对话

当用户问到实时信息、最新新闻、不确定的知识时，使用 web_search 工具进行联网搜索。
如果对方只是打招呼或简单聊天，直接回复即可。

可用工具：
- web_search(query, count): 联网搜索最新信息和新闻

回答时使用中文，适当使用 Markdown 格式提升可读性。
支持 LaTeX 数学公式（用 $$ 包裹）。"""


def _execute_tool_calls(response, llm_messages):
    """执行 LLM 的工具调用并返回更新后的消息列表"""
    for tc in response.tool_calls:
        try:
            if tc["name"] == "web_search":
                tool_result = web_search.invoke(tc["args"])
            else:
                tool_result = f"未知工具: {tc['name']}"
        except Exception as e:
            tool_result = f"工具调用失败: {e}"
        llm_messages.append(ToolMessage(
            content=str(tool_result)[:4000],
            tool_call_id=tc["id"],
            name=tc["name"],
        ))
    return llm_messages


def chat_agent_node(state: AgentState) -> Dict[str, Any]:
    """Chat Agent 节点 — 默认对话处理

    可调用 web_search 获取实时信息。
    完成后将 current_agent 设为 "synthesize" 以进入综合阶段。
    """
    messages = state.get("messages", [])
    model = state.get("model", "deepseek-chat")

    # 构建消息列表（包含系统提示）
    from langchain_core.messages import SystemMessage, HumanMessage
    llm_messages = [
        SystemMessage(content=CHAT_SYSTEM_PROMPT),
        *[_to_langchain_msg(m) for m in messages],
    ]

    try:
        # 绑定工具的 LLM（可调用 web_search）
        llm = get_llm(model).bind_tools([web_search])
        response = llm.invoke(llm_messages)

        reply = response.content or ""

        # 如果有工具调用，执行并再次调用 LLM
        if response.tool_calls:
            llm_messages.append(response)
            llm_messages = _execute_tool_calls(response, llm_messages)

            # 第二次调用生成最终回复
            response2 = llm.invoke(llm_messages)
            reply = response2.content or reply

        # 将回复追加到消息历史
        new_messages = messages + [
            {"role": "assistant", "content": reply, "timestamp": __import__("datetime").datetime.now().isoformat()}
        ]

        return {
            "messages": new_messages,
            "output_text": reply,
            "current_agent": "synthesize",
            "sub_task_queue": state.get("sub_task_queue", []),
        }

    except Exception as e:
        error_msg = f"生成回复失败: {str(e)}"
        return {
            "output_text": error_msg,
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


def chat_router(state: AgentState) -> str:
    """chat_router"""
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
    _log.info(f"[chat_router] tasks={len(plan_tasks)}, pending={len(pending)}")
    if pending:
        return "next_agent"
    return "synthesize"