"""
Research Agent（文献调研 Agent）

深度学习文献专家，负责检索 arXiv、筛选论文、分类总结。

模式：ReAct 循环
1. 用 LLM 判断是否需要搜索 arXiv → 调用 search_arxiv 工具
2. 分析返回的论文 → 按「已证实/争议/不足」分类
3. 归档到 memory.papers_archive
4. 生成调研摘要
"""
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool as tool_decorator

from backend.agent.state import AgentState
from backend.agent.llm import get_llm
from backend.tools.arxiv_tool import search_arxiv
from backend.tools.web_search_tool import web_search
from backend.memory.manager import get_context_for_agent

RESEARCH_SYSTEM_PROMPT = """你是一个深度学习文献调研专家。

你的工作流程：
1. 分析用户的研究问题
2. 如需检索论文，使用 search_arxiv 工具（可多次搜索不同关键词）
3. 阅读并分析每篇论文的核心方法、创新点和局限性
4. 按以下标准分类：
   - ✅ 已证实：多篇高水平论文一致验证的结论
   - ⚠️ 争议点：不同文献结论相悖
   - ❓ 资料不足：暂无公开研究
5. 归档到 papers_archive（记录标题、方法、分类、证据等级）
6. 给出详细的结构化调研结果

重要规则：
- 搜索到论文后，请仔细阅读论文标题和摘要
- 最终回复必须包含：该领域简介、核心方法分类、各方法优缺点、未来方向
- 每次调研都要给出完整全面的分析，不少于 300 字
- 使用中文回答，合理使用 Markdown 格式
- 不要只说"让我搜索"或"我会调研"，而是直接给出具体的调研结果

可用工具：
- search_arxiv(query, max_results): 搜索 arXiv 学术论文（适合找正式发表的学术文献）
- web_search(query, count): 使用阿里云百炼进行实时联网搜索（适合找最新新闻、实时信息、博客、百科等）

注意：如果已有足够知识可直接回答，无需调用工具。
搜索时使用具体的关键词，一次搜索不够可以多次。
对于学术文献优先使用 search_arxiv，对于最新新闻和实时信息使用 web_search。"""


def research_agent_node(state: AgentState) -> Dict[str, Any]:
    """Research Agent 节点 — 文献调研 (ReAct 循环)
    """
    import logging
    logger = logging.getLogger(__name__)

    messages = state.get("messages", [])
    model = state.get("model", "deepseek-chat")
    memory = state.get("memory", {})

    if not messages:
        logger.warning(f"[research_agent] EMPTY messages")
        return {"current_agent": "synthesize"}

    # 准备上下文
    context = get_context_for_agent("research_agent", memory)
    system_prompt = RESEARCH_SYSTEM_PROMPT
    if context:
        system_prompt += f"\n\n当前上下文:\n{context}"

    # 使用绑定工具的 LLM（arXiv + 联网搜索）
    tools_list = [search_arxiv, web_search]

    # 构建消息
    llm_messages = [
        SystemMessage(content=system_prompt),
        *[_to_langchain_msg(m) for m in messages],
    ]

    try:
        # === 阶段1: ReAct 循环 — 只搜索不生成 ===
        # 循环执行工具搜索收集论文，不要求 LLM 同时生成分析
        MAX_SEARCH_ROUNDS = 2
        total_search_calls = 0
        iteration = 0

        while iteration < MAX_SEARCH_ROUNDS:
            iteration += 1
            logger.info(f"[research_agent] 搜索第 {iteration} 轮")

            # 使用绑定工具的 LLM（arXiv + 联网搜索）
            llm_with_tools = get_llm(model).bind_tools(tools_list)
            response = llm_with_tools.invoke(llm_messages)

            if response.tool_calls:
                logger.info(f"[research_agent] 第 {iteration} 轮: {len(response.tool_calls)} 个工具调用")
                total_search_calls += len(response.tool_calls)

                # 把 AI 消息加入对话（包含 tool_calls）
                llm_messages.append(response)

                # 执行所有工具调用（支持 search_arxiv 和 web_search）
                for tc in response.tool_calls:
                    try:
                        if tc["name"] == "search_arxiv":
                            tool_result = search_arxiv.invoke(tc["args"])
                        elif tc["name"] == "web_search":
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
            else:
                # LLM 没有调用工具，可能直接生成了回答
                content = response.content or ""
                if content and len(content) > 20:
                    logger.info(f"[research_agent] 第 {iteration} 轮无工具调用，直接使用内容 (len={len(content)})")
                    reply = content
                    # ... 归档和返回 ...
                    timestamp = __import__("datetime").datetime.now().isoformat()
                    new_messages = messages + [
                        {"role": "assistant", "content": reply, "timestamp": timestamp}
                    ]
                    return {
                        "messages": new_messages,
                        "output_text": reply,
                        "memory": _archive_papers(reply, memory),
                        "current_agent": "synthesize",
                    }
                # 空内容且无工具调用 — 退出循环
                logger.info(f"[research_agent] 第 {iteration} 轮: 无工具调用且内容为空")
                break

        logger.info(f"[research_agent] 搜索阶段完成: {total_search_calls} 次搜索, {iteration} 轮")

        # === 阶段2: 强制生成最终分析（不带工具） ===
        # 使用普通 LLM（不绑定工具），强制模型基于搜索结果生成报告
        logger.info(f"[research_agent] 阶段2: 生成最终分析报告")
        generate_prompt = f"""你是一个深度学习文献调研专家。

请根据以上所有搜索到的论文信息，对用户的问题进行全面详细的分析报告。

要求：
1. 直接给出具体的调研结果和分析，不要再搜索
2. 包含该领域简介、核心方法分类、各方法优缺点、未来方向
3. 引用搜索到的具体论文（标题+要点）
4. 使用中文回答，合理使用 Markdown 格式
5. 回复不少于 500 字，越详细越好

用户的问题是：{messages[-1]['content'] if messages else '请进行全面调研'}

请现在写出完整的调研报告："""

        llm_no_tools = get_llm(model)
        final_messages = llm_messages + [HumanMessage(content=generate_prompt)]
        final_response = llm_no_tools.invoke(final_messages)
        reply = final_response.content or ""

        if not reply or len(reply) < 20:
            # 回退：用搜索阶段的最后文本
            reply = "基于以上搜索结果，已完成文献调研。请查看搜索到的论文信息。"

        logger.info(f"[research_agent] 最终回复长度={len(reply)} (通过强制生成)")

        # 归档论文到记忆
        updated_memory = _archive_papers(reply, memory)

        # 追加到消息历史
        timestamp = __import__("datetime").datetime.now().isoformat()
        new_messages = messages + [
            {"role": "assistant", "content": reply, "timestamp": timestamp}
        ]

        # 保留子任务队列（从多个来源尝试读取，确保跨节点不丢失）
        sub_tasks_to_pass = state.get("sub_task_queue", [])
        if not sub_tasks_to_pass and isinstance(memory, dict):
            tp = memory.get("task_plan", {})
            if isinstance(tp, dict):
                sub_tasks_to_pass = tp.get("sub_tasks", [])
        if not sub_tasks_to_pass:
            sub_tasks_to_pass = updated_memory.get("task_plan", {}).get("sub_tasks", []) if isinstance(updated_memory.get("task_plan"), dict) else []

        return {
            "messages": new_messages,
            "output_text": reply,
            "memory": updated_memory,
            "current_agent": "synthesize",
            "sub_task_queue": sub_tasks_to_pass,
        }

    except Exception as e:
        error_msg = f"文献调研失败: {str(e)}"
        logger.error(f"[research_agent] Exception: {e}")
        return {
            "output_text": error_msg,
            "current_agent": "synthesize",
        }


def _archive_papers(reply: str, memory: Dict[str, Any]) -> Dict[str, Any]:
    """从调研回复中检测并归档论文信息到记忆"""
    # 简化实现：记录调研回复中有"标题:"的行作为论文提及
    import re

    papers = memory.get("papers_archive", [])
    # 查找类似 "标题: XXXX" 或 "arxiv_id: XXXX" 的行
    titles = re.findall(r'(?:标题|Title)[：:]\s*(.+)', reply)

    for title in titles[:10]:
        title = title.strip().rstrip("。，,.")
        if title and not any(p.get("title") == title for p in papers):
            papers.append({
                "title": title,
                "source": "research_agent",
                "evidence_level": "insufficient",
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            })

    memory["papers_archive"] = papers
    return memory


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


def research_router(state: AgentState) -> str:
    """Research Agent 后置路由 — 从全局缓存检查剩余子任务"""
    import logging
    logger = logging.getLogger(__name__)
    session_id = state.get("session_id", "")
    plan_tasks = []
    # 从 supervisor 的全局缓存读取
    try:
        from backend.agent.supervisor import _SUBTASK_CACHE
        cached = _SUBTASK_CACHE.get(session_id, {})
        plan_tasks = cached.get("plan", [])
    except ImportError:
        pass
    if not plan_tasks:
        plan_tasks = state.get("sub_task_queue", [])
    pending = [t for t in plan_tasks if isinstance(t, dict) and t.get("status") == "pending"]
    logger.info(f"[research_router] tasks={len(plan_tasks)}, pending={len(pending)}")
    if pending:
        return "next_agent"
    return "synthesize"
