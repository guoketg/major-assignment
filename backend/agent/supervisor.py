"""
Supervisor Agent（智能路由）

职责：
1. 理解用户意图，决定需要哪个 Agent 处理
2. 对于复杂任务，分解为子任务队列
3. 管理路由决策

模式：LLM 调用 → 意图分类 → 路由决策
"""
import json
from typing import Dict, Any, List

from backend.agent.state import AgentState
from backend.agent.llm import get_llm
from backend.memory.manager import get_context_for_agent

# 全局子任务计划缓存（避免 LangGraph 深度拷贝丢失状态）
# key: session_id, value: {"plan": [...], "step": 0}
_SUBTASK_CACHE: Dict[str, Dict] = {}

# 意图 → Agent 映射
INTENT_MAP = {
    "chat": "chat",
    "research": "research",
    "innovate": "innovator",
    "experiment": "experiment",
    "report": "report",
    "plan": "planner",       # Planner Agent 处理复杂任务
    "complex": "planner",    # complex 的别名
}

# Supervisor 系统提示
SUPERVISOR_PROMPT = """你是一个科研助手系统的调度管理员（Supervisor）。

你的任务：分析用户的输入，判断用户的意图，并将其路由到合适的 Agent。

可能的意图分类：
- "chat" — 普通对话/闲聊/简单知识问答 → 路由到 Chat Agent
- "research" — 用户想查文献、了解研究现状、搜索论文 → 路由到 Research Agent
- "innovate" — 用户想构思创新方案、设计新方法 → 路由到 Innovator Agent
- "experiment" — 用户讨论实验、提交结果、设计实验 → 路由到 Experiment Agent
- "report" — 用户想导出报告、保存结果 → 路由到 Reporter

判断方法：
1. 如果用户提到"论文""文献""搜索""调研""SOTA""最新进展"等 → research
2. 如果用户提到"创新""方案""设计""改进""新方法"等 → innovate
3. 如果用户提到"实验""结果""准确率""消融""训练"等 → experiment
4. 如果用户提到"报告""导出""保存""文档"等 → report
5. 如果以上都不是，或用户只是打招呼/闲聊 → chat

如果任务复杂需要多个 Agent 协作，请设置 complex=True 并在 sub_tasks 中列出。

只返回 JSON 格式，不要包含其他内容：
{{"intent": "chat|research|innovate|experiment|report", "complex": false, "sub_tasks": [], "reasoning": "简短判断理由"}}

当前研究主题: {research_topic}
当前关注: {current_focus}
已有论文: {paper_count} 篇"""


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """Supervisor 节点 — 入口/路由/子任务调度节点

    读取用户最新消息，使用 LLM 判断意图，
    设置 state["current_agent"] 进行路由。

    特殊功能：如果 sub_task_queue 中有等待的子任务，推进到下一个子任务。
    如果 current_agent 已在初始状态中指定（用户手动选择 Agent），
    则直接跳过 LLM 意图分类，保持原有路由目标。
    """
    import logging
    _log = logging.getLogger(__name__)
    current = state.get("current_agent", "supervisor")
    session_id = state.get("session_id", "")
    memory = state.get("memory", {})

    _log.info(f"[supervisor] current={current}, session_id={session_id[:16] if session_id else 'EMPTY'}, cache_keys={list(_SUBTASK_CACHE.keys())[:3]}")

    # 从缓存或 state 中读取子任务计划
    cached = _SUBTASK_CACHE.get(session_id, {})
    task_plan = cached
    plan_tasks = cached.get("plan", [])
    if not plan_tasks:
        # 首次：从 state 读取并缓存
        sub_task_queue = state.get("sub_task_queue", [])
        task_plan = memory.get("task_plan", {}) if isinstance(memory, dict) else {}
        plan_tasks = task_plan.get("sub_tasks", []) if isinstance(task_plan, dict) else []
        if not plan_tasks:
            plan_tasks = sub_task_queue
        if plan_tasks:
            _SUBTASK_CACHE[session_id] = {"plan": plan_tasks, "step": 0}

    pending = [t for t in plan_tasks if t.get("status") == "pending"]
    running = [t for t in plan_tasks if t.get("status") == "running"]

    # 如果有 running 的子任务，标记为 complete
    for rt in running:
        rt["status"] = "complete"
        _log.info(f"[subtask] 完成: {rt.get('focus', '')[:60]}")
    if running and session_id in _SUBTASK_CACHE:
        _SUBTASK_CACHE[session_id]["step"] = sum(1 for t in plan_tasks if t.get("status") == "complete")

    # 推进到下一个 pending 子任务
    if pending:
        # 推进到下一个子任务
        next_task = pending[0]
        next_task["status"] = "running"

        agent_map = {
            "research": "research_agent",
            "chat": "chat_agent",
            "innovate": "innovator_agent",
            "experiment": "experiment_agent",
            "synthesize": "synthesizer",
        }
        next_agent = agent_map.get(next_task.get("agent", ""), "research_agent")
        # Map node name to route key (e.g. "research_agent" -> "research")
        node_to_route = {"chat_agent":"chat","research_agent":"research","innovator_agent":"innovate","experiment_agent":"experiment","synthesizer":"synthesize"}
        route_key = node_to_route.get(next_agent, "research")

        # 更新 task_plan
        if task_plan:
            task_plan["current_step"] = task_plan.get("total_steps", len(plan_tasks)) - len(pending) + 1
            memory["task_plan"] = task_plan

        msgs = state.get("messages", [])
        _log.info(f"[subtask] 推进到: msg_count={len(msgs)}, agent={next_agent}, focus={next_task.get('focus', '')[:60]}")

        return {
            "current_agent": route_key,
            "sub_task_queue": plan_tasks,
            "memory": memory,
        }

    if current != "supervisor" and not pending:
        return {"current_agent": current}

    messages = state.get("messages", [])
    if not messages:
        return {"current_agent": "end", "output_text": "没有消息需要处理。"}

    last_msg = messages[-1]["content"]
    memory = state.get("memory", {})

    # 构建提示
    context = get_context_for_agent("supervisor", memory)
    papers = memory.get("papers_archive", [])
    prompt = SUPERVISOR_PROMPT.format(
        research_topic=memory.get("research_topic", "未设置"),
        current_focus=memory.get("current_focus", "未设置"),
        paper_count=len(papers),
    )

    if context:
        prompt += f"\n\n额外上下文:\n{context}"

    try:
        llm = get_llm(state.get("model", "deepseek-chat"))
        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"用户消息: {last_msg}\n\n请判断意图并返回 JSON。"},
        ])

        content = response.content.strip()
        # 提取 JSON（LLM 可能用 ```json 包裹）
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            if content.startswith("json"):
                content = content[4:].strip()

        result = json.loads(content)
        intent = result.get("intent", "chat")
        complex_task = result.get("complex", False)
        sub_tasks = result.get("sub_tasks", [])
        reasoning = result.get("reasoning", "")

        # 确定路由目标
        agent_id = INTENT_MAP.get(intent, "chat")

        # 复杂任务 → 路由到 Planner 进行任务分解
        if complex_task and len(sub_tasks) > 0:
            agent_id = "planner"

        # 关键词检测：综述/撰写类请求强制设为复杂任务
        if not complex_task and agent_id == "research":
            review_keywords = ["综述", "撰写", "写一篇", "系统调研", "全面分析", "综合评述", "survey", "review"]
            last_msg_lower = last_msg.lower()
            if any(kw in last_msg_lower for kw in review_keywords):
                complex_task = True
                sub_tasks = []
                agent_id = "planner"

        # 记录研究主题
        update = {
            "current_agent": agent_id,
            "sub_task_queue": sub_tasks if complex_task else [],
            "output_text": "",
        }

        # 更新记忆中的研究主题
        if agent_id == "research" and not memory.get("research_topic"):
            memory["research_topic"] = last_msg[:100]
            update["memory"] = memory

        return update

    except (json.JSONDecodeError, Exception) as e:
        # 解析失败时默认走 chat
        return {
            "current_agent": "chat",
            "sub_task_queue": [],
            "output_text": "",
        }


def supervisor_router(state: AgentState) -> str:
    """Supervisor 路由 — 根据 current_agent 决定下一个节点"""
    return state.get("current_agent", "chat")


def _find_next_subtask(sub_task_queue: list) -> dict:
    """找到队列中第一个 pending 的子任务"""
    for st in sub_task_queue:
        if st.get("status") == "pending":
            return st
    return {}


def advance_subtask(state: AgentState) -> Dict[str, Any]:
    """推进到下一个子任务。标记当前子任务完成，启动下一个。

    如果有等待的子任务，返回下一个子任务的 agent 类型。
    如果没有更多子任务，返回 synthesize。
    """
    import logging
    logger = logging.getLogger(__name__)

    sub_task_queue = state.get("sub_task_queue", [])
    memory = state.get("memory", {})
    task_plan = memory.get("task_plan", {})

    if task_plan.get("sub_tasks"):
        # 从 memory.task_plan 读取
        plan_tasks = task_plan["sub_tasks"]
        current_step = task_plan.get("current_step", 0)
        total = task_plan.get("total_steps", len(plan_tasks))

        # 标记当前完成
        if current_step < total:
            plan_tasks[current_step]["status"] = "complete"

        # 启动下一个
        next_step = current_step + 1
        if next_step < total:
            plan_tasks[next_step]["status"] = "running"
            task_plan["current_step"] = next_step

            agent_map = {
                "research": "research_agent",
                "chat": "chat_agent",
                "innovate": "innovator_agent",
                "experiment": "experiment_agent",
                "synthesize": "synthesizer",
            }
            next_agent = agent_map.get(plan_tasks[next_step].get("agent", ""), "research_agent")

            memory["task_plan"] = task_plan
            logger.info(f"[subtask] 进度 {next_step}/{total}: {plan_tasks[next_step].get('focus', '')[:60]}")

            return {
                "current_agent": next_agent,
                "memory": memory,
                "sub_task_queue": plan_tasks,
            }
        else:
            # 所有子任务完成
            logger.info(f"[subtask] 所有 {total} 个子任务完成")
            return {"current_agent": "synthesize", "memory": memory}

    return {"current_agent": "synthesize"}
