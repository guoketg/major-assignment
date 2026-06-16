"""
LangGraph 多 Agent 图

构建 StateGraph，注册所有 Agent 节点和条件路由，
编译后可通过 astream_events 流式执行。
"""
import json
import os
import logging
from typing import AsyncGenerator, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.agent.state import AgentState, create_initial_state
from backend.agent.supervisor import supervisor_node, supervisor_router
from backend.agent.chat_agent import chat_agent_node, chat_router
from backend.agent.research_agent import research_agent_node, research_router
from backend.agent.innovator_agent import innovator_agent_node, innovator_router
from backend.agent.experiment_agent import experiment_agent_node, experiment_router
from backend.agent.synthesizer import synthesizer_node, synthesizer_router
from backend.agent.planner_agent import planner_agent_node, planner_router
from backend.memory.manager import (
    load_working_memory,
    save_working_memory,
)

logger = logging.getLogger(__name__)

# Agent 显示配置（与前端 AGENT_INFO 保持一致）
AGENT_DISPLAY = {
    "supervisor":       {"label": "🤖 智能路由",   "color": "#6366f1"},
    "chat_agent":       {"label": "💬 对话助手",   "color": "#22c55e"},
    "research_agent":   {"label": "🔍 文献调研",   "color": "#3b82f6"},
    "innovator_agent":  {"label": "💡 创新构思",   "color": "#f59e0b"},
    "experiment_agent": {"label": "🧪 实验分析",   "color": "#ef4444"},
    "reporter":         {"label": "📊 报告生成",   "color": "#8b5cf6"},
    "synthesizer":      {"label": "📋 综合输出",   "color": "#06b6d4"},
    "planner_agent":    {"label": "📋 任务规划",   "color": "#8b5cf6"},
}

# 节点名 -> 路由返回值 映射（用于条件边）
AGENT_ROUTE_MAP = {
    "supervisor": "supervisor",
    "chat": "chat_agent",
    "research": "research_agent",
    "innovator": "innovator_agent",
    "experiment": "experiment_agent",
    "report": "synthesizer",
    "synthesize": "synthesizer",
    "planner": "planner_agent",
    "end": END,
}


def build_graph() -> StateGraph:
    """构建并编译 LangGraph StateGraph"""
    workflow = StateGraph(AgentState)

    # === 注册节点 ===
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("chat_agent", chat_agent_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("innovator_agent", innovator_agent_node)
    workflow.add_node("experiment_agent", experiment_agent_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("planner_agent", planner_agent_node)

    # === 设置入口 ===
    workflow.set_entry_point("supervisor")

    # === Supervisor 条件路由 ===
    workflow.add_conditional_edges("supervisor", supervisor_router, AGENT_ROUTE_MAP)

    # === 各 Agent 的回退路由 ===
    workflow.add_conditional_edges("chat_agent", chat_router, {
        "next_agent": "supervisor",
        "synthesize": "synthesizer",
    })
    workflow.add_conditional_edges("research_agent", research_router, {
        "next_agent": "supervisor",
        "synthesize": "synthesizer",
    })
    workflow.add_conditional_edges("innovator_agent", innovator_router, {
        "next_agent": "supervisor",
        "synthesize": "synthesizer",
    })
    workflow.add_conditional_edges("experiment_agent", experiment_router, {
        "next_agent": "supervisor",
        "synthesize": "synthesizer",
    })

    # === Planner Agent 路由 ===
    workflow.add_conditional_edges("planner_agent", planner_router, AGENT_ROUTE_MAP)

    # === Synthesizer -> END ===
    workflow.add_conditional_edges("synthesizer", synthesizer_router, {
        "end": END,
    })

    # === 编译 ===
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    return app


class AgentGraph:
    """AgentGraph 封装，提供 run_stream() 异步流式执行方法"""

    def __init__(self):
        self.app = build_graph()
        self._cached_sub_tasks = []  # 缓存子任务规划
        self._plan_emitted = False   # 标记 plan 是否已发送（防重复）
        self._final_output_text = ""  # 保存最终合成的输出文本
        # 流水线数据收集器（用于持久化到对话记录）
        self._agent_pipeline_collected = []   # 所有 agent 轨迹 [{agent, label, status}]
        self._tool_calls_collected = []       # 所有工具调用 [{agent, tool, label, status}]
        self._sub_task_plan_collected = []    # 最终子任务计划 [{id, focus, agent, status}]

    async def run_stream(
        self,
        session_id: str,
        messages: list,
        model: str = "deepseek-chat",
        agent: str = "auto",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """异步流式执行 LangGraph，将事件转换为前端 SSE 兼容的 dict 事件"""
        memory = load_working_memory(session_id)

        initial_state = create_initial_state(
            session_id=session_id,
            model=model,
            messages=messages,
        )
        initial_state["memory"] = memory

        AGENT_OVERRIDE = {
            "chat": "chat",
            "research": "research",
            "innovate": "innovator",
            "experiment": "experiment",
        }
        if agent != "auto" and agent in AGENT_OVERRIDE:
            initial_state["current_agent"] = AGENT_OVERRIDE[agent]

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 25,
        }

        current_agent = "supervisor"
        accumulated_content = ""
        has_assistant_in_streaming = False
        self._cached_sub_tasks = []
        self._plan_emitted = False
        self._final_output_text = ""
        # 重置流水线数据收集器
        self._agent_pipeline_collected = []
        self._tool_calls_collected = []
        self._sub_task_plan_collected = []

        try:
            async for event in self.app.astream_events(
                initial_state, config=config, version="v2",
            ):
                kind = event.get("event", "")
                node_name = event.get("name", "")

                # == Agent 节点开始 ==
                if kind == "on_chain_start":
                    node = node_name.lower().replace(" ", "_")
                    if node in AGENT_DISPLAY:
                        current_agent = node
                        display = AGENT_DISPLAY[node]
                        # 收集流水线数据
                        self._agent_pipeline_collected.append({
                            "agent": node, "label": display["label"], "status": "running"
                        })
                        yield {
                            "type": "agent",
                            "agent": node,
                            "label": display["label"],
                            "status": "running",
                        }

                # == LLM 流式 token（跳过 supervisor） ==
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", {})
                    if current_agent == "supervisor":
                        continue

                    if hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        accumulated_content += content
                        has_assistant_in_streaming = True
                        yield {"type": "content", "content": content, "agent": current_agent}
                    if hasattr(chunk, "additional_kwargs"):
                        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                        if reasoning:
                            yield {"type": "content", "content": "", "reasoning_content": reasoning, "agent": current_agent}

                # == 工具调用 ==
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown_tool")
                    self._tool_calls_collected.append({
                        "agent": current_agent, "tool": tool_name,
                        "label": f"🔧 调用 {tool_name}", "status": "start"
                    })
                    yield {"type": "tool", "agent": current_agent, "tool": tool_name, "label": f"🔧 调用 {tool_name}", "status": "start"}

                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown_tool")
                    # 提取工具输出
                    tool_output = ""
                    try:
                        raw_output = event.get("data", {}).get("output", "")
                        if hasattr(raw_output, 'content'):
                            tool_output = str(raw_output.content) if raw_output.content else ""
                        elif isinstance(raw_output, (str, int, float)):
                            tool_output = str(raw_output)
                        else:
                            tool_output = str(raw_output)[:3000]
                    except Exception:
                        tool_output = ""
                    # 标记最后一个匹配的 start 为 complete
                    for tc in reversed(self._tool_calls_collected):
                        if tc["tool"] == tool_name and tc["agent"] == current_agent and tc["status"] == "start":
                            tc["status"] = "complete"
                            tc["output"] = tool_output
                            break
                    yield {"type": "tool", "agent": current_agent, "tool": tool_name, "label": f"🔧 调用 {tool_name}", "status": "complete", "output": tool_output}

                # == Agent 节点结束 ==
                elif kind == "on_chain_end":
                    # 路由器名称 -> Agent 节点名称映射
                    ROUTER_TO_NODE = {
                        "planner_router": "planner_agent",
                        "supervisor_router": "supervisor",
                        "chat_router": "chat_agent",
                        "research_router": "research_agent",
                        "innovator_router": "innovator_agent",
                        "experiment_router": "experiment_agent",
                        "synthesizer_router": "synthesizer",
                    }
                    node_key = node_name.lower().replace(" ", "_")
                    is_router_event = node_key in ROUTER_TO_NODE
                    if is_router_event:
                        node = ROUTER_TO_NODE[node_key]
                    else:
                        node = node_key
                    if node not in AGENT_DISPLAY:
                        continue

                    display = AGENT_DISPLAY[node]
                    # 更新收集器中的状态为 complete
                    for a in self._agent_pipeline_collected:
                        if a["agent"] == node:
                            a["status"] = "complete"
                            break
                    else:
                        self._agent_pipeline_collected.append({
                            "agent": node, "label": display["label"], "status": "complete"
                        })
                    yield {"type": "agent", "agent": node, "label": display["label"], "status": "complete"}

                    # ⚡ 以下内容事件只从实际节点（非 router）触发，防止重复
                    if not is_router_event:
                        try:
                            data = event.get("data", {})
                            output = data.get("output", {}) or {}
                            inp_state = data.get("input", {}) or {}

                            # 提取子任务数据（从 node output 或 input state）
                            text = ""
                            sub_tasks = []
                            task_plan = {}
                            if isinstance(output, dict):
                                text = str(output.get("output_text", "") or "")
                                sub_tasks = output.get("sub_task_queue", []) or []
                                mem = output.get("memory", {}) or {}
                                task_plan = mem.get("task_plan", {}) if isinstance(mem, dict) else {}
                            elif isinstance(inp_state, dict):
                                sub_tasks = inp_state.get("sub_task_queue", []) or []
                                mem = inp_state.get("memory", {}) or {}
                                task_plan = mem.get("task_plan", {}) if isinstance(mem, dict) else {}

                            # Planner 内容事件
                            if node == "planner_agent" and text:
                                yield {"type": "content", "content": text, "agent": node}

                            # Planner 子任务规划事件（仅发送一次）
                            if node == "planner_agent" and sub_tasks and not self._plan_emitted:
                                self._plan_emitted = True
                                self._cached_sub_tasks = sub_tasks
                                plan_data = [
                                    {"id": s.get("id", i+1), "focus": s.get("focus", ""), "agent": s.get("agent", ""), "status": s.get("status", "pending")}
                                    for i, s in enumerate(sub_tasks)
                                ]
                                self._sub_task_plan_collected = plan_data
                                yield {
                                    "type": "plan",
                                    "sub_tasks": [
                                        {"id": s.get("id", i+1), "focus": s.get("focus", ""), "agent": s.get("agent", ""), "status": s.get("status", "pending")}
                                        for i, s in enumerate(sub_tasks)
                                    ],
                                    "plan_summary": task_plan.get("plan_summary", ""),
                                }

                            # Synthesizer 最终内容 — _final 标记告诉前端这是完整输出（替换而非追加）
                            if node == "synthesizer" and text:
                                self._final_output_text = text
                                yield {"type": "content", "content": text, "agent": node, "_final": True}

                            # 子任务进度事件（从 output 或缓存读取）
                            active_tasks = sub_tasks if sub_tasks else self._cached_sub_tasks
                            if not active_tasks and task_plan:
                                active_tasks = task_plan.get("sub_tasks", [])
                            if active_tasks:
                                progress_data = [
                                    {"id": s.get("id", i+1), "focus": str(s.get("focus", ""))[:60], "agent": s.get("agent", ""), "status": s.get("status", "pending")}
                                    for i, s in enumerate(active_tasks)
                                ]
                                self._sub_task_plan_collected = progress_data
                                yield {
                                    "type": "subtask_progress",
                                    "sub_tasks": progress_data,
                                }

                        except Exception as exc:
                            logger.error(f"[{node}] on_chain_end error: {exc}")

            # === 获取最终状态 ===
            final_state = await self._get_final_state(session_id, initial_state, config)
            timestamp = __import__("datetime").datetime.now().isoformat()

            # 构建原始历史消息
            raw_history = []
            if final_state:
                if "memory" in final_state:
                    save_working_memory(session_id, final_state["memory"])
                raw_history = list(final_state.get("messages", messages) or messages)
            else:
                raw_history = list(messages) if messages else []

            # === 清理历史：基于输入消息保留所有 user-assistant 对，只追加本次最终回复 ===
            # 输入消息 messages 已经包含之前所有对话对，结构正确
            history = list(messages) if messages else []

            final_text = self._final_output_text or accumulated_content
            if final_text:
                # 追加本次 assistant 回复
                history.append({"role": "assistant", "content": final_text, "timestamp": timestamp})
            else:
                # 从 raw_history 取最后一条内容
                last = raw_history[-1] if raw_history else {}
                last_content = last.get('content', '') or ''
                history.append({"role": "assistant", "content": last_content or "处理完成", "timestamp": timestamp})

            logger.info(f"[DONE] cleaned history: count={len(history)} (raw was {len(raw_history)})")
            yield {
                "type": "done", "content": "", "done": True, "history": history,
                "agent_pipeline": self._agent_pipeline_collected,
                "tool_calls": self._tool_calls_collected,
                "sub_task_plan": self._sub_task_plan_collected,
            }

        except Exception as e:
            yield {"type": "error", "error": str(e), "done": True}

    async def _get_final_state(
        self, session_id: str, initial_state: AgentState, config: dict,
    ) -> Optional[Dict[str, Any]]:
        try:
            final_state = await self.app.aget_state(config)
            if final_state and final_state.values:
                return dict(final_state.values)
        except Exception:
            pass
        return None
