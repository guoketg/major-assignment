"""
Planner Agent（任务规划 Agent）

复杂任务拆解与执行规划。
将大型复杂任务（如撰写综述论文）分解为可逐步执行的子任务序列，
每个子任务聚焦一个具体方面，由对应 Agent 增量执行。
"""
import json
import logging
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.state import AgentState
from backend.agent.llm import get_llm

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """你是一个科研任务规划专家。你的职责是将复杂的科研任务分解为一系列可执行的子任务。

## 任务分解原则

1. 每个子任务应该聚焦一个具体的方面
2. 子任务之间应有逻辑顺序（前面的输出是后面的输入）
3. 每个子任务指定由哪个 Agent 执行
4. 子任务的描述应该清晰具体，包含搜索关键词

## Agent 类型说明

- "research": 文献调研 Agent — 搜索和分析学术论文
- "chat": 对话 Agent — 一般对话
- "innovate": 创新 Agent — 构思创新方案
- "experiment": 实验 Agent — 实验设计与分析
- "synthesize": 综合 Agent — 汇总所有子任务的结果

## 复杂任务判断标准

以下情况属于复杂任务，需要分解：
- 综述/调研论文撰写
- 需要多角度搜索的分析报告
- 包含多个步骤的综合任务

## 输出格式

必须返回以下 JSON 格式，不要包含其他内容：

```json
{
  "is_complex": true,
  "overall_goal": "任务总体目标描述",
  "sub_tasks": [
    {
      "id": 1,
      "agent": "research",
      "query": "具体搜索关键词",
      "focus": "这个子任务要研究的具体问题",
      "expected_output": "预期的输出内容"
    }
  ],
  "plan_summary": "简要说明这个规划的逻辑"
}
```

对于简单的任务（如打招呼、简单问答），设置 is_complex=false。

当前对话上下文将包含用户最近的几条消息。"""


def planner_agent_node(state: AgentState) -> Dict[str, Any]:
    """Planner Agent 节点 — 任务规划

    分析用户请求，判断是否为复杂任务。
    如果是，生成子任务队列并设置到 state 中。
    如果不是，直接路由到对应的简单 Agent。
    """
    messages = state.get("messages", [])
    model = state.get("model", "deepseek-chat")
    memory = state.get("memory", {})

    if not messages:
        return {"current_agent": "end"}

    # 获取用户最新消息
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    if not last_user_msg:
        return {"current_agent": "chat"}

    try:
        llm = get_llm(model)
        response = llm.invoke([
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"用户请求: {last_user_msg}\n\n请判断是否为复杂任务并生成规划。"},
        ])

        content = response.content.strip()
        # 提取 JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            if content.startswith("json"):
                content = content[4:].strip()

        plan = json.loads(content)
        is_complex = plan.get("is_complex", False)

        logger.info(f"[planner] is_complex={is_complex}, sub_tasks={len(plan.get('sub_tasks', []))}")

        if not is_complex:
            # 简单任务：直接路由到 supervisor 判断
            return {
                "current_agent": "supervisor",
                "output_text": "",
            }

        # 复杂任务：创建子任务队列
        sub_tasks = plan.get("sub_tasks", [])
        for st in sub_tasks:
            st["status"] = "pending"

        # 将规划存入记忆
        if "task_plan" not in memory:
            memory["task_plan"] = {}
        memory["task_plan"] = {
            "overall_goal": plan.get("overall_goal", ""),
            "plan_summary": plan.get("plan_summary", ""),
            "sub_tasks": sub_tasks,
            "current_step": 0,
            "total_steps": len(sub_tasks),
            "section_outputs": [],
        }

        logger.info(f"[planner] 复杂任务规划：{len(sub_tasks)} 个子任务")

        # 如果有子任务，路由到第一个子任务的 Agent
        if sub_tasks:
            first_task = sub_tasks[0]
            agent_map = {
                "research": "research_agent",
                "chat": "chat_agent",
                "innovate": "innovator_agent",
                "experiment": "experiment_agent",
                "synthesize": "synthesizer",
            }
            target = agent_map.get(first_task.get("agent", ""), "research_agent")
            # 注意：不要设置 status="running"，让 supervisor 的调度逻辑来处理

            return {
                "current_agent": "supervisor",  # 回到 supervisor 推进子任务
                "sub_task_queue": sub_tasks,
                "memory": memory,
                "output_text": f"📋 **任务规划完成**：共 {len(sub_tasks)} 个子任务\n"
                              f"**目标**：{plan.get('overall_goal', '')}\n"
                              f"**规划**：{plan.get('plan_summary', '')}\n"
                              f"**开始执行子任务 1/{len(sub_tasks)}**：{first_task.get('focus', '')}",
            }

        return {"current_agent": "synthesize"}

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"[planner] 规划失败: {e}")
        # 解析失败时回退到 supervisor
        return {"current_agent": "supervisor"}


def planner_router(state: AgentState) -> str:
    """Planner Agent 后置路由"""
    agent = state.get("current_agent", "supervisor")
    return agent
