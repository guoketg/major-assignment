# 多 Agent 协作科研分析系统 — LangGraph 编排与协作流程

## 0. 状态跟踪表

> 此表跟踪本 PRD 涉及功能的实现进度，随项目进展持续更新。

| 组件 | 前端 | 后端 | 测试 | 用户反馈 | 总体状态 |
|------|:----:|:----:|:----:|:--------:|:--------:|
| 📐 **AgentState TypedDict** | — | ✅ agent/state.py | — | — | ✅ **已完成** |
| 🔀 **StateGraph 图构建** | — | ✅ agent/graph.py | ✅ 测试通过 | — | ✅ **已完成** |
| 🔄 **Supervisor 条件路由** | — | ✅ LLM 意图分类 | ✅ 测试通过 | — | ✅ **已完成** |
| 📍 **ReAct Agent 节点模板** | — | ✅ 所有 Agent 节点 | ✅ 测试通过 | — | ✅ **已完成** |
| 🌀 **SSE + LangGraph 适配** | ✅ | ✅ astream_events | ✅ 测试通过 | — | ✅ **已完成** |
| 👁️ **Agent 前端可视化** | ✅ **流水线位置/唯一Key** | ✅ **模拟层** | ✅ **测试通过** | ✅ **流水线在用户/AI之间** | ✅ **已完成** |
| 📋 **SSE 事件映射** | ✅ | ✅ astream_events | ✅ | ✅ **调研回复5500字完整内容** | ✅ **已完成** |
| 📋 **子任务队列调度** | ✅ **前端子任务进度显示** | ✅ **Planner + Supervisor调度** | ✅ **5/5任务推进测试通过** | ✅ **子任务进度实时更新** | ✅ **已完成** |
| 🧹 **消息去重与中间输出清理** | ✅ **流式显示跳过中间内容** | ✅ **Synthesizer重建消息历史** | ✅ **多条assistant→1条清理测试通过** | ✅ **"反复输出"问题已修复** | ✅ **已完成** |

**图例：** ✅ 已完成 &nbsp;🚧 开发中 &nbsp;⏳ 待测试 &nbsp;❌ 未开始 &nbsp;📝 已规划

---

## 1. LangGraph 状态定义

```python
from typing import TypedDict, List, Dict, Optional, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

class AgentState(TypedDict):
    # === 核心会话 ===
    messages: List[Dict]              # 完整对话历史 [{"role","content"},...]
    session_id: str                   # 会话 ID
    model: str                        # 当前使用的模型名

    # === 路由控制 ===
    current_agent: str                # 当前应执行的 Agent: "chat"|"research"|"innovator"|"experiment"|"reporter"|"end"
    sub_task_queue: List[Dict]        # 待执行的子任务队列 [{agent, input, status},...]

    # === 记忆 ===
    memory: Dict                      # 结构化记忆体
        # papers_archive: List[PaperSummary]
        # innovation_candidates: List[InnovationPlan]
        # experiment_log: List[ExperimentRecord]

    # === 输出 ===
    output_text: str                  # 最终要展示给用户的文本
    output_artifacts: Dict            # 生成的文件路径 {word_doc, drawio_png, ...}
```

## 2. 图结构

```python
# 构建图
workflow = StateGraph(AgentState)

# 注册节点
workflow.add_node("supervisor", supervisor_node)      # 入口/路由节点
workflow.add_node("chat_agent", chat_agent_node)       # 对话 Agent
workflow.add_node("research_agent", research_agent_node)  # 调研 Agent
workflow.add_node("innovator_agent", innovator_agent_node) # 创新 Agent
workflow.add_node("experiment_agent", experiment_agent_node) # 实验 Agent
workflow.add_node("reporter", reporter_node)           # 报告生成
workflow.add_node("synthesizer", synthesizer_node)     # 结果综合

# 设置入口
workflow.set_entry_point("supervisor")

# Supervisor → 条件路由到各 Agent
workflow.add_conditional_edges(
    "supervisor",
    router_function,  # 根据 state["current_agent"] 决定
    {
        "chat": "chat_agent",
        "research": "research_agent",
        "innovator": "innovator_agent",
        "experiment": "experiment_agent",
        "reporter": "reporter",
        "synthesize": "synthesizer",
        "end": END,
    }
)

# 各 Agent 完成后回到 Supervisor（或去下一个 Agent）
for agent in ["chat_agent", "research_agent", "innovator_agent", "experiment_agent"]:
    workflow.add_conditional_edges(
        agent,
        after_agent_router,  # 判断是否需要继续下一个 Agent
        {
            "next_agent": "supervisor",    # 还有子任务 → 回 Supervisor 重新路由
            "synthesize": "synthesizer",   # 所有任务完成 → 综合输出
            "end": END,
        }
    )

workflow.add_edge("reporter", "synthesizer")
workflow.add_edge("synthesizer", END)

# 编译
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
```

## 3. 路由逻辑

### 3.1 主路由器（Supervisor → Agent）

```python
def router_function(state: AgentState) -> str:
    """根据 Supervisor 的决策路由到对应 Agent"""
    return state["current_agent"]
```

### 3.2 后置路由器（Agent → 下一步）

```python
def after_agent_router(state: AgentState) -> str:
    """Agent 执行后判断下一步"""
    queue = state.get("sub_task_queue", [])

    # 有等待中的子任务 → 回到 Supervisor 处理下一个
    pending = [t for t in queue if t["status"] == "pending"]
    if pending:
        return "next_agent"

    # 需要综合输出
    if state.get("output_text"):
        return "synthesize"

    # 没有更多任务
    return "end"
```

## 4. 节点实现模式

每个 Agent 节点遵循统一的 **ReAct（Reasoning + Acting）** 模式：

```python
def agent_node(state: AgentState) -> AgentState:
    """通用 Agent 节点模板"""
    
    # === 1. 感知 (Perception) ===
    # 从 state 中读取当前消息、历史、记忆
    current_input = extract_current_input(state)
    
    # === 2. 记忆检索 (Memory Recall) ===
    relevant_memory = retrieve_relevant_memory(state["memory"], current_input)
    
    # === 3. 规划 (Planning) ===
    plan = planner_llm.invoke(
        build_plan_prompt(current_input, relevant_memory, available_tools)
    )
    
    # === 4. 行动循环 (Action Loop) ===
    # ReAct: 思考 → 调用工具 → 观察结果 → 继续思考
    while not plan_complete:
        action = decide_next_action(state)
        if action["type"] == "tool_call":
            result = execute_tool(action["tool_name"], action["args"])
            state = update_state_with_tool_result(state, result)
        elif action["type"] == "llm_generate":
            response = llm.generate(action["prompt"])
            state["output_text"] = response
            
    # === 5. 记忆更新 (Memory Update) ===
    state["memory"] = update_memory(state["memory"], state)
    
    return state
```

## 5. 流式输出适配

当前后端使用 SSE 流式输出。LangGraph 支持流式执行，需要改造 `chat_stream` 端点：

```python
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    # 1. 准备初始状态
    initial_state = AgentState(
        messages=conversations.get(request.session_id, []),
        session_id=request.session_id,
        model=MODEL_MAP.get(request.model, "deepseek-chat"),
        current_agent="supervisor",
        sub_task_queue=[],
        memory={},
        output_text="",
        output_artifacts={},
    )
    
    # 2. 执行 LangGraph 并流式输出
    async def generate():
        # 逐节点执行图
        for event in app.astream_events(initial_state, config={"recursion_limit": 50}):
            if event["event"] == "on_node_end":
                node_name = event["name"]
                state = event["data"]["state"]
                
                # 每次节点完成都推送 SSE 事件
                yield format_sse_event(node_name, state)
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## 6. 前端适配

前端需要展示 Agent 的思考过程：

```typescript
// SSE 事件结构
interface AgentEvent {
    agent: string;           // 当前 Agent 名称
    status: 'thinking' | 'acting' | 'done';
    content?: string;        // Agent 输出的文本
    tool_calls?: {           // 工具调用记录
        tool: string;
        args: any;
        result: any;
    }[];
    memory_update?: any;     // 记忆更新
}
```

前端展示方式：
- **Agent 切换指示器**：显示当前是哪个 Agent 在工作
- **工具调用日志**：可折叠的「🔧 调用 arXiv 搜索」卡片
- **思维过程**：类似思考模式的 `🤔 分析中...` → 完成后折叠
- **最终回复**：综合所有 Agent 输出后的最终文本

## 7. 完整执行流程示例

```
用户: "帮我调研一下CLIP噪声标签学习的最新进展"

→ Supervisor: 意图分类 = "research"
→ Research Agent:
    ├─ 🔍 搜索: search_arxiv("CLIP noisy labels robust learning")
    ├─ 🔍 搜索: search_arxiv("multi-modal noise-tolerant learning 2024")
    ├─ 📖 解析: parse_paper(arxiv:2306.11113)
    ├─ 📖 解析: parse_paper(arxiv:2402.03300)
    └─ ✅ 完成: 归档 15 篇论文到 memory.papers_archive

→ Supervisor: 还有子任务? 无 → 综合输出
→ Synthesizer: 整合调研结果
→ END: 返回最终回复
```
