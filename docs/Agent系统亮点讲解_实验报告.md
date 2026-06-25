# 多 Agent 协作科研分析系统——Agent 架构亮点讲解

> 面向"期末实验报告"的 Agent 子系统深度分析，侧重架构设计与技术创新点。

---

## 一、系统总览：7 个专业 Agent + 1 个 Smart Router

本项目基于 **LangGraph StateGraph** 构建了一个多 Agent 协作科研分析系统，实现了从**意图识别 → 任务分解 → 多 Agent 协同执行 → 综合输出**的完整智能链路。

**Agent 阵容**：

| Agent | 角色 | 核心能力 |
|--------|------|----------|
| 🤖 Supervisor | 智能路由调度器 | LLM 意图分类、复杂任务分解、子任务队列推进 |
| 🔍 Research Agent | 文献调研专家 | ReAct 循环搜索 arXiv + 联网搜索，证据分级归档 |
| 💡 Innovator Agent | 创新构思专家 | 基于调研结果构思创新方案，多维对比评估 |
| 🧪 Experiment Agent | 实验设计专家 | 消融实验设计、结果分析、瓶颈定位 |
| 📋 Planner Agent | 任务规划专家 | 复杂任务自动拆解为可执行子任务序列 |
| 📋 Synthesizer | 综合输出引擎 | 整合多 Agent 输出，生成完整综述报告 |
| 💬 Chat Agent | 对话助手 | 处理一般性对话和简单问答 |

---

## 二、核心架构亮点

### 亮点 1：基于 LangGraph StateGraph 的图编排引擎

`backend/agent/graph.py` 完整实现了 7 个节点的 StateGraph 构建与编译：

```python
# graph.py 第58-107行
def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    # 注册 7 个节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research_agent", research_agent_node)
    # ... 共 7 个节点
    workflow.set_entry_point("supervisor")
    # Supervisor 条件路由 → 各 Agent
    workflow.add_conditional_edges("supervisor", supervisor_router, AGENT_ROUTE_MAP)
    # 各 Agent → supervisor / synthesizer
    workflow.add_conditional_edges("research_agent", research_router, {...})
    # ...
    return workflow.compile(checkpointer=MemorySaver())
```

**架构优势**：
- 所有节点共享统一 `AgentState`，消息和记忆在节点间自动流转
- 条件边实现灵活的路由分发，支持"跳回 Supervisor 处理下一子任务"或"进入 Synthesizer 合成输出"
- `MemorySaver` 作为内建检查点机制，支持会话恢复

---

### 亮点 2：LLM 驱动的智能 Supervisor 路由

Supervisor（`supervisor.py`）不是简单的关键词匹配，而是**调用 LLM 进行语义级意图分类**：

```python
# supervisor.py 第34-55行 SUPERVISOR_PROMPT
# 系统提示词定义了5种意图分类标准：
# chat / research / innovate / experiment / report
# 要求 LLM 返回 JSON: {"intent": "...", "complex": false, ...}
```

**关键技术点**：
1. **JSON 提取容错**：自动处理 LLM 输出中的 ` ```json ` 包裹
2. **关键词辅助检测**：综述类关键词（"综述""撰写""survey"）强制标记为复杂任务
3. **全局子任务缓存 `_SUBTASK_CACHE`**：绕过 LangGraph 深度拷贝限制，确保子任务计划在各节点间可靠传递
4. **用户手动路由**：支持前端指定 Agent（`agent != "auto"`），此时跳过 LLM 意图分类

---

### 亮点 3：Planner Agent 的层次化任务分解

`planner_agent.py` 实现了**两层路由**策略：

- **简单任务** → 直接交还 Supervisor 判断意图
- **复杂任务**（如"写一篇综述"）→ LLM 自动分解为多个可执行的子任务序列：

```python
# planner_agent.py 第47-66行 输出格式
{
  "is_complex": true,
  "overall_goal": "写一篇关于 Transformer 效率优化的综述",
  "sub_tasks": [
    {"id": 1, "agent": "research", "query": "sparse attention", "focus": "...", "status": "pending"},
    {"id": 2, "agent": "research", "query": "linear attention", "focus": "...", "status": "pending"},
    // ...
  ],
  "plan_summary": "..."
}
```

**任务驱动链**: Planner → Supervisor 调度 → Agent 执行 → Router 检测剩余子任务 → 回到 Supervisor 推进下一个子任务

---

### 亮点 4：Research Agent 的两阶段 ReAct 循环

`research_agent.py` 采用了一种**搜索-生成分离**的两阶段策略，解决了传统 ReAct 模式下 LLM 过早终止搜索的问题：

```
阶段1（搜索循环）：
  绑定工具 LLM → 判断是否需要搜索 → 调用 search_arxiv/web_search → 最多 2 轮
阶段2（强制生成）：
  剥离工具绑定 → 专用 prompt 强制 LLM 基于搜索结果生成详细报告 → 不少于 500 字
```

```python
# research_agent.py 第87-163行
while iteration < MAX_SEARCH_ROUNDS:
    llm_with_tools = get_llm(model).bind_tools(tools_list)
    response = llm_with_tools.invoke(llm_messages)
    # 执行搜索工具...

# 阶段2: 使用不绑定工具的 LLM 强制生成报告
llm_no_tools = get_llm(model)
final_response = llm_no_tools.invoke([...generate_prompt...])
```

**设计优势**：分离"搜索"和"生成"两个目标的提示词，避免 LLM 倾向性地"不搜索直接回答"，确保每次调研都能获取足够的文献支撑。

---

### 亮点 5：Synthesizer 的多输出智能整合

`synthesizer.py` 实现了两层综合策略：

1. **LLM 智能综合**：当存在 2+ 份子任务输出时，收集所有 assistant 消息，调用 LLM 生成一篇结构完整的综述报告（≥800 字）
2. **文档自动生成**：检测用户意图关键词（"文档""导出""保存"），自动调用 `python-docx` 将结果保存为 Word 文档并提供下载链接

此外，Synthesizer 执行**消息历史压缩**：将多轮 agent 消息清理为仅保留 user 消息 + 最终合成输出，避免对话历史膨胀。

---

### 亮点 6：可插拔技能增强系统

`skills.py` 和 `skills_store.py` 实现了一个**技能（Skill）注入系统**：

- **内置技能**：`docs`（结构化文档输出）、`pdf`（PDF 优化输出）
- **自定义技能**：用户可通过 API 创建自定义技能，持久化到 JSON 文件
- **注入机制**：技能的系统提示词在 `graph.run_stream()` 中被注入到消息列表首部，所有 Agent 共享此增强上下文

```python
# graph.py 第138-143行
skill_prompt = get_skill_prompt(skill)
if skill_prompt:
    augmented_messages = [{"role": "system", "content": skill_prompt}] + messages
```

技能系统提供完整的 RESTful CRUD API（`/skills GET/POST/PUT/DELETE`），内置技能不可删除覆盖，保证系统稳定性。

---

### 亮点 7：精细化的 Token 成本追踪

系统实现了**三层**成本统计：

| 层级 | 实现 | 存储 |
|------|------|------|
| 单次 LLM 调用 | `extract_token_usage()` + `calculate_cost()` | 内存 |
| 单次会话 | `AgentGraph._token_usage` + `_per_agent_tokens` | 内存 → 前端的 `token_update` SSE 事件 |
| 全局累计 | `total_usage.py` JSON 持久化 + 线程锁 | `agent_memory/total_usage.json` |

支持 **DeepSeek 两档定价**（chat: ¥1/¥2 每百万 token，reasoner: ¥4/¥16 每百万 token），按实际模型计费。实时推送 Token 用量到前端展示。

---

### 亮点 8：SSE 流式事件的精细化设计

`AgentGraph.run_stream()` 通过 `astream_events` 捕获 LangGraph 内部事件，转换出 8 种语义明确的 SSE 事件类型：

| 事件类型 | 触发时机 | 用途 |
|----------|----------|------|
| `agent` | Agent 节点开始/完成 | 前端 Agent 流水线动画 |
| `content` | LLM 流式 token 输出 | 打字机效果 |
| `tool` | 工具调用开始/结束 | 工具调用可视化 |
| `plan` | Planner 生成子任务计划 | 任务卡片渲染 |
| `subtask_progress` | 子任务状态变更 | 进度条更新 |
| `token_update` | Token 用量变化 | 成本实时展示 |
| `skill` | 技能激活 | 技能标签展示 |
| `done` | 执行完成 | 历史保存 + 统计汇总 |

所有事件通过 FastAPI `StreamingResponse` 以 SSE 格式实时推送至 React 前端。

---

### 亮点 9：工作记忆（Working Memory）系统

各 Agent 节点通过 `state["memory"]` 共享结构化工作记忆体：

```python
# state.py 第34-41行
memory = {
    "papers_archive": [],       # 文献调研结果
    "innovation_candidates": [], # 创新方案候选
    "experiment_log": [],       # 实验记录
    "research_topic": "",       # 当前研究主题
    "baseline_model": "",       # 基线模型
    "current_focus": "",        # 当前关注点
}
```

- Research Agent 自动归档论文（正则提取标题）
- Innovator Agent 自动记录创新方案
- Experiment Agent 自动记录实验日志
- 记忆通过 `backend/memory/manager.py` 的 `load_working_memory()` / `save_working_memory()` 实现 JSON 文件持久化

---

### 亮点 10：完整的异常处理与降级策略

系统在多个层面设计了容错机制：

- **LLM JSON 解析失败** → 降级为 `chat` Agent
- **Planner 规划失败** → 回退到 Supervisor 重新判断
- **工具调用异常** → 捕获异常后继续，不阻断流程
- **执行超时** → 600 秒超时保护
- **会话限制**：200 会话上限 + 20 万 Token / 会话预算 + 100 轮次上限
- **会话溢出** → 自动删除最旧会话

---

## 三、完整执行流程示意

```
用户输入 "帮我写一篇关于 Transformer 效率优化的综述"
    │
    ▼
[Supervisor] 意图分类 → 检测到综述关键词 → 路由到 Planner
    │
    ▼
[Planner] 分解为 3 个子任务：
    ├─ 子任务1: 搜索 "sparse attention efficiency"  (Research Agent)
    ├─ 子任务2: 搜索 "linear attention methods"      (Research Agent)
    └─ 子任务3: 综合写综述                           (Synthesizer)
    │
    ▼
[Supervisor → Research] 执行子任务1 → ReAct 搜索 → 归档论文 → 回 Supervisor
    │
    ▼
[Supervisor → Research] 执行子任务2 → ReAct 搜索 → 归档论文 → 回 Supervisor
    │
    ▼
[Supervisor → Synthesizer] 收集子任务1+2 的输出 → LLM 综合为 800+ 字综述
    │
    ▼
    → SSE 流式推送到前端 → Word 文档自动生成 → 结束
```

---

## 四、技术栈总结

| 层次 | 技术 |
|------|------|
| Agent 编排框架 | LangGraph StateGraph（条件边 + 内存检查点） |
| LLM 引擎 | DeepSeek Chat / DeepSeek Reasoner（OpenAI 兼容 SDK） |
| 工具生态 | arXiv API、阿里云百炼联网搜索、python-docx |
| 后端框架 | FastAPI + SSE 流式响应 |
| 前端 | React（Agent 流水线可视化 + 打字机效果） |
| 持久化 | JSON 文件（会话历史 + 工作记忆 + 全局用量） |
| 部署 | Docker Compose |

---

## 五、核心代码文件索引

| 文件 | 行数 | 说明 |
|------|------|------|
| `backend/agent/graph.py` | 490 | LangGraph 图构建 + SSE 流式调度 |
| `backend/agent/supervisor.py` | 304 | 智能路由 + 子任务队列调度 |
| `backend/agent/planner_agent.py` | 175 | 复杂任务层次化分解 |
| `backend/agent/research_agent.py` | 264 | ReAct 两阶段文献调研 |
| `backend/agent/innovator_agent.py` | 197 | 创新方案构思 + 多维评估 |
| `backend/agent/experiment_agent.py` | 180 | 消融实验设计 + 日志归档 |
| `backend/agent/synthesizer.py` | 147 | 多输出综合 + Word 文档导出 |
| `backend/agent/skills.py` + `skills_store.py` | 160 | 内置/自定义技能系统 |
| `backend/agent/total_usage.py` | 61 | 全局 Token 成本追踪 |
| `backend/agent/llm.py` | 117 | LLM 配置 + 计费模型 |
| `backend/agent/state.py` | 92 | AgentState 类型定义 |
| `backend/main.py` | 744 | FastAPI 入口 + 所有 API 端点 |
