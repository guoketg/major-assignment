# 多 Agent 协作科研分析系统 — Agent 定义与职责划分

## 0. 状态跟踪表

> 此表跟踪本 PRD 涉及功能的实现进度，随项目进展持续更新。

| Agent | 前端可视化 | 后端节点 | 路由逻辑 | 测试 | 总体状态 |
|-------|:---------:|:--------:|:--------:|:----:|:--------:|
| 🤖 Supervisor（智能路由） | ✅ 前端可视化 | ✅ **LLM意图分类** | ✅ **LangGraph图** | ✅ 测试通过 | ✅ **已完成** |
| 💬 Chat Agent（对话助手） | ✅ 前端可视化 | ✅ **LLM对话节点** | ✅ **LangGraph图** | ✅ 测试通过 | ✅ **已完成** |
| 🔍 Research Agent（文献调研） | ✅ 前端可视化 | ✅ **LLM+arXiv工具** | ✅ **ReAct循环** | ✅ 测试通过 | ✅ **已完成** |
| 💡 Innovator Agent（创新构思） | ✅ 前端可视化 | ✅ **LLM创新节点** | ✅ **LangGraph图** | ✅ 测试通过 | ✅ **已完成** |
| 🧪 Experiment Agent（实验分析） | ✅ 前端可视化 | ✅ **LLM实验节点** | ✅ **LangGraph图** | ✅ 测试通过 | ✅ **已完成** |
| 👁️ **Agent 前端可视化** | ✅ **已完成** | ✅ **模拟层已完成** | ✅ **测试通过** | — | ✅ **已完成** |
| 🔗 **LangGraph 后端引擎** | — | ✅ **StateGraph** | ✅ **astream_events** | ✅ 测试通过 | ✅ **已完成** |
| 🔧 **arXiv 工具封装** | — | ✅ **LangChain @tool** | ✅ **3s限流** | ✅ 测试通过 | ✅ **已完成** |
| 🗃️ **工作记忆（JSON）** | — | ✅ **记忆管理器** | ✅ **按Agent检索** | ✅ 测试通过 | ✅ **已完成** |
| 📋 **Synthesizer（综合输出）** | ✅ **前端展示** | ✅ **消息清理+docx关键词触发** | ✅ **只保留user+最终assistant** | ✅ **去重测试通过** | ✅ **已完成** |
| 🧹 **子任务中间消息清理** | ✅ **流式跳过中间Agent内容** | ✅ **Synthesizer重建历史** | ✅ **Router事件去重** | ✅ **多条→1条测试通过** | ✅ **已完成** |

**图例：** ✅ 已完成 &nbsp;🚧 开发中 &nbsp;⏳ 待测试 &nbsp;❌ 未开始 &nbsp;📝 已规划

---

## 1. 总体框架

本系统包含 **1 个 Supervisor + N 个 Specialist Agent**，每个 Agent 都拥有独立的感知、记忆、规划和行动能力，通过 LangGraph 的 StateGraph 协同工作。

```
Supervisor Agent (协调者)
├── Chat Agent (对话)       — 默认对话处理
├── Research Agent (调研)   — 文献检索与分析
├── Innovator Agent (创新)  — 算法构思与对比
└── Experiment Agent (实验) — 实验设计与消融分析
```

## 2. Supervisor Agent（协调者/路由）

| 维度 | 描述 |
|------|------|
| **角色** | 系统的大脑，理解用户意图，分配合适的 Agent |
| **感知** | 接收用户最新消息 + 完整对话历史 |
| **记忆** | 短期：当前会话状态；工作：各 Agent 返回的结果 |
| **规划** | 意图分类 → 判断需要哪个 Agent → 也可能需要多个 Agent 协作 |
| **行动** | 路由到目标 Agent / 综合多 Agent 输出 / 直接回答简单问题 |

**意图分类示例：**
- `search` → 用户想查文献 → Research Agent
- `innovate` → 用户想构思创新 → Innovator Agent
- `experiment` → 用户在讨论实验 → Experiment Agent
- `analyze` → 用户上传了实验结果 → Experiment Agent
- `chat` → 普通对话 → Chat Agent
- `export` → 用户想导出报告 → Reporter（特殊节点）
- `complex` → 复杂任务，需要多 Agent 协作 → 先调研→再创新→再实验

**核心伪代码：**
```python
def supervisor_agent(state: AgentState) -> AgentState:
    intent = classify_intent(state.messages[-1], state.memory)
    state["current_agent"] = intent
    state["plan"] = decompose_task(intent, state.messages[-1])
    return state
```

**路由逻辑：**
```python
def router(state: AgentState) -> str:
    return state["current_agent"]
```

## 3. Chat Agent（对话 Agent）

| 维度 | 描述 |
|------|------|
| **角色** | 默认会话处理器，处理普通对话和知识问答 |
| **感知** | 当前消息 + 对话历史 |
| **记忆** | 短期：整轮对话上下文 |
| **规划** | 无复杂规划，直接调用 LLM 生成回复 |
| **行动** | 调用 LLM 生成文本回复 |
| **触发条件** | 用户闲聊、问基础知识、未触发其他 Agent |

## 4. Research Agent（文献调研 Agent）

| 维度 | 描述 |
|------|------|
| **角色** | 深度学习文献专家，负责检索、筛选、总结论文 |
| **感知** | 用户查询 + 调研需求（关键词、领域、时间范围） |
| **记忆** | 工作：已读论文摘要库 (`papers_archive`) |
| **规划** | 拆解查询 → 多轮搜索 → 筛选 → 分类 → 总结 |
| **行动** | 工具调用：`search_arxiv`、`search_web`、`parse_paper` |
| **触发条件** | 用户提及论文、方法、SOTA、相关工作 |

**执行流程：**
```
1. 解析用户的研究问题
2. 构建设计多组搜索查询式
3. 调用 arXiv 搜索工具，获取论文列表
4. 对每篇论文提取标题、摘要、方法分类
5. 按「已证实/争议/不足」标准分类标注
6. 存入 memory["papers_archive"]
7. 返回结构化调研摘要给 Supervisor
```

**信息分类标准（所有 Agent 通用）：**
| 标签 | 含义 |
|------|------|
| ✅ **已证实** | 多篇高水平论文一致验证，或有理论保证 |
| ⚠️ **争议点** | 不同文献结论相悖，或仅在受限设置下有效 |
| ❓ **资料不足** | 经充分检索，尚无公开研究或仅粗略提及 |

## 5. Innovator Agent（创新构思 Agent）

| 维度 | 描述 |
|------|------|
| **角色** | 科研方法论专家，基于调研结果构思创新方案 |
| **感知** | 调研结果 (`papers_archive`) + 基线特性 + 用户要求 |
| **记忆** | 工作：`innovation_candidates`（创新方案对比表）|
| **规划** | 分析现有方法缺陷 → 构思 ≥3 种创新路径 → 多维度对比 |
| **行动** | LLM 推理 + 调用 Draw.io 生成架构草图 |
| **触发条件** | 调研完成后 / 用户要求构思创新 |

**对比维度：**
新颖性、潜在提升幅度、实现复杂度、计算开销、与基线兼容性

## 6. Experiment Agent（实验设计与分析 Agent）

| 维度 | 描述 |
|------|------|
| **角色** | 实验设计专家，负责消融实验方案 + 结果分析 |
| **感知** | 用户实验数据（准确率、曲线、消融结果）|
| **记忆** | 工作：`experiment_log`（实验步骤 + 结果 + 分析）|
| **规划** | 设计消融实验 → 拆分组件 → 分析用户反馈 → 迭代优化 |
| **行动** | LLM 推理 + 生成实验方案 → 更新 Word 文档 |
| **触发条件** | 用户提供实验结果 / 需要设计实验 |

**迭代闭环：**
```
设计方案 → 用户执行实验 → 用户反馈结果 → 
分析瓶颈 → 调整方案 → 再次设计 → ... → 收敛 → 确定最佳方案
```

## 7. Reporter（报告生成节点 — 非独立 Agent）

一个特殊的执行节点，非独立 Agent，由 Supervisor 按需调用。

| 维度 | 描述 |
|------|------|
| **职责** | 将记忆中的调研、创新、实验数据导出为 Word 文档 + Draw.io 图表 |
| **工具** | `python-docx` 创建/编辑 .docx 文件，Draw.io XML 操作绘制架构图 |
| **触发** | 用户指令：`导出报告`、`保存调研`、`画架构图` |

## 8. Agent 协作示例

**复杂任务：「帮我调研 CLIP 噪声标签学习的 SOTA，提出创新方案」**

```
Step 1: Supervisor 收到请求，判断为 `complex` 任务
Step 2: Supervisor 规划: 调研→创新
Step 3: → 路由到 Research Agent
Step 4: Research Agent 搜索 arXiv，构建 papers_archive
Step 5: → 返回 Supervisor
Step 6: Supervisor 判断调研完成，路由到 Innovator Agent
Step 7: Innovator Agent 阅读 papers_archive，生成 3 种创新方案
Step 8: → 返回 Supervisor
Step 9: Supervisor 综合调研+创新结果，生成最终回复
```
