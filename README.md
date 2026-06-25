# 🤖 多 Agent 协作科研分析系统

> **Vibe Coding 综合实验项目** — 基于 AI 智能体的人工智能项目开发
>
> **项目仓库**: https://github.com/guoketg/major-assignment

---

## 项目简介

本系统是一个基于 **LangGraph 多 Agent 架构** 的科研协作助手，使用真实的 **DeepSeek 大语言模型** 驱动，能够自动化完成文献调研、创新构思、实验设计等科研流程。

系统以 FastAPI 构建后端服务，React 构建前端界面，Docker 容器化部署，实现了流式 SSE 输出、三层记忆管理、MCP 协议联网搜索、会话持久化、安全护栏、LLMOps 可观测性、技能系统、全局 Token 追踪等完整功能。LangGraph 编排的 7 个 Agent 节点通过有向图协作，完成从用户意图识别到任务分解、专业处理、结果综合的完整工作流。

---

## 核心功能

| 功能模块 | 说明 |
|---------|------|
| 🔍 **文献调研** | 自动搜索 arXiv 学术论文，分类总结并归档到记忆系统 |
| 💡 **创新构思** | 基于已归档的文献信息，生成创新方案并评估新颖度/难度 |
| 🧪 **实验分析** | 设计结构化的实验步骤、评价指标和对照方案 |
| 💬 **智能对话** | 支持多轮、流式 SSE 输出、Markdown/LaTeX 实时渲染 |
| 🌐 **联网搜索** | 通过 MCP Streamable HTTP 协议实时检索最新网络信息 |
| 📄 **报告导出** | 自动将对话/调研结果导出为 Word 文档 |
| 👁️ **Agent 可视化** | 前端实时展示 Agent 流水线状态、工具调用记录和 Token 消耗 |
| 🛡️ **安全护栏** | 三层防护（输入/输出/工具），提示词注入检测、有害内容过滤、敏感信息脱敏 |
| 📊 **LLMOps 可观测性** | LLM 调用追踪、性能指标（p50/p95/p99 延迟）、告警、缓存、速率限制 |
| 🎯 **技能系统** | 内置文档/代码/翻译等技能 + 自定义技能 CRUD，注入 Agent 工作流 |

---

## 环境准备

### 环境说明

本项目使用 **Python venv**（而非 conda）管理 Python 环境。conda 默认安装会占用约 5GB 空间，在存储受限的环境中使用 venv 更为轻量。完整依赖见 `backend/requirements.txt`，可通过 pip 完全复现。

> 如必须使用 conda，等价命令如下：
> ```bash
> conda create -n major-assignment python=3.11 -y
> conda activate major-assignment
> python -m pip install -r backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 方式一：Docker 部署（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/guoketg/major-assignment.git
cd major-assignment

# 2. 配置环境变量（需提前申请 API Key）
echo "DEEPSEEK_API_KEY=your_key_here" > .env

# 3. 构建并启动
docker compose up -d --build

# 4. 访问
#    前端: http://localhost:3001
#    后端: http://localhost:8001
```

### 方式二：本地开发

```bash
# 后端
python -m venv venv
source venv/Scripts/activate                              # Windows Git Bash
# 或 venv\Scripts\activate                                # Windows cmd
pip install -r backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 前端（另一个终端）
cd frontend
npm install --registry=https://registry.npmmirror.com
REACT_APP_API_URL=http://localhost:8000 npm start
# 访问 http://localhost:3000
```

### 连接测试

```bash
# 确认后端服务正常
curl http://localhost:8000/health

# 真实模型连通性测试
python tools/test_chat.py
```

---

## 模型配置

### 环境变量

在项目根目录创建 `.env` 文件（参见 `.env.example`）：

```env
# DeepSeek API（必填）
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 可选配置
DEEPSEEK_MODEL=deepseek-chat                    # 默认模型名
OPENAI_BASE_URL=https://api.deepseek.com/v1      # API 端点

# 阿里云 DashScope API（联网搜索功能，可选）
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

### 模型选项

| 前端显示 | API 模型 | 特点 | 适用场景 |
|---------|---------|------|---------|
| DeepSeek V4 Flash | `deepseek-chat` | 快速响应，默认选项 | 日常对话、快速问答 |
| V4 Pro | `deepseek-chat` | 更高能力（同模型名，不同参数） | 复杂推理、文献分析 |
| 思考模式 | `deepseek-reasoner` | 深度推理，显示思考过程 | 复杂问题、数学推理 |

> **失败处理**：API 调用超时（180 秒）或失败时，LLM 层自动重试 2 次后抛出，系统返回友好错误信息并记录日志，不会崩溃。Agent 工具调用失败时会记录错误日志并继续执行，不阻断整体流程。

### LLMOps 可观测性

系统内置 LLM 运维可观测性基础设施，通过 API 暴露以下能力：

| 能力 | 说明 | 相关 API |
|------|------|----------|
| 调用追踪 | 结构化 LLM 调用记录，含 trace_id 串联 | `GET /llmops/traces` |
| 性能监控 | 延迟百分位 (p50/p95/p99)、成功率、Token 吞吐量 | `GET /llmops/metrics` |
| 响应缓存 | 内存 LRU 缓存 + TTL 过期 | `GET /llmops/cache` |
| 告警机制 | 错误率/延迟/成本阈值告警 | `GET /llmops/alerts` |
| 速率限制 | 令牌桶算法，默认 10 次/秒 | 自动生效 |

### 技能系统

系统内置 4 种技能（文档/代码/翻译/教学），支持用户自定义技能 CRUD。选择技能后，系统提示词会被注入到所有 Agent 工作流中，优化输出风格。

| 技能 | 说明 |
|------|------|
| 📄 文档 | 结构化文档输出，清晰的标题层级和格式 |
| 💻 代码 | 代码导向，优先提供可执行代码示例 |
| 🌐 翻译 | 专业翻译模式，保持原文风格和术语一致性 |
| 📚 教学 | 教学解释模式，由浅入深、分步骤讲解 |

---

## 系统架构

### 整体架构

```
用户浏览器 (React) ─── HTTP/SSE ─── FastAPI 后端 ─── LangGraph Agent 引擎
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
          ┌─────────▼────────┐   ┌─────────▼─────────┐   ┌────────▼────────┐
          │    工具层         │   │     安全层         │   │    外部服务      │
          │  arXiv 搜索       │   │  输入护栏          │   │  DeepSeek API   │
          │  MCP 联网搜索     │   │  输出护栏          │   │  arXiv API      │
          │  Word 文档        │   │  工具护栏          │   │  DashScope MCP  │
          └──────────────────┘   └───────────────────┘   └─────────────────┘
                    │                       │
          ┌─────────▼────────┐   ┌─────────▼─────────┐
          │    记忆层         │   │   可观测性层       │
          │  短期(消息列表)    │   │  LLM 调用追踪      │
          │  工作(JSON 文件)   │   │  性能指标收集      │
          │  长期(预留接口)    │   │  响应缓存+告警     │
          └──────────────────┘   └───────────────────┘
```

### Agent 流水线

系统基于 **LangGraph StateGraph** 构建了 7 个 Agent 节点，形成有向图工作流：

```
Supervisor ──→ Chat Agent       （日常对话）
    │         → Research Agent   （文献调研，ReAct 循环 + 工具调用）
    │         → Innovator Agent  （创新方案构思与评估）
    │         → Experiment Agent （结构化实验设计）
    │         → Planner Agent    （复杂任务分解为子任务队列）
    └────────→ Synthesizer       （汇总各 Agent 输出 → END）
```

各节点职责：

- **Supervisor（智能路由）** — 接收用户消息，通过 LLM 进行意图分类，决定路由目标
- **Chat Agent（对话助手）** — 处理普通对话和问答，不涉及工具调用
- **Research Agent（文献调研）** — 以 ReAct 循环模式运行，可调用 search_arxiv、web_search 等工具
- **Innovator Agent（创新构思）** — 基于已归档的论文信息生成创新方案并进行评估
- **Experiment Agent（实验分析）** — 设计结构化的实验方案，含步骤、指标、对照方法
- **Planner Agent（任务规划）** — 将复杂任务分解为多个子任务，支持并行执行
- **Synthesizer（综合输出）** — 汇总各 Agent 输出，生成结构化综合报告

### 工具集成

| 工具 | 封装方式 | 功能 |
|------|---------|------|
| `search_arxiv` | LangChain `@tool` | arXiv 学术论文搜索，支持关键词/排序/数量控制，3 秒限流 |
| `web_search` | MCP Streamable HTTP | 通过 DashScope MCP 协议实时联网搜索，JSON-RPC over SSE |
| `create_docx` | LangChain `@tool` | 创建 Word 文档 |
| `add_section` | LangChain `@tool` | 向文档添加章节 |
| `add_table` | LangChain `@tool` | 向文档添加数据表格 |

### 记忆系统

| 层级 | 存储介质 | 说明 |
|------|---------|------|
| 短期记忆 | `AgentState.messages` | 当前对话历史列表，LangGraph 自动管理 |
| 工作记忆 | JSON 文件 (`agent_memory/`) | 论文归档、创新方案、实验日志，跨会话持久化 |
| 长期记忆 | 向量数据库（预留接口） | 跨会话语义检索（待实现） |

---

## API 接口

### 核心对话 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/chat/stream` | 流式对话（SSE 协议，含 Agent 事件推送、安全护栏） |
| `POST` | `/chat` | 非流式对话 |
| `GET` | `/sessions` | 获取会话列表 |
| `POST` | `/sessions` | 创建新会话（含数量上限控制） |
| `GET` | `/history/{session_id}` | 获取历史消息（含流水线 meta 信息） |
| `DELETE` | `/history/{session_id}` | 删除会话 |
| `GET` | `/health` | 服务健康检查 |

### 资源 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/arxiv/search` | arXiv 论文搜索（3 秒限流） |
| `GET` | `/memory/{session_id}` | 查看工作记忆（论文/方案/实验） |
| `GET` | `/export/docx/{session_id}` | 导出对话为 Word 文档 |
| `GET` | `/download/reports/{filename}` | 下载生成的报告文件 |
| `GET` | `/usage/total` | 全局累计 Token 用量 |

### 技能管理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/skills` | 获取所有可用技能 |
| `GET` | `/skills/{skill_id}` | 获取技能详情（含 prompt） |
| `POST` | `/skills` | 创建自定义技能 |
| `PUT` | `/skills/{skill_id}` | 更新自定义技能 |
| `DELETE` | `/skills/{skill_id}` | 删除自定义技能 |

### LLMOps 可观测性 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/llmops/health` | LLMOps 健康状态（缓存/错误率/p95 延迟） |
| `GET` | `/llmops/metrics` | 性能指标快照（延迟百分位/成功率/吞吐量） |
| `GET` | `/llmops/traces` | 最近 LLM 调用追踪记录 |
| `GET` | `/llmops/traces/{session_id}` | 按会话查询追踪记录 |
| `GET` | `/llmops/alerts` | 告警列表（支持按确认状态筛选） |
| `POST` | `/llmops/alerts/{alert_id}/acknowledge` | 确认告警 |
| `GET` | `/llmops/cache` | 缓存统计（命中率/大小） |
| `POST` | `/llmops/cache/clear` | 清空缓存 |
| `POST` | `/llmops/metrics/persist` | 手动持久化当日指标 |

---

## 运行测试

本项目提供 **21 个自动化测试与辅助脚本**，覆盖 API、Agent 流水线、前端 UI、安全护栏和工具功能。

```bash
# 激活虚拟环境后执行：

# 1. 全面功能测试 — 覆盖健康检查、arXiv搜索、会话管理、Agent流水线等 8 项
PYTHONIOENCODING=utf-8 python tools/test_all_features.py

# 2. 流式对话测试（Playwright）
python tools/test_streaming.py

# 3. 后端流式 API 测试
python tools/test_streaming_api.py

# 4. 流式输出细节测试
python tools/test_streaming_detailed.py

# 5. 前端界面测试（Playwright）
python tools/test_front_end.py

# 6. 前端聊天功能测试（Playwright）
python tools/test_front_end_chat.py

# 7. Agent 可视化测试（Playwright）
python tools/test_agent_visualization.py

# 8. Research Agent 详细行为测试
python tools/test_research_agent_detail.py

# 9. Research Agent 基础测试
python tools/test_research_agent.py

# 10. ReAct 循环工具调用测试
python tools/test_react_loop.py

# 11. 联网搜索（MCP 协议）测试
python tools/test_web_search.py

# 12. 对话 API 基础测试
python tools/test_chat.py

# 13. 安全护栏测试
python tools/test_guardrails.py

# 14. MCP 工具列表测试
python tools/test_mcp_list.py

# 15. Agent 流水线修复测试
python tools/test_agent_pipeline_fix.py

# 16. 改进版 UI 测试
python tools/test_ui_improved.py

# 17. Agent 选择器检查
python tools/check_agent_selector.py

# 18. Agent UI 检查
python tools/check_agent_ui.py

# 19. Graph 修复工具
python tools/fix_graph.py

# 20. Markdown → Word 转换工具
python tools/md_to_docx.py

# 21. 后端更新工具
python tools/update_backend.py
```

> 注意：Playwright 测试和 Agent 流水线测试需要先通过 `docker compose up -d` 启动后端服务。

---

## 项目结构

```
major-assignment/
├── backend/
│   ├── main.py                 # FastAPI 入口，30+ API 端点
│   ├── requirements.txt        # Python 依赖列表
│   ├── agent/                  # LangGraph 多 Agent 系统
│   │   ├── graph.py            # StateGraph 构建与流式执行
│   │   ├── state.py            # AgentState TypedDict 定义
│   │   ├── llm.py              # LLM 配置（DeepSeek + LangChain）
│   │   ├── supervisor.py       # 智能路由节点
│   │   ├── chat_agent.py       # 对话助手节点
│   │   ├── research_agent.py   # 文献调研节点（ReAct 循环）
│   │   ├── innovator_agent.py  # 创新构思节点
│   │   ├── experiment_agent.py # 实验分析节点
│   │   ├── planner_agent.py    # 任务规划节点
│   │   ├── synthesizer.py      # 综合输出节点
│   │   ├── guardrails.py       # 安全护栏（输入/输出/工具三层防护）
│   │   ├── llmops.py           # LLMOps 可观测性（追踪/指标/缓存/告警/限流）
│   │   ├── skills.py           # 技能配置模块
│   │   ├── skills_store.py     # 自定义技能 CRUD 存储
│   │   └── total_usage.py      # 全局 Token 累计用量追踪
│   ├── tools/                  # 工具封装层
│   │   ├── arxiv_tool.py       # arXiv 学术搜索（@tool）
│   │   ├── web_search_tool.py  # MCP 联网搜索（@tool + MCP 协议）
│   │   └── docx_tool.py        # Word 文档生成（@tool）
│   ├── memory/
│   │   └── manager.py          # 三层记忆管理器
│   ├── conversations/          # 对话历史持久化文件
│   ├── agent_memory/           # 工作记忆 + 自定义技能持久化文件
│   ├── generated_reports/      # 生成的 Word 文档
│   └── logs/llmops/            # LLMOps 追踪/指标/告警日志
├── frontend/
│   ├── public/
│   │   └── index.html          # HTML 入口
│   ├── src/
│   │   ├── App.tsx             # React 单页应用（聊天 + Agent 可视化 + 论文搜索 + 技能选择）
│   │   └── index.tsx           # React 入口
│   ├── build/                  # 生产构建产物（含 KaTeX 字体）
│   ├── package.json
│   └── tsconfig.json
├── tools/                      # 21 个自动化测试与辅助脚本 + 11 张测试截图
│   ├── test_all_features.py    # 全面功能测试（8 项子测试）
│   ├── test_guardrails.py      # 安全护栏测试
│   ├── test_streaming.py       # 流式对话测试
│   ├── test_front_end.py       # 前端 UI 测试
│   ├── test_agent_visualization.py  # Agent 可视化测试
│   ├── test_web_search.py      # MCP 联网搜索测试
│   ├── test_mcp_list.py        # MCP 工具列表测试
│   ├── md_to_docx.py           # Markdown → Word 转换工具
│   └── ...
├── prd-request/                # 7 篇 PRD 设计文档
│   ├── prd1-intro.md
│   ├── 01-系统概述与架构愿景.md
│   ├── 02-Agent定义与职责划分.md
│   ├── 03-LangGraph编排与协作流程.md
│   ├── 04-记忆系统设计.md
│   ├── 05-工具集成与封装.md
│   └── 06-前后端集成方案.md
├── docs/                       # 项目文档与实验报告
│   ├── 综合实验报告.docx        # Word 格式综合实验报告
│   ├── 综合实验报告.md          # Markdown 格式综合实验报告（13 章）
│   ├── Agent系统亮点讲解_实验报告.md
│   ├── 代码审查报告.md
│   ├── 功能清单.md
│   ├── 测试用例文档.md
│   ├── 失败路径测试记录.md
│   ├── 失败路径测试输出.txt
│   ├── 测试运行输出.txt
│   ├── 项目文档.md
│   └── 项目整体流程示意图.html
├── Dockerfile.backend          # 后端容器镜像
├── Dockerfile.frontend         # 前端容器镜像（Nginx + React 构建产物）
├── docker-compose.yml          # 容器编排
├── .env.example                # 环境变量模板
├── package.json                # 根级 Playwright 测试依赖
├── CLAUDE.md                   # AI 智能体工作规范
└── .gitignore
```

---

## 安全机制

### 三层安全护栏 (Guardrails)

系统实现完整的三层安全防护体系（`backend/agent/guardrails.py`，约 700 行），遵循防御深度原则：

| 层级 | 功能 | 检测内容 |
|------|------|---------|
| **输入护栏** | 用户消息前置检查 | 提示词注入（40+ 中英文模式）、越狱检测、有害内容过滤、敏感信息扫描与脱敏、输入长度限制 |
| **输出护栏** | Agent 输出后置检查 | 有害内容检测、敏感信息脱敏（API Key/Token/密钥等 15+ 模式） |
| **工具护栏** | 工具调用验证 | 白名单机制、参数长度/范围限制、路径遍历防护（null 字符/`..`/文件名长度） |

拦截原因分类：`PROMPT_INJECTION`、`HARMFUL_CONTENT`、`SENSITIVE_INFO`、`JAILBREAK`、`TOOL_ABUSE`、`PATH_TRAVERSAL`

### 其他安全措施

| 防护措施 | 位置 | 说明 |
|---------|------|------|
| 路径安全检查 | `backend/main.py` | realpath 前缀比较，防止路径遍历攻击 |
| 密钥隔离 | 项目根 | `.env` + `.gitignore`，密钥不提交到仓库 |
| 输入校验 | `backend/main.py` | Pydantic 模型校验，message 1-10000 字符，agent 正则匹配 |
| arXiv 限流 | `backend/main.py` | 3 秒请求间隔，遵守 arXiv API 使用规范 |
| LLM 超时与重试 | `backend/agent/llm.py` | 180 秒超时 + 2 次自动重试 |
| Agent 执行超时 | `backend/main.py` | 600 秒超时保护（asyncio.timeout），防止无限执行 |
| CORS 限制 | `backend/main.py` | 仅允许指定前端地址，非通配符 |
| 日志脱敏 | `backend/main.py` | sanitize_log 函数，session_id 截断、消息内容截断 |
| 会话数量上限 | `backend/main.py` | MAX_SESSIONS=200，超出自动删除最旧会话 |
| Token 预算限制 | `backend/main.py` | 每会话 200000 Token 上限，100 轮次上限，80% 预警 |
| Session ID 增强 | `backend/main.py` | 完整 UUID 格式，替代简单 hex 串 |

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Python FastAPI |
| AI 编排 | LangGraph + LangChain |
| 大模型 | DeepSeek API（兼容 OpenAI SDK） |
| 前端框架 | React 18 + TypeScript |
| 内容渲染 | ReactMarkdown + remark-gfm + rehype-highlight + KaTeX |
| 前端构建 | react-scripts 5（开发）/ Nginx（生产容器） |
| 容器化 | Docker + Docker Compose |
| 自动化测试 | Playwright + Python requests |
| 通信协议 | SSE 流式推送 + MCP Streamable HTTP |
| 环境管理 | Python venv + pip（可选 conda） |
| 文档生成 | python-docx |
| 可观测性 | 自研 LLMOps 模块（追踪/指标/缓存/告警/限流） |
| 安全防护 | 自研三层护栏（输入/输出/工具，40+ 检测模式） |

---

## 已知限制

1. **模型依赖** — 系统依赖 DeepSeek API 在线服务，无网络时不可用
2. **速率限制** — arXiv API 要求至少 3 秒查询间隔；DeepSeek API 有调用频率限制
3. **长期记忆** — 向量数据库检索尚未实现，跨会话语义记忆为预留接口
4. **用户认证** — 缺少多用户隔离和权限管理机制
5. **护栏基于规则** — 安全护栏使用正则模式匹配，非 ML 模型，可能漏检复杂攻击
6. **模型覆盖** — 目前仅支持 DeepSeek 系列，尚未接入 OpenAI/Claude 等
7. **Docker 构建** — 需要稳定的网络以下载基础镜像
8. **Token 成本** — 使用在线 API 产生费用，单次复杂 Agent 任务约消耗数千 Token

## MCP 风格工具契约

至少 1 个 MCP 风格工具契约是课程硬性要求。本项目通过 `web_search` 工具实际集成 MCP Streamable HTTP 协议，调用 DashScope 的联网搜索服务。工具契约如下：

```text
协议化工具：web_search
用途：通过 DashScope MCP 协议实时联网搜索
输入：{"query": "字符串，1-500字符"}
输出：{"success": true/false, "results": ["搜索结果摘要列表"], "error": "错误信息"}
失败：query 为空返回 validation_error；MCP 连接失败返回 connection_error
安全边界：仅通过 DashScope 官方 MCP 端点调用，不直接访问任意 URL
协议：JSON-RPC 2.0 over SSE（MCP Streamable HTTP 传输）
```

详见 `backend/tools/web_search_tool.py`。

---

## 许可

本作品采用 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 许可协议。

> **Vibe Coding 综合实验项目** | 2026 年 6 月
