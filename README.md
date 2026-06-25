# 🤖 多 Agent 协作科研分析系统

> **Vibe Coding 综合实验项目** — 基于 AI 智能体的人工智能项目开发
>
> **项目仓库**: https://github.com/guoketg/major-assignment

---

## 项目简介

本系统是一个基于 **LangGraph 多 Agent 架构** 的科研协作助手，使用真实的 **DeepSeek 大语言模型** 驱动，能够自动化完成文献调研、创新构思、实验设计等科研流程。

系统以 FastAPI 构建后端服务，React 构建前端界面，Docker 容器化部署，实现了流式输出、三层记忆管理、MCP 风格工具契约、会话持久化等核心功能。LangGraph 编排的 7 个 Agent 节点通过有向图协作，完成从用户意图识别到任务分解、专业处理、结果综合的完整工作流。

---

## 核心功能

| 功能模块 | 说明 |
|---------|------|
| 🔍 **文献调研** | 自动搜索 arXiv 学术论文，分类总结并归档到记忆系统 |
| 💡 **创新构思** | 基于已归档的文献信息，生成创新方案并评估新颖度/难度 |
| 🧪 **实验分析** | 设计结构化的实验步骤、评价指标和对照方案 |
| 💬 **智能对话** | 支持多轮、流式 SSE 输出、Markdown/LaTeX 实时渲染 |
| 🌐 **联网搜索** | 通过 MCP 协议实时检索最新网络信息 |
| 📄 **报告导出** | 自动将调研结果导出为 Word 文档 |
| 👁️ **Agent 可视化** | 前端实时展示 Agent 流水线状态和工具调用记录 |

---

## 环境准备

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

> **说明**：本项目使用 **Python venv**（而非 conda）管理环境。conda 默认安装会占用约 5GB 空间，在存储受限的环境中使用 venv 更为轻量。完整依赖见 `backend/requirements.txt`，可通过 pip 完全复现。

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

---

## 系统架构

### 整体架构

```
用户浏览器 (React) ─── HTTP/SSE ─── FastAPI 后端 ─── LangGraph Agent 引擎
                                            │
                          ┌─────────────────┼─────────────────┐
                          │                 │                 │
                    ┌─────▼─────┐   ┌───────▼───────┐   ┌─────▼──────┐
                    │  工具层    │   │   记忆层       │   │ 外部服务    │
                    │ arXiv搜索  │   │ 短期(消息)    │   │ DeepSeek   │
                    │ MCP联网搜索│   │ 工作(JSON)    │   │ arXiv API  │
                    │ Word文档   │   │ 长期(预留)    │   │ DashScope  │
                    └───────────┘   └───────────────┘   └────────────┘
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

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/chat/stream` | 流式对话（SSE 协议，含 Agent 事件推送） |
| `POST` | `/chat` | 非流式对话 |
| `GET` | `/sessions` | 获取会话列表 |
| `POST` | `/sessions` | 创建新会话 |
| `GET` | `/history/{session_id}` | 获取历史消息（含流水线 meta 信息） |
| `DELETE` | `/history/{session_id}` | 删除会话 |
| `GET` | `/arxiv/search` | arXiv 论文搜索 |
| `GET` | `/memory/{session_id}` | 查看工作记忆 |
| `GET` | `/export/docx/{session_id}` | 导出对话为 Word 文档 |
| `GET` | `/download/reports/{filename}` | 下载生成的报告文件 |
| `GET` | `/health` | 服务健康检查 |

---

## 运行测试

本项目提供 **20 个自动化测试脚本**，覆盖 API、Agent 流水线、前端 UI 和工具功能。

```bash
# 激活虚拟环境后执行：

# 1. 全面功能测试 — 覆盖健康检查、arXiv搜索、会话管理、Agent流水线等 8 项
PYTHONIOENCODING=utf-8 python tools/test_all_features.py

# 2. 流式对话测试（Playwright）
python tools/test_streaming.py

# 3. 后端流式 API 测试
python tools/test_streaming_api.py

# 4. 前端界面测试（Playwright）
python tools/test_front_end.py

# 5. 前端聊天功能测试（Playwright）
python tools/test_front_end_chat.py

# 6. Agent 可视化测试（Playwright）
python tools/test_agent_visualization.py

# 7. Research Agent 详细行为测试
python tools/test_research_agent_detail.py

# 8. ReAct 循环工具调用测试
python tools/test_react_loop.py

# 9. 联网搜索（MCP 协议）测试
python tools/test_web_search.py

# 10. 对话 API 基础测试
python tools/test_chat.py
```

> 注意：Playwright 测试和 Agent 流水线测试需要先通过 `docker compose up -d` 启动后端服务。

---

## 项目结构

```
major-assignment/
├── backend/
│   ├── main.py                 # FastAPI 入口，11 个 API 端点
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
│   │   └── synthesizer.py      # 综合输出节点
│   ├── tools/                  # 工具封装层
│   │   ├── arxiv_tool.py       # arXiv 学术搜索（@tool）
│   │   ├── web_search_tool.py  # MCP 联网搜索（@tool + MCP 协议）
│   │   └── docx_tool.py        # Word 文档生成（@tool）
│   ├── memory/
│   │   └── manager.py          # 三层记忆管理器
│   ├── conversations/          # 对话历史持久化文件
│   ├── agent_memory/           # 工作记忆持久化文件
│   └── generated_reports/      # 生成的 Word 文档
├── frontend/
│   └── src/
│       ├── App.tsx             # React 单页应用（聊天 + Agent 可视化 + 论文搜索）
│       └── index.tsx           # React 入口
├── tools/                      # 20 个自动化测试与辅助脚本
│   ├── test_all_features.py    # 全面功能测试（8 项子测试）
│   ├── test_streaming.py       # 流式对话测试
│   ├── test_front_end.py       # 前端 UI 测试
│   ├── test_agent_visualization.py  # Agent 可视化测试
│   ├── md_to_docx.py           # Markdown → Word 转换工具
│   └── ...
├── prd-request/                # 6 篇 PRD 设计文档
│   ├── 01-系统概述与架构愿景.md
│   ├── 02-Agent定义与职责划分.md
│   ├── 03-LangGraph编排与协作流程.md
│   ├── 04-记忆系统设计.md
│   ├── 05-工具集成与封装.md
│   └── 06-前后端集成方案.md
├── docs/                       # 项目文档与实验报告
│   └── 综合实验报告.docx
├── Dockerfile.backend          # 后端容器镜像
├── Dockerfile.frontend         # 前端容器镜像
├── docker-compose.yml          # 容器编排
├── .env.example                # 环境变量模板
├── CLAUDE.md                   # AI 智能体工作规范
└── .gitignore
```

---

## 安全机制

| 防护措施 | 位置 | 说明 |
|---------|------|------|
| 路径安全检查 | `backend/main.py` | realpath 前缀比较，防止路径遍历攻击 |
| 密钥隔离 | 项目根 | `.env` + `.gitignore`，密钥不提交到仓库 |
| arXiv 限流 | `backend/tools/arxiv_tool.py` | 3 秒请求间隔，遵守 arXiv API 使用规范 |
| LLM 超时与重试 | `backend/agent/llm.py` | 180 秒超时 + 2 次自动重试 |
| Agent 执行超时 | `backend/main.py` | 600 秒超时保护，防止无限执行 |
| 输入校验 | `backend/main.py` | Pydantic 模型校验请求参数 |

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Python FastAPI |
| AI 编排 | LangGraph + LangChain |
| 大模型 | DeepSeek API（兼容 OpenAI SDK） |
| 前端框架 | React 18 + TypeScript |
| 内容渲染 | ReactMarkdown + KaTeX |
| 容器化 | Docker + Docker Compose |
| 自动化测试 | Playwright + Python requests |
| 通信协议 | SSE 流式推送 + MCP Streamable HTTP |
| 环境管理 | Python venv + pip |

---

## 已知限制

1. **模型依赖** — 系统依赖 DeepSeek API 在线服务，无网络时不可用
2. **速率限制** — arXiv API 要求至少 3 秒查询间隔；DeepSeek API 有调用频率限制
3. **长期记忆** — 向量数据库检索尚未实现，跨会话语义记忆为预留接口
4. **用户认证** — 缺少多用户隔离和权限管理机制
5. **成本监控** — Token 消耗仅在代码层面可追踪，缺乏前端可视化仪表板
6. **安全防护** — 输入长度限制、日志脱敏、null 字符过滤有待完善
7. **模型覆盖** — 目前仅支持 DeepSeek 系列，尚未接入 OpenAI/Claude 等
8. **Docker 构建** — 需要稳定的网络以下载基础镜像

---

## 许可

本作品采用 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 许可协议。

> **Vibe Coding 综合实验项目** | 2026 年 6 月
