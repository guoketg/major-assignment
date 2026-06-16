# 项目基础性介绍

## 1. 技术栈架构

本项目采用现代化的前后端分离架构，具体技术选型如下：

- **前端**：基于 TypeScript 与 React 框架构建用户界面，提供类型安全的前端开发体验
- **后端**：使用 FastAPI 框架构建高性能的 RESTful API 服务，支持异步处理与自动文档生成

## 2. 环境部署方式

项目采用 Docker 容器化部署方案，通过 docker-compose 实现服务编排与一键启动。

**启动命令**：
```bash
docker compose up -d
```

该命令将自动构建并启动所有相关服务容器，确保开发与生产环境的一致性。

## 3. 数据存储策略

项目实现了完善的数据持久化存储机制：

- **AI 对话记录**：所有用户与 AI 的对话历史均采用持久化存储，支持历史记录查询与上下文关联
- **文件数据**：项目生成的所有文件数据均实现持久性存储，确保数据安全与可追溯性

## 4. 数据库配置

项目集成了多种数据库系统以满足不同数据存储需求：

| 数据库 | 用途 | 特点 |
|--------|------|------|
| PostgreSQL | 基础数据存储 | 关系型数据库，用于存储结构化的业务数据、用户信息、对话元数据等 |
| MongoDB | 文件存储专用 | 文档型数据库，用于存储非结构化数据、文件元数据及大型数据对象 |

## 5. 大模型调用配置

项目使用 OpenAI 兼容接口调用 DeepSeek V4 Flash 大语言模型，实现高效的 AI 对话能力。

### 5.1 环境配置

**Python 虚拟环境创建**：
```bash
python -m venv venv
```

**激活虚拟环境**：
- Windows: `.\venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

**核心依赖安装**：
```bash
pip install openai python-dotenv
```

### 5.2 API 配置

在项目根目录的 `.env` 文件中配置密钥：
```env
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
```

### 5.3 调用示例

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}]
)
```

---

基于你的反馈，我重新设计了一个深度契合课程结业项目需求的智能体方案。这次定位聚焦于**计算机专业（深度学习领域）的科研分析专家**，融入 LangGraph 框架的规划-记忆-执行架构，封装了 ArXiv 检索、网络搜索、Word 文档操作和 Draw.io 绘图等工具，并突出了多创新方法比较与消融实验迭代。

以下是完整的设计文档，并以“多模态鲁棒性噪声标签学习，视觉基线 CLIP B/32”为例，展示一条完整的科研辅助流程。

---

## 基于 LangGraph 的深度学习科研分析智能体设计

### 1. 项目定位
一个面向深度学习研究者的**协同科研 Agent**。它不仅能做文献调研，更能辅助构思核心创新点，通过“提出方案→分析用户反馈的实验结果→迭代优化→消融实验确定最佳组件”的闭环，产出可落地的研究方案。项目封装为课程作业，重点展示 Agent 的复杂工具调用、状态记忆与规划能力。

### 2. 智能体核心原则
**角色**：擅长深度学习（特别是多模态学习、噪声标签、视觉-语言模型）的科研分析专家。  
**核心原则**：
- **证据驱动与诚实性**：严格区分已证实结果、学术争议和资料不足区域。无可靠证据时，明确声明“当前无足够研究”，拒绝编造。
- **多创新路径比较**：针对同一问题，系统性地提出至少三种不同角度的创新方法，并从*新颖性、潜在提升幅度、实现复杂度、计算开销、与基线（如 CLIP B/32）的兼容性*五个维度进行对比，供用户决策。
- **消融驱动优化**：通过设计消融实验，将复杂方案拆解为独立组件，根据用户反馈的实验数据，量化每个组件的贡献，确定最有效组合。
- **可复现与透明**：所有检索记录、分析过程、工具调用均记录在记忆模块中，并可导出为结构化 Word 文档。

### 3. 智能体架构（LangGraph 实现）
采用 LangGraph 的 `StateGraph` 构建主控流程，定义共享状态，并通过节点实现规划、执行和记忆更新。

#### 3.1 状态定义（AgentState）
```python
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    user_query: str                # 用户原始指令
    sub_tasks: List[Dict]         # 规划模块拆解的子任务队列
    memory: Dict                  # 持久记忆，包括：
        # "papers_archive": [{summary, evaluation, tags}]
        # "experiment_log": [{step, user_feedback, analysis}]
        # "innovation_candidates": [...]
    current_step: str             # 当前所处阶段：'survey', 'innovate', 'experiment', 'report'
    output_artifacts: Dict        # 已生成的制品路径或内容，如word文档id、drawio数据
```
状态在节点间流转，并支持检查点持久化（用于课程展示记忆能力）。

#### 3.2 图节点与规划-执行流程
图结构包含以下关键节点，通过条件边控制流转：

1. **规划节点（Planner）**：解读用户查询，分解为阶段性任务。例如：
   - 阶段1：调研 SOTA、方法分类与待解决方向。
   - 阶段2：分析基线（CLIP B/32）特性，构思多种创新算法并比较。
   - 阶段3：根据用户实验反馈迭代，设计消融实验。
   每个子任务附带了所需工具建议（`tools_to_call`）。

2. **检索与分析节点（Researcher）**：调用 ArXiv 和通用搜索工具，对子问题进行检索。然后调用摘要工具提取论文关键信息，并按照“已证实/争议/不足”分类缓存到 `memory["papers_archive"]`。

3. **创新构思节点（Innovator）**：基于调研结果和基线特性，生成多个创新方向。例如针对鲁棒噪声标签学习，可能生成：
   - **方向A**：基于 CLIP 零样本置信度加权的自适应损失。
   - **方向B**：利用多模态对比去噪的自监督预清洗阶段。
   - **方向C**：在 CLIP 视觉编码器上引入可学习的软标签校正网络。
   然后对每个方向进行多维度打分比较，形成结构化对比表存入记忆。

4. **实验设计节点（Experiment Designer）**：将选定创新细化为实验方案，包括网络结构修改、训练策略、评估指标。输出可直接在 Word 文档中保存的实验步骤，并利用 Draw.io 工具生成模型架构草图。

5. **反馈解析与迭代节点（Feedback Analyzer）**：接收用户提供的实验结果（如准确率、损失曲线、消融数据），更新 `experiment_log`，定位性能瓶颈，调整创新组件，重新规划消融实验。

6. **文档与绘图生成节点（Reporter）**：按需调用 Word 工具生成调研报告、实验日志；调用 Draw.io 工具绘制架构图、流程图或消融结果柱状图。

**执行路由**（简化）：
```
START → Planner → Researcher → (是否完成调研) → Innovator → Experiment Designer → 
→ 等待用户反馈 → Feedback Analyzer → (是否收敛) → 如果是，至 Reporter 并导出；否则返回 Innovator 微调
```

所有节点对 `AgentState` 的修改都会被记忆。用户可通过指令随时要求导出当前状态的 Word 文档（如调研简报或实验日志）。

### 4. 工具集封装
为满足项目要求，至少封装以下工具，并注册为 LangGraph 的 `ToolNode` 或直接在节点内调用。

#### 4.1 ArXiv 论文检索工具
- **功能**：根据关键词、作者、时间范围检索，返回标题、摘要、PDF 链接。
- **实现**：封装 `arxiv` 官方 API，构建查询式并过滤计算机科学子类（cs.CV, cs.LG）。  
- **示例调用**：`search_arxiv("multi-modal robust learning with noisy labels CLIP", max_results=20)`

#### 4.2 通用网络检索工具
- **功能**：获取 ArXiv 未覆盖的资源，如 GitHub 热门实现、技术博客、最新预印本新闻。
- **实现**：使用 SerpAPI 或 Bing Search API，限制信源为 `arxiv.org`, `github.com`, `paperswithcode.com` 等。

#### 4.3 论文深度解析工具（内置）
- **功能**：基于 PDF 链接下载并提取全文摘要、方法、实验设定，生成结构化摘要卡片。
- **价值**：为创新构思提供细致的 SOTA 细节对比。

#### 4.4 Word 文档操作工具集
- **新建文档**：创建 `.docx` 文件，预设模板（调研报告、实验日志）。
- **编辑段落**：追加/更新特定章节（如“创新点对比”、“消融实验结果”）。
- **删除与重置**：清除过时内容。
- **实现**：`python-docx` 封装，路径可由 Agent 管理。

#### 4.5 Draw.io 科研绘图工具
- **功能**：根据文本描述生成简单的框图或流程图（如模型架构、消融组件图）。
- **实现**：通过操作 `.drawio` 的 XML 源文件，使用预定义图形模板，或调用 `draw.io` 的命令行接口生成 PNG。Agent 将描述转化为绘图指令。

#### 4.6 （可选）代码模板与环境管理工具
用于生成基于 PyTorch+CLIP 的实验启动代码骨架，方便用户快速跑实验。

### 5. 记忆模块设计
记忆分三层：
- **短期记忆**：当前会话的 `AgentState`，包含未完成的子任务、上一步输出。
- **工作记忆**：已阅读论文的摘要库、创新点对比表、实验数据。以结构化字典存储在状态中，并随检查点持久化到本地文件。
- **长期知识库**：可跨会话的已归档论文特征向量库（使用 Chroma 或 FAISS），支持语义搜索“有哪些用 CLIP 处理噪声标签的方法”。在课程项目中可选做增量实现。

### 6. 信息分类标准（适用于所有分析）
针对每一个技术主张，Agent 必须标注：
- **已证实**：多篇高水平论文（如 CVPR/ICCV 论文）一致验证，或有理论保证。
- **争议点**：不同文献结论相悖，或仅在受限设置下有效。
- **资料不足**：经充分检索，尚无公开研究或仅粗略提及。

在创新构思阶段，Agent 会特别指出当前方案的局限性是否属于“争议点”，并据此设计规避风险的创新。

---

## 完整工作流程示例
**用户提问**：  
“我是做多模态噪声标签学习的研究生，现在希望用 CLIP B/32 作为视觉基线，调研这个方向的 SOTA，并帮我构思几种有创新性的算法，最终通过消融实验确定最有效的组件。请开始。”

**阶段一：课题调研**  
**Agent 规划** → 调用 `search_arxiv` 和网络搜索，检索式：
- `("robust learning" AND "noisy labels" AND "multi-modal")`
- `("CLIP" AND "noisy label" AND "vision-language")`
- `("label noise" AND "CLIP" AND "robust")`

从返回结果中筛选近三年高质量论文，使用解析工具提取方法分类。生成如下记忆结构：

```yaml
papers_archive:
  - title: "CLIP-Adapter for Robust Multi-modal Learning"
    category: 适配器方法
    core_idea: 在冻结的CLIP上加轻量级适配器学习噪声转移矩阵
    confirmed: 在多个噪声数据集上达到SOTA
    limitation: 需要一小部分干净验证集
  - title: "ProtoCLIP: Prototypical Contrastive Learning of Noisy Multi-modal Data"
    category: 原型学习
    ...
```

**阶段一输出**（存入Word调研简报）：  
- SOTA 方案分为三大类：损失校正法、多模态协同去噪、鲁棒表示学习。  
- 使用 CLIP B/32 的现有方法多依赖其零样本能力进行样本加权，但未充分利用图文对齐特性。  
- **用户可做的科研任务**：1) 设计无需干净验证集的在线噪声样本识别；2) 利用 CLIP 的文本塔进行语义清洁等。

**阶段二：创新算法分析与比较**  
**Agent Innovator** 基于基线和调研缺陷，生成三个创新方向，并进行多维比较：

| 创新方法 | 新颖性 | 潜在提升 | 实现难度 | 计算开销 | CLIP兼容性 | 简要描述 |
|----------|--------|----------|----------|----------|------------|----------|
| **A. 多模态一致性滤波** | 高 | 中 | 中 | 低 | 极高 | 结合图像-文本相似度与预测一致性动态过滤噪声样本，无需修改CLIP结构 |
| **B. 可学习软标签蒸馏** | 中 | 高 | 高 | 中 | 高 | 在CLIP视觉塔后接一个轻量级解码器，用文本塔生成软目标指导训练 |
| **C. 跨模态对比抗噪正则** | 极高 | 高 | 高 | 高 | 中 | 引入对抗扰动，强制图文特征对噪声标签不敏感，需微调部分CLIP层 |

Agent 推荐优先尝试 **A 和 B 的组合**，平衡创新与可行性。同时生成对应的 Draw.io 架构草图（如滤波器与蒸馏模块的串联图）。

**阶段三：实验迭代与消融**  
Agent 设计实验方案：  
- 基础实验：在标准噪声标签基准（如 CIFAR-100N）上，用 CLIP B/32 提取特征后，分别测试 A、B 及 A+B。  
- 记录用户反馈：“A+B 在 50% 对称噪声下准确率 78.2%，但 B 单独使用仅 74.1%，A 单独 76.5%”。

**Agent Feedback Analyzer** 解析：  
- 组件 A 贡献 +2.4%，组件 B 贡献 -0.2%？ 实际是组合比单纯 A 高 1.7%，因此 B 的贡献为 +1.7%（当与 A 结合时）。表明滤波后的干净数据为蒸馏提供了更好条件。  
- 决定设计消融实验：保留 A，将 B 的蒸馏温度、文本塔参与度作为变量。再次建议用户实验。

最终，通过三轮迭代，确定“多模态一致性滤波 + 低温度软蒸馏”为最佳组合，并在 Word 文档中记录完整消融表格和最优配置。Agent 更新 Draw.io 图，导出最终模型架构。

---

### 项目实现要点（LangGraph 集成提示）
- **图定义**：用 `StateGraph(AgentState)` 添加节点，用 `add_conditional_edges` 根据 `current_step` 路由。
- **工具节点**：可以单独封装为函数节点，或通过 `ToolExecutor` 与 agent 推理循环集成。
- **检查点**：使用 `MemorySaver` 或 `SqliteSaver` 在每次节点执行后保存状态，完美展示记忆持续性。
- **文档与绘图**：在 Reporter 节点中调用 `python-docx` 和 Draw.io XML 生成，并将文件路径存入 `output_artifacts`。

这个设计将原本简单的“检索简报”升级为了一个能陪伴研究者走完“idea→实验→确定方案”全流程的 Agent，贴合你的课程结业要求，且技术栈清晰，创新点突出。