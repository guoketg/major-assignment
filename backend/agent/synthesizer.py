"""
Synthesizer（综合输出节点）

收集所有 Agent 的输出，综合为最终回复。
可使用 docx 工具将结果保存为 Word 文档，同时展示在对话中。
"""
import os
import json
import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from backend.agent.state import AgentState
from backend.agent.llm import get_llm
from backend.tools.docx_tool import create_docx, add_section, add_table

logger = logging.getLogger(__name__)

SYNTHESIZER_COMBINE_PROMPT = """你是一个科研综述写作专家。你的任务是将多个专家的分析结果综合成一份连贯、完整的综述报告。

要求：
1. 综合所有专家的分析，去除重复内容
2. 组织成结构清晰的综述：包含引言、核心方法分类、各方法优缺点分析、未来方向
3. 使用中文
4. 使用 Markdown 格式（标题用 ## / ###）
5. 总字数不少于 800 字，保留具体的论文标题和技术细节
6. 保留核心论文标题和技术细节

下面是各专家的分析结果："""


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizer 节点 — 调用 LLM 综合多个 Agent 的输出 + 可选文档生成"""
    messages = state.get("messages", [])
    model = state.get("model", "deepseek-chat")

    # === 收集当前轮次的子任务输出（最后一条 user 消息之后的 assistant 消息） ===
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    subtask_outputs = []
    if last_user_idx >= 0:
        for m in messages[last_user_idx + 1:]:
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if content and len(content) > 50:
                    subtask_outputs.append(content)

    # === 调用 LLM 综合所有子任务输出 ===
    output_text = state.get("output_text", "")
    if len(subtask_outputs) >= 2:
        logger.info(f"[synthesizer] 正在综合 {len(subtask_outputs)} 份子任务输出...")
        try:
            sections = []
            for i, out in enumerate(subtask_outputs, 1):
                sections.append(f"### 专家 {i} 的分析\n\n{out[:4000]}")
            sections.append("---\n请将以上内容综合成一篇完整的综述报告。")

            llm = get_llm(model)
            response = llm.invoke([
                {"role": "system", "content": SYNTHESIZER_COMBINE_PROMPT},
                {"role": "user", "content": "\n\n".join(sections)},
            ])
            combined = response.content.strip()
            if combined and len(combined) > 100:
                output_text = combined
                logger.info(f"[synthesizer] LLM 综合完成: {len(output_text)} 字")
        except Exception as e:
            logger.warning(f"[synthesizer] LLM 综合失败，使用原始输出: {e}")

    if not output_text:
        output_text = "处理完成，但没有生成输出内容。"
        return {
            "current_agent": "end",
            "output_text": output_text,
        }

    # === 检查用户是否要求生成文档（关键词触发） ===
    docx_path = None
    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    user_wants_docx = any(
        kw in "".join(user_msgs[-3:]).lower()
        for kw in ["文档", "word", "docx", "导出", "保存", "生成文件", "下载"]
    )

    if user_wants_docx and len(output_text) > 300:
        try:
            import re, datetime
            title_match = re.search(r'^#\s+(.+?)$', output_text, re.MULTILINE)
            title = title_match.group(1).strip()[:50] if title_match else f"综述_{datetime.datetime.now().strftime('%Y%m%d')}"

            docx_result = create_docx.invoke({"title": title})
            if "文档已创建" in docx_result:
                filepath = docx_result.replace("文档已创建: ", "").strip()
                docx_path = filepath

                sections = re.split(r'(?=^#{1,3}\s+)', output_text, flags=re.MULTILINE)
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                    lines = section.split('\n', 1)
                    heading = lines[0].strip('# ').strip() if len(lines) > 0 else ""
                    content = lines[1].strip() if len(lines) > 1 else ""
                    if heading and content:
                        if '|' in content and '---' in content:
                            table_lines = [l for l in content.split('\n') if l.startswith('|') and '|' in l]
                            if len(table_lines) >= 3:
                                headers = [h.strip() for h in table_lines[0].split('|')[1:-1]]
                                data_rows = []
                                for row_line in table_lines[2:]:
                                    cells = [c.strip() for c in row_line.split('|')[1:-1]]
                                    if cells:
                                        data_rows.append(cells)
                                if headers and data_rows:
                                    add_table.invoke({"filepath": filepath, "headers": headers, "rows": data_rows})
                                    content = '\n'.join(l for l in content.split('\n') if not l.startswith('|') or l.startswith('|---'))
                        add_section.invoke({"filepath": filepath, "heading": heading, "content": content[:3000]})
        except Exception as e:
            logger.warning(f"[synthesizer] 生成 Word 文档失败: {e}")

    if docx_path:
        filename = os.path.basename(docx_path)
        output_text += f"\n\n📄 **综述论文已导出为 Word 文档**：[点击下载](http://localhost:8001/download/reports/{filename})"

    # === 重建消息历史：只保留所有 user 消息 + 一条最终 synthesis ===
    timestamp = __import__("datetime").datetime.now().isoformat()
    clean_messages = [m for m in messages if m.get("role") == "user"]
    clean_messages.append({"role": "assistant", "content": output_text, "timestamp": timestamp})

    logger.info(f"[synthesizer] 消息清理: {len(messages)} → {len(clean_messages)} (综合了 {len(subtask_outputs)} 份子任务输出)")

    return {
        "current_agent": "end",
        "output_text": output_text,
        "messages": clean_messages,
    }


def synthesizer_router(state: AgentState) -> str:
    """Synthesizer 路由 — 始终结束"""
    return "end"
