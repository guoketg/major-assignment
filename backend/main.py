"""
多轮对话API服务

基于FastAPI构建，支持多轮对话、历史记录管理、arXiv论文搜索、Agent可视化
"""
import os
import json
import time
import uuid
import asyncio
import glob
import logging
from datetime import datetime
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI
import httpx
import xmltodict
from backend.agent.graph import AgentGraph

load_dotenv()

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ====== 对话持久化存储 ======

CONVERSATIONS_DIR = "conversations"


def _ensure_conv_dir():
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)


def _conv_path(session_id: str) -> str:
    return os.path.join(CONVERSATIONS_DIR, f"{session_id}.json")


def _load_all_conversations():
    """启动时从文件加载所有历史对话"""
    _ensure_conv_dir()
    loaded = 0
    for path in glob.glob(os.path.join(CONVERSATIONS_DIR, "*.json")):
        # 跳过 meta 文件
        if path.endswith(".meta.json"):
            continue
        sid = os.path.basename(path).replace(".json", "")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 兼容新旧格式
            if isinstance(data, list):
                conversations[sid] = data
            elif isinstance(data, dict) and "history" in data:
                conversations[sid] = data
            else:
                continue
            loaded += 1
        except (json.JSONDecodeError, IOError):
            pass
    if loaded:
        print(f"  已加载 {loaded} 个历史会话")


def _save_conversation(session_id: str):
    """保存单个会话到文件（统一使用列表格式）"""
    if session_id in conversations:
        _ensure_conv_dir()
        path = _conv_path(session_id)
        try:
            conv = conversations[session_id]
            # 统一以列表格式保存（兼容 dict 格式降级）
            if isinstance(conv, list):
                data = conv
            elif isinstance(conv, dict):
                data = conv.get("history", [])
            else:
                return
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError:
            pass


def _delete_conversation_file(session_id: str):
    """删除会话文件及 meta 文件"""
    for suffix in ["", ".meta"]:
        path = os.path.join(CONVERSATIONS_DIR, f"{session_id}{suffix}.json")
        if os.path.exists(path):
            try:
                os.remove(path)
            except IOError:
                pass


def _pipeline_meta_path(session_id: str) -> str:
    """流水线 meta 文件路径"""
    return os.path.join(CONVERSATIONS_DIR, f"{session_id}.meta.json")


def _save_pipeline_meta(session_id: str, data: dict):
    """保存流水线 meta 数据到单独文件"""
    _ensure_conv_dir()
    path = _pipeline_meta_path(session_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError:
        pass


def _load_pipeline_meta(session_id: str) -> dict:
    """加载流水线 meta 数据"""
    path = _pipeline_meta_path(session_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _get_history(session_id: str) -> list:
    """安全获取会话历史列表（兼容新旧格式）"""
    conv = conversations.get(session_id, [])
    if isinstance(conv, list):
        return conv
    if isinstance(conv, dict):
        return conv.get("history", [])
    return []


# ====== FastAPI 应用 ======

app = FastAPI(title="多轮对话服务", version="1.0.0")


@app.on_event("startup")
async def startup():
    """启动时加载历史会话"""
    _load_all_conversations()


# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI客户端配置
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
client = OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
)

# 会话存储 (生产环境应使用数据库)
conversations: dict[str, list | dict] = {}

# 模型映射（前端显示名 -> API 模型名）
MODEL_MAP = {
    "deepseek-v4-flash": "deepseek-chat",
    "v4-pro": "deepseek-chat",
    "思考模式": "deepseek-reasoner",
}

# Agent Graph 实例（懒加载）
_agent_graph = None


def get_agent_graph() -> AgentGraph:
    """获取或创建全局 AgentGraph 实例"""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = AgentGraph()
    return _agent_graph


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = ""  # 可选，前端选择的模型名
    agent: str = "auto"  # 可选，"auto" | "chat" | "research" | "innovate" | "experiment"


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    history: list[dict]


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式处理用户消息，返回AI回复"""
    session_id = request.session_id
    user_message = request.message

    # 初始化会话历史（兼容新旧格式）
    if session_id not in conversations:
        conversations[session_id] = []
    history_list = _get_history(session_id)

    # 添加用户消息到历史（立即保存）
    history_list.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().isoformat()
    })
    conversations[session_id] = history_list
    _save_conversation(session_id)

    # 确定使用的模型
    selected_model = MODEL_MAP.get(request.model, "") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    async def generate():
        nonlocal selected_model
        try:
            # === 使用 LangGraph 多 Agent 引擎 ===
            agent_graph = get_agent_graph()

            async with asyncio.timeout(600):
                async for event in agent_graph.run_stream(
                    session_id=session_id,
                    messages=_get_history(session_id),
                    model=selected_model,
                    agent=request.agent,
                ):
                    # 直接转发 AgentGraph 的事件到 SSE
                    yield f"data: {json.dumps(event)}\n\n"

                    # 如果收到 done 或 error 事件，保存历史
                    if event.get("done") and event.get("history"):
                        history = event["history"]
                        # 检查 history 的完整性
                        if history and len(history) > 0:
                            last = history[-1]
                            if last.get("role") == "assistant" and not last.get("content"):
                                logger.error(f"[CRITICAL] Assistant content empty in done event! session={session_id}, history_len={len(history)}")
                            else:
                                logger.info(f"[SAVE] session={session_id}, history_len={len(history)}, last_role={last.get('role','?')}, last_content_len={len(last.get('content',''))}")
                        else:
                            logger.warning(f"[SAVE] Empty history for session={session_id}")
                        conversations[session_id] = history
                        _save_conversation(session_id)
                        # 保存流水线 meta 数据到单独文件
                        _save_pipeline_meta(session_id, {
                            "agent_pipeline": event.get("agent_pipeline", []),
                            "tool_calls": event.get("tool_calls", []),
                            "sub_task_plan": event.get("sub_task_plan", []),
                        })

        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'error': '执行超时（600秒），请简化您的请求或稍后重试', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
async def chat_non_stream(request: ChatRequest):
    """处理用户消息，返回AI回复（非流式）"""
    session_id = request.session_id
    user_message = request.message

    # 初始化会话历史
    if session_id not in conversations:
        conversations[session_id] = []
    history_list = _get_history(session_id)

    # 添加用户消息到历史（立即保存）
    history_list.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().isoformat()
    })
    conversations[session_id] = history_list
    _save_conversation(session_id)

    try:
        # 调用DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=_get_history(session_id)
        )

        assistant_reply = response.choices[0].message.content

        # 添加助手回复到历史
        history_list = _get_history(session_id)
        history_list.append({
            "role": "assistant",
            "content": assistant_reply,
            "timestamp": datetime.now().isoformat()
        })
        conversations[session_id] = history_list
        _save_conversation(session_id)

        return ChatResponse(
            session_id=session_id,
            reply=assistant_reply,
            history=_get_history(session_id)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API调用失败: {str(e)}")


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """获取会话历史（含流水线 meta 数据）"""
    if session_id not in conversations:
        return {"session_id": session_id, "history": []}

    history = _get_history(session_id)
    meta = _load_pipeline_meta(session_id)

    # 旧会话兼容：有消息但没有 meta 文件时自动生成基础流水线
    if not meta and history:
        has_assistant = any(isinstance(m, dict) and m.get("role") == "assistant" for m in history)
        if has_assistant:
            meta = {
                "agent_pipeline": [
                    {"agent": "supervisor", "label": "🤖 智能路由", "status": "complete"},
                    {"agent": "chat_agent", "label": "💬 对话助手", "status": "complete"},
                    {"agent": "synthesizer", "label": "📋 综合输出", "status": "complete"},
                ],
                "tool_calls": [],
                "sub_task_plan": [],
            }
            _save_pipeline_meta(session_id, meta)

    return {
        "session_id": session_id,
        "history": history,
        "meta": meta,
    }


@app.get("/sessions")
async def list_sessions():
    """获取所有会话列表"""
    session_list = []
    for session_id, conv in conversations.items():
        history = _get_history(session_id)
        if not history:
            continue
        first_user_msg = next(
            (msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
             for msg in history if isinstance(msg, dict) and msg.get("role") == "user"),
            "新会话"
        )
        session_list.append({
            "session_id": session_id,
            "title": first_user_msg,
            "message_count": len(history),
            "last_update": history[-1]["timestamp"] if history and isinstance(history[-1], dict) else None
        })
    session_list.sort(key=lambda x: x["last_update"] or "", reverse=True)
    return {"sessions": session_list}


@app.post("/sessions")
async def create_session():
    """创建新会话"""
    new_session_id = f"session_{uuid.uuid4().hex[:8]}"
    conversations[new_session_id] = []
    _save_conversation(new_session_id)
    return {"session_id": new_session_id}


@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    """删除会话历史"""
    if session_id in conversations:
        del conversations[session_id]
    _delete_conversation_file(session_id)
    return {"message": "会话已删除"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "time": datetime.now().isoformat()}


# ====== Word 文档导出 ======


@app.get("/export/docx/{session_id}")
async def export_session_docx(session_id: str):
    """将会话历史导出为 Word 文档"""
    from backend.tools.docx_tool import REPORTS_DIR

    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="会话不存在")

    history = _get_history(session_id)
    if not history:
        raise HTTPException(status_code=400, detail="会话没有消息")

    # 获取第一条用户消息作为文档标题
    title = next(
        (msg["content"][:20] + "..." if len(msg["content"]) > 20 else msg["content"]
         for msg in history if isinstance(msg, dict) and msg.get("role") == "user"),
        f"对话记录_{session_id}"
    )
    safe_title = "".join(c for c in title if c.isalnum() or c in " _-，。").strip()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = f"{safe_title}_{session_id}.docx"
    filepath = os.path.join(REPORTS_DIR, filename)

    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    # 设置默认字体
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Arial"
    font.size = Pt(11)

    # 标题
    doc.add_heading(f"AI 对话记录: {title}", 0)
    doc.add_paragraph(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph("")

    # 遍历消息
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        ts = msg.get("timestamp", "")

        if role == "user":
            p = doc.add_paragraph()
            run = p.add_run(f"👤 用户")
            run.bold = True
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(0x00, 0x7B, 0xFF)
            if ts:
                p.add_run(f"  ({ts[:19]})").font.size = Pt(9)
            doc.add_paragraph(content)
        elif role == "assistant":
            p = doc.add_paragraph()
            run = p.add_run(f"🤖 AI")
            run.bold = True
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(0x22, 0xC5, 0x5E)
            if ts:
                p.add_run(f"  ({ts[:19]})").font.size = Pt(9)

            if msg.get("reasoning_content"):
                doc.add_paragraph("【思考过程】")
                doc.add_paragraph(msg["reasoning_content"])

            doc.add_paragraph(content)
        else:
            doc.add_paragraph(f"[{role}]: {content}")

        doc.add_paragraph("")

    doc.save(filepath)

    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


@app.get("/download/reports/{filename}")
async def download_report(filename: str):
    """下载生成的 Word 报告文档"""
    from backend.tools.docx_tool import REPORTS_DIR
    filepath = os.path.join(REPORTS_DIR, filename)
    real_path = os.path.realpath(filepath)
    if not real_path.startswith(os.path.realpath(REPORTS_DIR)):
        raise HTTPException(status_code=403, detail="非法路径")
    if not os.path.exists(real_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        real_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


@app.get("/memory/{session_id}")
async def get_memory(session_id: str):
    """获取会话的工作记忆（论文归档、创新方案、实验日志）"""
    from backend.memory.manager import load_working_memory
    memory = load_working_memory(session_id)
    history = _get_history(session_id)
    return {
        "session_id": session_id,
        "memory": memory,
        "has_history": len(history) > 0,
        "stats": {
            "papers": len(memory.get("papers_archive", [])),
            "innovations": len(memory.get("innovation_candidates", [])),
            "experiments": len(memory.get("experiment_log", [])),
        },
    }


# ====== arXiv 论文搜索 ======

ARXIV_API_URL = "https://export.arxiv.org/api/query"
_last_arxiv_req_time = 0


@app.get("/arxiv/search")
async def arxiv_search(
    q: str = Query(..., description="搜索查询"),
    start: int = Query(0, ge=0),
    max_results: int = Query(10, ge=1, le=100),
    sortBy: str = Query("relevance", pattern="^(relevance|submittedDate|lastUpdatedDate)$"),
    sortOrder: str = Query("descending", pattern="^(ascending|descending)$"),
):
    """搜索arXiv论文"""
    global _last_arxiv_req_time

    now = time.time()
    since_last = now - _last_arxiv_req_time
    if since_last < 3:
        wait = 3 - since_last
        time.sleep(wait)

    params = {
        "search_query": q,
        "start": start,
        "max_results": max_results,
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(ARXIV_API_URL, params=params)
            resp.raise_for_status()
            _last_arxiv_req_time = time.time()

            data = xmltodict.parse(resp.text)
            feed = data.get("feed", {})
            entries = feed.get("entry", [])
            if isinstance(entries, dict):
                entries = [entries]

            papers = []
            for entry in entries:
                authors = entry.get("author", [])
                if isinstance(authors, dict):
                    authors = [authors]

                paper = {
                    "id": entry.get("id", ""),
                    "title": entry.get("title", "").strip().replace("\n", " "),
                    "summary": entry.get("summary", "").strip().replace("\n", " "),
                    "authors": [a.get("name", "") for a in authors],
                    "published": entry.get("published", ""),
                    "updated": entry.get("updated", ""),
                    "categories": [],
                    "pdf_link": "",
                    "links": [],
                }

                cats = entry.get("category", [])
                if isinstance(cats, dict):
                    cats = [cats]
                paper["categories"] = [c.get("@term", "") for c in cats]

                links = entry.get("link", [])
                if isinstance(links, dict):
                    links = [links]
                for link in links:
                    href = link.get("@href", "")
                    paper["links"].append(href)
                    if href.endswith("pdf"):
                        paper["pdf_link"] = href

                papers.append(paper)

            return {
                "total_results": int(feed.get("opensearch:totalResults", 0)),
                "start_index": int(feed.get("opensearch:startIndex", 0)),
                "items_per_page": int(feed.get("opensearch:itemsPerPage", 0)),
                "papers": papers,
            }

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="arXiv请求超时，请稍后重试")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"arXiv搜索失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
