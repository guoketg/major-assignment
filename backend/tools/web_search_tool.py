"""
DashScope MCP 联网搜索工具

使用阿里云 DashScope MCP 服务进行实时联网搜索，
获取最新的网络信息、新闻和实时数据。

MCP 服务器: https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp
协议: Streamable HTTP
"""
import os
import asyncio
import logging
from typing import Optional

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# MCP 服务配置
MCP_URL = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
MCP_TOOL_NAME = "bailian_web_search"  # DashScope MCP 暴露的工具名


def _call_mcp_tool(query: str, max_results: int = 10) -> str:
    """通过 MCP Streamable HTTP 协议调用 DashScope 联网搜索

    使用 httpx 直接与 MCP 服务器通信（SSE 传输 + JSON-RPC），
    避免依赖 asyncio 事件循环冲突。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        return "错误: 未配置 DASHSCOPE_API_KEY，请在 .env 中添加"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    try:
        # MCP Streamable HTTP 流程：
        # 1. POST /mcp 建立 SSE 连接 → 收到 endpoint URL
        # 2. 向 endpoint URL 发送 JSON-RPC tools/call 请求

        with httpx.Client(timeout=60, headers=headers) as client:
            # 步骤1: 初始化会话 — POST 获取 session/endpoint
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "0.1.0",
                    "capabilities": {},
                    "clientInfo": {"name": "research-agent", "version": "1.0.0"},
                },
            }
            resp = client.post(MCP_URL, json=init_payload)
            resp.raise_for_status()
            logger.info(f"[MCP] 会话初始化成功: {resp.status_code}")

            # 步骤2: 列出可用工具
            list_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            }
            resp = client.post(MCP_URL, json=list_payload)
            resp.raise_for_status()
            tools_data = resp.json()
            logger.info(f"[MCP] 可用工具: {tools_data}")

            # 步骤3: 调用 bailian_web_search 工具
            call_payload = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": MCP_TOOL_NAME,
                    "arguments": {
                        "query": query,
                        "count": min(max_results, 20),
                    },
                },
            }
            resp = client.post(MCP_URL, json=call_payload)
            resp.raise_for_status()
            result = resp.json()

            logger.info(f"[MCP] 调用结果: {str(result)[:200]}")

            # 解析 MCP 返回结果
            if "result" in result:
                content = result["result"].get("content", [])
                if content:
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text", "") or str(item.get("data", ""))
                            if text:
                                text_parts.append(text)
                    if text_parts:
                        return "\n\n".join(text_parts)
                return "搜索完成，但未找到相关内容。"
            elif "error" in result:
                error_msg = result["error"].get("message", str(result["error"]))
                return f"MCP 搜索错误: {error_msg}"
            else:
                return f"搜索结果: {str(result)[:2000]}"

    except httpx.TimeoutException:
        return "DashScope MCP 请求超时，请稍后重试。"
    except httpx.HTTPStatusError as e:
        return f"DashScope MCP HTTP 错误: {e.response.status_code} - {e.response.text[:200]}"
    except Exception as e:
        logger.error(f"[MCP] 搜索失败: {e}")
        return f"联网搜索失败: {str(e)}"


@tool
def web_search(query: str, count: int = 5) -> str:
    """使用阿里云百炼（DashScope）进行实时联网搜索。

    适合查找：最新的新闻事件、实时信息、网页内容、学术动态、百科知识、天气等。
    比 arXiv 搜索更适合捕捉最新网络信息和实时资讯。
    每次搜索消耗一次 API 调用额度。

    Args:
        query: 搜索关键词（如 "CLIP 模型 2025 最新进展"）
        count: 返回结果数量（1-20，默认5）

    Returns:
        结构化的搜索结果文本，包含标题、摘要、来源链接
    """
    return _call_mcp_tool(query, count)


def register_web_search_tools() -> list:
    """注册所有 MCP Web Search 工具，供 ToolNode 使用"""
    return [web_search]
