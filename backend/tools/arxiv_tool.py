"""
arXiv 论文搜索工具（LangChain Tool 封装）

复用现有 arXiv API 搜索逻辑，包装为 LangChain 兼容的 @tool，
供 Research Agent 在 ReAct 循环中调用。
"""
import time
from typing import Optional, List

import httpx
import xmltodict
from langchain_core.tools import tool

ARXIV_API_URL = "https://export.arxiv.org/api/query"
_last_req_time: float = 0


@tool
def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
) -> str:
    """搜索 arXiv 学术论文。

    当你需要查找最新的学术研究、论文、SOTA 方法时使用此工具。
    可指定搜索关键词、返回数量和排序方式。

    Args:
        query: 搜索关键词（arXiv 语法，如 "CLIP noisy label learning"）
        max_results: 返回论文数量（1-50）
        sort_by: 排序方式，可选 "relevance"（按相关度）或 "submittedDate"（按日期）

    Returns:
        结构化的论文列表文本，每篇包含标题、作者、摘要、分类、发布日期、PDF 链接
    """
    global _last_req_time

    # 限流 3 秒（arXiv API 要求）
    now = time.time()
    since_last = now - _last_req_time
    if since_last < 3:
        time.sleep(3 - since_last)

    params = {
        "search_query": query.replace(" ", "+"),
        "start": 0,
        "max_results": min(max_results, 50),
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    try:
        resp = httpx.get(ARXIV_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        _last_req_time = time.time()

        data = xmltodict.parse(resp.text)
        feed = data.get("feed", {})
        entries = feed.get("entry", [])
        if isinstance(entries, dict):
            entries = [entries]

        if not entries:
            return "arXiv 搜索未返回结果，请尝试调整搜索词。"

        total = int(feed.get("opensearch:totalResults", 0))
        lines = [f"arXiv 搜索完成，共找到 {total} 篇相关论文，显示前 {len(entries)} 篇：\n"]

        for i, entry in enumerate(entries, 1):
            title = entry.get("title", "").strip().replace("\n", " ")
            summary = entry.get("summary", "").strip().replace("\n", " ")[:300]
            authors_entry = entry.get("author", [])
            if isinstance(authors_entry, dict):
                authors_entry = [authors_entry]
            authors = ", ".join(a.get("name", "") for a in authors_entry[:5])
            if len(authors_entry) > 5:
                authors += " et al."

            published = (entry.get("published", "") or "")[:10]
            arxiv_id = (entry.get("id", "") or "").split("/")[-1].split("v")[0]

            cats = entry.get("category", [])
            if isinstance(cats, dict):
                cats = [cats]
            categories = ", ".join(c.get("@term", "") for c in cats)

            lines.append(f"--- 论文 {i} ---")
            lines.append(f"标题: {title}")
            lines.append(f"作者: {authors}")
            lines.append(f"日期: {published}")
            lines.append(f"分类: {categories}")
            lines.append(f"ID: {arxiv_id}")
            lines.append(f"摘要: {summary}")
            lines.append("")

        return "\n".join(lines)

    except httpx.TimeoutException:
        return "arXiv 请求超时，请稍后重试。"
    except Exception as e:
        return f"arXiv 搜索失败: {str(e)}"


# 工具列表，供 ToolNode 注册
arxiv_tools = [search_arxiv]
