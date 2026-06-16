"""
LLM 配置 — 所有 Agent 节点共享的 DeepSeek LLM 实例

使用 LangChain 的 ChatOpenAI 封装（兼容 DeepSeek API）。
"""
import os
from functools import lru_cache
from langchain_openai import ChatOpenAI


@lru_cache(maxsize=4)
def get_llm(model: str = "deepseek-chat") -> ChatOpenAI:
    """获取或创建 LLM 实例（缓存以复用连接）

    Args:
        model: 模型名，如 "deepseek-chat" / "deepseek-reasoner"

    Returns:
        ChatOpenAI 实例，已配置 DeepSeek API 端点
    """
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.7,
        streaming=True,
        timeout=180,
        max_retries=2,
    )


def get_llm_with_tools(model: str = "deepseek-chat") -> ChatOpenAI:
    """获取绑定工具的 LLM 实例（Research Agent 使用）

    与普通 LLM 的区别在于绑定了 arXiv 搜索工具，
    使 LLM 可以自主决定何时搜索论文。
    """
    from backend.tools.arxiv_tool import search_arxiv

    llm = get_llm(model)
    return llm.bind_tools([search_arxiv])
