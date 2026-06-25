"""
LLM 配置 — 所有 Agent 节点共享的 DeepSeek LLM 实例

使用 LangChain 的 ChatOpenAI 封装（兼容 DeepSeek API）。
支持 Token 使用统计追踪。
"""
import os
from functools import lru_cache
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

# DeepSeek 定价（每百万 Token，单位：元）
PRICING = {
    "deepseek-chat": {"input": 1.0, "output": 2.0, "cached": 0.1},
    "deepseek-reasoner": {"input": 4.0, "output": 16.0, "cached": 1.0},
}
DEFAULT_MODEL = "deepseek-chat"


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """根据模型和 Token 数计算成本（元）"""
    pricing = PRICING.get(model, PRICING[DEFAULT_MODEL])
    input_cost = prompt_tokens * pricing["input"] / 1_000_000
    output_cost = completion_tokens * pricing["output"] / 1_000_000
    return round(input_cost + output_cost, 6)


@lru_cache(maxsize=4)
def get_llm(model: str = DEFAULT_MODEL) -> ChatOpenAI:
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


def get_llm_with_tools(model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """获取绑定工具的 LLM 实例（Research Agent 使用）

    与普通 LLM 的区别在于绑定了 arXiv 搜索工具，
    使 LLM 可以自主决定何时搜索论文。
    """
    from backend.tools.arxiv_tool import search_arxiv

    llm = get_llm(model)
    return llm.bind_tools([search_arxiv])


def extract_token_usage(result) -> Optional[Dict[str, int]]:
    """从 LLM 调用结果中提取 Token 用量

    Args:
        result: ChatOpenAI 返回的 AIMessage 或 ChatResult

    Returns:
        {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N} 或 None
    """
    if result is None:
        return None

    # ChatResult (来自 generate / batch 调用)
    if isinstance(result, ChatResult):
        if result.llm_output and "token_usage" in result.llm_output:
            return result.llm_output["token_usage"]
        return None

    # AIMessage (来自 invoke)
    if hasattr(result, "usage_metadata") and result.usage_metadata:
        return {
            "prompt_tokens": result.usage_metadata.get("input_tokens", 0),
            "completion_tokens": result.usage_metadata.get("output_tokens", 0),
            "total_tokens": result.usage_metadata.get("total_tokens", 0),
        }

    # 回退：通过 response_metadata 获取
    if hasattr(result, "response_metadata"):
        meta = result.response_metadata or {}
        usage = meta.get("token_usage", {}) or meta.get("usage", {})
        if usage:
            return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0) or usage.get("completion_tokens", 0) + usage.get("prompt_tokens", 0),
            }

    return None


def accumulate_token_usage(
    current: Optional[Dict[str, int]],
    new: Optional[Dict[str, int]],
) -> Dict[str, int]:
    """累加两套 Token 使用统计"""
    result = dict(current or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    if new:
        result["prompt_tokens"] = result.get("prompt_tokens", 0) + new.get("prompt_tokens", 0)
        result["completion_tokens"] = result.get("completion_tokens", 0) + new.get("completion_tokens", 0)
        result["total_tokens"] = result.get("total_tokens", 0) + new.get("total_tokens", 0)
    return result
