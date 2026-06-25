"""
全局 Token 用量追踪模块

跨会话累计 Token 用量，持久化到 JSON 文件。
"""
import json
import os
import threading
from typing import Dict

USAGE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_memory", "total_usage.json")
_lock = threading.Lock()


def _load() -> Dict[str, int]:
    """加载全局累计用量"""
    if not os.path.exists(USAGE_FILE):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost": 0, "session_count": 0}
    try:
        with open(USAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost": 0, "session_count": 0}


def _save(data: Dict):
    """保存全局累计用量"""
    os.makedirs(os.path.dirname(USAGE_FILE), exist_ok=True)
    with open(USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_usage(prompt_tokens: int, completion_tokens: int, cost: float):
    """累加一次会话的 Token 用量到全局统计"""
    with _lock:
        data = _load()
        data["prompt_tokens"] = data.get("prompt_tokens", 0) + prompt_tokens
        data["completion_tokens"] = data.get("completion_tokens", 0) + completion_tokens
        data["total_tokens"] = data.get("total_tokens", 0) + prompt_tokens + completion_tokens
        data["total_cost"] = round(data.get("total_cost", 0) + cost, 6)
        _save(data)


def increment_session_count():
    """每次会话完成时 +1"""
    with _lock:
        data = _load()
        data["session_count"] = data.get("session_count", 0) + 1
        _save(data)


def get_total_usage() -> Dict:
    """获取全局累计用量"""
    return _load()


def reset_total_usage():
    """重置全局统计"""
    with _lock:
        _save({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost": 0, "session_count": 0})
