"""
全局 Token 用量追踪模块

按天追踪 Token 用量，持久化到 JSON 文件。
数据结构: { "daily": {"YYYY-MM-DD": {...}, ...}, "total": {...} }
"""
import json
import os
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List

# 使用东八区时间
_TZ = timezone(timedelta(hours=8))

USAGE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_memory", "total_usage.json")
_lock = threading.Lock()


def _empty_day() -> Dict:
    """创建一个空白日统计"""
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost": 0.0, "session_count": 0}


def _today_str() -> str:
    """获取今天的日期字符串 (东八区)"""
    return datetime.now(_TZ).strftime("%Y-%m-%d")


def _load() -> Dict:
    """加载用量数据"""
    if not os.path.exists(USAGE_FILE):
        return {"daily": {}, "total": _empty_day()}
    try:
        with open(USAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 兼容旧格式：如果加载到旧格式数据，迁移为新格式
            if "daily" not in data:
                old = {
                    "prompt_tokens": data.get("prompt_tokens", 0),
                    "completion_tokens": data.get("completion_tokens", 0),
                    "total_tokens": data.get("total_tokens", 0),
                    "total_cost": data.get("total_cost", 0),
                    "session_count": data.get("session_count", 0),
                }
                data = {"daily": {}, "total": old}
            # 确保 total 字段完整
            if "total" not in data:
                data["total"] = _empty_day()
            for key in ["prompt_tokens", "completion_tokens", "total_tokens", "total_cost", "session_count"]:
                if key not in data["total"]:
                    data["total"][key] = 0 if key != "total_cost" else 0.0
            return data
    except (json.JSONDecodeError, IOError):
        return {"daily": {}, "total": _empty_day()}


def _save(data: Dict):
    """保存用量数据"""
    os.makedirs(os.path.dirname(USAGE_FILE), exist_ok=True)
    with open(USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_usage(prompt_tokens: int, completion_tokens: int, cost: float):
    """累加一次会话的 Token 用量到今日和全局统计"""
    today = _today_str()
    with _lock:
        data = _load()

        # 累加到今日
        if today not in data["daily"]:
            data["daily"][today] = _empty_day()
        day = data["daily"][today]
        day["prompt_tokens"] += prompt_tokens
        day["completion_tokens"] += completion_tokens
        day["total_tokens"] += prompt_tokens + completion_tokens
        day["total_cost"] = round(day["total_cost"] + cost, 6)

        # 累加到总计
        data["total"]["prompt_tokens"] = data["total"].get("prompt_tokens", 0) + prompt_tokens
        data["total"]["completion_tokens"] = data["total"].get("completion_tokens", 0) + completion_tokens
        data["total"]["total_tokens"] = data["total"].get("total_tokens", 0) + prompt_tokens + completion_tokens
        data["total"]["total_cost"] = round(data["total"].get("total_cost", 0) + cost, 6)

        _save(data)


def increment_session_count():
    """每次会话完成时 +1 (今日 + 总计)"""
    today = _today_str()
    with _lock:
        data = _load()

        if today not in data["daily"]:
            data["daily"][today] = _empty_day()
        data["daily"][today]["session_count"] += 1
        data["total"]["session_count"] = data["total"].get("session_count", 0) + 1

        _save(data)


def get_total_usage() -> Dict:
    """获取全局累计用量"""
    data = _load()
    return data["total"]


def get_daily_usage() -> List[Dict]:
    """获取每日用量列表，按日期倒序排列"""
    data = _load()
    result = []
    for date_str, day_data in sorted(data["daily"].items(), reverse=True):
        result.append({
            "date": date_str,
            **day_data,
        })
    return result


def get_usage_for_date(date_str: str) -> Dict:
    """获取指定日期的用量，若不存在返回空"""
    data = _load()
    day = data["daily"].get(date_str)
    if day:
        return {"date": date_str, **day}
    return {"date": date_str, **_empty_day()}


def reset_total_usage():
    """重置全局统计"""
    with _lock:
        _save({"daily": {}, "total": _empty_day()})
