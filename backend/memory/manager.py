"""
三层记忆管理器

实现：
1. 短期记忆 — AgentState.messages（由 LangGraph 自动管理）
2. 工作记忆 — JSON 文件持久化（跨会话保持）
3. 长期记忆 — 预留向量数据库接口

当前主要实现工作记忆的读写+按Agent检索功能。
"""
import json
import os
from typing import Dict, Any, Optional

# 工作记忆存储目录
MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_memory")


# ========== 工作记忆 ==========


def ensure_memory_dir():
    """确保记忆目录存在"""
    os.makedirs(MEMORY_DIR, exist_ok=True)


def save_working_memory(session_id: str, memory: Dict[str, Any]) -> str:
    """保存工作记忆到 JSON 文件

    Args:
        session_id: 会话 ID
        memory: 记忆字典

    Returns:
        文件路径
    """
    ensure_memory_dir()
    path = os.path.join(MEMORY_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return path


def load_working_memory(session_id: str) -> Dict[str, Any]:
    """从 JSON 文件加载工作记忆

    Args:
        session_id: 会话 ID

    Returns:
        记忆字典，不存在时返回空默认结构
    """
    path = os.path.join(MEMORY_DIR, f"{session_id}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return _default_memory()
    return _default_memory()


def delete_working_memory(session_id: str) -> bool:
    """删除会话的工作记忆文件"""
    path = os.path.join(MEMORY_DIR, f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def _default_memory() -> Dict[str, Any]:
    """返回默认的空记忆结构"""
    return {
        "papers_archive": [],
        "innovation_candidates": [],
        "experiment_log": [],
        "research_topic": "",
        "baseline_model": "",
        "current_focus": "",
    }


# ========== 记忆检索 ==========


def get_context_for_agent(agent_name: str, memory: Dict[str, Any]) -> str:
    """为指定 Agent 提取相关记忆片段，格式化为提示文本

    Args:
        agent_name: Agent 名称
        memory: 完整工作记忆

    Returns:
        格式化的上下文提示文本
    """
    parts = []

    if agent_name == "supervisor":
        topic = memory.get("research_topic", "")
        if topic:
            parts.append(f"当前研究主题: {topic}")
        focus = memory.get("current_focus", "")
        if focus:
            parts.append(f"当前关注点: {focus}")

    elif agent_name == "research_agent":
        papers = memory.get("papers_archive", [])
        if papers:
            parts.append(f"已归档 {len(papers)} 篇论文:")
            for p in papers[-5:]:  # 最近5篇
                parts.append(f"  - {p.get('title', '无标题')} [{p.get('evidence_level', 'unknown')}]")
        topic = memory.get("research_topic", "")
        if topic:
            parts.append(f"研究方向: {topic}")

    elif agent_name == "innovator_agent":
        papers = memory.get("papers_archive", [])
        if papers:
            parts.append(f"现有 {len(papers)} 篇论文可供参考")
        existing = memory.get("innovation_candidates", [])
        if existing:
            parts.append(f"已有 {len(existing)} 个创新方案在列表中")
        topic = memory.get("research_topic", "")
        if topic:
            parts.append(f"研究方向: {topic}")

    elif agent_name == "experiment_agent":
        candidates = memory.get("innovation_candidates", [])
        if candidates:
            parts.append(f"当前有 {len(candidates)} 个创新方案待实验验证")
        logs = memory.get("experiment_log", [])
        if logs:
            parts.append(f"已有 {len(logs)} 条实验记录")
            for log in logs[-3:]:
                parts.append(f"  - 步骤{log.get('step', '?')}: {log.get('analysis', '')}")

    return "\n".join(parts) if parts else ""
