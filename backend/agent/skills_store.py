"""
自定义技能存储模块

管理用户自定义技能的 CRUD，持久化到 JSON 文件。
启动时加载内置技能 + 自定义技能合并使用。
"""
import json
import os
import copy
from typing import Dict, List, Optional

from backend.agent.skills import SKILL_CONFIG

SKILLS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_memory", "custom_skills.json")


def _load_custom_skills() -> Dict[str, dict]:
    """从文件加载自定义技能"""
    if not os.path.exists(SKILLS_FILE):
        return {}
    try:
        with open(SKILLS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, IOError):
        return {}


def _save_custom_skills(skills: Dict[str, dict]):
    """保存自定义技能到文件"""
    os.makedirs(os.path.dirname(SKILLS_FILE), exist_ok=True)
    with open(SKILLS_FILE, "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False, indent=2)


def get_all_skills() -> List[dict]:
    """获取所有可用技能（内置 + 自定义），返回前端格式的列表"""
    custom = _load_custom_skills()
    result = []
    # 内置技能
    for skill_id, cfg in SKILL_CONFIG.items():
        result.append({
            "id": skill_id,
            "label": cfg["label"],
            "desc": cfg.get("desc", ""),
            "builtin": True,
            "has_prompt": bool(cfg.get("system_prompt_append")),
        })
    # 自定义技能
    for skill_id, cfg in custom.items():
        result.append({
            "id": skill_id,
            "label": cfg.get("label", skill_id),
            "desc": cfg.get("desc", ""),
            "builtin": False,
            "has_prompt": bool(cfg.get("system_prompt_append")),
        })
    return result


def get_skill_detail(skill_id: str) -> Optional[dict]:
    """获取某个技能的完整信息（含 prompt）"""
    # 内置技能
    if skill_id in SKILL_CONFIG:
        cfg = SKILL_CONFIG[skill_id]
        return {
            "id": skill_id,
            "label": cfg["label"],
            "desc": cfg.get("desc", ""),
            "system_prompt_append": cfg.get("system_prompt_append", ""),
            "builtin": True,
        }
    # 自定义技能
    custom = _load_custom_skills()
    if skill_id in custom:
        cfg = custom[skill_id]
        return {
            "id": skill_id,
            "label": cfg.get("label", skill_id),
            "desc": cfg.get("desc", ""),
            "system_prompt_append": cfg.get("system_prompt_append", ""),
            "builtin": False,
        }
    return None


def create_skill(skill_id: str, label: str, desc: str, system_prompt_append: str) -> dict:
    """创建新的自定义技能。skill_id 不能与内置技能重名。"""
    if skill_id in SKILL_CONFIG:
        raise ValueError(f"不能覆盖内置技能: {skill_id}")
    custom = _load_custom_skills()
    if skill_id in custom:
        raise ValueError(f"技能已存在: {skill_id}")
    custom[skill_id] = {
        "label": label,
        "desc": desc,
        "system_prompt_append": system_prompt_append,
    }
    _save_custom_skills(custom)
    return {"id": skill_id, "label": label, "desc": desc, "builtin": False, "has_prompt": bool(system_prompt_append)}


def update_skill(skill_id: str, label: str, desc: str, system_prompt_append: str) -> dict:
    """更新自定义技能。内置技能只允许修改 prompt（原配置不变）。"""
    if skill_id in SKILL_CONFIG:
        raise ValueError(f"不能覆盖内置技能: {skill_id}")
    custom = _load_custom_skills()
    if skill_id not in custom:
        raise ValueError(f"技能不存在: {skill_id}")
    custom[skill_id] = {
        "label": label,
        "desc": desc,
        "system_prompt_append": system_prompt_append,
    }
    _save_custom_skills(custom)
    return {"id": skill_id, "label": label, "desc": desc, "builtin": False, "has_prompt": bool(system_prompt_append)}


def delete_skill(skill_id: str):
    """删除自定义技能。内置技能不能删除。"""
    if skill_id in SKILL_CONFIG:
        raise ValueError(f"不能删除内置技能: {skill_id}")
    custom = _load_custom_skills()
    if skill_id not in custom:
        raise ValueError(f"技能不存在: {skill_id}")
    del custom[skill_id]
    _save_custom_skills(custom)


def get_skill_prompt(skill_id: str) -> str:
    """获取技能的 system_prompt_append（先从内置查找，再查自定义）"""
    # 内置
    cfg = SKILL_CONFIG.get(skill_id)
    if cfg and cfg.get("system_prompt_append"):
        return cfg["system_prompt_append"]
    # 自定义
    custom = _load_custom_skills()
    cfg = custom.get(skill_id, {})
    return cfg.get("system_prompt_append", "")


def get_skill_label(skill_id: str) -> str:
    """获取技能的显示标签"""
    cfg = SKILL_CONFIG.get(skill_id)
    if cfg:
        return cfg["label"]
    custom = _load_custom_skills()
    cfg = custom.get(skill_id, {})
    return cfg.get("label", skill_id)


def get_valid_skills() -> List[str]:
    """获取所有有效的 skill ID 列表"""
    ids = list(SKILL_CONFIG.keys())
    custom = _load_custom_skills()
    ids.extend(custom.keys())
    return ids
