"""
技能配置模块

定义可选技能（Skills），每个技能包含系统提示词增强文本，
当用户在前端选择技能后，系统提示词会被注入到 Agent 工作流中，
使所有 Agent 的输出按照技能要求进行优化。
"""

SKILL_CONFIG = {
    "none": {
        "label": "🔄 默认",
        "desc": "无技能增强",
    },
    "docs": {
        "label": "📄 文档",
        "desc": "结构化文档输出",
        "system_prompt_append": """## 技能模式：文档生成 (docs)

你当前处于「文档生成」模式。请遵循以下输出规范：

1. **结构要求**：
   - 使用清晰的标题层级（H1 → H2 → H3），不要跳级
   - 开头提供文档概述和执行摘要
   - 末尾提供总结与展望

2. **格式要求**：
   - 关键概念使用 **加粗** 强调
   - 对比信息使用表格展示
   - 步骤说明使用有序列表
   - 功能清单使用无序列表

3. **内容要求**：
   - 每个章节有明确的主题句
   - 引用来源使用 `[来源](URL)` 格式标注
   - 代码示例使用围栏代码块并标注语言

4. **输出风格**：
   - 专业、严谨、易读
   - 适合导出为 Word 文档或直接作为项目文档使用""",
    },
    "pdf": {
        "label": "📑 PDF",
        "desc": "PDF优化输出",
        "system_prompt_append": """## 技能模式：PDF 优化 (pdf)

你当前处于「PDF 优化」模式。请遵循以下输出规范：

1. **页面适配**：
   - 每个一级标题前插入分页提示 `[分页]`
   - 表格宽度不超过 5 列（适合 A4 页面）
   - 段落长度控制在 8 行以内，避免跨页断裂

2. **排版要求**：
   - 使用标准标题编号（1、1.1、1.1.1）
   - 列表缩进统一，不超过 3 级
   - 代码块使用等宽字体标记

3. **图表说明**：
   - 每个表格必须有表头行
   - 表格后附简要说明
   - 数据使用对齐格式

4. **输出风格**：
   - 正式、简洁、适合打印
   - 避免过长段落
   - 关键数据突出显示""",
    },
}

# 有效的 skill ID 列表
VALID_SKILLS = list(SKILL_CONFIG.keys())


def get_skill_prompt(skill: str) -> str:
    """获取指定技能的 system prompt 附加文本

    Args:
        skill: 技能 ID，如 "docs" / "pdf" / "none"

    Returns:
        技能的 system_prompt_append 文本，如果 skill 为 "none" 或不存在则返回空字符串
    """
    if skill == "none" or skill not in SKILL_CONFIG:
        return ""
    return SKILL_CONFIG[skill].get("system_prompt_append", "")


def get_skill_label(skill: str) -> str:
    """获取技能的显示标签"""
    if skill not in SKILL_CONFIG:
        return skill
    return SKILL_CONFIG[skill]["label"]
