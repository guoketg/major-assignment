"""
安全护栏 (Safety Harness / Guardrails) 模块

提供多层次的 Agent 系统安全防护：
1. 输入护栏 (InputGuardrail)：用户输入内容审核、提示词注入检测、敏感信息扫描
2. 输出护栏 (OutputGuardrail)：Agent 输出过滤、有害内容拦截、敏感信息泄漏检测
3. 工具护栏 (ToolGuardrail)：工具调用验证、路径遍历防护、参数安全检查
4. 审计日志 (AuditLogger)：统一安全事件记录

设计原则：
- 防御深度：多层检查，任何一层触发即拦截
- 默认安全：未通过护栏的内容默认拒绝
- 可观测性：所有安全事件记录审计日志
- 可配置：支持按检查类型开关
"""

import os
import re
import logging
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto as enum_auto

# ─── 审计日志 ───────────────────────────────────────────────

logger = logging.getLogger("guardrails")
AUDIT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_audit_log_initialized = False


def _ensure_audit_log():
    """确保审计日志目录和文件存在"""
    global _audit_log_initialized
    if _audit_log_initialized:
        return
    os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
    _audit_log_initialized = True


def _write_audit_log(level: str, category: str, detail: str, session_id: str = "",
                     blocked: bool = False, extra: dict = None):
    """写入安全审计日志"""
    _ensure_audit_log()
    timestamp = datetime.now().isoformat()
    log_file = os.path.join(AUDIT_LOG_DIR, "guardrails_audit.log")

    entry = {
        "timestamp": timestamp,
        "level": level,
        "category": category,
        "blocked": blocked,
        "session_id": session_id,
        "detail": detail[:500],  # 截断避免日志膨胀
    }
    if extra:
        entry["extra"] = {k: str(v)[:200] for k, v in extra.items()}

    import json
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except IOError:
        pass

    if blocked:
        logger.warning(f"[GUARDRAIL_BLOCK] {category}: {detail[:120]}")
    else:
        logger.debug(f"[GUARDRAIL] {category}: {detail[:120]}")


# ─── 防护结果 ───────────────────────────────────────────────

class BlockReason(Enum):
    """拦截原因枚举"""
    NONE = enum_auto()
    HARMFUL_CONTENT = enum_auto()       # 有害/违规内容
    PROMPT_INJECTION = enum_auto()      # 提示词注入攻击
    SENSITIVE_INFO = enum_auto()         # 敏感信息
    JAILBREAK = enum_auto()             # 越狱尝试
    TOOL_ABUSE = enum_auto()            # 工具滥用
    PATH_TRAVERSAL = enum_auto()        # 路径遍历
    RATE_LIMIT = enum_auto()            # 频率限制
    TOKEN_BUDGET = enum_auto()          # Token 预算超限
    ILLEGAL_CONTENT = enum_auto()       # 违法内容


@dataclass
class GuardrailResult:
    """护栏检查结果"""
    passed: bool = True
    reason: BlockReason = BlockReason.NONE
    message: str = ""
    sanitized_content: Optional[str] = None  # 脱敏后的内容（如适用）
    details: Dict[str, Any] = field(default_factory=dict)


# ─── 输入护栏 ───────────────────────────────────────────────

# 提示词注入模式列表
PROMPT_INJECTION_PATTERNS = [
    # 指令覆盖类（英文）
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above|before)\s+(instructions?|prompts?|messages?)",
    r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above|before)\s+(instructions?|prompts?)",
    r"(?i)\bforget\s+(all\s+)?(previous|prior|above|before)\s+(instructions?|prompts?)",
    r"(?i)\boverride\s+(all\s+)?(instructions?|prompts?|system)",
    # 指令覆盖类（中文）
    r"(忽略|无视|忘记|遗忘|清除).{0,20}?(指令|提示|要求|规则|约束|限制|设定)",
    r"(覆盖|改写|修改|改变).{0,10}?(指令|系统|提示|设定)",
    # 角色切换类
    r"(?i)\b(pretend|act|roleplay)\s+(you\s+are|as\s+(a|an))\s",
    r"(?i)\byou\s+are\s+now\s+(a|an)\s",
    r"(?i)\bswitch\s+(your\s+)?(role|persona|identity)\s",
    # 角色切换类（中文）
    r"(假装|扮演|作为)\s*(你是|你是|成为)\s*(一个|一只)",
    r"从现在开始你(是|变成|成为)",
    # 规则解除类
    r"(?i)\b(remove|delete|change|modify)\s+(your\s+)?(constraints?|rules?|limitations?|restrictions?)",
    r"(?i)\bwithout\s+(any\s+)?(restrictions?|limitations?|filters?|guardrails?)",
    r"(?i)\bbypass\s+(the\s+)?(filter|moderation|safety|guardrail)",
    # 规则解除类（中文）
    r"(删除|移除|解除|取消|绕过)\s*(你的|所有|任何)\s*(限制|规则|约束|过滤|安全|护栏)",
    r"(不\s*受\s*(任何|所有|一切)\s*(限制|约束))",
    # DAN / 越狱类
    r"(?i)\bDAN\s+(mode|jailbreak)",
    r"(?i)\bdeveloper\s+mode",
    r"(?i)\bunrestricted\s+(mode|access|output)",
    r"(?i)\bno\s+(filters?|restrictions?|limitations?|boundaries?)",
    # 系统提示词窃取
    r"(?i)\b(print|show|reveal|output|display|tell\s+me)\s+(your\s+)?(system\s+prompt|instructions?|configuration|setup)",
    r"(?i)\bwhat\s+(is|are)\s+(your\s+)?(system\s+prompt|initial\s+instructions?)",
    r"(?i)\brepeat\s+the\s+(above|previous|first)\s+(message|text|prompt)",
    # 系统提示词窃取（中文）
    r"(打印|显示|说出|告诉|揭示|输出)\s*(你的|你的)\s*(系统\s*提示|指令|提示词|设定)",
    # token / 拼接注入
    r"(?i)\bconcat(enate)?\s+strings",
    r"(?i)\btranslate\s+this\s+(base64|hex|binary)",
    r"(?i)\bdecode\s+(the\s+)?following",
]

# 有害 / 违规内容关键词（中文 + 英文）
HARMFUL_CONTENT_PATTERNS = [
    # 暴力 / 伤害
    r"(?i)\b(kill|murder|assassinate|torture|massacre|genocide|bomb|shoot)\b",
    r"(杀人|谋杀|刺杀|酷刑|屠杀|种族灭绝|炸弹|开枪|自制\s*枪|制造\s*枪)",
    # 色情 / 儿童不当内容
    r"(?i)\b(hentai|porn|xxx|hardcore|nude|child\s*porn|cp\s*content|lolicon)\b",
    r"(色情|淫秽|裸体|儿童色情)",
    # 毒品
    r"(?i)\b(heroin|cocaine|meth|amphetamine|fentanyl|lsd|narcotics?)\b",
    r"(海洛因|可卡因|冰毒|芬太尼|毒品)",
    # 武器制造
    r"(?i)\b(how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|gun|explosive))",
    r"((制造|制作|自制)\s*(炸弹|武器|枪支|爆炸物))",
    # 黑客攻击
    r"(?i)\b(hack\s+(into|someone)|ddos\s+attack|ransomware|malware\s+creation)\b",
    r"(黑客攻击|DDOS攻击|勒索软件制作|恶意软件制作)",
    # 诈骗 / 钓鱼
    r"(?i)\b(phishing\s+(page|site)|fake\s+id\s+generator|credit\s+card\s+generator)\b",
    r"(钓鱼网站|假身份证|信用卡生成器)",
    # 政治极端内容（仅拦截明确违法）
    r"(?i)\b(terroris[mt]|extremist|insurrection|overthrow\s+government)\b",
]

# 敏感信息模式（密钥/令牌泄漏防护）
SENSITIVE_INFO_PATTERNS = [
    r"(?i)(sk-[a-zA-Z0-9]{20,})",                          # OpenAI API Key
    r"(?i)(sk-ant-[a-zA-Z0-9_\-]{20,})",                   # Anthropic API Key
    r"(?i)(AIza[0-9A-Za-z\-_]{20,})",                      # Google API Key (min 20 chars after prefix)
    r"(?i)(ghp_[a-zA-Z0-9]{36})",                          # GitHub Personal Access Token
    r"(?i)(gho_[a-zA-Z0-9]{36})",                          # GitHub OAuth
    r"(?i)(ghu_[a-zA-Z0-9]{36})",                          # GitHub User-to-Server
    r"(?i)(ghs_[a-zA-Z0-9]{36})",                          # GitHub Server-to-Server
    r"(?i)(github_pat_[a-zA-Z0-9_]{22,})",                 # GitHub Fine-grained PAT
    r"(?i)(eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,})",  # JWT
    r"(?i)(-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----)",     # 私钥头
    r"(?i)(-----BEGIN\s+CERTIFICATE-----)",                 # 证书
    r"(?i)(AKIA[0-9A-Z]{16})",                             # AWS Access Key
    r"(?i)(amzn\.mws\.[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",  # AWS MWS
    r"(?i)(https?://[^@\s]+@)",                            # URL 中包含密码
]

# 敏感信息脱敏替换
SENSITIVE_INFO_REPLACEMENTS = [
    (re.compile(r"(?i)sk-[a-zA-Z0-9]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)AIza[0-9A-Za-z\-_]{35}"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"(?i)(ghp_|gho_|ghu_|ghs_)[a-zA-Z0-9]{36}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"(?i)AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"(?i)eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}"), "[REDACTED_JWT]"),
    (re.compile(r"(?i)-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----[^-]*-----END\s+(RSA\s+)?PRIVATE\s+KEY-----", re.DOTALL),
     "[REDACTED_PRIVATE_KEY]"),
]

# 越狱关键词模式（多词联合匹配）
JAILBREAK_KEYWORD_GROUPS = [
    ["jailbreak", "prompt"],
    ["ignore", "instructions", "output"],
    ["pretend", "unrestricted", "any"],
    ["developer", "mode", "bypass"],
    ["disregard", "rules", "respond"],
]


# ─── 工具护栏 ───────────────────────────────────────────────

# 工具调用白名单
TOOL_WHITELIST = {
    "search_arxiv",           # arXiv 论文搜索
    "web_search",             # 联网搜索
    "create_docx",            # 创建 Word 文档
    "add_section",            # 添加章节
    "add_table",              # 添加表格
    # 可扩展：添加新工具到此列表
}

# 工具调用参数长度/范围限制
TOOL_PARAM_LIMITS = {
    "search_arxiv": {"query": 500, "max_results": 50},
    "web_search": {"query": 500},
    "create_docx": {"title": 200},
    "add_section": {"heading": 200, "content": 50000},
    "add_table": {"headers": 50, "rows": 500},
}

# 路径遍历防护：工具调用中涉及 filepath 参数时的特殊检查
PATH_TRAVERSAL_TOOLS = {"create_docx", "add_section", "add_table"}
ALLOWED_FILEPATH_PREFIX = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "generated_reports"
)


# ─── 输入护栏实现 ───────────────────────────────────────────

class InputGuardrail:
    """用户输入安全检查
    在用户消息进入 Agent 之前执行，阻止恶意/有害输入。
    """

    def __init__(self,
                 check_injection: bool = True,
                 check_harmful: bool = True,
                 check_sensitive: bool = True,
                 check_jailbreak: bool = True,
                 max_input_length: int = 10000,
                 sanitize_sensitive: bool = True):
        self.check_injection = check_injection
        self.check_harmful = check_harmful
        self.check_sensitive = check_sensitive
        self.check_jailbreak = check_jailbreak
        self.max_input_length = max_input_length
        self.sanitize_sensitive = sanitize_sensitive

    def check(self, content: str, session_id: str = "") -> GuardrailResult:
        """执行所有输入检查，返回检查结果"""
        if not content or not content.strip():
            return GuardrailResult(passed=False, reason=BlockReason.HARMFUL_CONTENT,
                                    message="输入内容为空")

        # 长度检查
        if len(content) > self.max_input_length:
            return GuardrailResult(
                passed=False, reason=BlockReason.TOKEN_BUDGET,
                message=f"输入过长（{len(content)}字符，上限{self.max_input_length}）"
            )

        # 1. 提示词注入检测
        if self.check_injection:
            result = self._detect_injection(content)
            if not result.passed:
                _write_audit_log("WARN", "input_injection", result.message,
                                 session_id, blocked=True)
                return result

        # 2. 越狱检测（多词联合匹配）
        if self.check_jailbreak:
            result = self._detect_jailbreak(content)
            if not result.passed:
                _write_audit_log("WARN", "input_jailbreak", result.message,
                                 session_id, blocked=True)
                return result

        # 3. 有害内容检测
        if self.check_harmful:
            result = self._detect_harmful(content)
            if not result.passed:
                _write_audit_log("WARN", "input_harmful", result.message,
                                 session_id, blocked=True)
                return result

        # 4. 敏感信息扫描 + 脱敏
        if self.check_sensitive:
            result = self._detect_sensitive(content)
            if not result.passed:
                _write_audit_log("WARN", "input_sensitive", result.message,
                                 session_id, blocked=True)
                return result
            if self.sanitize_sensitive and result.sanitized_content:
                # 敏感信息已脱敏，更新内容但标记为通过
                _write_audit_log("INFO", "input_sanitized",
                                 f"用户输入中的敏感信息已脱敏", session_id)

        _write_audit_log("INFO", "input_pass", "输入护栏检查通过", session_id)
        return GuardrailResult(passed=True, sanitized_content=result.sanitized_content)

    def _detect_injection(self, content: str) -> GuardrailResult:
        """检测提示词注入攻击"""
        for i, pattern in enumerate(PROMPT_INJECTION_PATTERNS):
            if re.search(pattern, content):
                return GuardrailResult(
                    passed=False,
                    reason=BlockReason.PROMPT_INJECTION,
                    message=f"检测到疑似提示词注入攻击（模式 #{i}）",
                    details={"pattern": pattern, "content_snippet": content[:100]}
                )
        return GuardrailResult(passed=True)

    def _detect_harmful(self, content: str) -> GuardrailResult:
        """检测有害/违规内容"""
        for i, pattern in enumerate(HARMFUL_CONTENT_PATTERNS):
            match = re.search(pattern, content)
            if match:
                return GuardrailResult(
                    passed=False,
                    reason=BlockReason.HARMFUL_CONTENT,
                    message=f"检测到疑似违规内容（关键词: {match.group()[:30]}）",
                    details={"pattern": pattern, "matched": match.group()[:30]}
                )
        return GuardrailResult(passed=True)

    def _detect_sensitive(self, content: str) -> GuardrailResult:
        """检测敏感信息（密钥泄漏），并执行脱敏"""
        sanitized = content
        detected_types = []

        # 先检测是否有敏感信息
        for pattern in SENSITIVE_INFO_PATTERNS:
            match = re.search(pattern, content)
            if match:
                detected_types.append(match.group(0)[:50])

        # 如果用户输入包含密钥/令牌，拒绝并警告
        if detected_types:
            return GuardrailResult(
                passed=False,
                reason=BlockReason.SENSITIVE_INFO,
                message=f"检测到用户输入中包含疑似密钥/令牌信息（类型: {', '.join(d[:20] for d in detected_types)}）",
                details={"detected": detected_types}
            )

        # 脱敏处理
        for pattern, replacement in SENSITIVE_INFO_REPLACEMENTS:
            old = sanitized
            sanitized = pattern.sub(replacement, sanitized)
            if sanitized != old:
                detected_types.append(replacement)

        if detected_types:
            return GuardrailResult(
                passed=True,
                sanitized_content=sanitized,
                details={"redacted": len(detected_types)}
            )

        return GuardrailResult(passed=True)

    def _detect_jailbreak(self, content: str) -> GuardrailResult:
        """检测越狱尝试（多关键词联合匹配）"""
        content_lower = content.lower()
        for group in JAILBREAK_KEYWORD_GROUPS:
            matches = sum(1 for kw in group if kw in content_lower)
            if matches >= 2:  # 匹配 2+ 个关键词即触发
                return GuardrailResult(
                    passed=False,
                    reason=BlockReason.JAILBREAK,
                    message=f"检测到疑似越狱尝试（联合关键词: {group}）",
                    details={"keywords": group, "matched_count": matches}
                )
        return GuardrailResult(passed=True)


# ─── 输出护栏实现 ───────────────────────────────────────────

class OutputGuardrail:
    """Agent 输出安全检查
    在 Agent 输出返回给用户之前执行，阻止有害输出和敏感信息泄漏。
    """

    def __init__(self,
                 check_harmful: bool = True,
                 check_sensitive: bool = True,
                 min_output_length: int = 0,
                 max_output_length: int = 100000,
                 sanitize_sensitive: bool = True):
        self.check_harmful = check_harmful
        self.check_sensitive = check_sensitive
        self.min_output_length = min_output_length
        self.max_output_length = max_output_length
        self.sanitize_sensitive = sanitize_sensitive

    def check(self, content: str, session_id: str = "") -> GuardrailResult:
        """执行输出安全检查，返回检查后的结果"""
        if not content:
            return GuardrailResult(passed=True)

        # 1. 有害内容检测
        if self.check_harmful:
            result = self._detect_harmful_output(content)
            if not result.passed:
                _write_audit_log("ERROR", "output_harmful", result.message,
                                 session_id, blocked=True)
                return result

        # 2. 敏感信息检测 + 脱敏
        sanitized = content
        if self.check_sensitive:
            result = self._detect_sensitive_output(content)
            if not result.passed:
                _write_audit_log("ERROR", "output_sensitive", result.message,
                                 session_id, blocked=True)
                return result
            if self.sanitize_sensitive and result.sanitized_content:
                sanitized = result.sanitized_content
                _write_audit_log("INFO", "output_sanitized",
                                 f"Agent 输出中的敏感信息已脱敏", session_id)

        _write_audit_log("INFO", "output_pass", "输出护栏检查通过", session_id)
        return GuardrailResult(passed=True, sanitized_content=sanitized)

    def _detect_harmful_output(self, content: str) -> GuardrailResult:
        """检测输出中的有害内容"""
        for i, pattern in enumerate(HARMFUL_CONTENT_PATTERNS):
            match = re.search(pattern, content)
            if match:
                return GuardrailResult(
                    passed=False,
                    reason=BlockReason.HARMFUL_CONTENT,
                    message=f"Agent 输出检测到违规内容",
                    details={"matched": match.group()[:30]}
                )
        return GuardrailResult(passed=True)

    def _detect_sensitive_output(self, content: str) -> GuardrailResult:
        """检测输出中的敏感信息（如 API Key 泄漏）"""
        sanitized = content
        detected = False

        for pattern, replacement in SENSITIVE_INFO_REPLACEMENTS:
            old = sanitized
            sanitized = pattern.sub(replacement, sanitized)
            if sanitized != old:
                detected = True

        if detected:
            return GuardrailResult(
                passed=True,  # 输出不阻止，而是脱敏
                sanitized_content=sanitized,
                message="Agent 输出中的敏感信息已自动脱敏",
                details={"redacted": True}
            )

        return GuardrailResult(passed=True)


# ─── 工具护栏实现 ───────────────────────────────────────────

class ToolGuardrail:
    """工具调用安全检查
    在 Agent 执行工具调用前验证参数合法性，防止工具滥用。
    """

    def __init__(self,
                 enforce_whitelist: bool = True,
                 enforce_param_limits: bool = True,
                 enforce_path_safety: bool = True):
        self.enforce_whitelist = enforce_whitelist
        self.enforce_param_limits = enforce_param_limits
        self.enforce_path_safety = enforce_path_safety

    def check_tool_call(self, tool_name: str, tool_input: dict,
                        session_id: str = "") -> GuardrailResult:
        """验证工具调用是否安全
        Args:
            tool_name: 工具名称
            tool_input: 工具参数字典
            session_id: 会话 ID
        Returns:
            GuardrailResult
        """
        # 1. 白名单检查
        if self.enforce_whitelist:
            if tool_name not in TOOL_WHITELIST:
                msg = f"工具 '{tool_name}' 不在白名单中，调用已拒绝"
                _write_audit_log("ERROR", "tool_whitelist", msg,
                                 session_id, blocked=True,
                                 extra={"tool": tool_name})
                return GuardrailResult(
                    passed=False, reason=BlockReason.TOOL_ABUSE,
                    message=msg
                )

        # 2. 参数长度/范围限制
        if self.enforce_param_limits:
            limits = TOOL_PARAM_LIMITS.get(tool_name, {})
            for param, max_val in limits.items():
                if param in tool_input:
                    val = tool_input[param]
                    if isinstance(val, str) and len(val) > max_val:
                        msg = (f"工具 '{tool_name}' 参数 '{param}' 过长"
                               f" ({len(val)} > {max_val})，调用已拒绝")
                        _write_audit_log("ERROR", "tool_param_limit", msg,
                                         session_id, blocked=True)
                        return GuardrailResult(
                            passed=False, reason=BlockReason.TOOL_ABUSE,
                            message=msg
                        )
                    if isinstance(val, (int, float)) and val > max_val:
                        msg = f"工具 '{tool_name}' 参数 '{param}' 超出范围（上限 {max_val}）"
                        _write_audit_log("ERROR", "tool_param_range", msg,
                                         session_id, blocked=True)
                        return GuardrailResult(
                            passed=False, reason=BlockReason.TOOL_ABUSE,
                            message=msg
                        )

        # 3. 路径遍历防护（针对涉及文件的工具）
        if self.enforce_path_safety and tool_name in PATH_TRAVERSAL_TOOLS:
            result = self._check_path_safety(tool_name, tool_input, session_id)
            if not result.passed:
                return result

        return GuardrailResult(passed=True)

    def _check_path_safety(self, tool_name: str, tool_input: dict,
                           session_id: str = "") -> GuardrailResult:
        """检查工具调用中的路径参数安全"""
        path_params = []

        if tool_name == "create_docx" and "title" in tool_input:
            path_params.append(("title", tool_input["title"]))
        elif tool_name in ("add_section", "add_table") and "filepath" in tool_input:
            path_params.append(("filepath", tool_input["filepath"]))

        for param_name, param_value in path_params:
            if "\x00" in param_value or ".." in param_value:
                msg = f"工具 '{tool_name}' 参数 '{param_name}' 含路径遍历字符"
                _write_audit_log("ERROR", "tool_path_traversal", msg,
                                 session_id, blocked=True,
                                 extra={"tool": tool_name, "param": param_name})
                return GuardrailResult(
                    passed=False, reason=BlockReason.PATH_TRAVERSAL,
                    message=msg
                )
            if len(param_value) > 200:
                msg = f"工具 '{tool_name}' 参数 '{param_name}' 文件名过长"
                _write_audit_log("WARN", "tool_path_length", msg,
                                 session_id, blocked=True)
                return GuardrailResult(
                    passed=False, reason=BlockReason.PATH_TRAVERSAL,
                    message=msg
                )

        return GuardrailResult(passed=True)


# ─── 统一护栏管理器 ─────────────────────────────────────────

class GuardrailManager:
    """统一护栏管理器
    整合输入/输出/工具三层护栏，对外提供统一的 check 方法。
    用法:
        gm = GuardrailManager()
        
        # 输入检查
        result = gm.check_input(user_message, session_id)
        if not result.passed:
            return error_response(result.message)
        
        # 输出检查
        result = gm.check_output(agent_output, session_id)
        
        # 工具调用检查
        result = gm.check_tool("search_arxiv", {"query": "..."}, session_id)
    """

    def __init__(self,
                 # 输入护栏配置
                 enable_input: bool = True,
                 # 输出护栏配置
                 enable_output: bool = True,
                 # 工具护栏配置
                 enable_tool: bool = True):
        self.enable_input = enable_input
        self.enable_output = enable_output
        self.enable_tool = enable_tool

        self._input_guard = InputGuardrail()
        self._output_guard = OutputGuardrail()
        self._tool_guard = ToolGuardrail()

    def check_input(self, content: str, session_id: str = "") -> GuardrailResult:
        """检查用户输入"""
        if not self.enable_input:
            return GuardrailResult(passed=True)

        result = self._input_guard.check(content, session_id)
        if not result.passed:
            logger.warning(
                f"[INPUT_BLOCKED] session={session_id[:8] if session_id else '?'}, "
                f"reason={result.reason.name}, msg={result.message[:80]}"
            )
        return result

    def check_output(self, content: str, session_id: str = "") -> GuardrailResult:
        """检查 Agent 输出"""
        if not self.enable_output:
            return GuardrailResult(passed=True)

        result = self._output_guard.check(content, session_id)
        if not result.passed:
            logger.warning(
                f"[OUTPUT_BLOCKED] session={session_id[:8] if session_id else '?'}, "
                f"reason={result.reason.name}, msg={result.message[:80]}"
            )
        return result

    def check_tool(self, tool_name: str, tool_input: dict,
                   session_id: str = "") -> GuardrailResult:
        """检查工具调用"""
        if not self.enable_tool:
            return GuardrailResult(passed=True)

        result = self._tool_guard.check_tool_call(tool_name, tool_input, session_id)
        if not result.passed:
            logger.warning(
                f"[TOOL_BLOCKED] session={session_id[:8] if session_id else '?'}, "
                f"tool={tool_name}, reason={result.reason.name}, msg={result.message[:80]}"
            )
        return result

    # ─── 工具调用拦截装饰器 ───

    def wrap_tool(self, tool_func, tool_name: str):
        """包装工具函数，添加安全检查

        Args:
            tool_func: 原始 LangChain @tool 函数
            tool_name: 工具名称（用于白名单）

        Returns:
            包装后的函数，参数不变但增加了调用前检查
        """
        original_func = tool_func.func if hasattr(tool_func, "func") else tool_func

        def safe_wrapper(*args, **kwargs):
            # 构建参数 dict
            import inspect
            sig = inspect.signature(original_func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            tool_input = dict(bound.arguments)

            # 安全检查
            result = self.check_tool(tool_name, tool_input)
            if not result.passed:
                return f"[安全护栏拦截] {result.message}"

            return original_func(*args, **kwargs)

        # 创建新的 @tool 包装
        from langchain_core.tools import tool as langchain_tool
        wrapped = langchain_tool(safe_wrapper)
        wrapped.name = getattr(tool_func, "name", tool_name)
        wrapped.description = getattr(tool_func, "description", "")
        return wrapped


# ─── 全局单例 ───────────────────────────────────────────────

_global_guardrail: Optional[GuardrailManager] = None


def get_guardrail() -> GuardrailManager:
    """获取全局护栏管理器实例"""
    global _global_guardrail
    if _global_guardrail is None:
        # 可通过环境变量控制检查项开关
        _global_guardrail = GuardrailManager(
            enable_input=os.getenv("GUARDRAIL_INPUT", "true").lower() == "true",
            enable_output=os.getenv("GUARDRAIL_OUTPUT", "true").lower() == "true",
            enable_tool=os.getenv("GUARDRAIL_TOOL", "true").lower() == "true",
        )
        logger.info(
            f"[GUARDRAIL] 护栏管理器已初始化 "
            f"(input={_global_guardrail.enable_input}, "
            f"output={_global_guardrail.enable_output}, "
            f"tool={_global_guardrail.enable_tool})"
        )
    return _global_guardrail


def reset_guardrail():
    """重置护栏管理器（测试用）"""
    global _global_guardrail
    _global_guardrail = None
