"""
LLMOps 模块 — LLM 运维可观测性基础设施

提供以下能力：
1. 调用追踪 (Trace) — 结构化 LLM 调用记录，含 trace_id 串联
2. 性能监控 (Metrics) — 延迟百分位、成功率、Token 吞吐量
3. 响应缓存 (Cache) — 内存 LRU 缓存，含 TTL 过期
4. 告警机制 (Alert) — 错误率/延迟/成本阈值告警
5. 速率限制 (Rate Limiter) — 令牌桶算法的通用速率限制器

设计原则：
- 非侵入式：通过装饰器和事件回调集成，不影响现有业务逻辑
- 线程安全：所有数据结构使用 threading.Lock 保护
- 持久化：追踪和指标数据定期写入 JSON 日志文件
- 可配置：关键阈值通过环境变量或参数控制
"""

import os
import re
import json
import time
import uuid
import hashlib
import logging
import threading
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Callable
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

logger = logging.getLogger("llmops")

# ─── 存储目录 ─────────────────────────────────────────────────

LLMOPS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "llmops")
TRACES_DIR = os.path.join(LLMOPS_DIR, "traces")
METRICS_DIR = os.path.join(LLMOPS_DIR, "metrics")
ALERTS_DIR = os.path.join(LLMOPS_DIR, "alerts")


def _ensure_dirs():
    for d in [LLMOPS_DIR, TRACES_DIR, METRICS_DIR, ALERTS_DIR]:
        os.makedirs(d, exist_ok=True)


_ensure_dirs()

# ─── 枚举 ─────────────────────────────────────────────────────


class SpanStatus(Enum):
    OK = "ok"
    ERROR = "error"
    CACHE_HIT = "cache_hit"


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ─── 数据类 ───────────────────────────────────────────────────


@dataclass
class TraceSpan:
    """单次 LLM 调用的追踪记录"""
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    model: str = ""
    agent: str = ""
    session_id: str = ""
    start_time: str = ""
    end_time: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    cost: float = 0.0
    status: str = SpanStatus.OK.value
    error_message: str = ""
    prompt_hash: str = ""
    cache_hit: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "model": self.model,
            "agent": self.agent,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": round(self.tokens_per_second, 1),
            "cost": round(self.cost, 6),
            "status": self.status,
            "error_message": self.error_message,
            "prompt_hash": self.prompt_hash,
            "cache_hit": self.cache_hit,
        }


@dataclass
class AlertRecord:
    """告警记录"""
    id: str
    timestamp: str
    level: AlertLevel
    category: str  # "error_rate" | "latency" | "cost" | "rate_limit" | "budget"
    message: str
    threshold: float
    current_value: float
    session_id: str = ""
    acknowledged: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "category": self.category,
            "message": self.message,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "session_id": self.session_id,
            "acknowledged": self.acknowledged,
        }


# ─── 工具函数 ─────────────────────────────────────────────────


def _hash_prompt(model: str, messages: list) -> str:
    """对 prompts 做确定性哈希，用于缓存 key"""
    content = json.dumps({"model": model, "messages": messages}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _calc_percentile(sorted_values: List[float], p: float) -> float:
    """计算百分位数（线性插值）"""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p / 100
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_values):
        return sorted_values[f] + c * (sorted_values[f + 1] - sorted_values[f])
    return sorted_values[f]


def _iso_now() -> str:
    return datetime.now().isoformat()


# ─── 1. Trace Manager ─────────────────────────────────────────


class TraceManager:
    """LLM 调用结构化追踪

    功能：
    - 为每次 LLM 调用生成 trace_id / span_id
    - 记录完整的调用元数据（模型、延迟、Token、成本、状态）
    - 写入 JSON Lines 格式的追踪日志文件
    - 支持按 trace_id 查询完整调用链
    """

    MAX_SPANS_PER_FILE = 10000  # 每个日志文件最多记录数

    def __init__(self):
        self._lock = threading.Lock()
        self._batch: List[TraceSpan] = []
        self._batch_lock = threading.Lock()
        self._current_trace_id: Optional[str] = None

    # ── ID 生成 ──

    def new_trace_id(self) -> str:
        tid = uuid.uuid4().hex[:16]
        self._current_trace_id = tid
        return tid

    def new_span_id(self) -> str:
        return uuid.uuid4().hex[:8]

    @property
    def current_trace_id(self) -> Optional[str]:
        return self._current_trace_id

    # ── Span 创建 ──

    def start_span(
        self,
        model: str = "",
        agent: str = "",
        session_id: str = "",
        parent_span_id: str = "",
    ) -> TraceSpan:
        trace_id = self._current_trace_id or self.new_trace_id()
        span = TraceSpan(
            trace_id=trace_id,
            span_id=self.new_span_id(),
            parent_span_id=parent_span_id,
            model=model,
            agent=agent,
            session_id=session_id,
            start_time=_iso_now(),
        )
        return span

    def end_span(
        self,
        span: TraceSpan,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        status: SpanStatus = SpanStatus.OK,
        error_message: str = "",
        prompt_hash: str = "",
        cache_hit: bool = False,
    ):
        span.end_time = _iso_now()
        span.prompt_tokens = prompt_tokens
        span.completion_tokens = completion_tokens
        span.total_tokens = prompt_tokens + completion_tokens
        span.status = status.value
        span.error_message = error_message
        span.prompt_hash = prompt_hash
        span.cache_hit = cache_hit

        # 计算延迟
        try:
            start_dt = datetime.fromisoformat(span.start_time)
            end_dt = datetime.fromisoformat(span.end_time)
            span.latency_ms = (end_dt - start_dt).total_seconds() * 1000
        except Exception:
            pass

        # 计算 tokens/秒
        if span.latency_ms > 0 and span.completion_tokens > 0:
            span.tokens_per_second = span.completion_tokens / (span.latency_ms / 1000)

        self._batch_span(span)

    def _batch_span(self, span: TraceSpan):
        with self._batch_lock:
            self._batch.append(span)
            if len(self._batch) >= 10:  # 每 10 条批量写入
                self._flush()

    def _flush(self):
        if not self._batch:
            return
        to_write = list(self._batch)
        self._batch.clear()
        self._write_spans(to_write)

    def _write_spans(self, spans: List[TraceSpan]):
        _ensure_dirs()
        # 按日期分文件
        date_str = datetime.now().strftime("%Y%m%d")
        trace_file = os.path.join(TRACES_DIR, f"trace_{date_str}.jsonl")
        try:
            with self._lock:
                with open(trace_file, "a", encoding="utf-8") as f:
                    for span in spans:
                        f.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")
        except IOError:
            pass

    def flush(self):
        """手动刷新缓冲区"""
        with self._batch_lock:
            self._flush()

    # ── 查询 ──

    def get_traces_by_session(self, session_id: str, limit: int = 50) -> List[dict]:
        """按 session_id 查询追踪记录（最近 N 条）"""
        results = []
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            trace_file = os.path.join(TRACES_DIR, f"trace_{date_str}.jsonl")
            if not os.path.exists(trace_file):
                return results
            with open(trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        span = json.loads(line.strip())
                        if span.get("session_id") == session_id:
                            results.append(span)
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass
        return results[-limit:]

    def get_recent_traces(self, limit: int = 100) -> List[dict]:
        """获取最近的追踪记录"""
        results = []
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            trace_file = os.path.join(TRACES_DIR, f"trace_{date_str}.jsonl")
            if not os.path.exists(trace_file):
                return results
            with open(trace_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        except IOError:
            pass
        return results


# ─── 2. Metrics Collector ─────────────────────────────────────


@dataclass
class MetricsSnapshot:
    """指标快照"""
    total_calls: int = 0
    success_calls: int = 0
    error_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_avg_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    tokens_per_second_avg: float = 0.0
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "success_calls": self.success_calls,
            "error_calls": self.error_calls,
            "success_rate": round(self.success_calls / max(self.total_calls, 1), 4),
            "error_rate": round(self.error_calls / max(self.total_calls, 1), 4),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hits / max(self.cache_hits + self.cache_misses, 1), 4),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost": round(self.total_cost, 6),
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "latency_p99_ms": round(self.latency_p99_ms, 1),
            "latency_avg_ms": round(self.latency_avg_ms, 1),
            "latency_min_ms": round(self.latency_min_ms, 1),
            "latency_max_ms": round(self.latency_max_ms, 1),
            "tokens_per_second_avg": round(self.tokens_per_second_avg, 1),
            "last_updated": self.last_updated,
        }


class MetricsCollector:
    """性能指标收集器

    功能：
    - 记录每次 LLM 调用的延迟、成功/失败、Token 用量
    - 计算百分位数延迟（p50/p95/p99）
    - 统计成功率、缓存命中率
    - 持久化指标到 JSON 文件（每日汇总）
    """

    MAX_LATENCY_SAMPLES = 2000  # 保留最近 N 个延迟样本用于百分位计算

    def __init__(self):
        self._lock = threading.Lock()
        self._latencies: List[float] = []  # 按时间排序的延迟列表 (ms)
        self._total_calls = 0
        self._success_calls = 0
        self._error_calls = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0
        self._tps_samples: List[float] = []
        self._last_update = ""

    def record(
        self,
        latency_ms: float = 0,
        success: bool = True,
        cache_hit: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
    ):
        with self._lock:
            self._total_calls += 1
            if success:
                self._success_calls += 1
            else:
                self._error_calls += 1

            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

            if latency_ms > 0:
                self._latencies.append(latency_ms)
                if len(self._latencies) > self.MAX_LATENCY_SAMPLES:
                    self._latencies = self._latencies[-self.MAX_LATENCY_SAMPLES:]

            if prompt_tokens:
                self._total_prompt_tokens += prompt_tokens
            if completion_tokens:
                self._total_completion_tokens += completion_tokens
            if cost:
                self._total_cost += cost

            if latency_ms > 0 and completion_tokens > 0:
                tps = completion_tokens / (latency_ms / 1000)
                self._tps_samples.append(tps)
                if len(self._tps_samples) > self.MAX_LATENCY_SAMPLES:
                    self._tps_samples = self._tps_samples[-self.MAX_LATENCY_SAMPLES:]

            self._last_update = _iso_now()

    def snapshot(self) -> MetricsSnapshot:
        """获取当前指标快照"""
        with self._lock:
            latencies = sorted(list(self._latencies))
            tps_list = list(self._tps_samples)
            return MetricsSnapshot(
                total_calls=self._total_calls,
                success_calls=self._success_calls,
                error_calls=self._error_calls,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                total_prompt_tokens=self._total_prompt_tokens,
                total_completion_tokens=self._total_completion_tokens,
                total_cost=self._total_cost,
                latency_p50_ms=_calc_percentile(latencies, 50),
                latency_p95_ms=_calc_percentile(latencies, 95),
                latency_p99_ms=_calc_percentile(latencies, 99),
                latency_avg_ms=round(sum(latencies) / max(len(latencies), 1), 1),
                latency_min_ms=round(min(latencies) if latencies else 0, 1),
                latency_max_ms=round(max(latencies) if latencies else 0, 1),
                tokens_per_second_avg=round(sum(tps_list) / max(len(tps_list), 1), 1),
                last_updated=self._last_update,
            )

    def get_per_agent_snapshot(self, agent_traces: List[dict]) -> dict:
        """按 Agent 聚合的指标快照"""
        agents = {}
        for span in agent_traces:
            agent = span.get("agent", "unknown")
            if agent not in agents:
                agents[agent] = {
                    "total_calls": 0, "error_calls": 0,
                    "total_tokens": 0, "total_cost": 0.0,
                    "latencies": [],
                }
            a = agents[agent]
            a["total_calls"] += 1
            if span.get("status") == "error":
                a["error_calls"] += 1
            a["total_tokens"] += span.get("total_tokens", 0)
            a["total_cost"] += span.get("cost", 0)
            if span.get("latency_ms", 0) > 0:
                a["latencies"].append(span["latency_ms"])

        result = {}
        for agent, stats in agents.items():
            lats = sorted(stats["latencies"])
            result[agent] = {
                "total_calls": stats["total_calls"],
                "error_calls": stats["error_calls"],
                "error_rate": round(stats["error_calls"] / max(stats["total_calls"], 1), 4),
                "total_tokens": stats["total_tokens"],
                "total_cost": round(stats["total_cost"], 6),
                "latency_avg_ms": round(sum(lats) / max(len(lats), 1), 1),
                "latency_p95_ms": round(_calc_percentile(lats, 95), 1),
            }
        return result

    def persist_daily_summary(self):
        """将每日汇总写入 metrics 文件"""
        try:
            snap = self.snapshot()
            date_str = datetime.now().strftime("%Y%m%d")
            path = os.path.join(METRICS_DIR, f"metrics_{date_str}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snap.to_dict(), f, ensure_ascii=False, indent=2)
        except IOError:
            pass


# ─── 3. Response Cache ────────────────────────────────────────


class ResponseCache:
    """LLM 响应缓存（内存 LRU + TTL）

    功能：
    - 基于 (model, messages) 的 SHA256 哈希做缓存 key
    - LRU 淘汰策略，默认最大 500 条
    - TTL 过期，默认 300 秒（5 分钟）
    - 线程安全
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 300):
        self._max_size = max(max_size, 1)
        self._ttl_seconds = max(ttl_seconds, 1)
        self._cache: OrderedDict[str, Tuple[float, str]] = OrderedDict()  # key -> (expire_at, content)
        self._lock = threading.Lock()

    def get(self, model: str, messages: list) -> Optional[str]:
        """查询缓存，命中返回内容，未命中返回 None"""
        key = _hash_prompt(model, messages)
        with self._lock:
            if key not in self._cache:
                return None
            expire_at, content = self._cache[key]
            if time.time() > expire_at:
                del self._cache[key]
                return None
            # LRU: 移到末尾
            self._cache.move_to_end(key)
            return content

    def set(self, model: str, messages: list, content: str) -> bool:
        """存入缓存，返回是否成功"""
        key = _hash_prompt(model, messages)
        expire_at = time.time() + self._ttl_seconds
        with self._lock:
            # 淘汰过期项
            self._evict_expired()
            # LRU: 如果满了淘汰最旧的
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (expire_at, content)
            return True

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, (exp, _) in self._cache.items() if now > exp]
        for k in expired:
            del self._cache[k]

    def clear(self):
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._cache)

    @property
    def stats(self) -> dict:
        return {
            "current_size": self.size,
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
        }


# ─── 4. Alert Manager ─────────────────────────────────────────


class AlertManager:
    """告警管理器

    功能：
    - 基于阈值的告警检测（错误率、延迟、成本、缓存命中率）
    - 告警冷却期（避免重复告警轰炸）
    - 存储告警历史到 JSON 文件
    - 支持自定义告警回调（webhook / 日志）
    """

    # 默认阈值
    DEFAULT_ERROR_RATE = float(os.getenv("LLMOPS_ALERT_ERROR_RATE", "0.1"))        # 10%
    DEFAULT_LATENCY_P95_MS = float(os.getenv("LLMOPS_ALERT_LATENCY_P95", "10000"))  # 10s
    DEFAULT_COST_PER_SESSION = float(os.getenv("LLMOPS_ALERT_COST_SESSION", "1.0"))  # ¥1
    DEFAULT_CACHE_HIT_RATE_MIN = float(os.getenv("LLMOPS_ALERT_CACHE_HIT_MIN", "0.0"))

    COOLDOWN_SECONDS = 300  # 同类告警冷却时间（秒）

    def __init__(self, callback: Optional[Callable[[AlertRecord], None]] = None):
        self._lock = threading.Lock()
        self._last_alert_time: Dict[str, float] = {}  # category -> last_alert_time
        self._alerts: List[AlertRecord] = []
        self._max_alerts = 500
        self._callback = callback

    def check_and_alert(
        self,
        snapshot: MetricsSnapshot,
        session_id: str = "",
    ) -> List[AlertRecord]:
        """检查指标并生成告警"""
        new_alerts = []
        now = time.time()

        checks = [
            ("error_rate", snapshot.error_calls / max(snapshot.total_calls, 1),
             self.DEFAULT_ERROR_RATE, "gt",
             f"LLM 调用错误率 {snapshot.error_calls}/{snapshot.total_calls} 超过阈值 {self.DEFAULT_ERROR_RATE:.0%}"),
            ("latency", snapshot.latency_p95_ms, self.DEFAULT_LATENCY_P95_MS, "gt",
             f"LLM P95 延迟 {snapshot.latency_p95_ms:.0f}ms 超过阈值 {self.DEFAULT_LATENCY_P95_MS:.0f}ms"),
            ("cost", snapshot.total_cost, self.DEFAULT_COST_PER_SESSION, "gt",
             f"累计成本 ¥{snapshot.total_cost:.4f} 超过阈值 ¥{self.DEFAULT_COST_PER_SESSION}"),
        ]

        for category, current, threshold, op, message in checks:
            triggered = (op == "gt" and current > threshold)
            if not triggered:
                continue

            # 冷却期检查
            last_ts = self._last_alert_time.get(category, 0)
            if now - last_ts < self.COOLDOWN_SECONDS:
                continue

            alert = AlertRecord(
                id=uuid.uuid4().hex[:8],
                timestamp=_iso_now(),
                level=AlertLevel.WARNING if category != "cost" else AlertLevel.INFO,
                category=category,
                message=message,
                threshold=threshold,
                current_value=round(current, 6),
                session_id=session_id,
            )
            new_alerts.append(alert)
            self._last_alert_time[category] = now

        if new_alerts:
            self._add_alerts(new_alerts)

        return new_alerts

    def alert_budget(self, token_usage_pct: float, session_id: str = "") -> Optional[AlertRecord]:
        """Token 预算告警（> 80% 时触发）"""
        if token_usage_pct <= 0.8:
            return None
        level = AlertLevel.CRITICAL if token_usage_pct >= 0.95 else AlertLevel.WARNING
        alert = AlertRecord(
            id=uuid.uuid4().hex[:8],
            timestamp=_iso_now(),
            level=level,
            category="budget",
            message=f"Token 预算使用率达 {token_usage_pct:.0%}",
            threshold=0.8,
            current_value=round(token_usage_pct, 4),
            session_id=session_id,
        )
        self._add_alerts([alert])
        return alert

    def alert_rate_limit(self, tool_name: str = "", session_id: str = "") -> Optional[AlertRecord]:
        """速率限制触发告警"""
        alert = AlertRecord(
            id=uuid.uuid4().hex[:8],
            timestamp=_iso_now(),
            level=AlertLevel.WARNING,
            category="rate_limit",
            message=f"速率限制触发" + (f" (tool={tool_name})" if tool_name else ""),
            threshold=0,
            current_value=0,
            session_id=session_id,
        )
        self._add_alerts([alert])
        return alert

    def _add_alerts(self, alerts: List[AlertRecord]):
        with self._lock:
            self._alerts.extend(alerts)
            while len(self._alerts) > self._max_alerts:
                self._alerts = self._alerts[-self._max_alerts:]

        # 持久化
        for alert in alerts:
            self._persist_alert(alert)

        # 回调
        if self._callback:
            for alert in alerts:
                try:
                    self._callback(alert)
                except Exception:
                    pass

    def _persist_alert(self, alert: AlertRecord):
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            path = os.path.join(ALERTS_DIR, f"alerts_{date_str}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")
        except IOError:
            pass

    def get_alerts(self, limit: int = 50, acknowledged: Optional[bool] = None) -> List[dict]:
        """查询最近的告警"""
        with self._lock:
            alerts = list(self._alerts)
            if acknowledged is not None:
                alerts = [a for a in alerts if a.acknowledged == acknowledged]
            return [a.to_dict() for a in alerts[-limit:]]

    def acknowledge(self, alert_id: str) -> bool:
        """确认告警"""
        with self._lock:
            for a in self._alerts:
                if a.id == alert_id:
                    a.acknowledged = True
                    return True
        return False


# ─── 5. Rate Limiter ──────────────────────────────────────────


class RateLimiter:
    """令牌桶算法速率限制器

    适用于限制 LLM API 调用频率，避免触发上游 API 限流。
    默认：10 次/秒，突发容量 20。
    """

    def __init__(self, rate: float = 10.0, burst: int = 20):
        """
        Args:
            rate: 每秒填充令牌数
            burst: 令牌桶容量（允许的最大突发请求数）
        """
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """尝试获取一个令牌，成功返回 True，否则返回 False"""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def wait_and_acquire(self, timeout: float = 30.0) -> bool:
        """等待直到获取令牌或超时"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.acquire():
                return True
            time.sleep(0.1)
        return False

    @property
    def available_tokens(self) -> float:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            return min(self._burst, self._tokens + elapsed * self._rate)


# ─── 6. LLMOps 统一管理器 ─────────────────────────────────────


class LLMOpsManager:
    """LLMOps 统一管理器（单例）

    用法:
        llmops = get_llmops()

        # 创建追踪 span
        trace_id = llmops.new_trace()
        span = llmops.start_span(model="deepseek-chat", agent="chat_agent", session_id="...")

        # 尝试缓存
        cached = llmops.cache.get(model, messages)
        if cached:
            span.cache_hit = True
            llmops.tracer.end_span(span, cache_hit=True)
            return cached

        # 速率限制检查
        if not llmops.rate_limiter.acquire():
            llmops.alerts.alert_rate_limit("chat_agent")
            raise Exception("速率限制")

        # 执行 LLM 调用
        t0 = time.monotonic()
        try:
            result = llm(...)
            elapsed = (time.monotonic() - t0) * 1000
            # 记录指标
            llmops.metrics.record(latency_ms=elapsed, success=True, ...)
            # 结束 span
            llmops.tracer.end_span(span, ...)
            # 写入缓存
            llmops.cache.set(model, messages, result)
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            llmops.metrics.record(latency_ms=elapsed, success=False, ...)
            llmops.tracer.end_span(span, status=SpanStatus.ERROR, error_message=str(e))
            raise

        # 周期性告警检查
        llmops.periodic_check()
    """

    def __init__(self):
        self.tracer = TraceManager()
        self.metrics = MetricsCollector()
        self.cache = ResponseCache(
            max_size=int(os.getenv("LLMOPS_CACHE_SIZE", "500")),
            ttl_seconds=int(os.getenv("LLMOPS_CACHE_TTL", "300")),
        )
        self.alerts = AlertManager(callback=self._default_alert_callback)
        self.rate_limiter = RateLimiter(
            rate=float(os.getenv("LLMOPS_RATE_LIMIT", "10.0")),
            burst=int(os.getenv("LLMOPS_RATE_BURST", "20")),
        )
        self._last_check_time = 0.0
        self._check_interval = 60.0  # 每分钟检查一次告警

    # ── 快捷方法 ──

    def new_trace(self) -> str:
        return self.tracer.new_trace_id()

    def start_span(self, model: str = "", agent: str = "", session_id: str = "",
                   parent_span_id: str = "") -> TraceSpan:
        return self.tracer.start_span(model=model, agent=agent,
                                       session_id=session_id, parent_span_id=parent_span_id)

    def end_span(self, span: TraceSpan, **kwargs):
        self.tracer.end_span(span, **kwargs)

    def periodic_check(self, session_id: str = ""):
        """周期性的告警检查和指标持久化"""
        now = time.time()
        if now - self._last_check_time < self._check_interval:
            return
        self._last_check_time = now

        snap = self.metrics.snapshot()
        self.alerts.check_and_alert(snap, session_id)
        self.metrics.persist_daily_summary()

    def _default_alert_callback(self, alert: AlertRecord):
        """默认告警回调：写入日志"""
        level_map = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.ERROR,
        }
        log_level = level_map.get(alert.level, logging.WARNING)
        logger.log(log_level, f"[ALERT:{alert.level.value.upper()}] {alert.category}: {alert.message}")


# ─── 全局单例 ─────────────────────────────────────────────────

_global_llmops: Optional[LLMOpsManager] = None


def get_llmops() -> LLMOpsManager:
    """获取全局 LLMOps 管理器实例"""
    global _global_llmops
    if _global_llmops is None:
        _global_llmops = LLMOpsManager()
        logger.info("[LLMOps] 管理器已初始化 "
                     f"(cache_size={_global_llmops.cache._max_size}, "
                     f"rate={_global_llmops.rate_limiter._rate}/s)")
    return _global_llmops


def reset_llmops():
    """重置 LLMOps 管理器（测试用）"""
    global _global_llmops
    if _global_llmops:
        _global_llmops.tracer.flush()
        _global_llmops.metrics.persist_daily_summary()
    _global_llmops = None


# ─── 7. 装饰器 ────────────────────────────────────────────────


def trace_llm_call(func):
    """LLM 调用追踪装饰器

    自动记录调用延迟、Token 用量、成功/失败状态。
    需要函数返回包含 usage_metadata 或 token_usage 的结果对象。

    用法:
        @trace_llm_call
        def my_llm_call(...) -> AIMessage:
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        llmops = get_llmops()
        span = llmops.start_span(
            model=kwargs.get("model", ""),
            agent=kwargs.get("agent", ""),
            session_id=kwargs.get("session_id", ""),
        )
        t0 = time.monotonic()
        try:
            result = func(*args, **kwargs)
            elapsed = (time.monotonic() - t0) * 1000

            # 尝试提取 token
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(result, "usage_metadata") and result.usage_metadata:
                prompt_tokens = result.usage_metadata.get("input_tokens", 0)
                completion_tokens = result.usage_metadata.get("output_tokens", 0)
            elif hasattr(result, "response_metadata"):
                meta = result.response_metadata or {}
                usage = meta.get("token_usage", {}) or meta.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

            cost = (prompt_tokens * 1.0 + completion_tokens * 2.0) / 1_000_000

            llmops.tracer.end_span(span, prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens)
            llmops.metrics.record(latency_ms=elapsed, success=True,
                                   prompt_tokens=prompt_tokens,
                                   completion_tokens=completion_tokens,
                                   cost=cost)
            return result
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            llmops.tracer.end_span(span, status=SpanStatus.ERROR,
                                    error_message=str(e))
            llmops.metrics.record(latency_ms=elapsed, success=False)
            raise

    return wrapper
