"""
LLMOps 模块单元测试

覆盖：TraceManager, MetricsCollector, ResponseCache,
      AlertManager, RateLimiter, LLMOpsManager, trace_llm_call 装饰器

运行方式（Docker 容器内）：
    docker compose exec backend python /app/test_llmops.py
"""

import os
import sys
import time
import json
import uuid
import logging
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from collections import OrderedDict

# 确保可以导入 backend 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.llmops import (
    TraceManager, TraceSpan, SpanStatus,
    MetricsCollector, MetricsSnapshot,
    ResponseCache,
    AlertManager, AlertRecord, AlertLevel,
    RateLimiter,
    LLMOpsManager,
    trace_llm_call,
    get_llmops, reset_llmops,
    _hash_prompt, _calc_percentile, _iso_now,
    LLMOPS_DIR, TRACES_DIR, METRICS_DIR, ALERTS_DIR,
)

# 关闭 llmops 日志输出，避免干扰测试
logging.getLogger("llmops").setLevel(logging.CRITICAL)


# ═══════════════════════════════════════════════════════════════
# Test 1: TraceManager  调用追踪
# ═══════════════════════════════════════════════════════════════

class TestTraceManager(unittest.TestCase):
    """TraceManager 单元测试"""

    def setUp(self):
        self.tracer = TraceManager()

    def test_new_trace_id_unique(self):
        """trace_id 生成唯一性"""
        ids = {self.tracer.new_trace_id() for _ in range(100)}
        self.assertEqual(len(ids), 100, "100个 trace_id 应全部唯一")

    def test_new_trace_id_hex16(self):
        """trace_id 应为 16 位十六进制字符串"""
        tid = self.tracer.new_trace_id()
        self.assertEqual(len(tid), 16)
        self.assertTrue(all(c in '0123456789abcdef' for c in tid))

    def test_new_span_id_unique(self):
        """span_id 生成唯一性"""
        ids = {self.tracer.new_span_id() for _ in range(100)}
        self.assertEqual(len(ids), 100, "100个 span_id 应全部唯一")

    def test_new_span_id_hex8(self):
        """span_id 应为 8 位十六进制字符串"""
        sid = self.tracer.new_span_id()
        self.assertEqual(len(sid), 8)
        self.assertTrue(all(c in '0123456789abcdef' for c in sid))

    def test_current_trace_id(self):
        """current_trace_id 属性反映当前 trace 序列"""
        self.assertIsNone(self.tracer.current_trace_id)
        tid = self.tracer.new_trace_id()
        self.assertEqual(self.tracer.current_trace_id, tid)

    def test_start_span_basic(self):
        """start_span 创建正确结构的 TraceSpan"""
        self.tracer.new_trace_id()
        span = self.tracer.start_span(
            model="test-model",
            agent="test_agent",
            session_id="sess-123",
        )
        self.assertIsInstance(span, TraceSpan)
        self.assertEqual(span.model, "test-model")
        self.assertEqual(span.agent, "test_agent")
        self.assertEqual(span.session_id, "sess-123")
        self.assertEqual(span.trace_id, self.tracer.current_trace_id)
        self.assertTrue(span.span_id)
        self.assertTrue(span.start_time)
        self.assertEqual(span.status, "ok")

    def test_start_span_without_trace_creates_one(self):
        """未 new_trace_id 时 start_span 自动创建"""
        span = self.tracer.start_span(agent="auto")
        self.assertIsNotNone(span.trace_id)
        self.assertEqual(len(span.trace_id), 16)

    def test_start_span_parent_chain(self):
        """start_span 支持 parent_span_id 链路"""
        self.tracer.new_trace_id()
        parent = self.tracer.start_span(agent="parent")
        child = self.tracer.start_span(
            agent="child",
            parent_span_id=parent.span_id,
        )
        self.assertEqual(child.parent_span_id, parent.span_id)
        self.assertEqual(child.trace_id, parent.trace_id)

    def test_end_span_success(self):
        """end_span 正确记录成功 span"""
        self.tracer.new_trace_id()
        span = self.tracer.start_span(agent="test")
        self.tracer.end_span(
            span,
            prompt_tokens=100,
            completion_tokens=200,
            status=SpanStatus.OK,
        )
        self.assertEqual(span.prompt_tokens, 100)
        self.assertEqual(span.completion_tokens, 200)
        self.assertEqual(span.total_tokens, 300)
        self.assertEqual(span.status, "ok")
        self.assertTrue(span.end_time)
        self.assertTrue(span.latency_ms >= 0)

    def test_end_span_error(self):
        """end_span 正确记录错误 span"""
        self.tracer.new_trace_id()
        span = self.tracer.start_span(agent="test")
        self.tracer.end_span(
            span,
            status=SpanStatus.ERROR,
            error_message="LLM API timeout",
        )
        self.assertEqual(span.status, "error")
        self.assertEqual(span.error_message, "LLM API timeout")

    def test_end_span_cache_hit(self):
        """end_span 正确记录缓存命中"""
        self.tracer.new_trace_id()
        span = self.tracer.start_span(agent="test")
        self.tracer.end_span(span, cache_hit=True, prompt_hash="abc123")
        self.assertTrue(span.cache_hit)
        self.assertEqual(span.prompt_hash, "abc123")
        self.assertEqual(span.status, "ok")

    def test_end_span_tokens_per_second(self):
        """end_span 正确计算 tokens_per_second"""
        # 使用快速完成模拟（将 start_time 提前）
        self.tracer.new_trace_id()
        span = self.tracer.start_span(agent="test")
        # 把 start_time 设为 1 秒前
        from datetime import datetime, timedelta
        span.start_time = (datetime.now() - timedelta(seconds=1)).isoformat()
        self.tracer.end_span(span, completion_tokens=500)
        self.assertGreater(span.tokens_per_second, 0)
        self.assertLess(span.tokens_per_second, 1000)  # 合理范围

    def test_span_to_dict(self):
        """TraceSpan.to_dict() 序列化正确"""
        self.tracer.new_trace_id()
        span = self.tracer.start_span(model="gpt-4", agent="chat", session_id="s1")
        self.tracer.end_span(span, prompt_tokens=50, completion_tokens=100)
        d = span.to_dict()
        self.assertIn("trace_id", d)
        self.assertIn("span_id", d)
        self.assertIn("latency_ms", d)
        self.assertIn("total_tokens", d)
        self.assertEqual(d["agent"], "chat")
        self.assertEqual(d["model"], "gpt-4")
        self.assertEqual(d["session_id"], "s1")
        self.assertEqual(d["prompt_tokens"], 50)
        self.assertEqual(d["completion_tokens"], 100)
        self.assertEqual(d["total_tokens"], 150)

    def test_flush_and_write(self):
        """flush 写入 JSONL 文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(TraceManager, '__init__', lambda self: None):
                pass
            # 直接通过 _write_spans 测试写入
            self.tracer.new_trace_id()
            span = self.tracer.start_span(agent="flush_test")
            self.tracer.end_span(span, prompt_tokens=10, completion_tokens=20)
            # 手动 flush
            self.tracer.flush()

    def test_get_recent_traces_empty(self):
        """空追踪时 get_recent_traces 返回空列表"""
        # 需要 patch TRACES_DIR 为空目录，避免读到之前测试的遗留数据
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.agent.llmops.TRACES_DIR", tmpdir):
                traces = self.tracer.get_recent_traces(limit=10)
                self.assertEqual(traces, [])

    def test_batch_flush_trigger(self):
        """积累 >=10 条 span 时自动 flush"""
        self.tracer.new_trace_id()
        for i in range(10):
            span = self.tracer.start_span(agent=f"agent_{i}")
            self.tracer.end_span(span, prompt_tokens=1, completion_tokens=1)
        # 第10条触发 flush，batch 清空
        self.assertEqual(len(self.tracer._batch), 0)

    def test_hash_prompt_deterministic(self):
        """_hash_prompt 确定性：相同输入产生相同哈希"""
        msgs = [{"role": "user", "content": "Hello"}]
        h1 = _hash_prompt("deepseek-chat", msgs)
        h2 = _hash_prompt("deepseek-chat", msgs)
        self.assertEqual(h1, h2)

    def test_hash_prompt_different(self):
        """_hash_prompt 区分不同输入"""
        h1 = _hash_prompt("model-a", [{"role": "user", "content": "A"}])
        h2 = _hash_prompt("model-a", [{"role": "user", "content": "B"}])
        h3 = _hash_prompt("model-b", [{"role": "user", "content": "A"}])
        self.assertNotEqual(h1, h2)
        self.assertNotEqual(h1, h3)


# ═══════════════════════════════════════════════════════════════
# Test 2: _calc_percentile  百分位计算
# ═══════════════════════════════════════════════════════════════

class TestPercentile(unittest.TestCase):
    """_calc_percentile 工具函数测试"""

    def test_empty_returns_zero(self):
        self.assertEqual(_calc_percentile([], 50), 0.0)

    def test_single_value(self):
        self.assertEqual(_calc_percentile([100.0], 50), 100.0)
        self.assertEqual(_calc_percentile([100.0], 95), 100.0)
        self.assertEqual(_calc_percentile([100.0], 0), 100.0)

    def test_median_odd(self):
        """奇数个值的 p50"""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(_calc_percentile(sorted(vals), 50), 3.0)

    def test_median_even(self):
        """偶数个值的 p50（线性插值）"""
        vals = [1.0, 2.0, 3.0, 4.0]
        # k = 3 * 50 / 100 = 1.5, f=1, c=0.5
        # result = vals[1] + 0.5 * (vals[2] - vals[1]) = 2 + 0.5 * 1 = 2.5
        self.assertEqual(_calc_percentile(sorted(vals), 50), 2.5)

    def test_p95(self):
        """p95 百分位"""
        vals = list(range(1, 101))  # 1..100
        # k = 99 * 95 / 100 = 94.05, f=94, c=0.05
        # vals[94] = 95, vals[95] = 96
        # result = 95 + 0.05 * 1 = 95.05
        self.assertAlmostEqual(_calc_percentile(vals, 95), 95.05, places=2)

    def test_p99(self):
        """p99 百分位"""
        vals = list(range(1, 101))
        self.assertAlmostEqual(_calc_percentile(vals, 99), 99.01, places=2)


# ═══════════════════════════════════════════════════════════════
# Test 3: MetricsCollector  性能指标收集器
# ═══════════════════════════════════════════════════════════════

class TestMetricsCollector(unittest.TestCase):
    """MetricsCollector 单元测试"""

    def setUp(self):
        self.metrics = MetricsCollector()

    def test_record_single_success(self):
        self.metrics.record(
            latency_ms=500.0, success=True,
            prompt_tokens=100, completion_tokens=200,
            cost=0.0005,
        )
        snap = self.metrics.snapshot()
        self.assertEqual(snap.total_calls, 1)
        self.assertEqual(snap.success_calls, 1)
        self.assertEqual(snap.error_calls, 0)
        self.assertEqual(snap.total_prompt_tokens, 100)
        self.assertEqual(snap.total_completion_tokens, 200)
        self.assertAlmostEqual(snap.total_cost, 0.0005, places=6)
        self.assertEqual(snap.latency_p50_ms, 500.0)
        self.assertEqual(snap.latency_min_ms, 500.0)
        self.assertEqual(snap.latency_max_ms, 500.0)

    def test_record_error(self):
        self.metrics.record(latency_ms=1000.0, success=False)
        snap = self.metrics.snapshot()
        self.assertEqual(snap.total_calls, 1)
        self.assertEqual(snap.success_calls, 0)
        self.assertEqual(snap.error_calls, 1)

    def test_record_multiple_latencies(self):
        """多次记录后百分位正确"""
        latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
        for lat in latencies:
            self.metrics.record(latency_ms=lat, success=True)
        snap = self.metrics.snapshot()
        self.assertEqual(snap.total_calls, 5)
        self.assertEqual(snap.success_calls, 5)
        self.assertEqual(snap.latency_min_ms, 100.0)
        self.assertEqual(snap.latency_max_ms, 500.0)
        self.assertEqual(snap.latency_avg_ms, 300.0)
        self.assertEqual(snap.latency_p50_ms, 300.0)

    def test_record_cache_hit(self):
        self.metrics.record(cache_hit=True)
        self.metrics.record(cache_hit=False)
        snap = self.metrics.snapshot()
        self.assertEqual(snap.cache_hits, 1)
        self.assertEqual(snap.cache_misses, 1)

    def test_snapshot_success_rate(self):
        """快照中成功率计算"""
        for _ in range(8):
            self.metrics.record(success=True)
        for _ in range(2):
            self.metrics.record(success=False)
        snap = self.metrics.snapshot()
        d = snap.to_dict()
        self.assertEqual(d["success_rate"], 0.8)
        self.assertEqual(d["error_rate"], 0.2)

    def test_snapshot_cache_hit_rate(self):
        for _ in range(3):
            self.metrics.record(cache_hit=True)
        for _ in range(7):
            self.metrics.record(cache_hit=False)
        snap = self.metrics.snapshot()
        d = snap.to_dict()
        self.assertEqual(d["cache_hit_rate"], 0.3)

    def test_snapshot_zero_division_safe(self):
        """空快照时各项比例安全（除零保护）"""
        snap = self.metrics.snapshot()
        d = snap.to_dict()
        self.assertEqual(d["success_rate"], 0.0)
        self.assertEqual(d["error_rate"], 0.0)
        self.assertEqual(d["cache_hit_rate"], 0.0)
        self.assertEqual(d["total_calls"], 0)

    def test_tps_tracking(self):
        """Token/秒吞吐量跟踪"""
        self.metrics.record(
            latency_ms=1000.0,       # 1 秒
            completion_tokens=500,    # 500 t/s
        )
        snap = self.metrics.snapshot()
        self.assertEqual(snap.tokens_per_second_avg, 500.0)

    def test_get_per_agent_snapshot(self):
        """按 Agent 维度聚合指标"""
        traces = [
            {"agent": "chat_agent", "status": "ok", "total_tokens": 100, "cost": 0.0001, "latency_ms": 200.0},
            {"agent": "chat_agent", "status": "error", "total_tokens": 50, "cost": 0.00005, "latency_ms": 500.0},
            {"agent": "research", "status": "ok", "total_tokens": 500, "cost": 0.001, "latency_ms": 1000.0},
        ]
        result = self.metrics.get_per_agent_snapshot(traces)
        self.assertIn("chat_agent", result)
        self.assertIn("research", result)
        self.assertEqual(result["chat_agent"]["total_calls"], 2)
        self.assertEqual(result["chat_agent"]["error_calls"], 1)
        self.assertEqual(result["chat_agent"]["error_rate"], 0.5)
        self.assertEqual(result["chat_agent"]["total_tokens"], 150)
        self.assertEqual(result["research"]["total_calls"], 1)
        self.assertEqual(result["research"]["error_calls"], 0)
        self.assertEqual(result["research"]["total_tokens"], 500)

    def test_get_per_agent_unknown(self):
        """agent 字段缺失时标记为 unknown"""
        traces = [{"status": "ok", "total_tokens": 10, "cost": 0, "latency_ms": 50}]
        result = self.metrics.get_per_agent_snapshot(traces)
        self.assertIn("unknown", result)

    def test_latency_sample_limit(self):
        """延迟样本超过最大值时自动清理"""
        for i in range(MetricsCollector.MAX_LATENCY_SAMPLES + 100):
            self.metrics.record(latency_ms=float(i), success=True)
        self.assertLessEqual(
            len(self.metrics._latencies),
            MetricsCollector.MAX_LATENCY_SAMPLES,
        )

    def test_persist_daily_summary(self):
        """persist_daily_summary 写入 metrics JSON 文件"""
        self.metrics.record(latency_ms=100.0, success=True,
                            prompt_tokens=10, completion_tokens=20)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.agent.llmops.METRICS_DIR", tmpdir):
                self.metrics.persist_daily_summary()
                files = os.listdir(tmpdir)
                self.assertEqual(len(files), 1)
                self.assertTrue(files[0].startswith("metrics_"))
                self.assertTrue(files[0].endswith(".json"))


# ═══════════════════════════════════════════════════════════════
# Test 4: ResponseCache  响应缓存
# ═══════════════════════════════════════════════════════════════

class TestResponseCache(unittest.TestCase):
    """ResponseCache 单元测试"""

    def setUp(self):
        self.cache = ResponseCache(max_size=5, ttl_seconds=60)

    def _make_msgs(self, text="Hello"):
        return [{"role": "user", "content": text}]

    def test_set_and_get(self):
        """基本 set/get 读写"""
        self.cache.set("deepseek-chat", self._make_msgs("Hi"), "response-A")
        result = self.cache.get("deepseek-chat", self._make_msgs("Hi"))
        self.assertEqual(result, "response-A")

    def test_get_miss(self):
        """未命中返回 None"""
        result = self.cache.get("deepseek-chat", self._make_msgs("No"))
        self.assertIsNone(result)

    def test_same_prompt_same_hash(self):
        """相同 prompt 使用相同缓存 key"""
        self.cache.set("deepseek-chat", self._make_msgs("same"), "val")
        self.assertEqual(self.cache.get("deepseek-chat", self._make_msgs("same")), "val")

    def test_different_model_different_cache(self):
        """不同模型独立缓存"""
        msgs = self._make_msgs("test")
        self.cache.set("model-a", msgs, "resp-a")
        self.cache.set("model-b", msgs, "resp-b")
        self.assertEqual(self.cache.get("model-a", msgs), "resp-a")
        self.assertEqual(self.cache.get("model-b", msgs), "resp-b")

    def test_lru_eviction(self):
        """LRU 淘汰最久未用条目"""
        cache = ResponseCache(max_size=3, ttl_seconds=300)
        for i in range(3):
            cache.set("m", self._make_msgs(f"msg_{i}"), f"resp_{i}")
        # 访问 msg_0 使其变"新"
        cache.get("m", self._make_msgs("msg_0"))
        # 存入第4个，应淘汰 msg_1（最久未访问）
        cache.set("m", self._make_msgs("msg_3"), "resp_3")
        self.assertIsNotNone(cache.get("m", self._make_msgs("msg_0")))
        self.assertIsNotNone(cache.get("m", self._make_msgs("msg_2")))
        self.assertIsNotNone(cache.get("m", self._make_msgs("msg_3")))
        self.assertIsNone(cache.get("m", self._make_msgs("msg_1")))

    def test_ttl_expiry(self):
        """TTL 过期后 get 返回 None"""
        # 注意: _ttl_seconds 最小为 1 秒，使用 mock 加速时间
        cache = ResponseCache(max_size=10, ttl_seconds=1)
        cache.set("m", self._make_msgs("expire"), "val")
        # 立即获取应命中
        self.assertIsNotNone(cache.get("m", self._make_msgs("expire")))
        # 模拟时间前进 1000 秒
        with patch("time.time", return_value=time.time() + 1000):
            result = cache.get("m", self._make_msgs("expire"))
            self.assertIsNone(result)

    def test_ttl_within_valid(self):
        """TTL 内 get 命中"""
        cache = ResponseCache(max_size=10, ttl_seconds=10)
        cache.set("m", self._make_msgs("valid"), "val")
        self.assertEqual(cache.get("m", self._make_msgs("valid")), "val")

    def test_clear(self):
        """clear 清空全部缓存"""
        self.cache.set("m", self._make_msgs("a"), "A")
        self.cache.set("m", self._make_msgs("b"), "B")
        self.cache.clear()
        self.assertIsNone(self.cache.get("m", self._make_msgs("a")))
        self.assertIsNone(self.cache.get("m", self._make_msgs("b")))

    def test_size_property(self):
        """size 属性反映实际条目数"""
        self.assertEqual(self.cache.size, 0)
        self.cache.set("m", self._make_msgs("1"), "A")
        self.assertEqual(self.cache.size, 1)
        self.cache.set("m", self._make_msgs("2"), "B")
        self.assertEqual(self.cache.size, 2)

    def test_stats(self):
        """stats 属性返回配置信息"""
        s = self.cache.stats
        self.assertEqual(s["current_size"], 0)
        self.assertEqual(s["max_size"], 5)
        self.assertEqual(s["ttl_seconds"], 60)

    def test_overwrite_same_key(self):
        """同一 key 重复 set 覆盖旧值"""
        msgs = self._make_msgs("overwrite")
        self.cache.set("m", msgs, "v1")
        self.cache.set("m", msgs, "v2")
        self.assertEqual(self.cache.get("m", msgs), "v2")
        self.assertEqual(self.cache.size, 1)

    def test_thread_safety(self):
        """多线程并发 set/get 不崩溃"""
        import threading
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    msgs = self._make_msgs(f"t{thread_id}_{i}")
                    self.cache.set("m", msgs, f"val_{thread_id}_{i}")
                    self.cache.get("m", msgs)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0, f"线程错误: {errors}")

    def test_max_size_boundary(self):
        """边界：max_size=1 时旧条目立即淘汰"""
        cache = ResponseCache(max_size=1, ttl_seconds=300)
        cache.set("m", self._make_msgs("a"), "A")
        cache.set("m", self._make_msgs("b"), "B")
        self.assertIsNone(cache.get("m", self._make_msgs("a")))
        self.assertEqual(cache.get("m", self._make_msgs("b")), "B")


# ═══════════════════════════════════════════════════════════════
# Test 5: AlertManager  告警管理器
# ═══════════════════════════════════════════════════════════════

class TestAlertManager(unittest.TestCase):
    """AlertManager 单元测试"""

    def setUp(self):
        self.alert_mgr = AlertManager()

    def _make_snapshot(self, **kwargs):
        """创建测试用 MetricsSnapshot"""
        defaults = {
            "total_calls": 10,
            "success_calls": 10,
            "error_calls": 0,
            "cache_hits": 0,
            "cache_misses": 10,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost": 0.0,
            "latency_p50_ms": 100.0,
            "latency_p95_ms": 200.0,
            "latency_p99_ms": 300.0,
            "latency_avg_ms": 100.0,
            "latency_min_ms": 50.0,
            "latency_max_ms": 500.0,
            "tokens_per_second_avg": 100.0,
            "last_updated": "",
        }
        defaults.update(kwargs)
        return MetricsSnapshot(**defaults)

    def test_no_alert_when_normal(self):
        """正常指标不触发告警"""
        snap = self._make_snapshot()
        alerts = self.alert_mgr.check_and_alert(snap)
        self.assertEqual(alerts, [])

    def test_error_rate_alert(self):
        """错误率超阈值触发告警"""
        snap = self._make_snapshot(
            total_calls=10, error_calls=5,  # 50% > 10%
        )
        alerts = self.alert_mgr.check_and_alert(snap)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].category, "error_rate")
        self.assertEqual(alerts[0].level, AlertLevel.WARNING)

    def test_latency_alert(self):
        """P95 延迟超阈值触发告警"""
        snap = self._make_snapshot(
            latency_p95_ms=15000,  # > 10000ms
        )
        alerts = self.alert_mgr.check_and_alert(snap)
        self.assertGreaterEqual(len(alerts), 1)
        self.assertTrue(any(a.category == "latency" for a in alerts))

    def test_cost_alert(self):
        """累计成本超阈值触发告警"""
        snap = self._make_snapshot(total_cost=5.0)  # > ¥1
        alerts = self.alert_mgr.check_and_alert(snap)
        self.assertGreaterEqual(len(alerts), 1)
        self.assertTrue(any(a.category == "cost" for a in alerts))

    def test_cost_level_info(self):
        """成本告警级别为 INFO"""
        snap = self._make_snapshot(total_cost=5.0)
        alerts = self.alert_mgr.check_and_alert(snap)
        cost_alerts = [a for a in alerts if a.category == "cost"]
        self.assertEqual(cost_alerts[0].level, AlertLevel.INFO)

    def test_cooldown_prevents_duplicate(self):
        """冷却期内不重复告警"""
        snap = self._make_snapshot(total_calls=10, error_calls=5)
        # 第一次触发
        alerts1 = self.alert_mgr.check_and_alert(snap)
        self.assertEqual(len(alerts1), 1, "第一次应触发告警")
        # 立即再次检查（冷却期内）
        alerts2 = self.alert_mgr.check_and_alert(snap)
        self.assertEqual(alerts2, [], "冷却期内不应重复触发")

    def test_cooldown_expires(self):
        """冷却期过后可再次触发"""
        snap = self._make_snapshot(total_calls=10, error_calls=5)
        self.alert_mgr.check_and_alert(snap)
        # 模拟冷却期已过
        self.alert_mgr._last_alert_time["error_rate"] = time.time() - 1000
        alerts2 = self.alert_mgr.check_and_alert(snap)
        self.assertEqual(len(alerts2), 1)

    def test_alert_budget_warning(self):
        """Token 预算 > 80% 触发 WARNING"""
        alert = self.alert_mgr.alert_budget(0.85, session_id="s1")
        self.assertIsNotNone(alert)
        self.assertEqual(alert.category, "budget")
        self.assertEqual(alert.level, AlertLevel.WARNING)

    def test_alert_budget_critical(self):
        """Token 预算 >= 95% 触发 CRITICAL"""
        alert = self.alert_mgr.alert_budget(0.96, session_id="s2")
        self.assertIsNotNone(alert)
        self.assertEqual(alert.level, AlertLevel.CRITICAL)

    def test_alert_budget_normal(self):
        """Token 预算 <= 80% 不触发告警"""
        alert = self.alert_mgr.alert_budget(0.75)
        self.assertIsNone(alert)
        alert2 = self.alert_mgr.alert_budget(0.80)
        self.assertIsNone(alert2)

    def test_alert_rate_limit(self):
        """速率限制告警"""
        alert = self.alert_mgr.alert_rate_limit(
            tool_name="web_search", session_id="s3"
        )
        self.assertIsNotNone(alert)
        self.assertEqual(alert.category, "rate_limit")
        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertIn("web_search", alert.message)

    def test_get_alerts_all(self):
        """获取告警列表"""
        self.alert_mgr.alert_budget(0.9)
        self.alert_mgr.alert_rate_limit(tool_name="test_tool")
        alerts = self.alert_mgr.get_alerts(limit=10)
        self.assertEqual(len(alerts), 2)

    def test_get_alerts_unacknowledged(self):
        """过滤未确认告警"""
        self.alert_mgr.alert_budget(0.9)
        self.alert_mgr.alert_rate_limit()
        unacked = self.alert_mgr.get_alerts(
            limit=10, acknowledged=False
        )
        self.assertEqual(len(unacked), 2)

    def test_get_alerts_acknowledged(self):
        """过滤已确认告警"""
        self.alert_mgr.alert_budget(0.9)
        alerts = self.alert_mgr.get_alerts(limit=10)
        alert_id = alerts[0]["id"]
        self.alert_mgr.acknowledge(alert_id)
        acked = self.alert_mgr.get_alerts(
            limit=10, acknowledged=True
        )
        self.assertEqual(len(acked), 1)
        self.assertEqual(acked[0]["id"], alert_id)

    def test_acknowledge_nonexistent(self):
        """确认不存在的告警返回 False"""
        result = self.alert_mgr.acknowledge("nonexistent-id")
        self.assertFalse(result)

    def test_callback_invoked(self):
        """告警触发回调函数"""
        callback_alerts = []
        mgr = AlertManager(callback=lambda a: callback_alerts.append(a))
        snap = self._make_snapshot(error_calls=5, total_calls=10)
        mgr.check_and_alert(snap)
        self.assertEqual(len(callback_alerts), 1)
        self.assertEqual(callback_alerts[0].category, "error_rate")

    def test_callback_exception_safe(self):
        """回调异常不中断告警流程"""
        def bad_callback(a):
            raise RuntimeError("callback failed")
        mgr = AlertManager(callback=bad_callback)
        snap = self._make_snapshot(total_calls=10, error_calls=5)
        # 不应抛出异常
        alerts = mgr.check_and_alert(snap)
        self.assertEqual(len(alerts), 1)

    def test_alert_max_limit(self):
        """告警超过最大数量时自动裁剪"""
        mgr = AlertManager()
        mgr._max_alerts = 5
        for i in range(10):
            mgr.alert_budget(0.9)
        alerts = mgr.get_alerts(limit=20)
        self.assertLessEqual(len(alerts), 5)

    def test_alert_to_dict(self):
        """AlertRecord.to_dict() 序列化正确"""
        alert = self.alert_mgr.alert_budget(0.9, session_id="s99")
        d = alert.to_dict()
        self.assertEqual(d["category"], "budget")
        self.assertEqual(d["level"], "warning")
        self.assertEqual(d["session_id"], "s99")
        self.assertFalse(d["acknowledged"])

    def test_alert_persist(self):
        """告警持久化到 JSONL 文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.agent.llmops.ALERTS_DIR", tmpdir):
                alerts = self.alert_mgr.check_and_alert(
                    self._make_snapshot(error_calls=5, total_calls=10)
                )
                self.assertEqual(len(alerts), 1)
                # 持久化发生在 _add_alerts 内部
                files = os.listdir(tmpdir)
                self.assertEqual(len(files), 1)
                self.assertTrue(files[0].startswith("alerts_"))
                self.assertTrue(files[0].endswith(".jsonl"))


# ═══════════════════════════════════════════════════════════════
# Test 6: RateLimiter  令牌桶速率限制器
# ═══════════════════════════════════════════════════════════════

class TestRateLimiter(unittest.TestCase):
    """RateLimiter 令牌桶算法测试"""

    def test_acquire_within_burst(self):
        """突发容量内获取令牌成功"""
        rl = RateLimiter(rate=10.0, burst=20)
        for _ in range(20):
            self.assertTrue(rl.acquire(), "burst=20 内应全部成功")
        # 第 21 次失败（令牌耗尽）
        self.assertFalse(rl.acquire())

    def test_token_refill(self):
        """令牌按速率恢复"""
        rl = RateLimiter(rate=100.0, burst=10)
        # 耗尽令牌
        for _ in range(10):
            rl.acquire()
        self.assertFalse(rl.acquire())
        # 等待 0.05 秒恢复 5 个令牌
        time.sleep(0.05)
        count = sum(1 for _ in range(10) if rl.acquire())
        self.assertGreaterEqual(count, 2, f"应恢复至少 2 个令牌，实际 {count}")
        self.assertLessEqual(count, 7)

    def test_available_tokens(self):
        """available_tokens 返回当前令牌数"""
        rl = RateLimiter(rate=10.0, burst=20)
        self.assertAlmostEqual(rl.available_tokens, 20.0, places=1)
        rl.acquire()
        self.assertAlmostEqual(rl.available_tokens, 19.0, places=1)

    def test_wait_and_acquire_success(self):
        """wait_and_acquire 等待后成功获取"""
        rl = RateLimiter(rate=100.0, burst=10)
        # 耗尽
        for _ in range(10):
            rl.acquire()
        # 等待获取（应很快恢复）
        result = rl.wait_and_acquire(timeout=1.0)
        self.assertTrue(result)

    def test_wait_and_acquire_timeout(self):
        """wait_and_acquire 超时返回 False"""
        rl = RateLimiter(rate=0.0, burst=0)  # 永远无法获取
        result = rl.wait_and_acquire(timeout=0.1)
        self.assertFalse(result)

    def test_custom_rate(self):
        """自定义速率参数"""
        rl = RateLimiter(rate=5.0, burst=5)
        for _ in range(5):
            self.assertTrue(rl.acquire())
        self.assertFalse(rl.acquire())


# ═══════════════════════════════════════════════════════════════
# Test 7: LLMOpsManager  统一管理器
# ═══════════════════════════════════════════════════════════════

class TestLLMOpsManager(unittest.TestCase):
    """LLMOpsManager 统一管理器测试"""

    def setUp(self):
        reset_llmops()
        self.ops = get_llmops()

    def tearDown(self):
        reset_llmops()

    def test_singleton(self):
        """get_llmops 返回单例"""
        ops1 = get_llmops()
        ops2 = get_llmops()
        self.assertIs(ops1, ops2)

    def test_reset_creates_new(self):
        """reset_llmops 后 get_llmops 返回新实例"""
        ops1 = get_llmops()
        reset_llmops()
        ops2 = get_llmops()
        self.assertIsNot(ops1, ops2)

    def test_new_trace(self):
        """new_trace 快捷方法"""
        tid = self.ops.new_trace()
        self.assertEqual(len(tid), 16)
        self.assertEqual(tid, self.ops.tracer.current_trace_id)

    def test_start_end_span(self):
        """start_span / end_span 快捷方法"""
        tid = self.ops.new_trace()
        span = self.ops.start_span(model="gpt-4", agent="chat", session_id="s1")
        self.ops.end_span(span, prompt_tokens=10, completion_tokens=20)
        self.assertEqual(span.total_tokens, 30)

    def test_periodic_check_first_call(self):
        """首次 periodic_check 触发告警检查（间隔已到）"""
        # 添加错误调用以触发告警
        for _ in range(5):
            self.ops.metrics.record(success=False)
        self.ops.periodic_check(session_id="test")
        # 检查告警
        alerts = self.ops.alerts.get_alerts(limit=10)
        self.assertGreater(len(alerts), 0)

    def test_periodic_check_throttle(self):
        """periodic_check 间隔内不重复执行"""
        self.ops.periodic_check()
        # 立即再次调用应跳过
        alerts_before = len(self.ops.alerts.get_alerts(limit=100))
        self.ops.periodic_check()
        alerts_after = len(self.ops.alerts.get_alerts(limit=100))
        self.assertEqual(alerts_before, alerts_after)

    def test_sub_components_initialized(self):
        """所有子组件正确初始化"""
        self.assertIsInstance(self.ops.tracer, TraceManager)
        self.assertIsInstance(self.ops.metrics, MetricsCollector)
        self.assertIsInstance(self.ops.cache, ResponseCache)
        self.assertIsInstance(self.ops.alerts, AlertManager)
        self.assertIsInstance(self.ops.rate_limiter, RateLimiter)

    def test_default_alert_callback(self):
        """默认告警回调写入日志"""
        with self.assertLogs("llmops", level="WARNING") as cm:
            alert = AlertRecord(
                id="test001",
                timestamp=_iso_now(),
                level=AlertLevel.WARNING,
                category="error_rate",
                message="Test alert",
                threshold=0.1,
                current_value=0.3,
            )
            self.ops._default_alert_callback(alert)
        self.assertTrue(any("[ALERT:WARNING]" in msg for msg in cm.output))


# ═══════════════════════════════════════════════════════════════
# Test 8: trace_llm_call 装饰器
# ═══════════════════════════════════════════════════════════════

class TestTraceLLMCall(unittest.TestCase):
    """trace_llm_call 装饰器测试"""

    def setUp(self):
        reset_llmops()

    def tearDown(self):
        reset_llmops()

    def test_decorated_func_called(self):
        """装饰的函数仍然正常工作"""
        @trace_llm_call
        def my_llm(messages, model="", agent="", session_id=""):
            return "hello"

        # 需要给 result 添加 response_metadata 才能提取 token
        result = my_llm([], model="deepseek", agent="chat", session_id="s1")
        self.assertEqual(result, "hello")

    def test_decorated_error_recorded(self):
        """装饰器在异常时记录错误 span"""
        @trace_llm_call
        def failing_llm(messages, model="", agent="", session_id=""):
            raise ValueError("API timeout")

        with self.assertRaises(ValueError):
            failing_llm([], model="test", agent="test", session_id="s2")

        # 验证 metrics 记录了错误
        snap = get_llmops().metrics.snapshot()
        self.assertEqual(snap.total_calls, 1)
        self.assertEqual(snap.error_calls, 1)
        self.assertEqual(snap.success_calls, 0)

    def test_decorated_success_recorded(self):
        """装饰器在成功时记录指标"""

        class FakeResponse:
            response_metadata = {}
            usage_metadata = {"input_tokens": 100, "output_tokens": 200}

        @trace_llm_call
        def my_llm(messages, model="", agent="", session_id=""):
            return FakeResponse()

        result = my_llm([], model="deepseek", agent="chat", session_id="s3")
        self.assertIsInstance(result, FakeResponse)

        snap = get_llmops().metrics.snapshot()
        self.assertEqual(snap.total_calls, 1)
        self.assertEqual(snap.success_calls, 1)
        self.assertEqual(snap.total_prompt_tokens, 100)
        self.assertEqual(snap.total_completion_tokens, 200)

    def test_decorated_token_from_response_metadata(self):
        """从 response_metadata.token_usage 提取 token"""

        class FakeResp:
            response_metadata = {
                "token_usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 100,
                }
            }

        @trace_llm_call
        def my_llm(messages, model="", agent="", session_id=""):
            return FakeResp()

        my_llm([], model="m", agent="a", session_id="s")
        snap = get_llmops().metrics.snapshot()
        self.assertEqual(snap.total_prompt_tokens, 50)
        self.assertEqual(snap.total_completion_tokens, 100)

    def test_decorated_token_from_usage_metadata(self):
        """从 usage_metadata 提取 token"""

        class FakeResp:
            response_metadata = {}
            usage_metadata = {"input_tokens": 30, "output_tokens": 70}

        @trace_llm_call
        def my_llm(messages, model="", agent="", session_id=""):
            return FakeResp()

        my_llm([], model="m", agent="a", session_id="s")
        snap = get_llmops().metrics.snapshot()
        self.assertEqual(snap.total_prompt_tokens, 30)
        self.assertEqual(snap.total_completion_tokens, 70)

    def test_decorated_preserves_func_name(self):
        """装饰器保留原函数元信息"""
        @trace_llm_call
        def special_function_name(messages, model="", agent="", session_id=""):
            return "ok"

        self.assertEqual(special_function_name.__name__, "special_function_name")


# ═══════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
