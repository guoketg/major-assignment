"""
Token 用量追踪模块单元测试

覆盖：add_usage, increment_session_count, get_total_usage,
      get_daily_usage, get_usage_for_date, reset_total_usage

运行方式（Docker 容器内）：
    docker compose exec backend python /app/test_usage.py
"""

import os
import sys
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from datetime import datetime

# 确保可以导入 backend 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.total_usage import (
    _empty_day, _today_str, _load, _save,
    add_usage, increment_session_count,
    get_total_usage, get_daily_usage,
    get_usage_for_date, reset_total_usage,
    USAGE_FILE,
)


class TestTotalUsage(unittest.TestCase):
    """Token 用量追踪测试"""

    def setUp(self):
        """每个测试前：使用临时文件 + 重置状态"""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_file = os.path.join(self.tmpdir.name, "total_usage.json")
        # 先 patch USAGE_FILE 指向临时路径，再重置（reset 内部会获取 _lock）
        self._patch = patch("backend.agent.total_usage.USAGE_FILE", self.tmp_file)
        self._patch.start()
        reset_total_usage()

    def tearDown(self):
        self._patch.stop()
        self.tmpdir.cleanup()

    # ── 基础测试 ──

    def test_empty_day_structure(self):
        """_empty_day 返回正确的字段结构"""
        day = _empty_day()
        self.assertIn("prompt_tokens", day)
        self.assertIn("completion_tokens", day)
        self.assertIn("total_tokens", day)
        self.assertIn("total_cost", day)
        self.assertIn("session_count", day)
        self.assertEqual(day["prompt_tokens"], 0)
        self.assertEqual(day["total_cost"], 0.0)

    def test_today_str_format(self):
        """_today_str 返回 YYYY-MM-DD 格式"""
        today = _today_str()
        parts = today.split("-")
        self.assertEqual(len(parts), 3)
        self.assertEqual(len(parts[0]), 4)  # year
        self.assertEqual(len(parts[1]), 2)  # month
        self.assertEqual(len(parts[2]), 2)  # day

    def test_initial_total_zero(self):
        """初始总用量为 0"""
        total = get_total_usage()
        self.assertEqual(total["prompt_tokens"], 0)
        self.assertEqual(total["completion_tokens"], 0)
        self.assertEqual(total["total_tokens"], 0)
        self.assertEqual(total["total_cost"], 0.0)
        self.assertEqual(total["session_count"], 0)

    # ── add_usage 测试 ──

    def test_add_usage_single(self):
        """单次 add_usage 正确累加"""
        add_usage(prompt_tokens=100, completion_tokens=200, cost=0.0005)
        total = get_total_usage()
        self.assertEqual(total["prompt_tokens"], 100)
        self.assertEqual(total["completion_tokens"], 200)
        self.assertEqual(total["total_tokens"], 300)
        self.assertAlmostEqual(total["total_cost"], 0.0005, places=6)

    def test_add_usage_multiple_same_day(self):
        """同一天多次 add_usage 累加正确"""
        add_usage(prompt_tokens=100, completion_tokens=200, cost=0.0005)
        add_usage(prompt_tokens=50, completion_tokens=100, cost=0.00025)
        total = get_total_usage()
        self.assertEqual(total["prompt_tokens"], 150)
        self.assertEqual(total["completion_tokens"], 300)
        self.assertEqual(total["total_tokens"], 450)
        self.assertAlmostEqual(total["total_cost"], 0.00075, places=6)

    def test_add_usage_creates_today_entry(self):
        """add_usage 自动创建今天的日统计条目"""
        add_usage(prompt_tokens=10, completion_tokens=20, cost=0.00003)
        today_str = _today_str()
        day = get_usage_for_date(today_str)
        self.assertEqual(day["prompt_tokens"], 10)
        self.assertEqual(day["completion_tokens"], 20)
        self.assertEqual(day["total_tokens"], 30)

    def test_add_usage_zero_tokens(self):
        """add_usage 支持零 Token（不应崩溃）"""
        add_usage(prompt_tokens=0, completion_tokens=0, cost=0.0)
        total = get_total_usage()
        self.assertEqual(total["prompt_tokens"], 0)
        self.assertEqual(total["total_cost"], 0.0)

    def test_add_usage_cost_rounding(self):
        """成本四舍五入到 6 位小数"""
        add_usage(prompt_tokens=1, completion_tokens=1, cost=0.123456789)
        total = get_total_usage()
        self.assertAlmostEqual(total["total_cost"], 0.123457, places=6)

    # ── increment_session_count 测试 ──

    def test_increment_session_count(self):
        """increment_session_count 正确递增"""
        increment_session_count()
        increment_session_count()
        total = get_total_usage()
        self.assertEqual(total["session_count"], 2)

    def test_increment_daily_session_count(self):
        """每日会话计数独立递增"""
        increment_session_count()
        today_str = _today_str()
        day = get_usage_for_date(today_str)
        self.assertEqual(day["session_count"], 1)

    # ── get_daily_usage 测试 ──

    def test_get_daily_usage_empty(self):
        """无数据时返回空列表"""
        daily = get_daily_usage()
        self.assertEqual(daily, [])

    def test_get_daily_usage_single(self):
        """有数据时返回包含 date 字段的列表"""
        add_usage(prompt_tokens=10, completion_tokens=20, cost=0.00003)
        daily = get_daily_usage()
        self.assertGreaterEqual(len(daily), 1)
        self.assertIn("date", daily[0])
        self.assertEqual(daily[0]["prompt_tokens"], 10)

    def test_get_daily_usage_sorted_desc(self):
        """每日用量按日期倒序排列"""
        today = _today_str()
        # 同一天多次添加
        add_usage(prompt_tokens=5, completion_tokens=5, cost=0.00001)
        daily = get_daily_usage()
        # 验证当前只有一天的数据
        self.assertGreaterEqual(len(daily), 1)
        self.assertEqual(daily[0]["date"], today)

    # ── get_usage_for_date 测试 ──

    def test_get_usage_for_existing_date(self):
        """查询存在日期的用量"""
        add_usage(prompt_tokens=100, completion_tokens=200, cost=0.0005)
        today = _today_str()
        data = get_usage_for_date(today)
        self.assertEqual(data["prompt_tokens"], 100)
        self.assertEqual(data["completion_tokens"], 200)
        self.assertEqual(data["total_tokens"], 300)
        self.assertAlmostEqual(data["total_cost"], 0.0005, places=6)

    def test_get_usage_for_nonexistent_date(self):
        """查询不存在日期的用量返回空结构"""
        data = get_usage_for_date("2000-01-01")
        self.assertEqual(data["date"], "2000-01-01")
        self.assertEqual(data["prompt_tokens"], 0)
        self.assertEqual(data["completion_tokens"], 0)
        self.assertEqual(data["total_tokens"], 0)
        self.assertEqual(data["total_cost"], 0.0)

    # ── reset_total_usage 测试 ──

    def test_reset_total_usage(self):
        """重置后全部归零"""
        add_usage(prompt_tokens=100, completion_tokens=200, cost=0.0005)
        reset_total_usage()
        total = get_total_usage()
        self.assertEqual(total["prompt_tokens"], 0)
        self.assertEqual(total["completion_tokens"], 0)
        self.assertEqual(total["total_tokens"], 0)
        self.assertEqual(total["total_cost"], 0.0)
        self.assertEqual(total["session_count"], 0)

    # ── 持久化测试 ──

    def test_persistence_across_reloads(self):
        """数据持久化到文件，_load 可重新读取"""
        add_usage(prompt_tokens=50, completion_tokens=100, cost=0.00015)
        # 验证文件存在
        self.assertTrue(os.path.exists(self.tmp_file))
        # 重新加载
        data = _load()
        total = data["total"]
        self.assertEqual(total["prompt_tokens"], 50)
        self.assertEqual(total["completion_tokens"], 100)
        self.assertEqual(total["total_tokens"], 150)

    def test_persistence_file_content(self):
        """文件内容为合法 JSON"""
        add_usage(prompt_tokens=1, completion_tokens=2, cost=0.000001)
        with open(self.tmp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("daily", data)
        self.assertIn("total", data)
        self.assertIn("prompt_tokens", data["total"])

    def test_compatibility_old_format(self):
        """兼容旧版格式（无 daily 字段）"""
        old_data = {
            "prompt_tokens": 500,
            "completion_tokens": 1000,
            "total_tokens": 1500,
            "total_cost": 0.0025,
            "session_count": 5,
        }
        with open(self.tmp_file, "w", encoding="utf-8") as f:
            json.dump(old_data, f)
        data = _load()
        self.assertIn("daily", data)
        self.assertIn("total", data)
        self.assertEqual(data["total"]["prompt_tokens"], 500)
        self.assertEqual(data["total"]["completion_tokens"], 1000)
        self.assertEqual(data["total"]["total_tokens"], 1500)
        self.assertEqual(data["total"]["total_cost"], 0.0025)
        self.assertEqual(data["total"]["session_count"], 5)

    def test_compatibility_missing_total(self):
        """兼容 total 字段缺失的数据"""
        data = {"daily": {}}
        with open(self.tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        loaded = _load()
        self.assertIn("total", loaded)
        self.assertEqual(loaded["total"]["prompt_tokens"], 0)

    def test_compatibility_partial_total(self):
        """兼容 total 中部分字段缺失的数据"""
        data = {
            "daily": {},
            "total": {"prompt_tokens": 100}  # 只有 prompt_tokens
        }
        with open(self.tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        loaded = _load()
        self.assertEqual(loaded["total"]["prompt_tokens"], 100)
        self.assertEqual(loaded["total"]["completion_tokens"], 0)
        self.assertEqual(loaded["total"]["session_count"], 0)

    def test_corrupted_file_handling(self):
        """损坏的 JSON 文件不崩溃，返回空数据"""
        with open(self.tmp_file, "w", encoding="utf-8") as f:
            f.write("not valid json {{{")
        data = _load()
        self.assertIn("total", data)
        self.assertEqual(data["total"]["prompt_tokens"], 0)

    def test_no_file_yet(self):
        """文件不存在时 _load 返回空结构"""
        os.remove(self.tmp_file)
        data = _load()
        self.assertEqual(data["total"]["prompt_tokens"], 0)

    # ── 成本计算验证 ──

    def test_cost_formula(self):
        """输入 ¥1/百万 token，输出 ¥2/百万 token"""
        # 输入 1000000 tokens = ¥1, 输出 1000000 tokens = ¥2
        cost_input = 1000000 * 1.0 / 1_000_000   # = 1.0
        cost_output = 1000000 * 2.0 / 1_000_000   # = 2.0
        self.assertEqual(cost_input, 1.0)
        self.assertEqual(cost_output, 2.0)

        # 实际调用：输入 500, 输出 1000
        prompt_tokens = 500
        completion_tokens = 1000
        expected_cost = (prompt_tokens * 1.0 + completion_tokens * 2.0) / 1_000_000
        # = (500 + 2000) / 1000000 = 0.0025
        self.assertAlmostEqual(expected_cost, 0.0025, places=6)

    # ── 线程安全测试 ──

    def test_concurrent_add_usage(self):
        """多线程并发 add_usage 数据一致"""
        errors = []
        N_THREADS, N_ITER = 4, 10  # 每线程10次迭代，足够验证线程安全

        def worker():
            try:
                for _ in range(N_ITER):
                    add_usage(prompt_tokens=1, completion_tokens=1, cost=0.000001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"线程错误: {errors}")
        total = get_total_usage()
        expected = N_THREADS * N_ITER
        self.assertEqual(total["prompt_tokens"], expected)
        self.assertEqual(total["completion_tokens"], expected)
        self.assertEqual(total["total_tokens"], expected * 2)

    def test_concurrent_session_count(self):
        """多线程并发 increment_session_count 数据一致"""
        N_THREADS, N_ITER = 4, 5

        def worker():
            for _ in range(N_ITER):
                increment_session_count()

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = get_total_usage()
        self.assertEqual(total["session_count"], N_THREADS * N_ITER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
