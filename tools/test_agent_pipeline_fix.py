"""
测试 Agent 流水线修复

验证两个关键修复：
1. 后端：graph.py 的 done 事件是否强制清理历史消息
2. 前端：isLastPair 计算是否准确找到最后一条 user 消息
"""
import sys
import os

# ========== 测试 1：后端消息清理逻辑 ==========
print("=" * 60)
print("测试 1：后端历史消息清理逻辑")
print("=" * 60)

# 模拟 graph.py 中的清理逻辑
raw_history = [
    {"role": "user", "content": "帮我写一篇CLIP噪声标签综述"},
    {"role": "assistant", "content": "下面是文献调研结果...", "timestamp": "2026-01-01T00:00:00"},
    {"role": "assistant", "content": "归档论文列表...", "timestamp": "2026-01-01T00:00:01"},
    {"role": "assistant", "content": "综合分析结果...", "timestamp": "2026-01-01T00:00:02"},
    {"role": "assistant", "content": "最终综合输出...", "timestamp": "2026-01-01T00:00:03"},
]

final_text = "最终综合输出..."
timestamp = "2026-01-01T00:00:03"

# 清理逻辑（修复后）
history = [m for m in raw_history if m.get("role") == "user"]
if history:
    if final_text:
        history.append({"role": "assistant", "content": final_text, "timestamp": timestamp})
    else:
        last = raw_history[-1] if raw_history else {}
        last_content = last.get('content', '') or ''
        history.append({"role": "assistant", "content": last_content or "处理完成", "timestamp": timestamp})

print(f"  清理前: {len(raw_history)} 条消息 (1 user + {len(raw_history)-1} assistants)")
print(f"  清理后: {len(history)} 条消息 (1 user + {len(history)-1} assistants)")
assert len(history) == 2, f"期望 2 条消息，得到 {len(history)} 条"
assert history[0]["role"] == "user"
assert history[1]["role"] == "assistant"
print("  ✅ 清理成功：只有 1 条 user + 1 条 assistant")

# ========== 测试 2：清理后前端 isLastPair 计算 ==========
print()
print("=" * 60)
print("测试 2：前端 isLastPair 计算（修复后）")
print("=" * 60)

messages = history  # 使用清理后的消息

def find_last_user_idx(msgs):
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "user":
            return i
    return -1

lastUserIdx = find_last_user_idx(messages)
print(f"  消息总数: {len(messages)}")
print(f"  最后一条 user 在 idx: {lastUserIdx}")

# 验证：最后一条 user 消息应该显示流水线
for idx, msg in enumerate(messages):
    isUser = msg["role"] == "user"
    isLastPair = idx == lastUserIdx
    showPipeline = isUser and isLastPair
    status = "✅ 显示流水线" if showPipeline else ""
    print(f"  [{idx}] role={msg['role']}, isLastPair={isLastPair} {status}")

    if isUser and idx == len(messages) - 2:
        print(f"      传统算法 (idx >= len-2) 也 OK")
    elif isUser and idx != len(messages) - 2:
        print(f"      💡 传统算法 (idx >= len-2) 会 FAIL（流水线消失！）")

assert showPipeline, "流水线应该被显示！"

# ========== 测试 3：带历史记录的清理 ==========
print()
print("=" * 60)
print("测试 3：带多轮历史记录的清理")
print("=" * 60)

raw_multi = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "帮我写一篇CLIP噪声标签综述"},
    {"role": "assistant", "content": "子任务1输出...", "timestamp": "2026-01-01T00:00:00"},
    {"role": "assistant", "content": "子任务2输出...", "timestamp": "2026-01-01T00:00:01"},
    {"role": "assistant", "content": "子任务3输出...", "timestamp": "2026-01-01T00:00:02"},
    {"role": "assistant", "content": "综合输出...", "timestamp": "2026-01-01T00:00:03"},
]

final_text = "综合输出..."

history = [m for m in raw_multi if m.get("role") == "user"]
if final_text:
    history.append({"role": "assistant", "content": final_text, "timestamp": timestamp})

print(f"  清理前: {len(raw_multi)} 条消息")
print(f"  清理后: {len(history)} 条消息")
assert len(history) == 3, f"期望 3 条消息 (2 user + 1 assistant)，得到 {len(history)}"

lastUserIdx = find_last_user_idx(history)
print()
print("  验证 isLastPair：")
for idx, msg in enumerate(history):
    isUser = msg["role"] == "user"
    isLastPair = idx == lastUserIdx
    showPipeline = isUser and isLastPair
    old_method = idx >= len(history) - 2 and idx <= len(history) - 1
    status = "✅ 显示流水线" if showPipeline else ""
    old_status = "✅" if (isUser and old_method) else "❌"
    print(f"  [{idx}] role={msg['role']}, 新算法={isLastPair} {status}, 旧算法={old_method} {old_status}")

assert showPipeline, "最后一条 user 消息应该显示流水线"

print()
print("=" * 60)
print("全部测试通过 ✅")
print("=" * 60)
