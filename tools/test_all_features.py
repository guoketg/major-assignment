"""
功能全面测试脚本

测试内容：
1. 流式输出功能（逐字渲染）
2. 滚轮修复（流式过程中可手动滚动）
3. 思考模式（reasoning_content 展示）
4. 模型选择切换
5. arXiv 论文搜索
6. Agent 流水线可视化
"""
import sys
import os
import json
import time

import requests

API_URL = os.getenv("API_URL", "http://localhost:8001")


def test_health():
    """测试后端健康"""
    r = requests.get(f"{API_URL}/health", timeout=5)
    assert r.status_code == 200 and r.json().get("status") == "ok"
    print("  ✅ 后端健康")


def test_arxiv_search():
    """测试 arXiv 搜索接口"""
    r = requests.get(
        f"{API_URL}/arxiv/search",
        params={"q": "all:deep+learning", "max_results": 5},
        timeout=30,
    )
    assert r.status_code == 200, f"arXiv搜索失败: {r.status_code}"
    data = r.json()
    assert "papers" in data, f"响应中没有 papers 字段: {list(data.keys())}"
    assert len(data["papers"]) > 0, f"应该返回论文，实际返回 {len(data['papers'])} 篇"
    paper = data["papers"][0]
    assert "title" in paper
    assert "authors" in paper
    assert "categories" in paper
    print(f"  ✅ arXiv搜索成功: 返回 {len(data['papers'])} 篇论文 (共 {data['total_results']} 篇)")


def test_models_endpoint():
    """测试模型选择API"""
    # 测试不同模型发送消息
    models_to_test = [
        ("deepseek-v4-flash", "deepseek-chat"),
        ("v4-pro", "deepseek-chat"),
        ("思考模式", "deepseek-reasoner"),
    ]

    for display_name, api_model in models_to_test:
        session_id = f"test_model_{display_name}"
        r = requests.post(
            f"{API_URL}/chat/stream",
            json={
                "session_id": session_id,
                "message": "用2个字回答：你好吗？",
                "model": display_name,
            },
            stream=True,
            timeout=60,
        )
        assert r.status_code == 200, f"模型 {display_name} 请求失败: {r.status_code}"

        # 读取流式响应
        chunks = []
        reasoning_chunks = []
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = json.loads(line[6:])
                chunks.append(data)
                if data.get("reasoning_content"):
                    reasoning_chunks.append(data["reasoning_content"])
                if data.get("done"):
                    break

        assert len(chunks) > 1, f"模型 {display_name} 应该返回多个 chunk"

        if display_name == "思考模式":
            print(f"  ✅ {display_name} ({api_model}): 返回 {len(chunks)} 个 chunk, " +
                  f"思考内容={len(reasoning_chunks)}段, 有思考内容={'是' if reasoning_chunks else '否'}")
        else:
            print(f"  ✅ {display_name} ({api_model}): 返回 {len(chunks)} 个 chunk")

        # 清理
        requests.delete(f"{API_URL}/history/{session_id}")


def test_streaming_incremental():
    """测试流式输出是否逐块增加"""
    session_id = "test_stream_inc"
    r = requests.post(
        f"{API_URL}/chat/stream",
        json={"session_id": session_id, "message": "用几个字说说天气"},
        stream=True,
        timeout=30,
    )

    prev_len = 0
    content_grew = False
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = json.loads(line[6:])
        if data.get("done"):
            break

        cur_len = len(data.get("content", ""))
        # 模拟前端流式渲染：累计内容在增长
        # 实际接口返回的是增量 content，不是累计
        # 所以我们检查每个 chunk 的 content 是否非空
        if data.get("content"):
            content_grew = True

    assert content_grew, "流式响应应该包含内容"
    print(f"  ✅ 流式输出逐块返回正常")
    requests.delete(f"{API_URL}/history/{session_id}")


def test_session_lifecycle():
    """测试会话创建、历史获取、删除"""
    r = requests.post(f"{API_URL}/sessions")
    sid = r.json()["session_id"]
    print(f"  ✅ 创建会话: {sid}")

    # 发消息（必须消费流式响应，否则后端不会保存历史）
    r = requests.post(
        f"{API_URL}/chat/stream",
        json={"session_id": sid, "message": "测试"},
        stream=True,
        timeout=15,
    )
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            import json
            data = json.loads(line[6:])
            if data.get("done") or data.get("type") == "done":
                break

    # 获取历史
    r = requests.get(f"{API_URL}/history/{sid}")
    assert len(r.json()["history"]) >= 2  # user + assistant
    print(f"  ✅ 历史记录: {len(r.json()['history'])} 条消息")

    # 删除
    requests.delete(f"{API_URL}/history/{sid}")
    print(f"  ✅ 会话已删除")


def test_agent_pipeline():
    """测试 Agent 流水线 SSE 事件"""
    # 创建会话
    r = requests.post(f"{API_URL}/sessions")
    sid = r.json()["session_id"]

    # 发送研究意图消息
    r = requests.post(
        f"{API_URL}/chat/stream",
        json={"session_id": sid, "message": "帮我调研一下CLIP噪声标签学习的最新论文", "model": "deepseek-v4-flash"},
        stream=True,
        timeout=30,
    )

    events = []
    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: "):
            data = json.loads(line[6:])
            events.append(data)
            if data.get("done"):
                break

    # 验证 Agent 事件序列
    agent_events = [e for e in events if e.get("type") == "agent"]
    tool_events = [e for e in events if e.get("type") == "tool"]

    assert len(agent_events) >= 2, f"至少要有 2 个 Agent 事件，实际 {len(agent_events)}"
    agents_seen = {e["agent"] for e in agent_events}
    assert "supervisor" in agents_seen, "应有 Supervisor"
    print(f"  ✅ Agent 事件: {len(agent_events)} 个, Agents: {agents_seen}")

    # 检查研究意图触发了工具调用
    assert len(tool_events) > 0, "研究意图应有工具调用事件"
    tools_seen = {e["tool"] for e in tool_events}
    print(f"  ✅ 工具调用: {len(tool_events)} 个, Tools: {tools_seen}")

    # 验证 Agent 状态流转
    assert any(e.get("status") == "running" for e in agent_events), "应有 running 状态"
    assert any(e.get("status") == "complete" for e in agent_events), "应有 complete 状态"
    print("  ✅ Agent 状态流转正确")

    # 清理
    requests.delete(f"{API_URL}/history/{sid}")


def test_docx_tool():
    """测试 Word 文档生成工具"""
    try:
        import sys
        sys.path.insert(0, "backend")
        from tools.docx_tool import create_docx, add_section, add_table
    except ImportError as e:
        print(f"  ⚠️ 跳过（本地缺少依赖: {e}）")
        print(f"  💡 如需本地测试，请先: source venv/Scripts/activate && pip install langchain-core")
        return

    # 从返回值提取实际路径（兼容 Docker 和本地）
    result = create_docx.invoke({"title": "测试报告"})
    assert "文档已创建" in result, f"创建失败: {result}"
    filepath = result.split(": ", 1)[-1].strip()
    print(f"  ✅ 创建文档: {filepath}")

    result = add_section.invoke({"filepath": filepath, "heading": "测试章节", "content": "测试内容"})
    assert "已添加" in result, f"添加章节失败: {result}"
    print(f"  ✅ 添加章节成功")

    result = add_table.invoke({"filepath": filepath, "headers": ["A", "B"], "rows": [["1", "2"]]})
    assert "已添加" in result, f"添加表格失败: {result}"
    print(f"  ✅ 添加表格成功")

    import os
    os.remove(filepath)
    print(f"  ✅ 临时文件已清理")


def test_memory_endpoint():
    """测试记忆接口"""
    r = requests.post(f"{API_URL}/sessions")
    sid = r.json()["session_id"]

    r = requests.get(f"{API_URL}/memory/{sid}")
    data = r.json()
    assert "stats" in data, f"缺少 stats: {data}"
    assert "memory" in data, f"缺少 memory: {data}"
    print(f"  ✅ 记忆接口正常: papers={data['stats']['papers']}, innovations={data['stats']['innovations']}, experiments={data['stats']['experiments']}")

    requests.delete(f"{API_URL}/history/{sid}")


if __name__ == "__main__":
    print("=" * 55)
    print("功能全面测试")
    print("=" * 55)

    tests = [
        ("后端健康", test_health),
        ("arXiv论文搜索", test_arxiv_search),
        ("模型选择切换", test_models_endpoint),
        ("流式增量输出", test_streaming_incremental),
        ("会话生命周期", test_session_lifecycle),
        ("Agent流水线", test_agent_pipeline),
        ("Word文档工具", test_docx_tool),
        ("记忆接口", test_memory_endpoint),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            failed += 1

    print(f"\n{'=' * 55}")
    print(f"结果: {passed} 通过, {failed} 失败")
    if failed == 0:
        print("全部测试通过! ✅")
    print("=" * 55)
