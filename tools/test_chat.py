"""
多轮对话前后端功能测试脚本

测试内容:
1. 后端API健康检查
2. 多轮对话功能
3. 历史记录管理

运行方式:
python tools/test_chat.py
"""
import requests
import time
import sys

BASE_URL = "http://localhost:8000"


def test_health_check():
    """测试1: 健康检查"""
    print("\n=== 测试1: 健康检查 ===")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        print(f"✓ 健康检查通过: {data}")
        return True
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return False


def test_multi_turn_conversation():
    """测试2: 多轮对话"""
    print("\n=== 测试2: 多轮对话 ===")
    session_id = f"test_session_{int(time.time())}"

    try:
        # 第一轮对话
        print("发送第1条消息...")
        r1 = requests.post(
            f"{BASE_URL}/chat",
            json={"session_id": session_id, "message": "你好，你叫什么名字？"},
            timeout=30
        )
        assert r1.status_code == 200
        data1 = r1.json()
        print(f"✓ 第1轮回复: {data1['reply'][:50]}...")
        assert len(data1['history']) == 2  # user + assistant

        # 第二轮对话
        print("发送第2条消息...")
        r2 = requests.post(
            f"{BASE_URL}/chat",
            json={"session_id": session_id, "message": "再说一次你的名字"},
            timeout=30
        )
        assert r2.status_code == 200
        data2 = r2.json()
        print(f"✓ 第2轮回复: {data2['reply'][:50]}...")
        assert len(data2['history']) == 4  # 2 user + 2 assistant

        # 验证上下文关联
        assert "历史" in str(data2['history']) or "名字" in str(data2['history']).lower()
        print("✓ 多轮对话测试通过，AI能够记住上下文")
        return True

    except Exception as e:
        print(f"✗ 多轮对话测试失败: {e}")
        return False


def test_get_history():
    """测试3: 获取历史记录"""
    print("\n=== 测试3: 获取历史记录 ===")
    session_id = f"test_session_{int(time.time())}"

    try:
        # 先发送消息
        requests.post(
            f"{BASE_URL}/chat",
            json={"session_id": session_id, "message": "测试消息"},
            timeout=30
        )

        # 获取历史
        response = requests.get(f"{BASE_URL}/history/{session_id}", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert len(data['history']) >= 2
        print(f"✓ 历史记录获取成功，共 {len(data['history'])} 条消息")
        return True

    except Exception as e:
        print(f"✗ 获取历史记录失败: {e}")
        return False


def test_delete_history():
    """测试4: 删除历史记录"""
    print("\n=== 测试4: 删除历史记录 ===")
    session_id = f"test_session_{int(time.time())}"

    try:
        # 先发送消息
        requests.post(
            f"{BASE_URL}/chat",
            json={"session_id": session_id, "message": "测试消息"},
            timeout=30
        )

        # 删除历史
        response = requests.delete(f"{BASE_URL}/history/{session_id}", timeout=5)
        assert response.status_code == 200

        # 验证已删除
        history_response = requests.get(f"{BASE_URL}/history/{session_id}", timeout=5)
        data = history_response.json()
        assert len(data['history']) == 0
        print("✓ 历史记录删除成功")
        return True

    except Exception as e:
        print(f"✗ 删除历史记录失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始运行多轮对话前后端测试")
    print("=" * 50)

    # 先检查后端是否运行
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"\n✗ 错误: 无法连接到后端服务 ({BASE_URL})")
        print("请确保后端服务已启动:")
        print("  cd backend")
        print("  venv\\Scripts\\activate  (Windows)")
        print("  pip install -r requirements.txt")
        print("  python main.py")
        sys.exit(1)

    results = {
        "健康检查": test_health_check(),
        "多轮对话": test_multi_turn_conversation(),
        "获取历史": test_get_history(),
        "删除历史": test_delete_history(),
    }

    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n✓ 所有测试通过!")
        sys.exit(0)
    else:
        print("\n✗ 部分测试失败")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
