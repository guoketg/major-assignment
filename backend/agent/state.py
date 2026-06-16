"""
AgentState 类型定义

LangGraph StateGraph 的状态结构，节点间通过此状态传递信息。
"""
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    """LangGraph 状态定义

    短期记忆（messages）在节点间自动流转，
    工作记忆（memory）通过本结构体传递并由 MemorySaver 持久化。
    """
    # === 核心会话 ===
    messages: List[Dict[str, Any]]
    """完整对话历史 [{"role", "content", ...}, ...]"""

    session_id: str
    """会话 ID"""

    model: str
    """当前使用的模型名（deepseek-chat / deepseek-reasoner）"""

    # === 路由控制 ===
    current_agent: str
    """当前应执行的 Agent: "supervisor" | "chat" | "research" | "innovator" | "experiment" | "report" | "synthesize" | "end" """

    sub_task_queue: List[Dict[str, Any]]
    """待执行的子任务队列 [{agent, input, status}, ...]"""

    # === 工作记忆 ===
    memory: Dict[str, Any]
    """结构化记忆体，包含:
        - papers_archive: List[PaperSummary]
        - innovation_candidates: List[InnovationPlan]
        - experiment_log: List[ExperimentRecord]
        - research_topic: str
        - baseline_model: str
        - current_focus: str
    """

    # === 输出 ===
    output_text: str
    """最终要展示给用户的文本"""

    output_artifacts: Dict[str, Any]
    """生成的文件路径 {word_doc, drawio_png, ...}"""

    reasoning_content: str
    """当前 Agent 的推理过程（思考模式专用）"""


def create_initial_state(
    session_id: str,
    model: str,
    messages: Optional[List[Dict]] = None,
) -> AgentState:
    """创建初始 AgentState"""
    return {
        "messages": messages or [],
        "session_id": session_id,
        "model": model,
        "current_agent": "supervisor",
        "sub_task_queue": [],
        "memory": {
            "papers_archive": [],
            "innovation_candidates": [],
            "experiment_log": [],
            "research_topic": "",
            "baseline_model": "",
            "current_focus": "",
        },
        "output_text": "",
        "output_artifacts": {},
        "reasoning_content": "",
    }
