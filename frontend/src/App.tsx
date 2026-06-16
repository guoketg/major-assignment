import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  reasoning_content?: string | null;
}

interface SSEData {
  content: string;
  reasoning_content?: string;
  done: boolean;
  history?: Message[];
  error?: string;
}

interface Session {
  session_id: string;
  title: string;
  message_count: number;
  last_update: string | null;
}

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// 可用模型列表
const MODELS = [
  { id: 'deepseek-v4-flash', label: 'DeepSeek V4 Flash', desc: '快速响应' },
  { id: 'v4-pro', label: 'V4 Pro', desc: '更强能力' },
  { id: '思考模式', label: '思考模式', desc: '深度推理（reasoner）' },
];

// arXiv 论文接口
interface ArxivPaper {
  id: string;
  title: string;
  summary: string;
  authors: string[];
  published: string;
  categories: string[];
  pdf_link: string;
  links: string[];
}

interface ArxivResponse {
  total_results: number;
  start_index: number;
  items_per_page: number;
  papers: ArxivPaper[];
}

// === Agent 可视化类型定义 ===
interface AgentTrace {
  agent: string;
  label: string;
  status: 'running' | 'complete' | 'error';
  routed_to?: string;
}

interface ToolCallTrace {
  agent: string;
  tool: string;
  label: string;
  status: 'start' | 'complete';
  toolId: string;  // 唯一 ID，用于 React key
  output?: string; // 工具返回结果（如搜索内容）
}

// 从工具输出中提取 URL 和关联文本
const extractReferences = (toolCalls: ToolCallTrace[]): {url: string; title: string; source: string}[] => {
  const refs: {url: string; title: string; source: string}[] = [];
  const seen = new Set<string>();
  for (const tc of toolCalls) {
    if (!tc.output) continue;
    const source = tc.tool === 'search_arxiv' ? 'arXiv' : tc.tool === 'web_search' ? 'Web' : tc.tool;

    // 匹配 URL 模式
    const urlRegex = /https?:\/\/[^\s<>"]+/g;
    const urls = tc.output.match(urlRegex) || [];
    for (const url of urls) {
      if (seen.has(url)) continue;
      seen.add(url);
      const lines = tc.output.split('\n');
      let title = url;
      for (const line of lines) {
        if (line.includes(url)) {
          const parts = line.split(url);
          const before = parts[0].replace(/^[-\s*#]*\s*/, '').trim();
          if (before && before.length > 3 && before.length < 120) {
            title = before;
          }
          break;
        }
      }
      refs.push({ url, title: title.length > 100 ? title.slice(0, 100) + '...' : title, source });
    }

    // arXiv 工具：从 ID 生成链接
    if (tc.tool === 'search_arxiv') {
      const idRegex = /^ID:\s*(\d{4}\.\d{4,5})/gm;
      let match;
      while ((match = idRegex.exec(tc.output)) !== null) {
        const arxivId = match[1];
        const arxivUrl = `https://arxiv.org/abs/${arxivId}`;
        if (seen.has(arxivUrl)) continue;
        seen.add(arxivUrl);
        // 提取论文标题
        const lines = tc.output.split('\n');
        let title = `arXiv:${arxivId}`;
        for (let i = 0; i < lines.length; i++) {
          if (lines[i].includes(`ID: ${arxivId}`) || lines[i].includes(`ID:${arxivId}`)) {
            // 往前找标题（通常在 "标题:" 行）
            for (let j = i - 1; j >= Math.max(0, i - 8); j--) {
              const tMatch = lines[j].match(/^标题:\s*(.+)/);
              if (tMatch) { title = tMatch[1].trim(); break; }
            }
            break;
          }
        }
        refs.push({ url: arxivUrl, title, source: 'arXiv' });
      }
    }
  }
  return refs;
};

// 预处理 AI 回复内容：将 arXiv ID 转为可点击链接
const linkifyContent = (content: string): string => {
  let result = content;
  // 匹配 arXiv:XXXX.XXXXX 或 arXiv:XXXX.XXXXXvX 格式
  result = result.replace(
    /arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)/gi,
    '[arXiv:$1](https://arxiv.org/abs/$1)'
  );
  // 匹配孤立 arXiv ID（如 2604.08571），但避免括号和已经转换的
  result = result.replace(
    /(?<!\[)(?<!\d)(\d{4}\.\d{4,5})(?!\d)(?!\])/g,
    (match) => `[${match}](https://arxiv.org/abs/${match})`
  );
  return result;
};

// Agent 图标映射
const AGENT_ICONS: { [key: string]: string } = {
  supervisor: '🤖',
  chat_agent: '💬',
  research_agent: '🔍',
  innovator_agent: '💡',
  experiment_agent: '🧪',
  planner_agent: '📋',
  reporter: '📊',
  synthesizer: '📋',
};

// Agent 颜色映射
const AGENT_COLORS: { [key: string]: string } = {
  supervisor: '#6366f1',
  chat_agent: '#22c55e',
  research_agent: '#3b82f6',
  innovator_agent: '#f59e0b',
  experiment_agent: '#ef4444',
  planner_agent: '#8b5cf6',
  reporter: '#8b5cf6',
  synthesizer: '#06b6d4',
};

// 前端 Agent 选择选项
const AGENT_OPTIONS = [
  { id: 'auto', label: '🤖 自动', desc: 'Supervisor 自动路由' },
  { id: 'chat', label: '💬 对话', desc: '普通对话' },
  { id: 'research', label: '🔍 调研', desc: '论文检索与分析' },
  { id: 'innovate', label: '💡 创新', desc: '构思创新方案' },
  { id: 'experiment', label: '🧪 实验', desc: '实验设计与分析' },
];

function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [streamingReasoning, setStreamingReasoning] = useState('');
  const [reasoningDone, setReasoningDone] = useState(false);

  // 模型选择
  const [selectedModel, setSelectedModel] = useState(MODELS[0].id);

  // Agent 手动选择（"auto" 表示由 Supervisor 自动路由）
  const [selectedAgent, setSelectedAgent] = useState('auto');

  // 页面模式
  const [mode, setMode] = useState<'chat' | 'arxiv' | 'memory' | 'tools'>('chat');

  // Agent 可视化状态
  const [agentPipeline, setAgentPipeline] = useState<AgentTrace[]>([]);
  const [toolCalls, setToolCalls] = useState<ToolCallTrace[]>([]);
  const [showAgentPipeline, setShowAgentPipeline] = useState(false);
  const [pipelineCollapsed, setPipelineCollapsed] = useState(false);
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set()); // 展开了输出结果的工具 ID
  const autoExpandedRef = useRef(false); // 首轮自动展开工具输出
  // 子任务规划与进度
  const [subTaskPlan, setSubTaskPlan] = useState<any[]>([]);
  const subTaskPlanRef = useRef<any[]>([]);  // 同步访问，绕过 React 异步状态更新延迟

  // 记忆状态
  const [memoryData, setMemoryData] = useState<{memory?: any; stats?: {papers: number; innovations: number; experiments: number}} | null>(null);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryError, setMemoryError] = useState('');

  // arXiv 搜索
  const [arxivQuery, setArxivQuery] = useState('');
  const [arxivResults, setArxivResults] = useState<ArxivPaper[]>([]);
  const [arxivTotal, setArxivTotal] = useState(0);
  const [arxivLoading, setArxivLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageAreaRef = useRef<HTMLDivElement>(null);
  const userScrolledUp = useRef(false);

  const scrollToBottom = (smooth = true) => {
    (messagesEndRef.current as HTMLElement)?.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto' } as ScrollIntoViewOptions);
  };

  // 加载会话列表
  const loadSessions = async () => {
    try {
      const response = await fetch(`${API_URL}/sessions`);
      const data = await response.json();
      setSessions(data.sessions || []);
    } catch (error) {
      console.error('加载会话列表失败:', error);
    }
  };

  // 加载指定会话的历史（含流水线数据）
  const loadSessionHistory = async (sessionId: string) => {
    try {
      const response = await fetch(`${API_URL}/history/${sessionId}`);
      const data = await response.json();
      setMessages(data.history || []);
      setCurrentSessionId(sessionId);
      setStreamingContent('');

      // 从 meta 中恢复流水线数据（持久化到文件）
      const meta = data.meta || {};
      if (meta.agent_pipeline && meta.agent_pipeline.length > 0) {
        setAgentPipeline(meta.agent_pipeline);
        setToolCalls(meta.tool_calls || []);
        setSubTaskPlan(meta.sub_task_plan || []);
        setShowAgentPipeline(true);
        // keep expanded after done
      } else {
        // 没有流水线数据时重置
        setAgentPipeline([]);
        setToolCalls([]);
        setSubTaskPlan([]);
        setExpandedTools(new Set());
        setShowAgentPipeline(false);
      }

      setTimeout(() => {
        if (meta.agent_pipeline && meta.agent_pipeline.length > 0) {
          const el = messageAreaRef.current;
          if (el) el.scrollTop = 0;
        } else {
          scrollToBottom();
        }
      }, 100);
    } catch (error) {
      console.error('加载历史失败:', error);
    }
  };

  // 创建新会话，返回 sessionId
  const createNewSession = async (): Promise<string | undefined> => {
    try {
      const response = await fetch(`${API_URL}/sessions`, { method: 'POST' });
      const data = await response.json();
      setMessages([]);
      setCurrentSessionId(data.session_id);
      setStreamingContent('');
      await loadSessions();
      return data.session_id;
    } catch (error) {
      console.error('创建会话失败:', error);
      return undefined;
    }
  };

  // 删除会话
  const deleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await fetch(`${API_URL}/history/${sessionId}`, { method: 'DELETE' });
      if (currentSessionId === sessionId) {
        setCurrentSessionId('');
        setMessages([]);
        setStreamingContent('');
      }
      await loadSessions();
    } catch (error) {
      console.error('删除会话失败:', error);
    }
  };
  useEffect(() => {
    loadSessions();
  }, []);

  // 滚动追踪：用户手动向上滚动后停止自动滚到底部
  const handleScroll = useCallback(() => {
    const el = messageAreaRef.current;
    if (!el) return;
    const threshold = 150;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
    userScrolledUp.current = !isAtBottom;
  }, []);

  const scrollIfAtBottom = useCallback(() => {
    // 流水线已在消息区外部固定显示，只需正常滚到底部查看最新内容
    const el = messageAreaRef.current;
    if (!el) return;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 150;
    if (isAtBottom) {
      (messagesEndRef.current as HTMLElement)?.scrollIntoView({ behavior: 'auto' } as ScrollIntoViewOptions);
    }
  }, []);

  // 新消息完成或切换历史时滚动
  // 如果流水线可见（即将有新 Agent 数据），滚动到顶部展示流水线；否则滚到底部
  useEffect(() => {
    if (showAgentPipeline) {
      // 流水线即将出现，滚动到顶部让用户看到流水线
      requestAnimationFrame(() => {
        const el = messageAreaRef.current;
        if (el) el.scrollTop = 0;
      });
    } else {
      scrollToBottom(true);
    }
    userScrolledUp.current = false;
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    let sessionId = currentSessionId;
    if (!sessionId) {
      // createNewSession 现在直接返回 sessionId，避免 React 异步状态更新问题
      const newId = await createNewSession();
      if (!newId) return;
      sessionId = newId;
      setCurrentSessionId(newId);
    }

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setStreamingContent('');
    setStreamingReasoning('');
    setReasoningDone(false);
    userScrolledUp.current = false;

    // 重置 Agent 可视化状态
    setAgentPipeline([]);
    setToolCalls([]);
    setSubTaskPlan([]);
    setExpandedTools(new Set());
    setShowAgentPipeline(true);
    // pipelines expanded by default

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: input, model: selectedModel, agent: selectedAgent }),
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullContent = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));

                // === 处理 Agent 事件（新格式）===
                if (data.type === 'agent') {
                  // Agent 状态更新事件
                  setAgentPipeline(prev => {
                    const wasEmpty = prev.length === 0;
                    const existing = prev.findIndex(a => a.agent === data.agent);
                    if (existing >= 0) {
                      const updated = [...prev];
                      updated[existing] = { ...updated[existing], status: data.status, routed_to: data.routed_to };
                      return updated;
                    }
                    // 首个 Agent 出现时，滚动到顶部展示流水线
                    if (wasEmpty) {
                      requestAnimationFrame(() => {
                        const el = messageAreaRef.current;
                        if (el) el.scrollTop = 0;
                      });
                    }
                    return [...prev, { agent: data.agent, label: data.label, status: data.status, routed_to: data.routed_to }];
                  });
                  continue;
                }

                if (data.type === 'tool') {
                  // 工具调用事件 — 使用唯一 ID 避免 React key 重复
                  setToolCalls(prev => {
                    if (data.status === 'start') {
                      const toolId = `tool-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
                      return [...prev, { agent: data.agent, tool: data.tool, label: data.label, status: 'start', toolId }];
                    }
                    // 完成时：标记最后一个匹配的 start，并保存输出
                    const idx = prev.length - 1 - [...prev].reverse().findIndex(
                      t => t.tool === data.tool && t.agent === data.agent && t.status === 'start'
                    );
                    if (idx >= 0 && idx < prev.length) {
                      const updated = [...prev];
                      updated[idx] = { ...updated[idx], status: 'complete', output: data.output || '' };
                      // 自动展开工具输出
                      const toolId = updated[idx].toolId;
                      if (data.output) {
                        setTimeout(() => setExpandedTools(prev => new Set([...prev, toolId])), 50);
                      }
                      return updated;
                    }
                    return prev;
                  });
                  continue;
                }

                if (data.type === 'agent_thinking') {
                  // Agent 开始思考 — 无需处理，Content 即将到来
                  continue;
                }

                if (data.type === 'plan') {
                  // 子任务规划事件（来自 Planner Agent）
                  if (data.sub_tasks) {
                    subTaskPlanRef.current = data.sub_tasks;
                    setSubTaskPlan(data.sub_tasks);
                  }
                  continue;
                }

                if (data.type === 'subtask_progress') {
                  // 子任务进度更新
                  if (data.sub_tasks) {
                    subTaskPlanRef.current = data.sub_tasks;
                    setSubTaskPlan(data.sub_tasks);
                  }
                  continue;
                }

                if (data.type === 'error') {
                  console.error('流式错误:', data.error);
                  break;
                }

                // === 处理内容事件 ===
                if (data.type === 'content' || 'content' in data) {
                  const contentStr = data.content || '';
                  if (!data.done) {
                    // 跳过中间 agent 内容：用 ref 绕过 React 异步状态更新（setState 不会立即更新闭包中的变量）
                    const hasPlan = subTaskPlanRef.current.length > 0;
                    if (data.agent === 'planner_agent' || (hasPlan && data.agent && data.agent !== 'synthesizer')) {
                      continue;
                    }
                    // _final 标记表示这是完整合成文本（来自 on_chain_end 事件），替换全部内容
                    // 普通流式 token（含 synthesizer 的 on_chat_model_stream）总是追加，保证流式效果
                    if (data._final) {
                      fullContent = contentStr;
                    } else {
                      fullContent += contentStr;
                    }
                    setStreamingContent(fullContent);

                    // 处理思考内容
                    if (data.reasoning_content) {
                      setStreamingReasoning(prev => prev + data.reasoning_content);
                    }
                    if (contentStr) {
                      setReasoningDone(true);
                    }

                    // 让出事件循环，让 React 有机会渲染每个块
                    await new Promise(resolve => setTimeout(resolve, 0));
                    // 渲染后检查滚动
                    scrollIfAtBottom();
                  }
                }

                // === 完成事件 ===
                if (data.done || data.type === 'done') {
                  if (data.history) {
                    setMessages(data.history);
                    await loadSessions();
                  }
                  // 保留 agent 工作流可见但折叠 — 不要让它消失
                  setShowAgentPipeline(true);
                  // keep expanded after done
                  // 清除流式内容，避免与 done 中的历史消息重复
                  setStreamingContent('');
                  setSelectedAgent('auto');
                  // 流水线在消息区外部：滚动到顶部展示流水线，用户可自行下滚查看回复
                  requestAnimationFrame(() => {
                    const el = messageAreaRef.current;
                    if (el) el.scrollTop = 0;
                  });
                  break;
                }
              } catch (e) {
                // 忽略解析错误
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('发送失败:', error);
      alert('发送失败，请检查后端服务是否启动');
    } finally {
      setLoading(false);
      setStreamingContent('');
      setStreamingReasoning('');
      setReasoningDone(false);
    }
  };

  const clearCurrentHistory = async () => {
    if (!currentSessionId) return;
    try {
      await fetch(`${API_URL}/history/${currentSessionId}`, { method: 'DELETE' });
      setMessages([]);
      setStreamingContent('');
      await loadSessions();
    } catch (error) {
      console.error('清除失败:', error);
    }
  };

  // 导出 Word 文档
  const exportWord = async () => {
    if (!currentSessionId) return;
    if (messages.length === 0) {
      alert('当前会话没有消息可导出');
      return;
    }
    try {
      const response = await fetch(`${API_URL}/export/docx/${currentSessionId}`);
      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: '导出失败' }));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }
      // 获取文件名
      const disposition = response.headers.get('content-disposition') || '';
      const match = disposition.match(/filename="?(.+?)"?$/);
      const filename = match ? match[1] : `对话记录_${currentSessionId}.docx`;

      // 触发下载
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error: any) {
      console.error('导出失败:', error);
      alert('导出 Word 失败: ' + (error.message || '未知错误'));
    }
  };

  // ====== 加载记忆 ======
  const loadMemory = async () => {
    if (!currentSessionId) {
      setMemoryError('请先创建或选择一个对话');
      return;
    }
    setMemoryLoading(true);
    setMemoryError('');
    try {
      const resp = await fetch(`${API_URL}/memory/${currentSessionId}`);
      if (!resp.ok) throw new Error(`加载失败: ${resp.status}`);
      const data = await resp.json();
      setMemoryData(data);
    } catch (err: any) {
      setMemoryError(err.message || '加载失败');
    } finally {
      setMemoryLoading(false);
    }
  };

  // 切换到记忆标签时自动加载
  useEffect(() => {
    if (mode === 'memory') {
      loadMemory();
    }
  }, [mode, currentSessionId]);

  // ====== arXiv 论文搜索 ======
  const searchArxiv = async (pageStart = 0) => {
    if (!arxivQuery.trim()) return;
    setArxivLoading(true);

    try {
      // arXiv API 使用 + 作为空格，不能使用 URLSearchParams 自动编码
      const arxivQ = arxivQuery.trim().replace(/ /g, '+');
      const params = `q=${arxivQ}&start=${pageStart}&max_results=20&sortBy=relevance&sortOrder=descending`;
      const resp = await fetch(`${API_URL}/arxiv/search?${params}`);
      if (!resp.ok) throw new Error(`搜索失败: ${resp.status}`);
      const data: ArxivResponse = await resp.json();
      setArxivResults(data.papers);
      setArxivTotal(data.total_results);
    } catch (err: any) {
      console.error('arXiv搜索失败:', err);
      alert('arXiv搜索失败: ' + (err.message || '未知错误'));
    } finally {
      setArxivLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      {/* 侧边栏 */}
      <div style={styles.sidebar}>
        <div style={styles.sidebarHeader}>
          <h2 style={styles.sidebarTitle}>对话历史</h2>
          <button onClick={createNewSession} style={styles.newChatBtn}>
            + 新对话
          </button>
        </div>
        <div style={styles.sessionList}>
          {sessions.length === 0 ? (
            <p style={styles.emptyText}>暂无会话记录</p>
          ) : (
            sessions.map(session => (
              <div
                key={session.session_id}
                onClick={() => loadSessionHistory(session.session_id)}
                style={{
                  ...styles.sessionItem,
                  backgroundColor: currentSessionId === session.session_id ? '#e3f2fd' : 'transparent',
                }}
              >
                <div style={styles.sessionInfo}>
                  <div style={styles.sessionTitle}>{session.title}</div>
                  <div style={styles.sessionMeta}>{session.message_count} 条消息</div>
                </div>
                <button
                  onClick={(e) => deleteSession(session.session_id, e)}
                  style={styles.deleteBtn}
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* 主内容区 */}
      <div style={styles.main}>
        {/* 顶部栏 - tabs + 模型选择 */}
        <div style={styles.header}>
          <div style={styles.headerLeft}>
            {/* 模式切换 */}
            <div style={styles.tabGroup}>
              <button
                onClick={() => setMode('chat')}
                style={{
                  ...styles.tabBtn,
                  ...(mode === 'chat' ? styles.tabBtnActive : {}),
                }}
              >
                💬 对话
              </button>
              <button
                onClick={() => setMode('arxiv')}
                style={{
                  ...styles.tabBtn,
                  ...(mode === 'arxiv' ? styles.tabBtnActive : {}),
                }}
              >
                📄 论文搜索
              </button>
              <button
                onClick={() => setMode('memory')}
                style={{
                  ...styles.tabBtn,
                  ...(mode === 'memory' ? styles.tabBtnActive : {}),
                }}
              >
                🧠 记忆
              </button>
              <button
                onClick={() => setMode('tools')}
                style={{
                  ...styles.tabBtn,
                  ...(mode === 'tools' ? styles.tabBtnActive : {}),
                }}
              >
                🔧 工具
              </button>
            </div>
          </div>
          <div style={styles.headerRight}>
            {mode === 'chat' && currentSessionId && (
              <>
                <button
                  onClick={exportWord}
                  disabled={messages.length === 0}
                  style={{
                    ...styles.exportBtn,
                    opacity: messages.length === 0 ? 0.5 : 1,
                    cursor: messages.length === 0 ? 'not-allowed' : 'pointer',
                  }}
                  title="导出为 Word 文档"
                >
                  📄 导出 Word
                </button>
                <button onClick={clearCurrentHistory} style={styles.clearBtn}>
                  清空对话
                </button>
              </>
            )}
          </div>
        </div>

        {/* 聊天消息区域 */}
        {mode === 'chat' && (
          <>
          {/* Agent 流水线 —— 位于用户消息和 AI 回复之间，始终可见 */}
          {showAgentPipeline && agentPipeline.length > 0 && (
            <div style={{...styles.agentPipelineContainer, marginBottom: 0, flexShrink: 0}}>
              <div style={styles.agentPipelineHeader} onClick={() => setPipelineCollapsed(!pipelineCollapsed)}>
                <span style={{fontWeight: 500, fontSize: 13, color: '#555'}}>
                  🤖 Agent 流水线 · {AGENT_OPTIONS.find(a => a.id === selectedAgent)?.label || selectedAgent}
                </span>
                <span style={{fontSize: 11, color: '#999'}}>
                  {pipelineCollapsed ? '展开 ▼' : '收起 ▲'}
                </span>
              </div>
              {!pipelineCollapsed && (
                <div style={styles.agentPipelineBody}>
                  {agentPipeline.map((trace, ti) => (
                    <div key={trace.agent} style={styles.agentTraceRow}>
                      {ti > 0 && <div style={styles.agentConnectLine} />}
                      <div style={{...styles.agentCard, borderLeftColor: AGENT_COLORS[trace.agent] || '#ccc'}}>
                        <div style={styles.agentCardRow}>
                          <span style={styles.agentCardIcon}>{AGENT_ICONS[trace.agent] || '🤖'}</span>
                          <span style={styles.agentCardLabel}>{trace.label}</span>
                          <span style={styles.agentCardStatus}>
                            {trace.status === 'running' ? <span style={styles.statusRunning}>⏳ 进行中</span> : trace.status === 'complete' ? <span style={styles.statusComplete}>✅ 完成</span> : <span style={styles.statusError}>❌ 错误</span>}
                          </span>
                        </div>
                        {toolCalls.filter(t => t.agent === trace.agent).map(tc => (
                          <div key={tc.toolId}>
                            <div style={{...styles.toolCallRow, cursor: tc.output ? 'pointer' : 'default'}}
                              onClick={() => {
                                if (tc.output) {
                                  setExpandedTools(prev => {
                                    const next = new Set(prev);
                                    if (next.has(tc.toolId)) next.delete(tc.toolId);
                                    else next.add(tc.toolId);
                                    return next;
                                  });
                                }
                              }}
                            >
                              <span style={styles.toolCallIcon}>└─</span>
                              <span style={styles.toolCallLabel}>🔧 {tc.label || tc.tool}</span>
                              <span style={styles.toolCallStatus}>
                                {tc.status === 'start' ? <span style={styles.statusRunning}>⏳</span> : <span style={styles.statusComplete}>✅</span>}
                                {tc.output && (expandedTools.has(tc.toolId) ? ' 🔽' : ' 🔍')}
                              </span>
                            </div>
                            {tc.output && expandedTools.has(tc.toolId) && (
                              <div style={{
                                marginLeft: 24, marginTop: 4, marginBottom: 6,
                                padding: '8px 10px', backgroundColor: '#f0f4f8',
                                borderRadius: 6, fontSize: 11, color: '#555',
                                maxHeight: 200, overflowY: 'auto', whiteSpace: 'pre-wrap',
                                border: '1px solid #e0e0e0', lineHeight: 1.5,
                              }}>
                                {tc.output.length > 2000 ? tc.output.slice(0, 2000) + '\n\n... (内容过长，已截断)' : tc.output}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                  {subTaskPlan.length > 0 && (
                    <div style={{marginTop: 10, padding: '8px 10px', backgroundColor: '#f8f9fa', borderRadius: 8, border: '1px solid #eee'}}>
                      <div style={{fontSize: 12, fontWeight: 600, color: '#555', marginBottom: 8}}>
                        📋 任务进度 ({subTaskPlan.filter((s) => s.status === 'complete').length}/{subTaskPlan.length})
                      </div>
                      {subTaskPlan.map((st, si) => (
                        <div key={si} style={{display: 'flex', alignItems: 'center', gap: 6, padding: '3px 0', fontSize: 11, color: '#666', opacity: st.status === 'pending' ? 0.5 : 1}}>
                          <span style={{fontSize: 10, width: 16}}>{st.status === 'running' ? '⏳' : st.status === 'complete' ? '✅' : '⏺️'}</span>
                          <span style={{flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontWeight: st.status === 'running' ? 600 : 400, color: st.status === 'running' ? '#333' : '#888'}}>
                            {si + 1}. [{st.agent}] {st.focus}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
          <div ref={messageAreaRef} onScroll={handleScroll} style={styles.messageArea}>
            {messages.length === 0 && !streamingContent ? (
              <div style={styles.emptyState}>
                <div style={styles.emptyIcon}>💬</div>
                <p>{currentSessionId ? '开始和 AI 对话吧' : '选择或创建一个新对话开始'}</p>
              </div>
            ) : (
              <>
                {messages.flatMap((msg, idx) => {
                  const isUser = msg.role === 'user';
                  const msgEl = (
                    <div
                      key={`msg-${idx}`}
                      style={{
                        ...styles.messageRow,
                        justifyContent: isUser ? 'flex-end' : 'flex-start',
                      }}
                    >
                      <div
                        style={{
                          ...styles.messageBubble,
                          backgroundColor: isUser ? '#007bff' : '#fff',
                          color: isUser ? '#fff' : '#333',
                          border: !isUser ? '1px solid #e0e0e0' : 'none',
                        }}
                      >
                        {isUser ? (
                          <div>{msg.content}</div>
                        ) : (
                          <div style={styles.markdownContent}>
                            {msg.reasoning_content && (
                              <details style={styles.reasoningDetails}>
                                <summary style={styles.reasoningSummary}>🧠 思考过程</summary>
                                <div style={styles.reasoningContent}>{msg.reasoning_content}</div>
                              </details>
                            )}
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm, remarkMath]}
                              rehypePlugins={[rehypeKatex]}
                              components={{
                                a: ({href, children, ...props}: any) => (
                                  <a href={href} target="_blank" rel="noopener noreferrer" style={{color: '#3b82f6'}} {...props}>
                                    {children} 🔗
                                  </a>
                                ),
                              }}
                            >
                              {linkifyContent(msg.content)}
                            </ReactMarkdown>
                          </div>
                        )}
                      </div>
                    </div>
                  );

                  return [msgEl];
                })}

                {/* 流式输出 */}
                {streamingContent && (
                  <div style={styles.messageRow}>
                    <div style={{ ...styles.messageBubble, backgroundColor: '#fff', border: '1px solid #e0e0e0' }}>
                      <div style={styles.markdownContent}>
                        {streamingReasoning && (
                          <details open={!reasoningDone} style={styles.reasoningDetails}>
                            <summary style={styles.reasoningSummary}>
                              {reasoningDone ? '🧠 思考过程' : '🤔 思考中...'}
                            </summary>
                            <div style={styles.reasoningContent}>{streamingReasoning}</div>
                          </details>
                        )}
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                          components={{
                            a: ({href, children, ...props}: any) => (
                              <a href={href} target="_blank" rel="noopener noreferrer" style={{color: '#3b82f6'}} {...props}>
                                {children} 🔗
                              </a>
                            ),
                          }}
                        >
                          {linkifyContent(streamingContent)}
                        </ReactMarkdown>
                        <span style={styles.cursor}>|</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* 参考文献列表（从工具调用结果中提取） */}
                {(() => {
                  const refs = extractReferences(toolCalls);
                  if (refs.length === 0) return null;
                  return (
                    <div style={{
                      marginTop: 16, padding: '12px 14px',
                      backgroundColor: '#f8fafc', borderRadius: 10,
                      border: '1px solid #e2e8f0', fontSize: 12,
                    }}>
                      <div style={{fontWeight: 600, color: '#475569', marginBottom: 8}}>
                        📚 参考文献 ({refs.length})
                      </div>
                      {refs.map((ref, ri) => (
                        <div key={ri} style={{
                          padding: '4px 0', borderBottom: ri < refs.length - 1 ? '1px solid #e2e8f0' : 'none',
                          display: 'flex', alignItems: 'center', gap: 6,
                        }}>
                          <span style={{color: '#94a3b8', fontSize: 11, minWidth: 18}}>[{ri + 1}]</span>
                          <span style={{flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>
                            {ref.title}
                          </span>
                          <span style={{
                            padding: '1px 6px', borderRadius: 4, fontSize: 10,
                            backgroundColor: ref.source === 'arXiv' ? '#dbeafe' : '#dcfce7',
                            color: ref.source === 'arXiv' ? '#2563eb' : '#16a34a',
                          }}>
                            {ref.source}
                          </span>
                          <a href={ref.url} target="_blank" rel="noopener noreferrer"
                            style={{color: '#3b82f6', fontSize: 11, textDecoration: 'none', whiteSpace: 'nowrap'}}>
                            打开 🔗
                          </a>
                        </div>
                      ))}
                    </div>
                  );
                })()}
              </>
            )}
            {loading && !streamingContent && (
              <div style={styles.messageRow}>
                <div style={{ ...styles.messageBubble, backgroundColor: '#fff', border: '1px solid #e0e0e0' }}>
                  <div style={styles.thinking}>AI 正在思考...</div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          </>
        )}

        {/* 记忆面板 */}
        {mode === 'memory' && (
          <div style={{ ...styles.arxivPanel, padding: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h3 style={{ margin: 0, fontSize: 16, color: '#333' }}>🧠 工作记忆</h3>
              <button onClick={loadMemory} disabled={memoryLoading} style={{ padding: '6px 14px', fontSize: 12, border: '1px solid #ddd', borderRadius: 6, backgroundColor: '#fff', cursor: 'pointer' }}>
                {memoryLoading ? '刷新中...' : '🔄 刷新'}
              </button>
            </div>

            {!currentSessionId && (
              <div style={styles.emptyState}><p>请先创建一个对话，然后发送消息让 Agent 工作</p></div>
            )}

            {memoryError && (
              <div style={{ padding: 16, backgroundColor: '#fff3f3', borderRadius: 8, color: '#d32f2f', fontSize: 13 }}>{memoryError}</div>
            )}

            {memoryLoading && <div style={{ textAlign: 'center', padding: 40, color: '#999' }}>加载中...</div>}

            {memoryData && !memoryLoading && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {/* 统计概览 */}
                <div style={{ display: 'flex', gap: 12 }}>
                  {[
                    { label: '📄 已归档论文', count: memoryData.stats?.papers || 0, color: '#3b82f6' },
                    { label: '💡 创新方案', count: memoryData.stats?.innovations || 0, color: '#f59e0b' },
                    { label: '🧪 实验记录', count: memoryData.stats?.experiments || 0, color: '#ef4444' },
                  ].map(stat => (
                    <div key={stat.label} style={{ flex: 1, backgroundColor: '#fff', border: '1px solid #e0e0e0', borderRadius: 10, padding: '16px', textAlign: 'center' }}>
                      <div style={{ fontSize: 28, fontWeight: 600, color: stat.color }}>{stat.count}</div>
                      <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>{stat.label}</div>
                    </div>
                  ))}
                </div>

                {/* 论文归档 */}
                <div style={{ backgroundColor: '#fff', border: '1px solid #e0e0e0', borderRadius: 10, padding: 16 }}>
                  <h4 style={{ margin: '0 0 12px 0', fontSize: 14, color: '#333' }}>📄 已归档论文 ({memoryData.stats?.papers || 0})</h4>
                  {memoryData.memory?.papers_archive?.length > 0 ? (
                    memoryData.memory.papers_archive.map((p: any, i: number) => (
                      <div key={i} style={{ padding: '8px 0', borderBottom: i > 0 ? '1px solid #f0f0f0' : 'none', fontSize: 13, color: '#555' }}>
                        <span style={{ fontWeight: 500, color: '#333' }}>{p.title}</span>
                        {p.evidence_level && <span style={{ marginLeft: 8, fontSize: 11, color: p.evidence_level === 'confirmed' ? '#22c55e' : p.evidence_level === 'disputed' ? '#f59e0b' : '#999' }}>[{p.evidence_level}]</span>}
                      </div>
                    ))
                  ) : (
                    <p style={{ fontSize: 13, color: '#999' }}>暂无归档论文</p>
                  )}
                </div>

                {/* 创新方案 */}
                <div style={{ backgroundColor: '#fff', border: '1px solid #e0e0e0', borderRadius: 10, padding: 16 }}>
                  <h4 style={{ margin: '0 0 12px 0', fontSize: 14, color: '#333' }}>💡 创新方案 ({memoryData.stats?.innovations || 0})</h4>
                  {memoryData.memory?.innovation_candidates?.length > 0 ? (
                    memoryData.memory.innovation_candidates.map((c: any, i: number) => (
                      <div key={i} style={{ padding: '8px 0', borderBottom: i > 0 ? '1px solid #f0f0f0' : 'none', fontSize: 13, color: '#555' }}>
                        <span style={{ fontWeight: 500, color: '#333' }}>{c.name}</span>
                        <span style={{ marginLeft: 8, fontSize: 11, color: '#999' }}>
                          {c.novelty && `新颖:${c.novelty}`} {c.difficulty && `难度:${c.difficulty}`} {c.status && `[${c.status}]`}
                        </span>
                      </div>
                    ))
                  ) : (
                    <p style={{ fontSize: 13, color: '#999' }}>暂无创新方案</p>
                  )}
                </div>

                {/* 实验日志 */}
                <div style={{ backgroundColor: '#fff', border: '1px solid #e0e0e0', borderRadius: 10, padding: 16 }}>
                  <h4 style={{ margin: '0 0 12px 0', fontSize: 14, color: '#333' }}>🧪 实验记录 ({memoryData.stats?.experiments || 0})</h4>
                  {memoryData.memory?.experiment_log?.length > 0 ? (
                    memoryData.memory.experiment_log.map((log: any, i: number) => (
                      <div key={i} style={{ padding: '8px 0', borderBottom: i > 0 ? '1px solid #f0f0f0' : 'none', fontSize: 13, color: '#555' }}>
                        <span style={{ fontWeight: 500, color: '#333' }}>步骤 {log.step}</span>
                        <span style={{ marginLeft: 8 }}>{log.analysis?.slice(0, 100)}</span>
                      </div>
                    ))
                  ) : (
                    <p style={{ fontSize: 13, color: '#999' }}>暂无实验记录</p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* arXiv 搜索面板 */}
        {mode === 'arxiv' && (
          <div style={styles.arxivPanel}>
            {/* 搜索栏 */}
            <div style={styles.arxivSearchBar}>
              <input
                type="text"
                value={arxivQuery}
                onChange={e => setArxivQuery(e.target.value)}
                onKeyPress={e => e.key === 'Enter' && searchArxiv()}
                placeholder="搜索 arXiv 论文，如：cat:cs.LG+AND+ti:transformer"
                style={styles.arxivInput}
              />
              <button
                onClick={() => searchArxiv(0)}
                disabled={arxivLoading}
                style={{
                  ...styles.sendBtn,
                  opacity: arxivLoading ? 0.6 : 1,
                  cursor: arxivLoading ? 'not-allowed' : 'pointer',
                }}
              >
                {arxivLoading ? '搜索中...' : '搜索'}
              </button>
            </div>

            {/* 搜索结果 */}
            <div style={styles.arxivResults}>
              {arxivResults.length === 0 && !arxivLoading && (
                <div style={styles.emptyState}>
                  <div style={styles.emptyIcon}>📄</div>
                  <p>输入搜索词查询 arXiv 论文</p>
                  <p style={{ fontSize: 12, color: '#999', marginTop: 8 }}>
                    示例：all:machine+learning | ti:transformer | cat:cs.LG+AND+au:"John Doe"
                  </p>
                </div>
              )}
              {arxivLoading && (
                <div style={{ textAlign: 'center', padding: 40, color: '#999' }}>
                  正在搜索 arXiv，等待 3 秒限流...
                </div>
              )}
              {arxivResults.map((paper, idx) => (
                <div key={idx} style={styles.arxivPaperCard}>
                  <div style={styles.arxivPaperTitle}>
                    <a href={paper.id} target="_blank" rel="noopener noreferrer">
                      {paper.title}
                    </a>
                  </div>
                  <div style={styles.arxivPaperMeta}>
                    <span>👥 {paper.authors.slice(0, 5).join('; ')}{paper.authors.length > 5 ? '...' : ''}</span>
                  </div>
                  <div style={styles.arxivPaperMeta}>
                    <span>📅 {paper.published.slice(0, 10)}</span>
                    <span style={{ marginLeft: 16 }}>🏷️ {paper.categories.join(', ')}</span>
                    {paper.pdf_link && (
                      <a href={paper.pdf_link} target="_blank" rel="noopener noreferrer" style={{ marginLeft: 16 }}>
                        📥 PDF
                      </a>
                    )}
                  </div>
                  <div style={styles.arxivPaperSummary}>
                    {paper.summary.slice(0, 300)}{paper.summary.length > 300 ? '...' : ''}
                  </div>
                </div>
              ))}
              {arxivTotal > 0 && (
                <div style={{ textAlign: 'center', padding: 16, color: '#999', fontSize: 13 }}>
                  共 {arxivTotal} 篇结果
                </div>
              )}
            </div>
          </div>
        )}

        {/* 工具面板 */}
        {mode === 'tools' && (
          <div style={{ ...styles.arxivPanel, padding: 20, overflow: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h3 style={{ margin: 0, fontSize: 16, color: '#333' }}>🔧 系统工具一览</h3>
              <span style={{ fontSize: 12, color: '#999' }}>共 5 个工具 · 绑定到 4 个 Agent</span>
            </div>

            {/* Agent → 工具 映射 */}
            <div style={{ marginBottom: 20, backgroundColor: '#fff', border: '1px solid #e0e0e0', borderRadius: 10, padding: 16 }}>
              <h4 style={{ margin: '0 0 12px 0', fontSize: 14, color: '#333' }}>🤖 Agent 工具绑定</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                  { agent: '💬 对话助手', color: '#22c55e', tools: ['web_search'] },
                  { agent: '🔍 文献调研', color: '#3b82f6', tools: ['search_arxiv', 'web_search'] },
                  { agent: '💡 创新构思', color: '#f59e0b', tools: ['web_search', 'create_docx', 'add_section', 'add_table'] },
                  { agent: '🧪 实验分析', color: '#ef4444', tools: ['web_search', 'create_docx', 'add_section', 'add_table'] },
                  { agent: '📋 综合输出', color: '#06b6d4', tools: ['create_docx', 'add_section', 'add_table'] },
                ].map(item => (
                  <div key={item.agent} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', backgroundColor: '#f8f9fa', borderRadius: 8 }}>
                    <span style={{ fontSize: 14, fontWeight: 500, color: item.color, minWidth: 110 }}>{item.agent}</span>
                    <span style={{ fontSize: 11, color: '#666' }}>→</span>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {item.tools.map(t => {
                        const colors: any = {
                          search_arxiv: '#3b82f6', web_search: '#8b5cf6',
                          create_docx: '#06b6d4', add_section: '#10b981', add_table: '#10b981',
                        };
                        return (
                          <span key={t} style={{
                            padding: '2px 8px', borderRadius: 4, fontSize: 11,
                            backgroundColor: (colors[t] || '#999') + '20',
                            color: colors[t] || '#999', fontWeight: 500,
                          }}>
                            {t}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 工具详情列表 */}
            {[
              {
                name: 'search_arxiv', icon: '📚', color: '#3b82f6',
                desc: '搜索 arXiv 学术论文。适合查找正式发表的学术文献、SOTA 方法。',
                params: 'query (必填), max_results (默认10), sort_by (默认relevance)',
                agents: ['🔍 文献调研'],
              },
              {
                name: 'web_search', icon: '🌐', color: '#8b5cf6',
                desc: '阿里云百炼 MCP 实时联网搜索。适合查找最新新闻、实时信息、博客和百科。需要 DASHSCOPE_API_KEY。',
                params: 'query (必填), count (默认5)',
                agents: ['💬 对话助手', '🔍 文献调研', '💡 创新构思', '🧪 实验分析'],
              },
              {
                name: 'create_docx', icon: '📄', color: '#06b6d4',
                desc: '创建一个新的 Word 文档 (.docx)，用于保存调研报告、实验记录等。',
                params: 'title (必填)',
                agents: ['💡 创新构思', '🧪 实验分析', '📋 综合输出'],
              },
              {
                name: 'add_section', icon: '📝', color: '#10b981',
                desc: '在已存在的 Word 文档中添加一个新章节（含标题和正文）。',
                params: 'filepath (必填), heading (必填), content (必填)',
                agents: ['💡 创新构思', '🧪 实验分析', '📋 综合输出'],
              },
              {
                name: 'add_table', icon: '📊', color: '#10b981',
                desc: '在 Word 文档中添加一个对比表格，用于展示方法对比、实验结果等。',
                params: 'filepath (必填), headers (必填), rows (必填)',
                agents: ['💡 创新构思', '🧪 实验分析', '📋 综合输出'],
              },
            ].map(tool => (
              <div key={tool.name} style={{
                backgroundColor: '#fff', border: '1px solid #e0e0e0',
                borderLeft: `4px solid ${tool.color}`,
                borderRadius: 10, padding: 16, marginBottom: 12,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <span style={{ fontSize: 18 }}>{tool.icon}</span>
                  <span style={{ fontSize: 15, fontWeight: 600, color: '#333' }}>{tool.name}</span>
                </div>
                <p style={{ fontSize: 13, color: '#555', lineHeight: 1.5, margin: '0 0 8px 0' }}>{tool.desc}</p>

                {/* 参数 */}
                <div style={{ fontSize: 12, color: '#888', marginBottom: 8 }}>
                  <span style={{ fontWeight: 500, color: '#666' }}>参数:</span> {tool.params}
                </div>

                {/* 绑定的 Agent */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 11, color: '#999' }}>绑定:</span>
                  {tool.agents.map(a => (
                    <span key={a} style={{
                      padding: '2px 8px', borderRadius: 4, fontSize: 11,
                      backgroundColor: '#f0f0f0', color: '#666',
                    }}>
                      {a}
                    </span>
                  ))}
                </div>
              </div>
            ))}

            {/* 配置提示 */}
            <div style={{ marginTop: 16, padding: 12, backgroundColor: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, fontSize: 12, color: '#92400e' }}>
              <strong>💡 提示:</strong> 工具的 API Key 在 <code>.env</code> 文件中配置（DEEPSEEK_API_KEY, DASHSCOPE_API_KEY）。
              每个工具调用会消耗 API 额度。Agent 在 ReAct 循环中会自动选择合适的工具。
            </div>
          </div>
        )}

        {/* 输入区域（仅聊天模式） */}
        {mode === 'chat' && (
          <div style={styles.inputArea}>
            {/* Agent 选择栏 */}
            <div style={styles.agentSelector}>
              {AGENT_OPTIONS.map(ao => (
                <button
                  key={ao.id}
                  onClick={() => setSelectedAgent(ao.id)}
                  disabled={loading}
                  style={{
                    ...styles.agentChip,
                    ...(selectedAgent === ao.id ? styles.agentChipActive : {}),
                    opacity: loading ? 0.6 : 1,
                    cursor: loading ? 'not-allowed' : 'pointer',
                  }}
                  title={ao.desc}
                >
                  {ao.label}
                </button>
              ))}
            </div>
            {/* 输入栏 */}
            <div style={styles.inputRow}>
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyPress={e => e.key === 'Enter' && sendMessage()}
                placeholder="输入消息... (支持 Markdown 和 LaTeX)"
                disabled={loading}
                style={styles.input}
              />
              <select
                value={selectedModel}
                onChange={e => setSelectedModel(e.target.value)}
                style={styles.modelSelectInline}
                disabled={loading}
                title="切换模型"
              >
                {MODELS.map(m => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
              <button
                onClick={sendMessage}
                disabled={loading}
                style={{
                  ...styles.sendBtn,
                  opacity: loading ? 0.6 : 1,
                  cursor: loading ? 'not-allowed' : 'pointer',
                }}
              >
                发送
              </button>
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
        .katex { font-size: 1.1em; }
        code {
          background-color: #f5f5f5;
          padding: 2px 6px;
          border-radius: 4px;
          font-family: 'Consolas', monospace;
        }
        pre {
          background-color: #f5f5f5;
          padding: 12px;
          border-radius: 8px;
          overflow-x: auto;
        }
        blockquote {
          border-left: 3px solid #ddd;
          margin-left: 0;
          padding-left: 16px;
          color: #666;
        }
        ${extraStyles}
      `}</style>
    </div>
  );
}

// 额外 CSS 注入到 <style> 中
const extraStyles = `
  .reasoning-details {
    margin-bottom: 12px;
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 0;
  }
  .reasoning-details summary {
    cursor: pointer;
    padding: 8px 12px;
    font-size: 13px;
    color: #666;
    font-weight: 500;
    user-select: none;
  }
  .reasoning-details summary:hover {
    background-color: #e9ecef;
    border-radius: 8px;
  }
  .reasoning-details .reasoning-body {
    padding: 0 12px 12px 12px;
    font-size: 13px;
    color: #888;
    line-height: 1.6;
    border-top: 1px solid #e9ecef;
    margin-top: 4px;
    padding-top: 8px;
    white-space: pre-wrap;
  }
`;

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    display: 'flex',
    height: '100vh',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    backgroundColor: '#f8f9fa',
  },
  sidebar: {
    width: '280px',
    backgroundColor: '#fff',
    borderRight: '1px solid #e0e0e0',
    display: 'flex',
    flexDirection: 'column',
  },
  sidebarHeader: {
    padding: '20px',
    borderBottom: '1px solid #e0e0e0',
  },
  sidebarTitle: {
    margin: '0 0 15px 0',
    fontSize: '18px',
    color: '#333',
  },
  newChatBtn: {
    width: '100%',
    padding: '12px',
    backgroundColor: '#007bff',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '14px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  sessionList: {
    flex: 1,
    overflowY: 'auto',
    padding: '10px',
  },
  emptyText: {
    textAlign: 'center',
    color: '#999',
    padding: '20px',
    fontSize: '14px',
  },
  sessionItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 15px',
    borderRadius: '8px',
    cursor: 'pointer',
    marginBottom: '4px',
    transition: 'background-color 0.2s',
  },
  sessionInfo: {
    flex: 1,
    overflow: 'hidden',
  },
  sessionTitle: {
    fontSize: '14px',
    fontWeight: 500,
    color: '#333',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  sessionMeta: {
    fontSize: '12px',
    color: '#999',
    marginTop: '4px',
  },
  deleteBtn: {
    background: 'none',
    border: 'none',
    fontSize: '18px',
    color: '#999',
    cursor: 'pointer',
    padding: '0 5px',
  },
  main: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    maxWidth: '900px',
    margin: '0 auto',
    width: '100%',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '15px 20px',
    borderBottom: '1px solid #e0e0e0',
    backgroundColor: '#fff',
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
  },
  headerRight: {
    display: 'flex',
    alignItems: 'center',
  },
  sessionId: {
    fontSize: '14px',
    color: '#666',
  },
  clearBtn: {
    padding: '8px 16px',
    backgroundColor: '#dc3545',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '13px',
    cursor: 'pointer',
    marginLeft: 8,
  } as React.CSSProperties,
  exportBtn: {
    padding: '8px 16px',
    backgroundColor: '#28a745',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '13px',
    cursor: 'pointer',
  } as React.CSSProperties,
  messageArea: {
    flex: 1,
    overflowY: 'auto',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
  },
  emptyState: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#999',
  },
  emptyIcon: {
    fontSize: '48px',
    marginBottom: '16px',
  },
  messageRow: {
    display: 'flex',
    marginBottom: '16px',
  },
  messageBubble: {
    maxWidth: '75%',
    padding: '12px 16px',
    borderRadius: '12px',
    lineHeight: 1.6,
  },
  markdownContent: {
    fontSize: '15px',
    lineHeight: 1.7,
  },
  cursor: {
    animation: 'blink 1s infinite',
  },
  thinking: {
    color: '#999',
    fontStyle: 'italic',
  },
  reasoningDetails: {
    marginBottom: 12,
    background: '#f8f9fa',
    border: '1px solid #e9ecef',
    borderRadius: 8,
    overflow: 'hidden',
  } as React.CSSProperties,
  reasoningSummary: {
    cursor: 'pointer',
    padding: '8px 12px',
    fontSize: 13,
    color: '#666',
    fontWeight: 500,
  } as React.CSSProperties,
  reasoningContent: {
    padding: '8px 12px 12px 12px',
    fontSize: 13,
    color: '#888',
    lineHeight: 1.6,
    whiteSpace: 'pre-wrap' as const,
    borderTop: '1px solid #e9ecef',
  } as React.CSSProperties,

  // 模型选择
  modelSelect: {
    padding: '6px 12px',
    fontSize: 13,
    border: '1px solid #ddd',
    borderRadius: 6,
    backgroundColor: '#fff',
    color: '#333',
    outline: 'none',
    cursor: 'pointer',
    marginRight: 12,
  } as React.CSSProperties,

  // 模式切换标签
  tabGroup: {
    display: 'flex',
    gap: 0,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    padding: 2,
  } as React.CSSProperties,
  tabBtn: {
    padding: '6px 16px',
    fontSize: 13,
    border: 'none',
    borderRadius: 6,
    backgroundColor: 'transparent',
    color: '#666',
    cursor: 'pointer',
    transition: 'all 0.2s',
  } as React.CSSProperties,
  tabBtnActive: {
    backgroundColor: '#fff',
    color: '#333',
    fontWeight: 500,
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  } as React.CSSProperties,

  // arXiv面板
  arxivPanel: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  } as React.CSSProperties,
  arxivSearchBar: {
    display: 'flex',
    gap: 12,
    padding: 16,
    borderBottom: '1px solid #e0e0e0',
    backgroundColor: '#fff',
  } as React.CSSProperties,
  arxivInput: {
    flex: 1,
    padding: '10px 16px',
    fontSize: 14,
    border: '1px solid #ddd',
    borderRadius: 8,
    outline: 'none',
  } as React.CSSProperties,
  arxivResults: {
    flex: 1,
    overflowY: 'auto',
    padding: 16,
  } as React.CSSProperties,
  arxivPaperCard: {
    backgroundColor: '#fff',
    border: '1px solid #e0e0e0',
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
  } as React.CSSProperties,
  arxivPaperTitle: {
    fontSize: 15,
    fontWeight: 500,
    marginBottom: 8,
    lineHeight: 1.4,
  } as React.CSSProperties,
  arxivPaperMeta: {
    fontSize: 12,
    color: '#666',
    marginBottom: 6,
    lineHeight: 1.5,
  } as React.CSSProperties,
  arxivPaperSummary: {
    fontSize: 13,
    color: '#888',
    lineHeight: 1.6,
    marginTop: 8,
  } as React.CSSProperties,
  inputArea: {
    padding: '12px 20px 20px 20px',
    backgroundColor: '#fff',
    borderTop: '1px solid #e0e0e0',
  } as React.CSSProperties,
  agentSelector: {
    display: 'flex',
    gap: 6,
    marginBottom: 10,
    flexWrap: 'wrap' as const,
  } as React.CSSProperties,
  agentChip: {
    padding: '5px 12px',
    fontSize: 13,
    border: '1px solid #e0e0e0',
    borderRadius: 20,
    backgroundColor: '#f8f9fa',
    color: '#666',
    cursor: 'pointer',
    transition: 'all 0.15s',
    whiteSpace: 'nowrap' as const,
  } as React.CSSProperties,
  agentChipActive: {
    backgroundColor: '#e8f4fd',
    borderColor: '#3b82f6',
    color: '#3b82f6',
    fontWeight: 500,
  } as React.CSSProperties,
  inputRow: {
    display: 'flex',
    gap: 10,
    alignItems: 'center',
  } as React.CSSProperties,
  modelSelectInline: {
    padding: '10px 12px',
    fontSize: 13,
    border: '1px solid #ddd',
    borderRadius: 8,
    backgroundColor: '#f8f9fa',
    color: '#555',
    outline: 'none',
    cursor: 'pointer',
    minWidth: 120,
  } as React.CSSProperties,
  input: {
    flex: 1,
    padding: '14px 18px',
    fontSize: '15px',
    border: '1px solid #ddd',
    borderRadius: '25px',
    outline: 'none',
    transition: 'border-color 0.2s',
  },
  sendBtn: {
    padding: '14px 28px',
    fontSize: '15px',
    backgroundColor: '#007bff',
    color: '#fff',
    border: 'none',
    borderRadius: '25px',
    fontWeight: 500,
  },

  // === Agent 流水线样式 ===
  agentPipelineContainer: {
    marginBottom: 16,
    backgroundColor: '#fff',
    border: '1px solid #e0e0e0',
    borderRadius: 10,
    overflow: 'hidden',
    position: 'sticky',
    top: 0,
    zIndex: 10,
  } as React.CSSProperties,
  agentPipelineHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '10px 14px',
    cursor: 'pointer',
    backgroundColor: '#fafafa',
    borderBottom: '1px solid #f0f0f0',
  } as React.CSSProperties,
  agentPipelineBody: {
    padding: '12px 14px',
  } as React.CSSProperties,
  agentTraceRow: {
    position: 'relative',
    marginBottom: 8,
  } as React.CSSProperties,
  agentConnectLine: {
    position: 'absolute',
    left: 12,
    top: -8,
    width: 2,
    height: 12,
    backgroundColor: '#e0e0e0',
  } as React.CSSProperties,
  agentCard: {
    padding: '10px 14px',
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    borderLeft: '3px solid #ccc',
  } as React.CSSProperties,
  agentCardRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  } as React.CSSProperties,
  agentCardIcon: {
    fontSize: 16,
  } as React.CSSProperties,
  agentCardLabel: {
    flex: 1,
    fontSize: 14,
    fontWeight: 500,
    color: '#333',
  } as React.CSSProperties,
  agentCardStatus: {
    fontSize: 12,
  } as React.CSSProperties,
  statusRunning: {
    color: '#f59e0b',
    animation: 'pulse 1.5s infinite',
  } as React.CSSProperties,
  statusComplete: {
    color: '#22c55e',
  } as React.CSSProperties,
  statusError: {
    color: '#ef4444',
  } as React.CSSProperties,
  toolCallRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    marginTop: 6,
    paddingLeft: 24,
    fontSize: 12,
    color: '#666',
  } as React.CSSProperties,
  toolCallIcon: {
    color: '#ccc',
    fontSize: 10,
  } as React.CSSProperties,
  toolCallLabel: {
    flex: 1,
  } as React.CSSProperties,
  toolCallStatus: {
    fontSize: 12,
  } as React.CSSProperties,
};

export default App;
