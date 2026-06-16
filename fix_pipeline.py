import re

with open('F:/code/MyPython/major-assignment/frontend/src/App.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace per-index collapse state with simple boolean
old1 = '''  // 每个 user 消息索引独立的展开状态，互不影响
  // 默认展开；Set 中记录的是被用户折叠的索引
  const agentPipelineRef = useRef<AgentTrace[]>([]);  // ref 同步版本，避免闭包陈旧问题
  const [collapsedPipelines, setCollapsedPipelines] = useState<Set<number>>(new Set());
  const isPipelineExpanded = (idx: number) => !collapsedPipelines.has(idx);
  const togglePipeline = (idx: number) => {
    setCollapsedPipelines(prev => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx); else next.add(idx);
      return next;
    });
  };'''

new1 = '''  const agentPipelineRef = useRef<AgentTrace[]>([]);  // ref 同步版本，避免闭包陈旧问题
  const [pipelineCollapsed, setPipelineCollapsed] = useState(false);'''

content = content.replace(old1, new1)

# 2. Remove pipeline rendering from inside flatMap
# Find the if(showPipeAfter) block + pipeline rendering, replace with just return [msgEl]
pipeline_pattern = r"                  if \(showPipeAfter\) \{\s+                    return \[[^\]]*?\];\s+                  \}\s+                  return \[msgEl\];"

# Actually let's be more precise - find the exact block
lines = content.split('\n')
modified_lines = []
i = 0
in_flatmap_pipeline = False
while i < len(lines):
    line = lines[i]

    # Detect the start of the pipeline-in-flatmap block: "if (showPipeAfter) {"
    stripped = line.strip()
    if stripped.startswith('if (showPipeAfter) {'):
        in_flatmap_pipeline = True
        # Skip until we find "return [msgEl];"
        i += 1
        while i < len(lines):
            if 'return [msgEl];' in lines[i]:
                modified_lines.append(lines[i])
                i += 1
                break
            i += 1
        in_flatmap_pipeline = False
        continue

    modified_lines.append(line)
    i += 1

content = '\n'.join(modified_lines)

# 3. Add the standalone pipeline renderer BEFORE the message area
# Find the message area opening
msg_area_marker = '''        {/* 聊天消息区域 */}
        {mode === 'chat' && (
          <div ref={messageAreaRef} onScroll={handleScroll} style={styles.messageArea}>'''

pipeline_component = '''        {/* 聊天消息区域 */}
        {mode === 'chat' && (
          <>
          {/* 固定在消息区顶部的 Agent 流水线（不参与滚动） */}
          {showAgentPipeline && agentPipeline.length > 0 && (
            <div style={{...styles.agentPipelineContainer, marginBottom: 0, flexShrink: 0}}>
              <div
                style={styles.agentPipelineHeader}
                onClick={() => setPipelineCollapsed(!pipelineCollapsed)}
              >
                <span style={{ fontWeight: 500, fontSize: 13, color: '#555' }}>
                  🤖 Agent 流水线
                </span>
                <span style={{ fontSize: 11, color: '#999' }}>
                  {pipelineCollapsed ? '展开 ▼' : '收起 ▲'}
                </span>
              </div>
              {!pipelineCollapsed && (
                <div style={styles.agentPipelineBody}>
                  {agentPipeline.map((trace, ti) => (
                    <div key={trace.agent} style={styles.agentTraceRow}>
                      {ti > 0 && <div style={styles.agentConnectLine} />}
                      <div style={{
                        ...styles.agentCard,
                        borderLeftColor: AGENT_COLORS[trace.agent] || '#ccc',
                      }}>
                        <div style={styles.agentCardRow}>
                          <span style={styles.agentCardIcon}>
                            {AGENT_ICONS[trace.agent] || '🤖'}
                          </span>
                          <span style={styles.agentCardLabel}>{trace.label}</span>
                          <span style={styles.agentCardStatus}>
                            {trace.status === 'running' ? (
                              <span style={styles.statusRunning}>⏳ 进行中</span>
                            ) : trace.status === 'complete' ? (
                              <span style={styles.statusComplete}>✅ 完成</span>
                            ) : (
                              <span style={styles.statusError}>❌ 错误</span>
                            )}
                          </span>
                        </div>
                        {toolCalls.filter(t => t.agent === trace.agent).map(tc => (
                          <div key={tc.toolId} style={styles.toolCallRow}>
                            <span style={styles.toolCallIcon}>└─</span>
                            <span style={styles.toolCallLabel}>
                              🔧 {tc.label || tc.tool}
                            </span>
                            <span style={styles.toolCallStatus}>
                              {tc.status === 'start'
                                ? <span style={styles.statusRunning}>⏳</span>
                                : <span style={styles.statusComplete}>✅</span>
                              }
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                  {subTaskPlan.length > 0 && (
                    <div style={{ marginTop: 10, padding: '8px 10px', backgroundColor: '#f8f9fa', borderRadius: 8, border: '1px solid #eee' }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: '#555', marginBottom: 8 }}>
                        📋 任务进度 ({subTaskPlan.filter((s: any) => s.status === 'complete').length}/{subTaskPlan.length})
                      </div>
                      {subTaskPlan.map((st: any, si: number) => (
                        <div key={si} style={{
                          display: 'flex', alignItems: 'center', gap: 6,
                          padding: '3px 0', fontSize: 11, color: '#666',
                          opacity: st.status === 'pending' ? 0.5 : 1,
                        }}>
                          <span style={{ fontSize: 10, width: 16 }}>
                            {st.status === 'running' ? '⏳' : st.status === 'complete' ? '✅' : '⏺️'}
                          </span>
                          <span style={{
                            flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                            fontWeight: st.status === 'running' ? 600 : 400,
                            color: st.status === 'running' ? '#333' : '#888',
                          }}>
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

# Replace the closing of messageArea
old_close = '''          <div ref={messagesEndRef} />
          </div>
        )}'''

new_close = '''          <div ref={messagesEndRef} />
          </div>
          </>
        )}'''

content = content.replace(msg_area_marker, pipeline_component)
content = content.replace(old_close, new_close)

with open('F:/code/MyPython/major-assignment/frontend/src/App.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print('SUCCESS: Pipeline moved outside message area')
