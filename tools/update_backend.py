"""
更新后端 main.py 添加 reasoning_content 支持
"""
import json

FILE = "backend/main.py"

with open(FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the generate function boundaries
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if "async def generate():" in line:
        start_idx = i
    if start_idx is not None and i > start_idx and line.strip().startswith("return StreamingResponse"):
        end_idx = i - 1
        break

if start_idx is None or end_idx is None:
    print(f"ERROR: Could not find generate function. start={start_idx}, end={end_idx}")
    exit(1)

print(f"Found generate() at line {start_idx+1} to {end_idx+1}")

# Build replacement lines
NL = []

# Indentation levels
B = "    "       # 4 spaces - base
B2 = B * 2       # 8 spaces - function body
B3 = B * 3       # 12 spaces - inside try
B4 = B * 4       # 16 spaces - inside for/if

NL.append(f"{B2}async def generate():\n")
NL.append(f"{B3}try:\n")
NL.append(f"{B4}full_reply = \"\"\n")
NL.append(f"{B4}full_reasoning = \"\"\n")
NL.append(f"{B4}# 调用DeepSeek API - 流式\n")
NL.append(f"{B4}stream = client.chat.completions.create(\n")
NL.append(f"{B5}model=os.getenv(\"DEEPSEEK_MODEL\", \"deepseek-chat\"),\n")
NL.append(f"{B5}messages=conversations[session_id],\n")
NL.append(f"{B5}stream=True\n")
NL.append(f"{B4})\n")
NL.append("\n")
NL.append(f"{B4}for chunk in stream:\n")
NL.append(f"{B5}delta = chunk.choices[0].delta\n")
NL.append(f"{B5}content = delta.content or \"\"\n")
NL.append(f"{B5}reasoning_content = getattr(delta, 'reasoning_content', None) or \"\"\n")
NL.append("\n")
NL.append(f"{B5}event_data = {{'content': content, 'done': False}}\n")
NL.append("\n")
NL.append(f"{B5}if reasoning_content:\n")
NL.append(f"{B5}    full_reasoning += reasoning_content\n")
NL.append(f"{B5}    event_data['reasoning_content'] = reasoning_content\n")
NL.append("\n")
NL.append(f"{B5}if content:\n")
NL.append(f"{B5}    full_reply += content\n")
NL.append("\n")
NL.append(f"{B5}# 发送 SSE 格式的数据\n")
NL.append(f"{B5}yield f\"data: {json.dumps(event_data)}\\n\\n\"\n")
NL.append("\n")
NL.append(f"{B4}# 流结束后，保存完整回复到历史\n")
NL.append(f"{B4}conversations[session_id].append({{\n")
NL.append(f"{B5}\"role\": \"assistant\",\n")
NL.append(f"{B5}\"content\": full_reply,\n")
NL.append(f"{B5}\"reasoning_content\": full_reasoning if full_reasoning else None,\n")
NL.append(f"{B5}\"timestamp\": datetime.now().isoformat()\n")
NL.append(f"{B4}}})\n")
NL.append("\n")
NL.append(f"{B4}# 发送完成信号和完整历史\n")
NL.append(f"{B4}yield f\"data: {json.dumps({{'content': '', 'done': True, 'history': conversations[session_id]}})}\\n\\n\"\n")
NL.append("\n")
NL.append(f"{B3}except Exception as e:\n")
NL.append(f"{B4}yield f\"data: {json.dumps({{'error': str(e), 'done': True}})}\\n\\n\"\n")

# Replace
lines[start_idx : end_idx + 1] = NL

with open(FILE, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("替换成功!")
