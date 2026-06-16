"""Fix graph.py: move plan/subtask events outside isinstance check"""
import re

with open('backend/agent/graph.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The redundant isinstance block (kept from original, overwrites our extracted data)
# Remove the useless second isinstance block that was kept as dead code
old = '''                            if isinstance(output, dict):
                                text = output.get("output_text", "") or ""
                                sub_tasks = output.get("sub_task_queue", [])
                                memory = output.get("memory", {})
                                task_plan = memory.get("task_plan", {}) if isinstance(memory, dict) else {}'''

# Replace it with just the first line's continuation
new = '''                            # text/sub_tasks/task_plan already extracted above from inp_state or output'''

content = content.replace(old, new)

# Now also fix the planner/synth/subtask condition - they should fire even for str output
# Remove the redundant "if isinstance(output, dict):" wrapping for plan/subtask events
# These conditions are already handled inside the respective blocks

with open('backend/agent/graph.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed!")
