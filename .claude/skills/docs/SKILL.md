---
name: docs
description: >-
  Manage project documentation — generate, update, organize, and convert docs.
  Use this skill whenever the user talks about documentation, docs, reports, PRD,
  updating status tables, generating Word reports from Markdown, or converting
  between document formats. Also use it when a feature is completed and the PRD
  status tracking table needs updating per CLAUDE.md rule #7. Trigger this when
  the user says "update the docs", "generate the report", "convert to docx",
  "update PRD status", "organize documentation", or any similar request.
---

# Docs Skill

Manage and generate documentation for this project following established conventions.

## Project documentation structure

```
docs/                   # Final documentation and reports
├── 项目文档.md          # Project documentation
├── 功能清单.md          # Feature checklist
├── 测试用例文档.md      # Test case documentation
├── 测试运行输出.txt      # Test run output
├── 失败路径测试记录.md   # Failure path test records
├── 代码审查报告.md      # Code review report
├── 综合实验报告.md      # Comprehensive experiment report (Markdown source)
├── 综合实验报告.docx    # Comprehensive experiment report (Word export)
└── 综合实验报告.bak.docx # Backup

prd-request/            # PRD design documents (with status tracking tables)
├── 01-系统概述与架构愿景.md
├── 02-Agent定义与职责划分.md
├── 03-LangGraph编排与协作流程.md
├── 04-记忆系统设计.md
├── 05-工具集成与封装.md
└── 06-前后端集成方案.md

CLAUDE.md               # AI agent workflow rules (rule #7: update PRD status)
generated_reports/       # Auto-generated Word reports from the app
```

## Key rules from CLAUDE.md

- **Rule 1**: All test/utility files go in `tools/`
- **Rule 2**: All PRD files go in `prd-request/`
- **Rule 7**: After completing a feature, update the status tracking table (`## 0. 状态跟踪表`) in the corresponding PRD file under `prd-request/`, marking frontend/backend/test/overall status as ✅ 已完成

## Converting Markdown to DOCX

The project has `tools/md_to_docx.py` for converting Markdown reports to Word format. Use it:

```bash
# From project root:
python tools/md_to_docx.py
```

The script reads from the hardcoded `docs/综合实验报告.md` path. To convert a different file, either:
1. Edit the `__main__` section in `tools/md_to_docx.py`, or
2. Import and call `convert_md_to_docx(md_path, docx_path)` directly:
   ```python
   import sys; sys.path.insert(0, '.')
   from tools.md_to_docx import convert_md_to_docx
   convert_md_to_docx("docs/some_file.md", "docs/some_file.docx")
   ```

## Updating PRD status tables

Every PRD file under `prd-request/` has a status tracking table in section `## 0. 状态跟踪表`. When you complete a feature module:

1. Identify which PRD file corresponds to the feature
2. Open the file and find the `## 0. 状态跟踪表` section
3. Update the relevant row's columns: 前端/后端/测试/总体 → `✅ 已完成`
4. Keep the table format consistent with existing entries

The table typically looks like:

```markdown
## 0. 状态跟踪表

| 功能模块 | 前端状态 | 后端状态 | 测试状态 | 总体状态 |
|---------|---------|---------|---------|---------|
| 功能名称 | ✅ 已完成 | ✅ 已完成 | ✅ 已完成 | ✅ 已完成 |
```

## Generating new documentation

When creating new documentation:

1. **Place it in the right directory:**
   - Final project docs → `docs/`
   - Design/PRD docs → `prd-request/`
   - Test-related docs → reference them in `tools/`, output goes to `docs/`

2. **Use consistent naming:**
   - Chinese names for project-specific docs (following existing convention)
   - English names for general/tool docs

3. **Include a status table** in every new PRD document:
   ```markdown
   ## 0. 状态跟踪表

   | 功能模块 | 前端状态 | 后端状态 | 测试状态 | 总体状态 |
   |---------|---------|---------|---------|---------|
   | ... | ⬜ 待开始 | ⬜ 待开始 | ⬜ 待开始 | ⬜ 待开始 |
   ```

4. **Update the feature checklist** in `docs/功能清单.md` when adding major new features.

## Organizing scattered docs

If documentation files accumulate outside the standard directories:
1. Analyze each file's content to determine its category
2. Move PRD/design docs → `prd-request/`
3. Move reports/output docs → `docs/`
4. Move utility scripts → `tools/`
5. Remove duplicates and stale backups
6. Update any cross-references between docs

## Report generation workflow

When asked to generate a comprehensive report:

1. Collect all relevant sources (PRDs, test outputs, feature checklists)
2. Compose the report in Markdown, saving to `docs/`
3. Convert to DOCX using `tools/md_to_docx.py`
4. Verify both .md and .docx files are saved correctly
