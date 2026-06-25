---
name: pdf
description: >-
  Work with PDF files — read, generate, convert, merge, split, extract text, and fill forms.
  Use this skill whenever the user mentions PDFs, wants to read a PDF, generate a PDF report,
  convert documents to/from PDF (Markdown, HTML, DOCX, images), merge or split PDFs,
  extract text or tables from PDFs, or fill PDF forms. Even if the user doesn't explicitly
  say "PDF", use this skill when they ask to generate reports, export documents, or process
  uploaded documents that might be PDFs.
---

# PDF Skill

Work with PDF files in this project — reading, generating, converting, and manipulating PDFs.

## Project context

This is a Python/FastAPI + React project with Docker deployment. The project already uses `python-docx` for Word document generation. PDF support complements the existing document pipeline.

All PDF-related scripts should be placed in `tools/` and follow the existing naming convention (`tools/pdf_*.py`). Backend tools should be registered in `backend/tools/` following the LangChain `@tool` pattern used by `docx_tool.py`.

## Choosing the right library

| Task | Library | Import |
|------|---------|--------|
| Generate PDF from scratch | `fpdf2` | `from fpdf import FPDF` |
| HTML/CSS → PDF | `weasyprint` | `import weasyprint` |
| Read/extract text from PDF | `pdfplumber` | `import pdfplumber` |
| High-performance text extraction | `pymupdf` (fitz) | `import fitz` |
| Manipulate/merge/split PDFs | `pymupdf` (fitz) | `import fitz` |
| Fill PDF forms | `pypdftk` or `pdf-lib` (JS) | — |

Prefer `fpdf2` for programmatic PDF generation (lightweight, pure Python). Prefer `pdfplumber` for text extraction (better table detection). Use `pymupdf` when you need speed or advanced manipulation (merge, split, rotate, annotations).

## Reading PDFs

### Extract all text with pdfplumber

```python
import pdfplumber

with pdfplumber.open("file.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        print(text)
```

### Extract tables from a specific page

```python
import pdfplumber

with pdfplumber.open("file.pdf") as pdf:
    page = pdf.pages[0]
    tables = page.extract_tables()
    for table in tables:
        for row in table:
            print(row)
```

### Extract text with pymupdf (faster for large PDFs)

```python
import fitz  # pymupdf

doc = fitz.open("file.pdf")
for page in doc:
    text = page.get_text()
    print(text)
```

## Generating PDFs

### From scratch with fpdf2

```python
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)
pdf.cell(200, 10, text="Hello PDF", align="C")
pdf.output("output.pdf")
```

### From Markdown or HTML

First convert Markdown to HTML (using `markdown` library), then use `weasyprint`:

```python
import markdown
import weasyprint

with open("report.md", "r", encoding="utf-8") as f:
    md_text = f.read()

html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

# Wrap in basic HTML with styling
full_html = f"""
<html><head><meta charset="utf-8">
<style>body {{ font-family: Arial, sans-serif; max-width: 800px; margin: auto; }}</style>
</head><body>{html}</body></html>
"""

weasyprint.HTML(string=full_html).write_pdf("report.pdf")
```

### From DOCX (convert existing Word docs)

```python
# Best approach: convert DOCX → HTML → PDF
# Or use a headless LibreOffice in Docker:
# libreoffice --headless --convert-to pdf file.docx
```

## Merging and splitting

```python
import fitz

# Merge multiple PDFs
result = fitz.open()
for pdf_path in ["file1.pdf", "file2.pdf"]:
    result.insert_pdf(fitz.open(pdf_path))
result.save("merged.pdf")

# Split: extract pages 0-2 as a new PDF
doc = fitz.open("input.pdf")
new = fitz.open()
new.insert_pdf(doc, from_page=0, to_page=2)
new.save("split_output.pdf")
```

## Integration patterns for this project

### As a LangChain tool (backend/tools/pdf_tool.py)

Follow the same `@tool` decorator pattern from `backend/tools/docx_tool.py`:

```python
from langchain_core.tools import tool

@tool
def create_pdf(title: str, content: str) -> str:
    """Generate a PDF report from markdown content."""
    # ... generate PDF
    return f"PDF created: {path}"
```

### As a standalone script (tools/pdf_*.py)

Follow the pattern from `tools/md_to_docx.py` — accept input file path, output file path, print confirmation.

### Register in Docker

If a library needs system dependencies (e.g., weasyprint needs GTK), add them to `Dockerfile.backend`:

```dockerfile
RUN apt-get update && apt-get install -y libgtk-3-dev
```

## Dependencies

Add to `backend/requirements.txt` as needed:
```
fpdf2>=2.7
pdfplumber>=0.10
pymupdf>=1.23
weasyprint>=60
markdown>=3.5
```

Install with: `pip install <package>` (the virtual environment must be activated first per CLAUDE.md rule #4).
