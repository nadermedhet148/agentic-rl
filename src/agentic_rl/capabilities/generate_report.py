from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agentic_rl.capabilities.base import Capability, Outcome, Tier

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Report title."},
        "content": {
            "type": "string",
            "description": (
                "Report body. Blank lines start a new paragraph; a line starting with "
                "'# ' is a section heading; a line starting with '- ' is a bullet point."
            ),
        },
        "filename": {"type": "string", "description": "Optional file name, e.g. 'summary.pdf'."},
    },
    "required": ["title", "content"],
    "additionalProperties": False,
}

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_filename(requested: str | None, title: str) -> str:
    """Turns arbitrary (LLM-controlled) title/filename text into a bare, safe
    file name confined to a single path segment — no '..', '/', or '\\' can
    survive the whitelist substitution, so this can never escape reports_dir."""
    base = (requested or title).strip()
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    slug = _UNSAFE_CHARS.sub("-", base).strip("-")[:60]
    return f"{slug or 'report'}.pdf"


def _latin1_safe(text: str) -> str:
    """fpdf2's core fonts (Helvetica/Times/Courier) only support Latin-1/WinAnsi —
    replace anything outside that range instead of crashing on e.g. emoji, smart
    quotes, or non-Latin text that might show up in an LLM answer or search
    snippet. A real Unicode font could be embedded later if this matters more."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


class GenerateReportCapability(Capability):
    """Renders text content as a simple PDF report and saves it under
    reports_dir, served back at /reports/<filename> (see api/app.py). Always
    write-tier — it persists a file to disk."""

    name = "generate_report"
    description = (
        "Generate a PDF report from a title and text content (supports '# heading' and "
        "'- bullet' lines) and save it. Use this once you already have the content to "
        "report on — it doesn't fetch anything itself."
    )
    input_schema = INPUT_SCHEMA

    def __init__(self, reports_dir: Path, max_content_chars: int = 20_000):
        self._reports_dir = reports_dir
        self._max_content_chars = max_content_chars

    def tier_for(self, params: dict[str, Any]) -> Tier:
        return Tier.WRITE

    async def execute(self, params: dict[str, Any]) -> Outcome:
        title = params.get("title")
        content = params.get("content")
        if not title or not content:
            return Outcome(ok=False, error="title and content are required")
        content = str(content)[: self._max_content_chars]

        filename = _safe_filename(params.get("filename"), str(title))
        target = (self._reports_dir / filename).resolve()
        reports_root = self._reports_dir.resolve()
        if reports_root not in target.parents:
            return Outcome(ok=False, error="invalid filename")

        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            # new_x/new_y reset the cursor to the left margin/next line after each
            # cell — fpdf2's default (since 2.x) otherwise leaves x at the cell's
            # right edge, which starves the next multi_cell of horizontal space.
            pdf.multi_cell(0, 10, _latin1_safe(str(title)), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            pdf.set_font("Helvetica", size=12)
            for raw_line in content.split("\n"):
                line = _latin1_safe(raw_line.strip())
                if not line:
                    pdf.ln(4)
                elif line.startswith("# "):
                    pdf.set_font("Helvetica", "B", 14)
                    pdf.multi_cell(0, 8, line[2:].strip(), new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", size=12)
                elif line.startswith("- "):
                    pdf.multi_cell(0, 6, f"- {line[2:].strip()}", new_x="LMARGIN", new_y="NEXT")
                else:
                    pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
            self._reports_dir.mkdir(parents=True, exist_ok=True)
            pdf.output(str(target))
        except Exception as exc:  # noqa: BLE001 - surfaced as a normal failed outcome
            return Outcome(ok=False, error=f"{type(exc).__name__}: {exc}")

        return Outcome(
            ok=True,
            status="200",
            payload={"filename": filename, "url": f"/reports/{filename}", "size_bytes": target.stat().st_size},
        )
