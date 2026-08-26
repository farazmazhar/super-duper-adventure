"""CLI rendering — turn agent answers + render_hints into readable ASCII output.

Renders the same `{kind, data, spec}` payloads the FE consumes, as text:
tables (aligned columns), bar charts (ASCII), KPI cards, and clean sections.
"""

from __future__ import annotations

from typing import Any


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def render_table(rows: list[dict[str, Any]] | None, title: str | None = None) -> str:
    """Render rows as a left-aligned ASCII table."""
    rows = rows or []
    if not rows:
        return "  (no data)"
    cols = list(rows[0].keys())
    # column widths
    widths = {c: max(len(c), max((len(_fmt(r.get(c))) for r in rows), default=0)) for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)

    lines: list[str] = []
    if title:
        lines.append(f"▌{title}")
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    lines.append(header)
    lines.append(sep)
    for r in rows:
        lines.append("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Charts (ASCII bar charts from {kind: chart, data, spec})
# ---------------------------------------------------------------------------
def render_chart(payload: dict[str, Any]) -> str:
    """Render a chart payload as an ASCII bar chart (best-effort)."""
    spec = payload.get("spec") or {}
    data = payload.get("data") or []
    if not data:
        return "  (no chart data)"
    chart_type = spec.get("type", "bar")
    x_key = spec.get("x")
    y_key = spec.get("y")
    if chart_type == "line":
        return _render_line(data, x_key, y_key, spec.get("title", ""))
    return _render_bar(data, x_key, y_key, spec.get("title", ""))


def _render_bar(data: list[dict[str, Any]], x_key: str | None, y_key: str | None, title: str) -> str:
    rows = data[:12]
    if not x_key or not y_key or x_key not in rows[0] or y_key not in rows[0]:
        return render_table(data, title)
    max_v = max((r.get(y_key) or 0 for r in rows), default=1) or 1
    width = 40
    lines = [f"▌{title}" if title else "▌Chart"]
    for r in rows:
        label = str(r.get(x_key, ""))[:18].ljust(18)
        v = r.get(y_key) or 0
        bar = "█" * max(1, round(v / max_v * width))
        lines.append(f"  {label} {bar} {v}")
    return "\n".join(lines)


def _render_line(data: list[dict[str, Any]], x_key: str | None, y_key: str | None, title: str) -> str:
    rows = data[:30]
    if not rows or not x_key or not y_key or x_key not in rows[0] or y_key not in rows[0]:
        return render_table(data, title)
    ys = [r.get(y_key) or 0 for r in rows]
    max_v = max(ys) or 1
    height = 10
    lines = [f"▌{title}" if title else "▌Chart"]
    for h in range(height, 0, -1):
        threshold = max_v * h / height
        line = "".join("█" if v >= threshold else " " for v in ys)
        lines.append(f"  {line}")
    labels = [str(r.get(x_key, ""))[5:10] for r in rows]
    lines.append("  " + "".join(l[:1] for l in labels))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
def render_cards(kpis: list[dict[str, Any]] | None) -> str:
    """Render KPI cards as a compact grid."""
    kpis = kpis or []
    if not kpis:
        return "  (no metrics)"
    lines: list[str] = []
    for kpi in kpis:
        label = kpi.get("label", "")
        value = kpi.get("value", "")
        delta = kpi.get("delta")
        d = f"  ({delta})" if delta else ""
        lines.append(f"  ◆ {label:<22} {_fmt(value)}{d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer sections
# ---------------------------------------------------------------------------
def render_answer_text(answer: Any) -> str:
    """Render a full answer (dict or pydantic) as readable text with the visual."""
    if hasattr(answer, "model_dump"):
        answer = answer.model_dump()
    if not isinstance(answer, dict):
        return str(answer)

    out: list[str] = []
    summary = answer.get("summary") or ""
    confidence = answer.get("confidence", "")
    if summary:
        out.append(f"◆ {summary}")
    if confidence:
        out.append(f"  Confidence: {confidence}")

    # primary visual from render_hint
    hint = answer.get("render_hint") or {}
    if hasattr(hint, "model_dump"):
        hint = hint.model_dump()
    payload = hint.get("payload") if isinstance(hint, dict) else {}
    if not isinstance(payload, dict):
        payload = {}
    kind = payload.get("kind") or hint.get("kind", "markdown")
    if kind == "table":
        out.append("")
        out.append(render_table(payload.get("data"), payload.get("spec", {}).get("title")))
    elif kind == "chart":
        out.append("")
        out.append(render_chart(payload))
    elif kind == "cards":
        out.append("")
        out.append(render_cards(payload.get("data") or payload.get("kpis")))

    # sections
    for section_name in ("facts", "interpretation", "recommendation", "other_sections"):
        sections = answer.get(section_name) or []
        for sec in sections:
            heading = sec.get("heading", section_name)
            content = (sec.get("content") or "").strip()
            if not content:
                continue
            out.append("")
            out.append(f"── {heading} ──")
            # simple bullet formatting for multi-line / list-ish content
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                out.append(f"  • {line}")
            cites = sec.get("citations") or []
            if cites:
                out.append(f"  📎 {', '.join(cites)}")

    return "\n".join(out)
