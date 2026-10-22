"""The map itself: what each model can and cannot be asked, as one table."""
from __future__ import annotations

COLUMNS = ("tool_call", "json_schema", "reasoning", "code", "long_context", "french", "instruction", "vision", "fim", "agentic")
SHORT = {"tool_call": "tools", "json_schema": "json", "reasoning": "reasoning", "code": "code", "long_context": "long ctx",
         "french": "french", "instruction": "rules", "vision": "vision", "fim": "fim", "agentic": "agent"}


def verdict(score) -> str:
    """strong / partial / weak / not applicable."""
    if score is None:
        return "—"
    return "✓" if score >= 0.8 else "~" if score >= 0.5 else "✗"


def cell(task: str, entry: dict | None) -> str:
    if entry is None:
        return "?"
    score = entry.get("score")
    if task == "long_context" and score is not None:
        found = entry.get("max_tokens_found") or 0
        return f"{verdict(score)} {found // 1024}k"
    return f"{verdict(score)} {score:.2f}" if score is not None else "—"


def render(results: list[dict]) -> str:
    """results: one dict per model as saved by the runner. Sorted by group name."""
    head = f"{'model':<24}" + "".join(f"{SHORT[c]:>10}" for c in COLUMNS) + f"{'tok/s':>8}"
    lines = [head, "-" * len(head)]
    for r in sorted(results, key=lambda x: x["group"]):
        res = r["results"]
        tps = (res.get("speed") or {}).get("gen_tps")
        lines.append(f"{r['group']:<24}" + "".join(f"{cell(c, res.get(c)):>10}" for c in COLUMNS) + f"{(tps if tps else '?')!s:>8}")
    lines.append("")
    lines.append("✓ >= 0.80   ~ 0.50-0.79   ✗ < 0.50   — not applicable (no image support / no /infill)   ? not measured")
    return "\n".join(lines)
