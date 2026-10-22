"""Pure scoring: no network, no model. Everything a task decides about an answer lives here, so it is tested."""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile

from . import french

REFUSALS = ("i can't", "i cannot", "i can not", "i'm sorry", "i am sorry", "i won't", "as an ai", *french.REFUSAL_MARKERS)


def strip_thinking(text: str) -> str:
    """The answer without a <think>…</think> block (an unclosed one means the model never answered)."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return "" if text.lstrip().startswith("<think>") else text.strip()


def last_token(text: str) -> str:
    """The last number, else the last word, of an answer: 'The answer is 42.' -> '42'."""
    text = strip_thinking(text)
    numbers = re.findall(r"-?\d+(?::\d+)?", text.replace(",", ""))
    if numbers:
        return numbers[-1]
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text)
    return words[-1] if words else ""


def code_block(text: str) -> str:
    """The first fenced code block of an answer, else the answer itself."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", strip_thinking(text), re.DOTALL)
    return (m.group(1) if m else strip_thinking(text)).strip()


def run_python(code: str, tests: str, timeout: float = 8.0) -> bool:
    """True when `code` followed by `tests` runs clean in a throwaway interpreter (no env, no network keys, CPU and memory capped)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "case.py")
        with open(path, "w") as f:
            f.write(code + "\n\n" + tests + "\n")
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=tmp, env={}, capture_output=True, timeout=timeout,
                               preexec_fn=_limit, check=False)
        except subprocess.TimeoutExpired:
            return False
        return p.returncode == 0


def _limit() -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def refused(text: str) -> bool:
    low = strip_thinking(text).lower()
    return len(low) < 150 or any(p in low for p in REFUSALS)


def french_ratio(text: str) -> float:
    words = re.findall(r"[a-zà-ÿ']+", strip_thinking(text).lower())
    return sum(1 for w in words if w in french.WORDS) / len(words) if words else 0.0


def tool_call_args(call: dict) -> dict:
    raw = (call.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw or "{}")
        return out if isinstance(out, dict) else {}
    except ValueError:
        return {}


def text_calls(text: str) -> list[dict]:
    """The function calls a model WROTE as text, in the usual shapes, as OpenAI-style tool calls.

    A model can know how to call a function and still not be heard by the server, which only
    recognises the format its chat template declares: a JSON object or list, a Python literal list,
    in a code fence or in <tool_call> tags. This reads those, so the map can say "calls tools, but
    only through a text parser" rather than "cannot".
    """
    calls = []
    body = strip_thinking(text)
    for chunk in re.findall(r"<tool_call>(.*?)</tool_call>", body, re.DOTALL) + re.findall(r"```(?:json|python)?\s*(.*?)```", body, re.DOTALL) + [body]:
        chunk = chunk.strip()
        for start, end in (("[", "]"), ("{", "}")):
            i, j = chunk.find(start), chunk.rfind(end)
            if i < 0 or j <= i:
                continue
            for parse in (json.loads, ast.literal_eval):
                try:
                    data = parse(chunk[i:j + 1])
                except (ValueError, SyntaxError):
                    continue
                for item in data if isinstance(data, list) else [data]:
                    if isinstance(item, dict) and "name" in item and ("arguments" in item or "parameters" in item):
                        args = item.get("arguments", item.get("parameters"))
                        calls.append({"function": {"name": item["name"], "arguments": args if isinstance(args, dict) else {}}})
                break
            if calls:
                return calls
    return calls


def tool_call_name(call: dict) -> str:
    return str((call.get("function") or {}).get("name") or "")
