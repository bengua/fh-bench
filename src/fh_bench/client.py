"""A chat client for any OpenAI-compatible server (llama.cpp, Ollama, vLLM, LiteLLM...), standard library only."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Reply:
    """What came back, or why nothing did. `error` is set when the call failed or the server refused."""

    text: str = ""
    tool_calls: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    gen_tps: float = 0.0       # generated tokens per second, as the server timed them (llama.cpp only; else 0)
    prompt_tps: float = 0.0
    finish: str = ""          # why the server stopped: stop, length (the budget ran out), tool_calls
    reasoning: str = ""       # the model's separate reasoning channel, when the server splits it from the answer
    error: str = ""


def root(base: str) -> str:
    """The server root: `http://h:8080`, `http://h:8080/` and `http://h:8080/v1` all name the same server."""
    base = base.rstrip("/")
    return base.removesuffix("/v1")


def headers(api_key: str = "") -> dict:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = "Bearer " + api_key
    return h


def post(url: str, body: dict, timeout: float = 600.0, api_key: str = "") -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), headers(api_key))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def request_body(messages: list, model: str, max_tokens: int, temperature: float, **extra) -> dict:
    body: dict = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if model:
        body["model"] = model
    body.update({k: v for k, v in extra.items() if v})
    return body


def chat(base: str, messages: list, *, tools: list | None = None, response_format: dict | None = None,
         max_tokens: int = 600, temperature: float = 0.0, timeout: float = 600.0, model: str = "", api_key: str = "",
         transport=post) -> Reply:
    body = request_body(messages, model, max_tokens, temperature, tools=tools, response_format=response_format)
    started = time.perf_counter()
    try:
        d = transport(root(base) + "/v1/chat/completions", body, timeout, api_key)
    except urllib.error.HTTPError as e:
        return Reply(error=f"HTTP {e.code}: {e.read().decode(errors='replace')[:160]}", seconds=time.perf_counter() - started)
    except (OSError, ValueError) as e:
        return Reply(error=f"{type(e).__name__}: {e}"[:200], seconds=time.perf_counter() - started)
    choice = (d.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = d.get("usage") or {}
    timings = d.get("timings") or {}
    return Reply(text=str(message.get("content") or ""), tool_calls=message.get("tool_calls") or [],
                 finish=str(choice.get("finish_reason") or ""), reasoning=str(message.get("reasoning_content") or ""),
                 prompt_tokens=int(usage.get("prompt_tokens") or 0), completion_tokens=int(usage.get("completion_tokens") or 0),
                 seconds=time.perf_counter() - started, gen_tps=float(timings.get("predicted_per_second") or 0),
                 prompt_tps=float(timings.get("prompt_per_second") or 0))


def open_stream(url: str, body: dict, timeout: float, api_key: str = ""):
    """The response lines of a streamed request (an iterable of bytes)."""
    req = urllib.request.Request(url, json.dumps(body).encode(), headers(api_key))
    return urllib.request.urlopen(req, timeout=timeout)


def stream_speed(base: str, messages: list, *, max_tokens: int = 300, model: str = "", api_key: str = "",
                 timeout: float = 600.0, opener=open_stream, clock=time.perf_counter) -> dict:
    """Speed measured the same way on every server: a streamed reply, timed by the client.

    prefill = the wait until the first token; generation = the tokens that followed, over the time they took.
    The token counts come from the server's `usage` when it sends one, else one streamed chunk counts as one
    token (true of llama.cpp, Ollama and vLLM). When the server also reports its own timings (llama.cpp),
    those are kept alongside, under `server_*`.
    """
    body = request_body(messages, model, max_tokens, 0.0)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    started = clock()
    first = last = None
    chunks, usage, timings = 0, {}, {}
    try:
        with opener(root(base) + "/v1/chat/completions", body, timeout, api_key) as lines:
            for raw in lines:
                line = raw.decode(errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    d = json.loads(data)
                except ValueError:
                    continue
                usage = d.get("usage") or usage
                timings = d.get("timings") or timings
                delta = ((d.get("choices") or [{}])[0].get("delta") or {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    now = clock()
                    first = now if first is None else first
                    last = now
                    chunks += 1
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:120]}"}
    except (OSError, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}"[:160]}
    if first is None:
        return {"error": "no token streamed"}
    generated = int(usage.get("completion_tokens") or chunks)
    prompt = int(usage.get("prompt_tokens") or 0)
    gen_seconds = (last - first) if last > first else 0.0
    out = {"ttft_s": round(first - started, 3), "seconds": round(last - started, 2), "completion_tokens": generated,
           "prompt_tokens": prompt, "gen_tps": round((generated - 1) / gen_seconds, 1) if gen_seconds and generated > 1 else 0.0,
           "prefill_tps": round(prompt / (first - started)) if prompt and first > started else 0, "counted": "usage" if usage else "chunks"}
    if timings:
        out["server_gen_tps"] = round(float(timings.get("predicted_per_second") or 0), 1)
        out["server_prefill_tps"] = round(float(timings.get("prompt_per_second") or 0))
    return out


def reachable(base: str, api_key: str = "", timeout: float = 5.0) -> bool:
    """True when the server answers: llama.cpp's /health, else the standard /v1/models."""
    for path in ("/health", "/v1/models"):
        req = urllib.request.Request(root(base) + path, headers=headers(api_key))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status == 200:
                    return True
        except (OSError, ValueError):
            continue
    return False
