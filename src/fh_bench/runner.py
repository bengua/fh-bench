"""Load each configured server, run the battery, keep and report the results."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time

from . import agentic, client, tasks
from .config import Server

TASK_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, RuntimeError, subprocess.SubprocessError)
EVENT_CMD_ENV = "FH_BENCH_EVENT_CMD"


def game_running(which=shutil.which, run=subprocess.run) -> bool:
    """A game holds an NVIDIA card: do not take it from under it. False when there is no nvidia-smi."""
    if not which("nvidia-smi"):
        return False
    try:
        p = run(["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return any(w in p.stdout.lower() for w in (".exe", "steam", "wine"))


def busy_reason(sessions_running=None) -> str:
    """Why the machine should not be used for a long measurement now ('' when it is free).

    `sessions_running` is the number of agent sessions at work on the optional agent harness
    (asked when None and the harness is configured)."""
    if game_running():
        return "a game is running"
    if sessions_running is None:
        if not agentic.enabled():
            return ""
        try:
            sessions_running = agentic.running_sessions()
        except (OSError, ValueError, KeyError):
            return ""                                           # harness unreachable: nothing is running there
    return f"{sessions_running} agent session(s) at work" if sessions_running else ""


def ceiling_reached(output: str) -> bool:
    """turnstile, used as a loader, refuses past a number of swaps an hour: worth waiting for, not a failure."""
    return "swaps in the last hour" in output


def wait_up(server: Server, tries: int = 60, sleep=time.sleep, reachable=client.reachable) -> bool:
    for _ in range(tries):
        if reachable(server.base_url, server.api_key):
            return True
        sleep(2)
    return False


def load(server: Server, say=print, patience: int = 3600, sleep=time.sleep, reachable=client.reachable) -> bool:
    """Run the server's optional `load` command, then wait until the server answers.

    A loader that refuses because of an hourly swap ceiling (turnstile's wording) is waited for and asked
    again, up to `patience` seconds, instead of failing in silence."""
    if server.load:
        waited = 0
        while True:
            try:
                p = subprocess.run(server.load, capture_output=True, text=True, timeout=420, check=False)
            except (OSError, subprocess.SubprocessError) as e:
                say(f"  ✗ {server.name}: the load command failed ({type(e).__name__}: {e})"[:200])
                return False
            said = (p.stdout + p.stderr).strip()
            if p.returncode == 0 and not said.startswith("refused"):
                break
            if ceiling_reached(said) and waited < patience:
                say(f"  … {server.name}: the loader is at its swap ceiling, waiting ({waited}s)")
                sleep(120)
                waited += 120
                continue
            say(f"  ✗ {server.name}: the load command said {said[-140:]}")
            return False
    if wait_up(server, sleep=sleep, reachable=reachable):
        return True
    say(f"  ✗ {server.name}: {server.base_url} does not answer")
    return False


def after(command: list[str], say=print) -> None:
    """The optional command run once the bench is done (puts the machine back as it was)."""
    if not command:
        return
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
        say(f"  ↩ {' '.join(command)}: exit {p.returncode}")
    except (OSError, subprocess.SubprocessError) as e:
        say(f"  ↩ {' '.join(command)}: {type(e).__name__}")


def run_server(server: Server, names: list[str], say=print) -> dict:
    ctx = tasks.Ctx(base=server.base_url, server=server.name, model=server.model, api_key=server.api_key,
                    context=server.context, vision=server.vision)
    out = {}
    for name in names:
        started = time.perf_counter()
        ctx.trace.clear()
        ctx.stats.update(calls=0, tokens=0, truncated=0)
        try:
            if name == "agentic":
                out[name] = agentic.evaluate(server.model or server.name, (out.get("tool_call") or {}).get("score"))
            else:
                r = tasks.BATTERY[name](ctx)
                out[name] = {"score": r.score, "n": r.n, **r.detail, "samples": list(ctx.trace),
                             "tokens_per_call": round(ctx.stats["tokens"] / ctx.stats["calls"]) if ctx.stats["calls"] else None,
                             "truncated": ctx.stats["truncated"]}
        except TASK_ERRORS as e:                                 # a task that crashes is a result, not a stop
            out[name] = {"score": None, "n": 0, "crash": f"{type(e).__name__}: {e}"[:160]}
        say(f"    {name:<13} {out[name]['score']!s:<6} ({time.perf_counter() - started:.0f}s)")
    return out


# A variant (another quantisation, card or engine) serves the same model differently: it is measured on speed AND
# on what a router relies on, so that a server which cannot call a tool is never chosen for work that needs one.
VARIANT_TASKS = ["speed", "tool_call", "json_schema"]


def measure(group: str, servers: list[Server], names: list[str], variants: bool = True, say=print) -> dict | None:
    """The whole battery on the first server of the group, the short one on the others."""
    primary = servers[0]
    say(f"■ {group}  ({primary.name}, window {primary.context})")
    if not load(primary, say):
        return None
    result = {"group": group, "server": primary.name, "context": primary.context, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "results": run_server(primary, names, say), "variants": {}}
    if variants:
        for s in servers[1:]:
            say(f"  ▫ {s.name}")
            if load(s, say):
                result["variants"][s.name] = run_server(s, VARIANT_TASKS, say)
    return result


def file_name(group: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in group) + ".json"


def save(result: dict, directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, file_name(result["group"]))
    try:
        with open(path) as f:
            kept = json.load(f)
        kept["results"].update(result["results"])
        kept["variants"].update(result["variants"])
        result = {**result, "results": kept["results"], "variants": kept["variants"]}
    except (OSError, ValueError, KeyError):
        pass
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    return path


def emit(result: dict, sender=None) -> int:
    """One `bench.result` event per task, one `bench.speed` per variant and one `bench.variant` per scored variant task."""
    sender = sender or send_event
    n = 0
    for task, r in result["results"].items():
        sender("bench.result", {"group": result["group"], "server": result["server"], "task": task, "context": result["context"],
                                "measured_at": result["measured_at"], **{k: v for k, v in r.items()}})
        n += 1
    for server, tasks_ in result["variants"].items():
        s = tasks_.get("speed", {})
        sender("bench.speed", {"group": result["group"], "server": server, "gen_tps": s.get("gen_tps"), "prefill_tps": s.get("prefill_tps"),
                               "measured_at": result["measured_at"]})
        n += 1
        for task, r in tasks_.items():
            if task != "speed" and r.get("score") is not None:
                sender("bench.variant", {"group": result["group"], "server": server, "task": task, "score": r["score"],
                                         "measured_at": result["measured_at"]})
                n += 1
    return n


def send_event(kind: str, detail: dict, out=print) -> bool:
    """One event as a JSON line on stdout; or, when $FH_BENCH_EVENT_CMD is set, handed to that command
    as two arguments (the kind, then the JSON detail) — e.g. a script that posts it to your own event store."""
    line = json.dumps(detail, separators=(",", ":"))
    command = os.environ.get(EVENT_CMD_ENV, "")
    if not command:
        out(json.dumps({"kind": kind, **detail}, separators=(",", ":")))
        return True
    try:
        p = subprocess.run([*shlex.split(command), kind, line], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  ! {EVENT_CMD_ENV}: {type(e).__name__}: {e}"[:160])
        return False
    if p.returncode != 0:
        print(f"  ! {EVENT_CMD_ENV} exit {p.returncode}: {p.stderr.strip()[:120]}")
    return p.returncode == 0
