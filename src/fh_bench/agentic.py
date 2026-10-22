"""Optional stage: an agent session on a small repository, judged by running its tests.

Everything else in the battery asks a model questions; this lets it WORK, through an agent harness with an
HTTP API (OpenFox). A session is pinned on the model under test, and the verdict is not what the model says
it did but what runs afterwards. Two cases: read a file and answer from it, fix a bug and prove it.

OFF by default. It runs only when FH_BENCH_OPENFOX names the harness (e.g. http://127.0.0.1:10369) and
FH_BENCH_OPENFOX_TOKEN_FILE (default ~/.config/fh-bench/openfox-token) holds its session token. Without
them the `agentic` task is reported as skipped and every other task runs as usual.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

DATA = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "fh-bench")
OPENFOX = os.environ.get("FH_BENCH_OPENFOX", "")
TOKEN_FILE = os.path.expanduser(os.environ.get("FH_BENCH_OPENFOX_TOKEN_FILE", "~/.config/fh-bench/openfox-token"))
WORKDIR = os.path.expanduser(os.environ.get("FH_BENCH_WORKDIR", os.path.join(DATA, "agentic-work")))


def enabled(base: str | None = None) -> bool:
    """The agent stage runs only when a harness is configured."""
    return bool(OPENFOX if base is None else base)

CALC = "def average(xs):\n    return sum(xs) / (len(xs) + 1)\n"
TEST_CALC = ("import unittest\nfrom calc import average\n\n\nclass T(unittest.TestCase):\n"
             "    def test_three(self):\n        self.assertEqual(average([2, 4, 6]), 4)\n\n"
             "    def test_one(self):\n        self.assertEqual(average([10]), 10)\n\n\nif __name__ == '__main__':\n    unittest.main()\n")
NOTES = "Maintenance notes\nThe deployment window is Thursday 02:30, never during a game session.\n"

READ_TASK = "Read notes.txt and tell me, in one sentence, when the deployment window is."
FIX_TASK = "The tests in test_calc.py fail. Find the bug in calc.py, fix it, and run `python3 test_calc.py` to prove they pass."


def reset_workspace(directory: str = WORKDIR) -> None:
    """A fresh repository: a bug, its failing tests, and a note to read."""
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)
    for name, text in (("calc.py", CALC), ("test_calc.py", TEST_CALC), ("notes.txt", NOTES)):
        with open(os.path.join(directory, name), "w") as f:
            f.write(text)


def tests_pass(directory: str = WORKDIR) -> bool:
    p = subprocess.run([sys.executable, "test_calc.py"], cwd=directory, capture_output=True, timeout=30, check=False)
    return p.returncode == 0


def answer_ok(text: str) -> bool:
    """The day and the hour, in English or French (a harness may make the agent answer in French)."""
    low = text.lower()
    return any(d in low for d in ("thursday", "jeudi")) and any(h in low for h in ("02:30", "2:30", "2h30", "2 h 30", "2h"))


class OpenFox:
    def __init__(self, base: str = OPENFOX, token_file: str = TOKEN_FILE):
        self.base = base.rstrip("/")
        with open(token_file) as f:
            self.token = f.read().strip()

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 30.0) -> dict:
        req = urllib.request.Request(self.base + path, json.dumps(body).encode() if body is not None else None,
                                     {"x-session-token": self.token, "Content-Type": "application/json"}, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)


def running_sessions(fox: OpenFox | None = None) -> int:
    """How many OpenFox sessions are working right now."""
    fox = fox or OpenFox()
    return sum(1 for s in fox.call("GET", "/api/sessions?limit=50").get("sessions", []) if s.get("isRunning"))


def wait_for_model(get_model, model: str, seconds: float = 90.0, sleep=time.sleep, now=time.monotonic) -> bool:
    """A harness may take a few seconds to notice the model that was just loaded; wait until it is on `model`."""
    deadline = now() + seconds
    while now() < deadline:
        if get_model() == model:
            return True
        sleep(3)
    return False


def next_quiet(quiet: int, running: bool, messages: list) -> int:
    """Consecutive quiet polls. Not counted until an assistant message exists: right after the prompt a
    session is not yet 'running', and counting that moment ended the case before the agent began."""
    started = any(m.get("role") == "assistant" for m in messages)
    return quiet + 1 if (not running and started) else 0


def run_case(fox: OpenFox, project: str, model: str, prompt: str, patience: float = 360.0) -> dict:
    """One agent session on `model`; returns what it did (tool calls, seconds, final words)."""
    sid = fox.call("POST", "/api/sessions", {"projectId": project, "title": "bench: " + prompt[:40]})["session"]["id"]
    try:
        config = fox.call("GET", "/api/config")
        for p in config.get("providers") or []:
            if any(m.get("id") == model for m in p.get("models") or []):
                fox.call("POST", f"/api/sessions/{sid}/provider", {"providerId": p["id"], "model": model})
        fox.call("POST", f"/api/sessions/{sid}/message", {"content": prompt})
        started, quiet = time.monotonic(), 0
        while time.monotonic() - started < patience:
            time.sleep(5)
            d = fox.call("GET", f"/api/sessions/{sid}")
            quiet = next_quiet(quiet, bool(d["session"].get("isRunning")), d.get("messages", []))
            if quiet >= 2:
                break
        messages = d.get("messages", [])
        calls = sum(len(m.get("toolCalls") or []) for m in messages if m.get("role") == "assistant")
        last = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "assistant" and m.get("content")), "")
        return {"tool_calls": calls, "seconds": round(time.monotonic() - started), "last": last[:300], "timed_out": quiet < 2}
    finally:
        try:
            fox.call("DELETE", f"/api/sessions/{sid}")
        except OSError:
            pass


def evaluate(model: str, tool_score: float | None, fox: OpenFox | None = None) -> dict:
    """The agentic score of a model already loaded: the fraction of the two cases that really worked."""
    if tool_score is not None and tool_score < 0.5:
        return {"score": 0.0, "n": 0, "skipped": "no native tool calling (tool_call < 0.5)"}
    if fox is None and not enabled():
        return {"score": None, "n": 0, "skipped": "agent harness not configured (set FH_BENCH_OPENFOX)"}
    fox = fox or OpenFox()
    if not wait_for_model(lambda: fox.call("GET", "/api/config").get("model"), model):
        return {"score": None, "n": 0, "skipped": "the agent harness did not switch to the model"}
    projects = fox.call("GET", "/api/projects").get("projects", [])
    reset_workspace()
    project = next((p["id"] for p in projects if p.get("name") == "bench-agentic"), None)
    if project is None:
        project = fox.call("POST", "/api/projects", {"name": "bench-agentic", "workdir": WORKDIR})["project"]["id"]
    read = run_case(fox, project, model, READ_TASK)
    fix = run_case(fox, project, model, FIX_TASK)
    passed = [1.0 if answer_ok(read["last"]) else 0.0, 1.0 if tests_pass() else 0.0]
    return {"score": round(sum(passed) / 2, 3), "n": 2, "read_ok": bool(passed[0]), "fix_ok": bool(passed[1]),
            "tool_calls": read["tool_calls"] + fix["tool_calls"], "seconds": read["seconds"] + fix["seconds"],
            "timed_out": read["timed_out"] or fix["timed_out"]}
