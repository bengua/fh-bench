"""The scorers and the battery against scripted servers: a right answer scores, a wrong one does not."""
import json

from fh_bench import cli, client, config, french, mapping, runner, scorers, tasks


def reply(text="", calls=None, **kw):
    return client.Reply(text=text, tool_calls=calls or [], **kw)


def ctx_with(fn, **kw):
    return tasks.Ctx(base="http://x", chat=lambda base, messages, **k: fn(messages[-1]["content"], k), **kw)


def call(name, **args):
    return {"function": {"name": name, "arguments": json.dumps(args)}}


# ── scorers ──────────────────────────────────────────────────────────────────

def test_thinking_is_stripped_and_an_unfinished_thought_is_no_answer():
    assert scorers.strip_thinking("<think>hmm</think>\nParis") == "Paris"
    assert scorers.strip_thinking("<think>still thinking") == ""
    assert scorers.strip_thinking("plain") == "plain"


def test_the_answer_is_the_last_number_or_word():
    assert scorers.last_token("So 17 * 23 = 391.") == "391"
    assert scorers.last_token("The youngest is Carol") == "Carol"
    assert scorers.last_token("<think>2+2</think>It arrives at 17:25") == "17:25"
    assert scorers.last_token("1,000 euros") == "1000"


def test_code_comes_from_the_fenced_block():
    assert scorers.code_block("Here:\n```python\nx = 1\n```\nDone") == "x = 1"
    assert scorers.code_block("x = 2") == "x = 2"


def test_generated_code_is_run_and_a_bug_fails():
    assert scorers.run_python("def f(): return 1", "assert f() == 1")
    assert not scorers.run_python("def f(): return 2", "assert f() == 1")
    assert not scorers.run_python("def f(:", "")                                       # a syntax error is a failure
    assert not scorers.run_python("while True: pass", "", timeout=2)                   # a loop is stopped
    assert not scorers.run_python("import os\nassert os.environ.get('HOME')", "")      # it runs without the environment


def test_refusals_and_french_are_detected():
    assert scorers.refused("I'm sorry, I can't help with that.") and scorers.refused("no")
    assert not scorers.refused("A pin tumbler lock has a row of pins of different lengths. " * 4)
    assert scorers.french_ratio(french.SAMPLE) > 0.2
    assert scorers.french_ratio("The sky is blue because of scattering") == 0.0


# ── the battery ──────────────────────────────────────────────────────────────

def test_tool_calling_scores_what_was_called_and_what_was_not():
    def good(prompt, kw):
        if "weather in Lyon" in prompt:
            return reply(calls=[call("get_weather", city="Lyon")])
        if "add 1234" in prompt:
            return reply(calls=[call("add", a=1234, b=5678)])
        if "backup" in prompt:
            return reply(calls=[call("search_docs", query="backup rotation", limit=3)])
        if "timer" in prompt:
            return reply(calls=[call("set_timer", seconds=90, label="pasta")])
        if "capital" in prompt:
            return reply("Paris")
        return reply(calls=[call("get_weather", city="Paris"), call("get_weather", city="Tokyo")])

    assert tasks.tool_call(ctx_with(good)).score == 1.0
    as_text = tasks.tool_call(ctx_with(lambda p, k: reply('{"name": "get_weather", "arguments": {"city": "Lyon"}}')))
    assert as_text.score == 0.0 and "wrote the call as text" in as_text.detail["notes"]
    assert tasks.tool_call(ctx_with(lambda p, k: client.Reply(error="HTTP 400"))).score == 0.0


def test_constrained_json_must_carry_the_right_values():
    answers = {"Marie": {"name": "Marie Curie", "birth_year": 1867, "city": "Warsaw", "nobel_prizes": 2},
               "Order": {"order": 8841, "items": 3, "price": 59.9},
               "Alan": {"name": "Alan Turing", "city": "London", "year": 1912},
               "cake": {"label": "positive", "confidence": 0.9},
               "Meeting": {"date": "2026-10-14", "time": "15:30", "room": "B12"}}

    def good(prompt, kw):
        return reply(json.dumps(next(v for k, v in answers.items() if k in prompt)))

    assert tasks.json_schema(ctx_with(good)).score == 1.0
    assert tasks.json_schema(ctx_with(lambda p, k: reply("not json"))).score == 0.0


def test_reasoning_reads_the_final_answer_whatever_the_wording():
    table = {q: a for q, a in tasks.REASONING}
    assert tasks.reasoning(ctx_with(lambda p, k: reply(f"<think>x</think>The answer is {table[p]}."))).score == 1.0
    assert tasks.reasoning(ctx_with(lambda p, k: reply("<think>never finishes"))).score == 0.0


def test_the_duplicate_order_trap_is_caught():
    """Two 27B models answered 'duplicates in order of the second occurrence': the test exists to catch it."""
    wrong = "def dedupe_ordered(items):\n    seen, out = set(), []\n    for x in items:\n        if x in seen and x not in out:\n            out.append(x)\n        seen.add(x)\n    return out"
    right = "def dedupe_ordered(items):\n    from collections import Counter\n    c = Counter(items)\n    out = []\n    for x in items:\n        if c[x] > 1 and x not in out:\n            out.append(x)\n    return out"
    _, tests = tasks.CODE[0]
    assert not scorers.run_python(wrong, tests)
    assert scorers.run_python(right, tests)


def test_code_scores_the_tasks_whose_tests_pass():
    solutions = {
        "dedupe_ordered": "from collections import Counter\ndef dedupe_ordered(items):\n    c = Counter(items)\n    out = []\n    for x in items:\n        if c[x] > 1 and x not in out:\n            out.append(x)\n    return out",
    }

    def only_first(prompt, kw):
        return reply("```python\n" + (solutions["dedupe_ordered"] if "dedupe_ordered" in prompt else "def nothing(): pass") + "\n```")

    r = tasks.code(ctx_with(only_first))
    assert r.n == 6 and r.score == round(1 / 6, 3)


def test_the_haystack_holds_the_needle_at_about_the_asked_size():
    text = tasks.haystack(4096)
    assert tasks.NEEDLE in text
    assert 4096 * 3.0 < len(text) < 4096 * 4.5


def test_long_context_climbs_until_the_model_loses_the_needle_or_the_window():
    def finds_up_to_16k(prompt, kw):
        return reply("4417-KESTREL" if len(prompt) < 16384 * 4 else "I don't know", prompt_tps=1000.0)

    r = tasks.long_context(ctx_with(finds_up_to_16k, context=131072))
    assert r.detail["max_tokens_found"] == 16384 and r.score == round(2 / 3, 3)         # 4k ok, 16k ok, 32k lost
    short = tasks.long_context(ctx_with(lambda p, k: reply("4417-KESTREL"), context=8192))
    assert short.n == 1 and short.detail["max_tokens_found"] == 4096                  # the window caps the climb


def test_vision_is_unsupported_without_a_projector_and_scored_with_one():
    assert tasks.vision(ctx_with(lambda p, k: reply("red"))).score is None
    seen = []

    def sees(prompt, kw):
        seen.append(prompt)
        return reply("red")

    r = tasks.vision(tasks.Ctx(base="x", vision=True, chat=lambda b, m, **k: sees(m[-1]["content"][1]["text"], k)))
    assert r.n == 4 and r.score == 0.25                                                # only the red square was right


def test_a_png_is_a_png():
    data = tasks.png(8, 8, lambda x, y: (255, 0, 0))
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and b"IHDR" in data and data.endswith(b"IEND\xaeB`\x82")


def test_instructions_and_french_and_refusals_are_judged_by_code():
    perfect = {
        "three fruits": "- apple\n- pear\n- plum", "word YES": "YES", "exactly 8 words": "The blue sea rolls under a pale sky.",
        "JSON object": '{"a": 1, "b": 2}', "lowercase": "blue, the sky",
    }
    assert tasks.instruction(ctx_with(lambda p, k: reply(next(v for key, v in perfect.items() if key in p)))).score == 1.0
    assert tasks.refusal(ctx_with(lambda p, k: reply("I'm sorry, I can't help."))).score == 0.0
    assert tasks.refusal(ctx_with(lambda p, k: reply("Here is how it works, step by step. " * 10))).score == 1.0


# ── the runner and the map ───────────────────────────────────────────────────

MAPTEXT = """
# comment
Q-9B-llamacpp-c0|8105|0|Q-9B|-m /models/a.gguf -c 196608 --parallel 1
Q-9B-llamacpp-c1|8095|1|Q-9B|-m /models/a.gguf -c 196608
G-12B-llamacpp-c1|8101|1|G-12B|-m /models/g.gguf --mmproj /models/p.gguf -c 65536
H-3B-llamacpp-cpu|8084|cpu|H-3B|-m /models/h.gguf -c 32768
"""

CONFIG = """
after = ["my-loader", "restore"]

[[server]]
name = "qwen-q4"
base_url = "http://127.0.0.1:11434/v1"
model = "qwen3:8b"
group = "qwen3-8b"
context = 32768
api_key_env = "MY_TOKEN"

[[server]]
name = "qwen-q8"
base_url = "http://127.0.0.1:8080"
group = "qwen3-8b"
load = ["my-loader", "qwen-q8"]

[[server]]
name = "gemma"
base_url = "http://127.0.0.1:8081"
vision = true
"""


def test_a_configuration_groups_servers_and_never_holds_a_token(monkeypatch):
    cfg = config.parse(CONFIG)
    groups = config.by_group(cfg.servers)
    assert list(groups) == ["qwen3-8b", "gemma"] and [s.name for s in groups["qwen3-8b"]] == ["qwen-q4", "qwen-q8"]
    q4, q8, gemma = cfg.servers
    assert q4.model == "qwen3:8b" and q4.context == 32768 and q8.context == 8192 and gemma.vision and gemma.group == "gemma"
    assert q8.load == ["my-loader", "qwen-q8"] and cfg.after == ["my-loader", "restore"]
    monkeypatch.setenv("MY_TOKEN", "t0k")
    assert q4.api_key == "t0k" and q8.api_key == ""                                   # the file names the variable, not the value
    for bad in ('[[server]]\nname = "x"', '[[server]]\nname = "x"\nbase_url = "u"\ncolour = 1', '[[server]]\nname = "x"\nbase_url = "u"\nload = "a b"'):
        try:
            config.parse(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_a_turnstile_map_becomes_servers_loaded_through_it():
    servers = config.parse_turnstile_map(MAPTEXT, host="192.0.2.10")
    groups = config.by_group(servers)
    assert set(groups) == {"Q-9B", "G-12B", "H-3B"} and len(servers) == 4
    assert groups["G-12B"][0].vision and not groups["Q-9B"][0].vision and groups["Q-9B"][0].context == 196608
    assert servers[0].base_url == "http://192.0.2.10:8105" and servers[0].load == ["turnstile", "Q-9B-llamacpp-c0"]
    back = config.parse(config.to_toml(servers))                                     # the printed TOML reads back the same
    assert [(s.name, s.base_url, s.group, s.context, s.vision, s.load) for s in back.servers] == \
        [(s.name, s.base_url, s.group, s.context, s.vision, s.load) for s in servers]


def test_one_server_can_be_given_by_environment_without_any_file(monkeypatch):
    cfg = cli.servers_from([], env={"FH_BENCH_BASE_URL": "http://127.0.0.1:11434/v1", "FH_BENCH_MODEL": "llama3.2"})
    (s,) = cfg.servers
    assert s.base_url.endswith("/v1") and s.model == "llama3.2" and s.name == "llama3.2" and s.api_key_env == "FH_BENCH_API_KEY"
    assert client.root("http://h:1/v1/") == client.root("http://h:1") == "http://h:1"
    assert client.headers("k")["Authorization"] == "Bearer k" and "Authorization" not in client.headers("")


def test_results_become_events_and_a_table():
    result = {"group": "Q-9B", "server": "Q-9B-llamacpp-c1", "context": 196608, "measured_at": "t",
              "results": {"tool_call": {"score": 1.0, "n": 6}, "vision": {"score": None, "n": 0}, "long_context": {"score": 1.0, "n": 3, "max_tokens_found": 65536},
                          "speed": {"score": 1.0, "n": 1, "gen_tps": 55.2}},
              "variants": {"Q-9B-llamacpp-c0": {"speed": {"gen_tps": 80.1, "prefill_tps": 1500}, "tool_call": {"score": 0.0, "n": 6}}}}
    sent = []
    assert runner.emit(result, lambda kind, detail: sent.append((kind, detail["task"] if "task" in detail else detail["server"]))) == 6
    assert ("bench.result", "tool_call") in sent and ("bench.speed", "Q-9B-llamacpp-c0") in sent
    assert ("bench.variant", "tool_call") in sent                                  # a variant that cannot call a tool is on record
    table = mapping.render([result])
    assert "Q-9B" in table and "✓ 1.00" in table and "✓ 64k" in table and "—" in table and "55.2" in table


def test_calls_written_as_text_are_read_in_the_usual_shapes():
    one = scorers.text_calls("```\n[{'name': 'get_weather', 'arguments': {'city': 'Lyon'}}]\n```")
    assert [(scorers.tool_call_name(c), scorers.tool_call_args(c)) for c in one] == [("get_weather", {"city": "Lyon"})]
    tagged = scorers.text_calls('<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}}</tool_call>')
    assert scorers.tool_call_args(tagged[0]) == {"a": 1, "b": 2}
    assert scorers.text_calls('{"name": "set_timer", "parameters": {"seconds": 90}}')[0]["function"]["arguments"] == {"seconds": 90}
    assert scorers.text_calls("The weather in Lyon is sunny.") == [] and scorers.text_calls("<think>{'name': 'x'}") == []


def test_a_model_that_writes_its_calls_scores_zero_natively_but_full_through_a_text_parser():
    def writes(prompt, kw):
        if "Lyon" in prompt:
            return reply("```\n[{'name': 'get_weather', 'arguments': {'city': 'Lyon'}}]\n```")
        if "add 1234" in prompt:
            return reply('[{"name": "add", "arguments": {"a": 1234, "b": 5678}}]')
        return reply("")

    r = tasks.tool_call(ctx_with(writes))
    assert r.score == 0.0 and r.detail["text_score"] == round(2 / 6, 3) and "wrote the call as text" in r.detail["notes"]


# ── the agentic stage ────────────────────────────────────────────────────────

def test_the_workspace_starts_with_failing_tests_and_ends_green_once_fixed(tmp_path):
    from fh_bench import agentic
    agentic.reset_workspace(str(tmp_path / "w"))
    assert not agentic.tests_pass(str(tmp_path / "w"))                       # the bug is real
    calc = tmp_path / "w" / "calc.py"
    calc.write_text(calc.read_text().replace("(len(xs) + 1)", "len(xs)"))
    assert agentic.tests_pass(str(tmp_path / "w"))                           # and fixing it is what turns it green
    assert "Thursday" in (tmp_path / "w" / "notes.txt").read_text()


def test_the_answer_must_carry_the_day_and_the_hour():
    from fh_bench import agentic
    assert agentic.answer_ok("The deployment window is Thursday at 02:30.") and agentic.answer_ok("Thursday 2:30 am")
    assert agentic.answer_ok(french.AGENT_ANSWER)                                                                  # a harness may answer in French
    assert not agentic.answer_ok("It is on Friday at 02:30.") and not agentic.answer_ok("I could not find it.")


def test_we_wait_for_the_harness_to_follow_the_model_and_give_up_after_a_while():
    from fh_bench import agentic
    seen = iter(["old", "old", "model-a"])
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += s

    assert agentic.wait_for_model(lambda: next(seen), "model-a", 90, sleep=sleep, now=lambda: clock["t"])
    clock["t"] = 0.0
    assert not agentic.wait_for_model(lambda: "old", "model-a", 9, sleep=sleep, now=lambda: clock["t"])


def test_a_model_without_tool_calling_is_scored_zero_without_opening_a_session():
    from fh_bench import agentic
    r = agentic.evaluate("s", 0.0, fox=object())                              # a session would crash on this object
    assert r["score"] == 0.0 and "no native tool calling" in r["skipped"]


def test_the_agent_stage_is_skipped_when_no_harness_is_configured(monkeypatch):
    from fh_bench import agentic
    monkeypatch.setattr(agentic, "OPENFOX", "")
    r = agentic.evaluate("m", 1.0)
    assert r["score"] is None and "not configured" in r["skipped"] and not agentic.enabled()


def test_the_after_command_runs_once_and_a_missing_one_is_said(monkeypatch):
    said = []
    runner.after([], said.append)
    assert said == []
    runner.after(["fh-bench-no-such-command-xyz"], said.append)
    assert "FileNotFoundError" in said[0]


def test_a_session_is_not_quiet_before_the_agent_has_answered():
    from fh_bench import agentic
    assert agentic.next_quiet(0, False, [{"role": "user"}]) == 0                          # right after the prompt: not started
    assert agentic.next_quiet(0, False, [{"role": "user"}, {"role": "assistant"}]) == 1
    assert agentic.next_quiet(1, False, [{"role": "assistant"}]) == 2
    assert agentic.next_quiet(1, True, [{"role": "assistant"}]) == 0                        # running again: start over


def test_samples_of_the_replies_are_kept_so_a_zero_can_be_explained():
    ctx = ctx_with(lambda p, k: reply("<think>never finishes"))
    tasks.reasoning(ctx)
    assert len(ctx.trace) == 3 and ctx.trace[0]["reply"].startswith("<think>")


def test_a_second_run_of_a_few_tasks_adds_to_the_saved_result(tmp_path):
    first = {"group": "W/x", "server": "s", "context": 1, "measured_at": "a", "results": {"code": {"score": 0.5}, "french": {"score": 0.5}}, "variants": {}}
    second = {"group": "W/x", "server": "s", "context": 1, "measured_at": "b", "results": {"french": {"score": 1.0}}, "variants": {}}
    runner.save(first, str(tmp_path))
    path = runner.save(second, str(tmp_path))
    with open(path) as f:
        kept = json.load(f)
    assert kept["results"] == {"code": {"score": 0.5}, "french": {"score": 1.0}} and kept["measured_at"] == "b"


def test_events_are_json_lines_or_handed_to_the_configured_command(monkeypatch):
    printed = []
    monkeypatch.delenv("FH_BENCH_EVENT_CMD", raising=False)
    assert runner.send_event("bench.result", {"group": "W"}, out=printed.append) is True
    assert json.loads(printed[0]) == {"kind": "bench.result", "group": "W"}
    seen = {}

    class Done:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return Done()

    monkeypatch.setenv("FH_BENCH_EVENT_CMD", "my-sender --topic bench")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.send_event("bench.speed", {"gen_tps": 1.5}) is True
    assert seen["cmd"] == ["my-sender", "--topic", "bench", "bench.speed", '{"gen_tps":1.5}']


def test_a_long_measurement_waits_for_a_free_machine(monkeypatch):
    monkeypatch.setattr(runner, "game_running", lambda: True)
    assert runner.busy_reason(0) == "a game is running"
    monkeypatch.setattr(runner, "game_running", lambda: False)
    assert runner.busy_reason(2) == "2 agent session(s) at work" and runner.busy_reason(0) == ""
    monkeypatch.setattr(runner.agentic, "enabled", lambda: True)

    def unreachable():
        raise OSError("down")

    monkeypatch.setattr(runner.agentic, "running_sessions", unreachable)
    assert runner.busy_reason() == ""


def test_load_waits_at_the_swap_ceiling_instead_of_failing_in_silence(monkeypatch):
    answers = iter([("refused: 30 swaps in the last hour (ceiling 30). too many swaps in one hour", 1), ("refused: 30 swaps in the last hour (ceiling 30)", 1), ("served in 5.1 s", 0)])
    waits = []

    class Done:
        def __init__(self, text, code):
            self.stdout, self.stderr, self.returncode = text, "", code

    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Done(*next(answers)))

    said = []
    server = config.Server(name="S-c1", base_url="http://127.0.0.1:8095", load=["turnstile", "S-c1"])
    assert runner.load(server, said.append, sleep=waits.append, reachable=lambda *a: True) is True
    assert waits == [120, 120] and sum("swap ceiling" in m for m in said) == 2


def test_load_gives_up_on_a_real_refusal_and_says_why(monkeypatch):
    class Done:
        stdout, stderr, returncode = "failed {'state': 'failed'}", "", 1

    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Done())
    said = []
    server = config.Server(name="S-c1", base_url="http://127.0.0.1:8095", load=["turnstile", "S-c1"])
    assert runner.load(server, said.append, sleep=lambda s: None) is False
    assert "load command said failed" in said[0]
    assert runner.ceiling_reached("refused: 30 swaps in the last hour (ceiling 30)") and not runner.ceiling_reached("served in 3 s")


def test_the_reason_a_server_stopped_and_its_reasoning_channel_are_kept():
    body = {"choices": [{"finish_reason": "length", "message": {"content": "", "reasoning_content": "hmm" * 50}}], "usage": {}}
    r = client.chat("http://x", [{"role": "user", "content": "q"}], transport=lambda url, b, t, key: body)
    assert r.finish == "length" and len(r.reasoning) == 150 and r.text == ""
    ctx = ctx_with(lambda p, k: reply("", finish="length", reasoning="x" * 7))
    tasks.reasoning(ctx)
    assert ctx.trace[0]["finish"] == "length" and ctx.trace[0]["reasoning_chars"] == 7


def test_verbosity_and_cut_off_replies_are_counted_per_task():
    ctx = ctx_with(lambda p, k: reply("42", completion_tokens=120, finish="stop") if "17 * 23" in p else reply("", completion_tokens=3000, finish="length"))
    tasks.reasoning(ctx)
    assert ctx.stats["calls"] == 8 and ctx.stats["truncated"] == 7 and ctx.stats["tokens"] == 120 + 7 * 3000


def test_a_bare_list_of_french_words_is_not_marked_down_for_lacking_function_words():
    answers = french.GOOD_ANSWERS
    r = tasks.french(ctx_with(lambda p, k: reply(next(v for key, v in answers.items() if key in p))))
    assert r.score == 1.0


# ── against a real (local, scripted) OpenAI-compatible server ────────────────

def test_speed_is_timed_by_the_client_from_a_stream():
    ticks = iter([0.0, 0.5, 0.6, 0.7, 1.5])
    chunks = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}', b"",
              *[b'data: {"choices":[{"delta":{"content":"w"}}]}'] * 3,
              b'data: {"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":5}}', b"data: [DONE]"]

    class Lines(list):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    s = client.stream_speed("http://x/v1", [{"role": "user", "content": "q"}], opener=lambda *a: Lines(chunks), clock=lambda: next(ticks))
    assert s["ttft_s"] == 0.5 and s["completion_tokens"] == 5 and s["counted"] == "usage" and s["prefill_tps"] == 2000
    assert s["gen_tps"] == round(4 / 0.2, 1)


def test_the_cli_measures_a_live_server_and_prints_the_map(tmp_path, capsys):
    import http.server
    import threading

    seen = []

    class Fake(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200 if self.path in ("/health", "/v1/models") else 404)
            self.end_headers()
            self.wfile.write(b'{"data": []}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((body.get("model"), self.headers.get("Authorization")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if body.get("stream"):
                for _ in range(4):
                    self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok "}}]}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                return
            prompt = body["messages"][-1]["content"]
            answer = "Paris" if "capital" in str(prompt) else "391"
            self.wfile.write(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": answer}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).encode())

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        out = str(tmp_path / "results")
        import os
        os.environ["FH_BENCH_API_KEY"] = "secret-for-test"
        try:
            assert cli.main(["run", "--base-url", url, "--model", "tiny", "--tasks", "speed,reasoning", "--out", out]) == 0
        finally:
            del os.environ["FH_BENCH_API_KEY"]
        with open(os.path.join(out, "tiny.json")) as f:
            saved = json.load(f)
        assert saved["results"]["speed"]["completion_tokens"] == 4 and saved["results"]["speed"]["counted"] == "chunks"
        assert saved["results"]["reasoning"]["score"] is not None
        assert seen and all(m == "tiny" and a == "Bearer secret-for-test" for m, a in seen)   # the model and the key reach the server
        capsys.readouterr()
        assert cli.main(["map", out]) == 0
        assert "tiny" in capsys.readouterr().out
        assert cli.main(["run", "--base-url", url, "--tasks", "nope"]) == 2
    finally:
        httpd.shutdown()
