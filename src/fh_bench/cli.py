"""fh-bench - measure local models served over the OpenAI API, and print the map of what each can be asked.

    fh-bench list   [SERVERS]                       the servers to measure, grouped by model
    fh-bench run    [SERVERS] [--servers a,b] [--tasks t1,t2] [--no-variants] [--emit] [--only-if-idle] [--out DIR]
    fh-bench map    [DIR]                           the table of what each model can be asked
    fh-bench emit   [DIR]                           the saved results as events (JSON lines, or $FH_BENCH_EVENT_CMD)
    fh-bench import-turnstile MAP [--host H]        turn a turnstile slot map into a fh-bench.toml (printed)

SERVERS, one of:
    --base-url URL [--model M] [--name N] [--context N] [--vision]    one server ($FH_BENCH_BASE_URL, $FH_BENCH_MODEL...)
    --config FILE                       a TOML file ($FH_BENCH_CONFIG, ./fh-bench.toml, ~/.config/fh-bench/config.toml)
"""
from __future__ import annotations

import json
import os
import sys

from . import config, mapping, runner, tasks

DATA = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "fh-bench")
OUT = os.environ.get("FH_BENCH_RESULTS") or os.path.join(DATA, "results")
SINGLE_KEY_ENV = "FH_BENCH_API_KEY"


def option(args: list[str], flag: str, default=None):
    if flag in args:
        i = args.index(flag)
        if i + 1 >= len(args):
            raise SystemExit(f"{flag} needs a value")
        value = args[i + 1]
        del args[i:i + 2]
        return value
    return default


def flag(args: list[str], name: str) -> bool:
    if name in args:
        args.remove(name)
        return True
    return False


def servers_from(args: list[str], env=os.environ) -> config.Config:
    """One server from --base-url (or $FH_BENCH_BASE_URL), else the configuration file."""
    base = option(args, "--base-url", env.get("FH_BENCH_BASE_URL"))
    model = option(args, "--model", env.get("FH_BENCH_MODEL", ""))
    name = option(args, "--name", env.get("FH_BENCH_NAME", ""))
    context = int(option(args, "--context", env.get("FH_BENCH_CONTEXT", "8192")))
    vision = flag(args, "--vision") or env.get("FH_BENCH_VISION", "") in ("1", "true", "yes")
    path = option(args, "--config")
    if base:
        return config.Config([config.Server(name=name or model or "default", base_url=base, model=model, context=context,
                                            vision=vision, api_key_env=SINGLE_KEY_ENV)])
    return config.load(path)


def results(directory: str) -> list[dict]:
    files = sorted(f for f in os.listdir(directory) if f.endswith(".json")) if os.path.isdir(directory) else []
    out = []
    for name in files:
        with open(os.path.join(directory, name)) as f:
            out.append(json.load(f))
    return out


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__, file=sys.stderr)
        return 0 if args else 2
    cmd = args.pop(0)
    if cmd == "map":
        loaded = results(args[0] if args else OUT)
        print(mapping.render(loaded) if loaded else "no results yet: run `fh-bench run`")
        return 0
    if cmd == "emit":
        print(f"{sum(runner.emit(r) for r in results(args[0] if args else OUT))} events sent", file=sys.stderr)
        return 0
    if cmd == "import-turnstile":
        host = option(args, "--host", "127.0.0.1")
        if not args:
            print("usage: fh-bench import-turnstile MAP [--host H]", file=sys.stderr)
            return 2
        with open(args[0], encoding="utf-8") as f:
            print(config.to_toml(config.parse_turnstile_map(f.read(), host)), end="")
        return 0
    if cmd not in ("list", "run"):
        print(__doc__, file=sys.stderr)
        return 2
    try:
        cfg = servers_from(args)
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"fh-bench: {e}", file=sys.stderr)
        return 2
    groups = config.by_group(cfg.servers)
    if cmd == "list":
        for g, servers in groups.items():
            p = servers[0]
            print(f"{g:<26} {len(servers)} server(s), battery on {p.name} at {p.base_url} (window {p.context}{', vision' if p.vision else ''})")
        return 0
    names = option(args, "--tasks", ",".join([*tasks.BATTERY, "agentic"])).split(",")
    unknown = [n for n in names if n not in tasks.BATTERY and n != "agentic"]
    if unknown:
        print(f"fh-bench: unknown task(s) {', '.join(unknown)}; known: {', '.join([*tasks.BATTERY, 'agentic'])}", file=sys.stderr)
        return 2
    wanted = option(args, "--servers")
    directory = option(args, "--out", OUT)
    variants = not flag(args, "--no-variants")
    emit = flag(args, "--emit")
    if flag(args, "--only-if-idle"):
        why = runner.busy_reason()
        if why:
            print(f"skipped: {why}")
            return 0
    chosen = [g for g in groups if not wanted or g in wanted.split(",")]
    try:
        for g in chosen:
            result = runner.measure(g, groups[g], names, variants)
            if result:
                print("  saved", runner.save(result, directory))
                if emit:
                    runner.emit(result)
    finally:
        runner.after(cfg.after)
    return 0


if __name__ == "__main__":
    sys.exit(main())
