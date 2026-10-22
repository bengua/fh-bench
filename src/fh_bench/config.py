"""Which servers to measure: a TOML file, or one server given on the command line.

A server is anything that speaks the OpenAI chat API: llama.cpp's llama-server, Ollama, vLLM,
LiteLLM, LM Studio, a hosted endpoint. Servers that share a `group` serve the same model in
different ways (another quantisation, another card, another engine): the first one listed gets
the whole battery, the others the short variant battery.
"""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field

CONFIG_ENV = "FH_BENCH_CONFIG"
CONFIG_PATHS = ("fh-bench.toml", os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
                                              "fh-bench", "config.toml"))


@dataclass
class Server:
    name: str                       # the label in the results
    base_url: str                   # e.g. http://127.0.0.1:8080 or http://127.0.0.1:11434/v1
    model: str = ""                 # the `model` field of each request (Ollama, vLLM and LiteLLM need it)
    group: str = ""                 # servers sharing a group are variants of one model; defaults to `name`
    context: int = 8192             # the window the server was started with: caps the long-context climb
    vision: bool = False            # the server accepts images
    api_key_env: str = ""           # the NAME of an environment variable holding a bearer token, never the token
    load: list[str] = field(default_factory=list)   # optional command run before measuring (starts or swaps the model)

    def __post_init__(self) -> None:
        self.group = self.group or self.name

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""


@dataclass
class Config:
    servers: list[Server]
    after: list[str] = field(default_factory=list)  # optional command run once the bench is done (puts the machine back)


KEYS = {"name", "base_url", "model", "group", "context", "vision", "api_key_env", "load"}


def parse(text: str) -> Config:
    data = tomllib.loads(text)
    servers = []
    for i, entry in enumerate(data.get("server") or []):
        unknown = set(entry) - KEYS
        if unknown:
            raise ValueError(f"server #{i + 1}: unknown key(s) {', '.join(sorted(unknown))}")
        if not entry.get("name") or not entry.get("base_url"):
            raise ValueError(f"server #{i + 1}: `name` and `base_url` are required")
        load = entry.get("load") or []
        if isinstance(load, str) or not all(isinstance(a, str) for a in load):
            raise ValueError(f"server {entry['name']}: `load` is a list of arguments, e.g. [\"my-loader\", \"model-a\"]")
        servers.append(Server(**{**entry, "load": list(load), "context": int(entry.get("context", 8192))}))
    after = data.get("after") or []
    if isinstance(after, str):
        raise TypeError("`after` is a list of arguments, e.g. [\"my-loader\", \"restore\"]")
    return Config(servers, list(after))


def find(path: str | None = None) -> str | None:
    """The configuration file: the one given, else $FH_BENCH_CONFIG, else ./fh-bench.toml, else ~/.config/fh-bench/config.toml."""
    if path:
        return path
    if os.environ.get(CONFIG_ENV):
        return os.environ[CONFIG_ENV]
    return next((p for p in CONFIG_PATHS if os.path.isfile(p)), None)


def load(path: str | None = None) -> Config:
    found = find(path)
    if not found:
        raise FileNotFoundError("no configuration: pass --base-url, or write fh-bench.toml (see examples/fh-bench.example.toml)")
    with open(found, encoding="utf-8") as f:
        return parse(f.read())


def by_group(servers: list[Server]) -> dict[str, list[Server]]:
    groups: dict[str, list[Server]] = {}
    for s in servers:
        groups.setdefault(s.group, []).append(s)
    return groups


# ── optional: import the slots of a turnstile map ───────────────────────────
# turnstile (a GPU slot allocator published alongside fh-bench) describes each slot as
# name|port|card|weights|arguments. This turns such a map into fh-bench servers, loaded through it.

def parse_turnstile_map(text: str, host: str = "127.0.0.1") -> list[Server]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split("|")
        if len(p) >= 5 and p[1].isdigit():
            args = "|".join(p[4:])
            m = re.search(r"(?:^|\s)-c\s+(\d+)", args)
            out.append(Server(name=p[0], base_url=f"http://{host}:{p[1]}", group=p[3], context=int(m.group(1)) if m else 8192,
                              vision="--mmproj" in args, load=["turnstile", p[0]]))
    return out


def to_toml(servers: list[Server]) -> str:
    def q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    blocks = []
    for s in servers:
        lines = ["[[server]]", f"name = {q(s.name)}", f"base_url = {q(s.base_url)}", f"group = {q(s.group)}",
                 f"context = {s.context}", f"vision = {'true' if s.vision else 'false'}"]
        if s.model:
            lines.append(f"model = {q(s.model)}")
        if s.load:
            lines.append("load = [" + ", ".join(q(a) for a in s.load) + "]")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
