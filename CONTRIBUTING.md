# Contributing

Contributions are welcome: bug reports, new tasks, support for another server.

## The one rule that matters

**A change is verified by running it.** Before opening a pull request, run the
thing it changes, not only its tests. A test written alongside its code inherits
that code's blind spots: a green CI means the checks passed, not that the change
is right.

## Workflow

1. Fork the repository and create a branch from `main`.
2. Install the development tools: `pip install -e '.[dev]'`.
3. Make your change, with a test that executes it.
4. Run `./verify.sh` (ruff + pytest). No GPU and no model server are needed:
   the tests use a local fake OpenAI server.
5. Open a pull request describing the problem as observed (file, symptom) and
   what the change does about it.

## Before you open a pull request

- The change has been executed, not only written.
- `README.md` still describes what the tool does.
- `CHANGELOG.md` has an entry under `Unreleased`.
- Code, comments and commit messages are in English.

## Adding a task

A task is a function in `src/fh_bench/tasks.py` that takes a `Ctx` and returns
a `Result`, registered in `BATTERY`. It must be judged by code (a parser, a
schema, a test run), never by another model.
