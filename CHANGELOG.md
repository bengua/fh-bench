# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- First public release.
- Works with any OpenAI-compatible server (Ollama, llama.cpp, vLLM, LiteLLM),
  configured by command-line options, environment variables or a TOML file.
- Optional bearer token (`FH_BENCH_API_KEY`, or `api_key_env` per server).
- `fh-bench import-turnstile` converts a turnstile slot map into a configuration file.
- Events (`--emit`) go to standard output as JSON lines, or to `$FH_BENCH_EVENT_CMD`.

### Changed

- The agentic task is optional and skipped unless `FH_BENCH_OPENFOX` is set.
