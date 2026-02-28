# AGENTS.md

Guide for autonomous coding agents working in this repository.

## Repository Overview

- Project: `grader-ai`
- Language: Python `>=3.12`
- Dependency and command runner: `uv`
- Source package: `src/grader_ai/`
- Console entrypoint: `grader-ai = grader_ai.cli:main`

Primary modules:

- `src/grader_ai/cli.py`: CLI commands (`grade`, `serve`)
- `src/grader_ai/app.py`: Gradio UI and live progress updates
- `src/grader_ai/core.py`: orchestration and parallel grading engine
- `src/grader_ai/extraction.py`: extract submissions from `.tex` / `.zip`
- `src/grader_ai/parsing.py`: parse reference/submission LaTeX
- `src/grader_ai/grading.py`: OpenAI tool-call grading contract

## Setup and Environment

Install dependencies:

```bash
uv sync
```

Environment variables are loaded from `.env` via `python-dotenv`:

- `OPENAI_API_KEY` (required)
- `OPENAI_BASE_URL` (optional, for OpenAI-compatible providers)

Agent rules:

- Never hardcode secrets.
- Never commit private keys or tokens.
- Treat `.env` as sensitive local configuration.

## Build / Lint / Test Commands

There is no separate compile step; use these standard commands.

Install or refresh dependencies:

```bash
uv sync
```

Lint:

```bash
uv run ruff check .
```

Lint with automatic fixes:

```bash
uv run ruff check . --fix
```

Run all tests:

```bash
uv run pytest
```

Run one test file:

```bash
uv run pytest tests/test_example.py
```

Run one specific test function (pytest node id):

```bash
uv run pytest tests/test_example.py::test_specific_behavior
```

Run tests by keyword:

```bash
uv run pytest -k "specific_behavior"
```

Verbose test output:

```bash
uv run pytest -vv
```

Current repo note:

- No `tests/` files were found when this document was generated.
- If adding tests, use pytest discovery conventions (`tests/test_*.py`).

## Runtime Commands

Grade from CLI:

```bash
uv run grader-ai grade \
  --model gpt-4o-mini \
  --reference path/to/reference.tex \
  --submission path/to/submission_or_dir_or_zip \
  --output path/to/output_dir
```

Grade with parallel workers:

```bash
uv run grader-ai grade -m gpt-4o-mini -r ref.tex -s submissions.zip -o out --parallel 4
```

Launch web UI:

```bash
uv run grader-ai serve --host 0.0.0.0 --port 7860
```

## Architecture Guardrails

- Keep `core.py` free from file I/O.
- Keep filesystem concerns in CLI/UI/extraction layers.
- Preserve JSON report schema compatibility.
- Maintain `.tex` and `.zip` support (including nested zip flows).

## Quick Agent Checklist

1. Run `uv run ruff check --fix` and `uv run ruff format`
2. Run `uv run pytest` (or a targeted single-test node id)
3. Verify affected runtime path(s) still work
4. Ensure no secrets or machine-specific artifacts were introduced
5. Keep edits minimal and aligned with existing module boundaries
