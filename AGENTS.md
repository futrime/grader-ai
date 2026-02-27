# AGENTS.md

This file gives coding agents project-specific instructions for `grader-ai`.

## Project Snapshot

- Language: Python
- Packaging: `pyproject.toml` (PEP 621 metadata)
- Build backend: `uv_build`
- Python requirement: `>=3.12`
- Importable package: `src/grader_ai/`
- CLI entry point: `grader-ai` -> `grader_ai:main`

## Repository Rules Discovery

Checked for additional agent rules and found none:

- `.cursor/rules/`: not present
- `.cursorrules`: not present
- `.github/copilot-instructions.md`: not present

If any of these files are added later, agents should treat them as higher-priority
repository instructions and update this document accordingly.

## Setup

Preferred setup from repo root:

```bash
uv sync
```

Fallback setup when `uv` is unavailable:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Build, Lint, and Test Commands

This repository is minimal: no lint or test tool is pinned in `pyproject.toml`.
Use these commands as defaults and report missing tools clearly.

### Build / Run

```bash
# build sdist + wheel
uv build

# editable install smoke check
uv pip install -e .

# run CLI entry point
uv run grader-ai
```

### Lint / Format (Ruff only)

```bash
uv run ruff check .
uv run ruff format .
```

Use Ruff as the only code quality tool in this repository.
Do not add or run mypy/flake8/pylint/black/isort unless the user requests it.
If `ruff` is missing, report that linting is not configured in this repo yet.

### Tests

There is no `tests/` directory yet, but use `pytest` conventions when tests are added.

```bash
# run all tests
uv run pytest

# run a single test file
uv run pytest tests/test_example.py

# run a single test function
uv run pytest tests/test_example.py::test_specific_behavior

# filter by expression
uv run pytest -k "keyword"

# stop on first failure
uv run pytest -x

# verbose + print output for one test
uv run pytest -vv -s tests/test_example.py::test_specific_behavior
```

## Expected Agent Workflow

1. Read `pyproject.toml` before proposing or running project commands.
2. Prefer `uv run ...` for Python tooling in this repository.
3. Keep changes scoped to the user request; avoid broad refactors.
4. Run the narrowest relevant verification command first, then widen as needed.
5. If a command cannot run, state exactly why and give fallback.
6. Use Ruff only for linting/formatting tasks.

## Code Style Guidelines

Follow the Google Python Style Guide directly:
https://google.github.io/styleguide/pyguide.html

Only repository-specific additions:

- Prefer absolute imports from `grader_ai`.
- Add type hints for public functions and methods.
- Use `logging` for diagnostics and keep CLI output concise.
- Use `pytest` conventions (`tests/test_*.py`, `test_*` names) when tests exist.

## Change Management for Agents

- Do not add heavy dependencies without clear justification.
- Update `pyproject.toml` when adding runtime or dev dependencies.
- Document any new lint/test/type commands in this file.
- Prefer incremental, reviewable patches.
- Explicitly mention checks you could not run and why.

## Quick Command Reference

```bash
uv sync
uv run grader-ai
uv build
uv run pytest tests/test_example.py::test_specific_behavior
uv run ruff check .
uv run ruff format .
```
