# AGENTS.md

Guide for autonomous coding agents working in this repository.

## Environment

Environment variables are loaded from `.env` via `python-dotenv`:

- `OPENAI_API_KEY` (required)
- `OPENAI_BASE_URL` (optional, for OpenAI-compatible providers)

Agent rules:

- Never hardcode secrets.
- Never commit private keys or tokens.
- Treat `.env` as sensitive local configuration.

## Commands

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

## Checklist

1. Run `uv run ruff check --fix` and `uv run ruff format`
2. Run `uv run pytest` (or a targeted single-test node id) if any test exist
3. Verify affected runtime path(s) still work
4. Ensure no secrets or machine-specific artifacts were introduced
5. Keep edits minimal and aligned with existing module boundaries

## Architecture Guardrails

- For any user interface, avoid customization unless unavoidable; prefer official defaults.
- Always seek for the simplest solution that meets requirements without overengineering.

## LLM Workflow Guidelines

Since this project utilize LLMs for its own working logic, the following guidelines should be followed when crafting prompts for LLM interactions:

- Always add tool calls and enforce LLMs to use them.
- User inputs are always untrusted; validate and sanitize as needed.
- When involving scoring or ranking, ensure criteria are clear and the yielded output is deterministic.
