# grader-ai

> Grade LaTeX assignment submissions with an OpenAI-compatible LLM.

`grader-ai` extracts `\answer{...}` macros from a reference file and `\solution{...}`
macros from a student submission, compares each pair with an LLM, writes per-problem
results to JSONL, and prints a final score summary.

## Install

This project targets Python 3.12+.

```bash
uv sync
```

## Usage

Run the CLI with a reference TeX file and a student submission TeX file:

```bash
uv run grader-ai -r examples/reference.tex -s examples/submission.tex -o results.jsonl -m gpt-4o-mini
```

Validate parsing and macro pairing only (skip API calls):

```bash
uv run grader-ai --reference examples/reference.tex --submission examples/submission.tex --dry-run
```

Expected macro format:

- Reference file contains one or more `\answer{...}` entries.
- Submission file contains matching `\solution{...}` entries.
- Counts must match (`N` answers and `N` solutions).

## Configuration

Set environment variables in your shell or a `.env` file:

```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

- `OPENAI_API_KEY`: API key for your OpenAI-compatible provider.
- `OPENAI_BASE_URL`: Base URL for the provider API.

CLI options:

- `-r, --reference`: path to the reference TeX file.
- `-s, --submission`: path to the student submission TeX file.
- `-o, --output`: path to JSONL output file (required for grading mode).
- `-m, --model`: model name (default: `gpt-4o-mini`).
- `-n, --dry-run`: parse and validate only, without grading API calls.

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and run checks:

```bash
uv run ruff check .
uv run pytest
```

4. Open a pull request.

## License

MIT © Zijian Zhang
