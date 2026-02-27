# grader-ai

> Grade LaTeX assignment submissions with an OpenAI-compatible LLM.

`grader-ai` parses problems and answers from a reference TeX file, parses student
solutions from a submission TeX file, asks an LLM to score each response, and
writes a JSON report.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Background

The CLI currently extracts:

- Problem text from reference macros: `\problemTF{...}`, `\problemMC{...}`,
  `\problemPS{...}`, `\problemAI{...}`
- Reference answers from `\answer{...}`
- Student responses from `\solution{...}`

Each graded item is aligned by order and scored by the model from `0` to that
problem's credit value.

## Install

This project targets Python 3.12+.

```bash
uv sync
```

## Usage

Run the CLI with a reference TeX file, submission TeX file, output path, and
model name:

```bash
uv run grader-ai \
  --reference examples/reference.tex \
  --submission examples/submission.tex \
  --output results.json \
  --model gpt-4o-mini
```

CLI options:

- `-r, --reference`: path to the reference TeX file (required)
- `-s, --submission`: path to the student submission TeX file (required)
- `-o, --output`: path to JSON output file (required)
- `-m, --model`: model name passed to the OpenAI-compatible API (required)

Output is a single JSON report with this structure:

- `reference`: input reference filename
- `submission`: input submission filename
- `grades`: list of per-problem grading results (`problem`, `credits`, `answer`,
  `response`, `score`, `feedback`)
- `total_score`: sum of all per-problem scores

## Configuration

`grader-ai` loads `.env` automatically (`python-dotenv`) and initializes the
OpenAI client from environment variables.

```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

- `OPENAI_API_KEY`: API key for your OpenAI-compatible provider.
- `OPENAI_BASE_URL`: optional base URL for OpenAI-compatible providers.

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

MIT (c) Zijian Zhang
