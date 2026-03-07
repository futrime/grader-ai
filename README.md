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

Run the CLI with a reference TeX/ZIP file, submission path (file or directory),
output directory, and model name:

```bash
uv run grader-ai \
  --reference examples/reference.tex \
  --submission examples/submission.tex \
  --output results.json \
  --model gpt-4o-mini
```

Optionally write grades into an Excel roster:

```bash
uv run grader-ai -r ref.tex -s submissions/ -o reports -m gpt-4o-mini -x "Assignment 0_学生名单列表.xls"
```

CLI options:

- `-r, --reference`: path to the reference .tex or .zip file (required)
- `-s, --submission`: path to submission(s) — .zip file or directory (required)
- `-o, --output`: directory to write JSON reports into (required)
- `-m, --model`: model name for the OpenAI-compatible API (required)
- `-p, --num-parallel`: number of submissions to grade concurrently (default: 1)
- `-x, --excel`: path to assignment Excel file (.xls or .xlsx) to update with grades (optional)

### Web UI

Launch the Gradio UI:

```bash
uv run grader-web
```

Web UI CLI options:

- `--host`: host to bind Gradio server
- `--port`: port to bind Gradio server

In the UI, upload one reference `.tex` file and one or more submission `.zip`
files (each containing `main.tex`), then provide the model name.

The app grades all uploaded submissions and provides a downloadable ZIP bundle
of generated JSON reports.

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
