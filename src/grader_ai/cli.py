"""Command-line interface for grader-ai."""

import json
import logging
import threading
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path

import dotenv
from openai import OpenAI
from tqdm import tqdm

from grader_ai.core import ProgressEvent, ProgressStage, grade_all
from grader_ai.extraction import extract_reference, extract_submissions


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="grader-ai", description="Grade assignments with LLMs")
    subparsers = parser.add_subparsers(dest="command")

    grade_parser = subparsers.add_parser("grade", help="Grade submissions via CLI.")
    grade_parser.add_argument("-m", "--model", type=str, required=True)
    grade_parser.add_argument("-r", "--reference", type=Path, required=True)
    grade_parser.add_argument("-s", "--submission", type=Path, required=True)
    grade_parser.add_argument("-o", "--output", type=Path, required=True)
    grade_parser.add_argument("-p", "--parallel", type=int, default=1)

    serve_parser = subparsers.add_parser("serve", help="Launch the Gradio web UI.")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=7860)
    serve_parser.add_argument("--share", action="store_true", default=False)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    dotenv.load_dotenv()

    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "grade":
        _run_grade(args)
        return
    if args.command == "serve":
        _run_serve(args)
        return
    parser.print_help()
    raise SystemExit(1)


def _run_grade(args: Namespace) -> None:
    if args.parallel < 1:
        raise SystemExit("error: --parallel must be at least 1")

    reference_content = extract_reference(args.reference)
    submissions = extract_submissions(args.submission)
    if not submissions:
        raise SystemExit("error: no submissions found")

    tqdm.write(f"Extracted {len(submissions)} submission(s) from {args.submission}.")
    args.output.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    submission_bar = tqdm(total=len(submissions), desc="Submissions", unit="sub")
    problem_bar = tqdm(total=0, desc="Problems", unit="prob")
    known_total_problems = 0

    def on_progress(event: ProgressEvent) -> None:
        nonlocal known_total_problems
        with lock:
            if event.stage is ProgressStage.PROBLEM_GRADED:
                if event.problem_index == 0:
                    known_total_problems += event.total_problems
                    problem_bar.total = known_total_problems
                    problem_bar.refresh()
                problem_bar.update(1)
            elif event.stage is ProgressStage.SUBMISSION_COMPLETE:
                submission_bar.update(1)
                if event.error:
                    tqdm.write(f"ERROR [{event.submission_name}]: {event.error}")
                elif event.report is not None:
                    submission_bar.set_postfix_str(
                        f"{event.submission_name} {event.report.total_score}/{event.report.max_score}",
                        refresh=True,
                    )

    reports, errors = grade_all(
        client=OpenAI(),
        model=args.model,
        reference_name=args.reference.name,
        reference_content=reference_content,
        submissions=submissions,
        parallel=args.parallel,
        on_progress=on_progress,
    )

    problem_bar.close()
    submission_bar.close()

    for report in sorted(reports, key=lambda item: item.submission):
        output_file = args.output / f"{report.submission}.json"
        output_file.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        tqdm.write(
            f"{report.submission}: {report.total_score}/{report.max_score} -> {output_file}"
        )
        for warning in report.warnings:
            logging.warning("[%s] %s", report.submission, warning)

    total_score = sum(report.total_score for report in reports)
    max_score = sum(report.max_score for report in reports)
    tqdm.write(
        f"Done. {len(reports)} graded, {len(errors)} failed. Total: {total_score}/{max_score}."
    )
    if errors:
        raise RuntimeError(
            f"Grading failed for {len(errors)} submission(s): "
            + ", ".join(sorted(errors))
        )


def _run_serve(args: Namespace) -> None:
    from grader_ai.app import launch

    launch(host=args.host, port=args.port, share=args.share)
