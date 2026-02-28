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


def main() -> None:
    """Entry point for the ``grader-ai`` console script."""
    logging.basicConfig(level=logging.INFO)

    dotenv.load_dotenv()

    parser = ArgumentParser(prog="grader-ai", description="Grade assignments with LLMs")
    subparsers = parser.add_subparsers(dest="command")

    # -- grade subcommand ------------------------------------------------
    grade_parser = subparsers.add_parser("grade", help="Grade submissions via CLI.")
    grade_parser.add_argument("-m", "--model", type=str, required=True)
    grade_parser.add_argument("-r", "--reference", type=Path, required=True)
    grade_parser.add_argument("-s", "--submission", type=Path, required=True)
    grade_parser.add_argument("-o", "--output", type=Path, required=True)
    grade_parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=1,
        help="Number of submissions to grade in parallel (default: 1).",
    )

    # -- serve subcommand ------------------------------------------------
    serve_parser = subparsers.add_parser("serve", help="Launch the Gradio web UI.")
    serve_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)."
    )
    serve_parser.add_argument(
        "--port", type=int, default=7860, help="Port to listen on (default: 7860)."
    )
    serve_parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable Gradio link.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    if args.command == "grade":
        _run_grade(args)
    elif args.command == "serve":
        _run_serve(args)


def _run_grade(args: Namespace) -> None:
    """Execute the batch-grading workflow."""
    if args.parallel < 1:
        raise SystemExit("error: --parallel must be at least 1")

    reference_content = extract_reference(args.reference)
    submissions = extract_submissions(args.submission)

    total_submissions = len(submissions)
    tqdm.write(f"Extracted {total_submissions} submission(s) from {args.submission}.")

    if total_submissions == 0:
        raise SystemExit("error: no submissions found")

    args.output.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    # -- Progress tracking with tqdm ------------------------------------
    # We use two bars:
    #   1. An outer bar tracking submissions completed.
    #   2. An inner bar tracking problems graded within the current
    #      submission(s).
    # When parallel > 1 multiple submissions are in-flight simultaneously,
    # so the inner bar shows aggregate problem-level progress.

    lock = threading.Lock()
    submission_bar = tqdm(
        total=total_submissions,
        desc="Submissions",
        unit="sub",
        position=0,
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )
    problem_bar = tqdm(
        total=0,
        desc="  Problems ",
        unit="prob",
        position=1,
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )
    # Track how many total problems we know about so far, since
    # submissions are parsed at different times when parallel > 1.
    known_total_problems = 0

    def _on_progress(event: ProgressEvent) -> None:
        nonlocal known_total_problems

        with lock:
            if event.stage is ProgressStage.PARSING:
                submission_bar.set_postfix_str(
                    f"parsing {event.submission_name}", refresh=True
                )

            elif event.stage is ProgressStage.GRADING_PROBLEM:
                # First time we see problems for this submission, expand
                # the problem bar total.
                if event.problem_index == 0:
                    known_total_problems += event.total_problems
                    problem_bar.total = known_total_problems
                    problem_bar.refresh()

                problem_bar.set_postfix_str(
                    f"{event.submission_name} Q{event.problem_index + 1}"
                    f"/{event.total_problems}",
                    refresh=True,
                )

            elif event.stage is ProgressStage.GRADED_PROBLEM:
                problem_bar.update(1)
                if event.grade_result is not None:
                    score = event.grade_result.score
                    credits = event.grade_result.credits
                    problem_bar.set_postfix_str(
                        f"{event.submission_name} Q{event.problem_index + 1}"
                        f" {score}/{credits}",
                        refresh=True,
                    )

            elif event.stage is ProgressStage.SUBMISSION_DONE:
                submission_bar.update(1)
                if event.report is not None:
                    submission_bar.set_postfix_str(
                        f"{event.submission_name}"
                        f" {event.report.total_score}/{event.report.max_score}",
                        refresh=True,
                    )

            elif event.stage is ProgressStage.SUBMISSION_ERROR:
                submission_bar.update(1)
                submission_bar.set_postfix_str(
                    f"{event.submission_name} FAILED", refresh=True
                )
                tqdm.write(f"ERROR [{event.submission_name}]: {event.error}")

    reports, errors = grade_all(
        client=client,
        model=args.model,
        reference_name=args.reference.name,
        reference_content=reference_content,
        submissions=submissions,
        parallel=args.parallel,
        on_progress=_on_progress,
    )

    # Close progress bars before final output.
    problem_bar.close()
    submission_bar.close()

    # -- Write reports ---------------------------------------------------
    tqdm.write("")  # blank line after bars
    for report in sorted(reports, key=lambda r: r.submission):
        output_file = args.output / f"{report.submission}.json"
        output_file.write_text(json.dumps(asdict(report), indent=2))
        tqdm.write(
            f"  {report.submission}: "
            f"{report.total_score}/{report.max_score} pts "
            f"-> {output_file}"
        )
        if report.warnings:
            for warning in report.warnings:
                logging.warning("[%s] %s", report.submission, warning)

    # -- Summary ---------------------------------------------------------
    total_score = sum(r.total_score for r in reports)
    max_score = sum(r.max_score for r in reports)
    tqdm.write(
        f"\nDone. {len(reports)} graded, {len(errors)} failed. "
        f"Total: {total_score}/{max_score} pts."
    )

    if errors:
        raise RuntimeError(
            f"Grading failed for {len(errors)} submission(s): "
            + ", ".join(sorted(errors))
        )


def _run_serve(args: Namespace) -> None:
    """Launch the Gradio web interface."""
    from grader_ai.app import launch

    launch(host=args.host, port=args.port, share=args.share)
