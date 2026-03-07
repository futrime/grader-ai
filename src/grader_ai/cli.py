"""Command line entrypoint for grader-ai."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from dotenv import load_dotenv

from grader_ai.grader import (
    AnyEvent,
    ProblemFinishedEvent,
    ProblemStartedEvent,
    RunFinishedEvent,
    RunStartedEvent,
    SubmissionFinishedEvent,
    SubmissionStartedEvent,
    run,
)

logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    args = _parse_args()

    logging.basicConfig(level=logging.INFO)

    submission_files = _discover_submission_files(args.submission)

    def on_update(event: AnyEvent) -> None:
        if isinstance(event, RunStartedEvent):
            logger.info(
                "Started grading %d submissions...", len(event.submission_files)
            )

        elif isinstance(event, RunFinishedEvent):
            logger.info("Finished grading %d submissions", len(event.report_files))

        if isinstance(event, SubmissionStartedEvent):
            logger.info(
                "Started grading submission '%s' with %d problems...",
                submission_files[event.submission_idx],
                event.num_problems,
            )

        elif isinstance(event, SubmissionFinishedEvent):
            if event.error is None:
                logger.info(
                    "Finished grading submission '%s'",
                    submission_files[event.submission_idx],
                )

            else:
                logger.exception(
                    "Failed to grade submission '%s'",
                    submission_files[event.submission_idx],
                    exc_info=event.error,
                )

        elif isinstance(event, ProblemStartedEvent):
            logger.info(
                "Started grading problem %d in submission '%s'...",
                event.problem_idx,
                submission_files[event.submission_idx],
            )

        elif isinstance(event, ProblemFinishedEvent):
            logger.info(
                "Graded problem %d in submission '%s'",
                event.problem_idx,
                submission_files[event.submission_idx],
            )

    run(
        reference_file=args.reference,
        submission_files=submission_files,
        reports_dir=args.output,
        model=args.model,
        num_parallel=args.num_parallel,
        on_update=on_update,
        excel_path=args.excel,
    )


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="Grade LaTeX submissions with an LLM")
    parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        required=True,
        help="Path to reference .tex or .zip file",
    )
    parser.add_argument(
        "-s",
        "--submission",
        type=Path,
        required=True,
        help="Path to submission(s) (.zip or directory for multiple)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Directory to write JSON reports into",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model name for the OpenAI-compatible API",
    )
    parser.add_argument(
        "-p",
        "--num-parallel",
        type=int,
        default=1,
        help="Number of submissions to grade concurrently",
    )
    parser.add_argument(
        "-x",
        "--excel",
        type=Path,
        default=None,
        help="Path to assignment Excel file (.xls or .xlsx) to update with grades",
    )

    args = parser.parse_args()

    return args


def _discover_submission_files(submission_path: Path) -> list[Path]:
    if submission_path.is_dir():
        return [p for p in submission_path.iterdir()]

    else:
        return [submission_path]
