"""Command line entrypoint for grader-ai."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from dotenv import load_dotenv

from grader_ai.core import (
    AnyEvent,
    ProblemFinishedEvent,
    SubmissionFinishedEvent,
    SubmissionStartedEvent,
    run,
)

logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    args = _parse_args()

    logging.basicConfig(level=logging.INFO)

    def on_update(event: AnyEvent) -> None:
        if isinstance(event, SubmissionStartedEvent):
            logger.info(
                "Started grading submission '%s' with %d problems...",
                event.submission,
                event.num_problems,
            )

        elif isinstance(event, ProblemFinishedEvent):
            logger.info(
                "Graded problem %d in submission '%s'",
                event.problem_idx,
                event.submission,
            )

        elif isinstance(event, SubmissionFinishedEvent):
            if event.error is not None:
                logger.exception(
                    "Failed to grade submission '%s'",
                    event.submission,
                    exc_info=event.error,
                )
            else:
                logger.info("Finished grading submission '%s'", event.submission)

    run(
        reference_file=args.reference,
        submission_files=_discover_submission_files(args.submission),
        reports_dir=args.output,
        model=args.model,
        num_parallel=args.num_parallel,
        on_update=on_update,
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

    args = parser.parse_args()

    return args


def _discover_submission_files(submission_path: Path) -> list[Path]:
    if submission_path.is_dir():
        return [p for p in submission_path.iterdir()]

    else:
        return [submission_path]
