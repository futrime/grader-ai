"""Command line entrypoint for grader-ai."""

import logging
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv

from grader_ai.core import run


def main() -> None:
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
        "--submissions",
        type=Path,
        required=True,
        help="Path to submissions (.tex/.zip or directory)",
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

    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    run(
        reference_file=args.reference,
        submissions_dir=args.submissions,
        reports_dir=args.output,
        model=args.model,
        num_parallel=args.num_parallel,
    )
