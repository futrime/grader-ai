import json
import logging
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path

import dotenv
from openai import OpenAI

from grader_ai.extraction import extract_reference, extract_submissions
from grader_ai.grading import GradeResult, grade
from grader_ai.parsing import parse


@dataclass(frozen=True)
class _Args:
    reference: Path
    submission: Path
    output: Path
    model: str


@dataclass(frozen=True)
class _Report:
    reference: str
    submission: str
    grades: list[GradeResult]
    total_score: int


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = _parse_args()

    dotenv.load_dotenv()

    reference_content = extract_reference(args.reference)
    submissions = extract_submissions(args.submission)

    logging.info(
        "Extracted %d submission(s) from %s.", len(submissions), args.submission
    )

    args.output.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    for submission in submissions:
        parse_results = parse(reference_content, submission.content)

        logging.info(
            "Parsed %d problem(s) for submission '%s'.",
            len(parse_results),
            submission.name,
        )

        grading_results: list[GradeResult] = []
        for parsed in parse_results:
            result = grade(
                client=client,
                model=args.model,
                parsed=parsed,
            )
            grading_results.append(result)

        report = _Report(
            reference=args.reference.name,
            submission=submission.name,
            grades=grading_results,
            total_score=sum(r.score for r in grading_results),
        )

        output_file = args.output / f"{submission.name}.json"
        output_file.write_text(json.dumps(asdict(report), indent=2))

        logging.info("Wrote report to %s.", output_file)


def _parse_args() -> _Args:
    parser = ArgumentParser()

    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-r", "--reference", type=Path, required=True)
    parser.add_argument("-s", "--submission", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)

    args = parser.parse_args()

    return _Args(**vars(args))
