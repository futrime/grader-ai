import json
import logging
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path

import dotenv
from openai import OpenAI

from grader_ai.grading import GradeResult, grade
from grader_ai.parsing import parse


@dataclass(frozen=True)
class _Args:
    reference: Path
    submission: Path
    output: Path
    model: str


@dataclass(frozen=True)
class Report:
    reference: str
    submission: str
    grades: list[GradeResult]
    total_score: int


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = _parse_args()

    dotenv.load_dotenv()

    reference_content = args.reference.read_text()
    submission_content = args.submission.read_text()

    parse_results = parse(reference_content, submission_content)

    logging.info(
        "Parsed %s problems from reference and submission.", len(parse_results)
    )

    client = OpenAI()

    grading_results: list[GradeResult] = []
    for parsed in parse_results:
        result = grade(
            client=client,
            model=args.model,
            parsed=parsed,
        )
        grading_results.append(result)

    report = Report(
        reference=args.reference.name,
        submission=args.submission.name,
        grades=grading_results,
        total_score=sum(result.score for result in grading_results),
    )

    with args.output.open("a") as f:
        json.dump(asdict(report), f)


def _parse_args() -> _Args:
    parser = ArgumentParser()

    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-r", "--reference", type=Path, required=True)
    parser.add_argument("-s", "--submission", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)

    args = parser.parse_args()

    return _Args(**vars(args))
