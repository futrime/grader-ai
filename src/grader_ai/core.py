"""Core grading orchestration and data helpers."""

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

from grader_ai.extraction import Submission, extract_reference, extract_submissions
from grader_ai.grading import GradeResult, grade
from grader_ai.parsing import parse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Report:
    reference: str
    submission: str
    grade_results: list[GradeResult]
    error: str | None = None


@dataclass(frozen=True)
class State:
    submission: str
    num_problems: int
    num_graded: int


def run(
    *,
    reference_file: Path,
    submissions_dir: Path,
    reports_dir: Path,
    model: str,
    num_parallel: int,
    on_update: Callable[[State], None] | None = None,
) -> None:
    client = OpenAI()

    reference = extract_reference(reference_file)
    submissions = extract_submissions(submissions_dir)

    def grade_submission(submission: Submission) -> Report:
        logger.info("Grading submission '%s'...", submission.name)

        try:
            parse_results = parse(reference, submission.content)

            grade_results: list[GradeResult] = []

            for parsed in parse_results:
                grade_result = grade(client, model, parsed)

                grade_results.append(grade_result)

                if on_update is not None:
                    on_update(
                        State(
                            submission=submission.name,
                            num_problems=len(parse_results),
                            num_graded=len(grade_results),
                        )
                    )

            return Report(
                reference=reference_file.name,
                submission=submission.name,
                grade_results=grade_results,
            )

        except Exception as e:
            logger.exception(
                "Failed to grade submission '%s'", submission.name, exc_info=e
            )

            return Report(
                reference=reference_file.name,
                submission=submission.name,
                grade_results=[],
                error=str(e),
            )

    reports: list[Report] = []

    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        for report in executor.map(grade_submission, submissions):
            reports.append(report)

    logger.info("Grading complete, writing reports to '%s'...", reports_dir)

    for report in reports:
        with open(reports_dir / f"{report.submission}.json", "w") as f:
            json.dump(asdict(report), f)
