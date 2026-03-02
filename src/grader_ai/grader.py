"""Core grading orchestration and data helpers."""

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

from grader_ai.extraction import extract_reference, extract_submission
from grader_ai.grading import GradeResult, grade
from grader_ai.parsing import parse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Report:
    reference_file: str
    submission_file: str
    grade_results: list[GradeResult]
    error: str | None


@dataclass(frozen=True)
class RunStartedEvent:
    submission_files: list[Path]


@dataclass(frozen=True)
class RunFinishedEvent:
    report_files: list[Path]


@dataclass(frozen=True)
class SubmissionStartedEvent:
    submission_idx: int
    num_problems: int


@dataclass(frozen=True)
class SubmissionFinishedEvent:
    submission_idx: int
    error: Exception | None


@dataclass(frozen=True)
class ProblemStartedEvent:
    submission_idx: int
    problem_idx: int


@dataclass(frozen=True)
class ProblemFinishedEvent:
    submission_idx: int
    problem_idx: int


type AnyEvent = (
    RunStartedEvent
    | RunFinishedEvent
    | ProblemStartedEvent
    | ProblemFinishedEvent
    | SubmissionStartedEvent
    | SubmissionFinishedEvent
)


def run(
    *,
    reference_file: Path,
    submission_files: list[Path],
    model: str,
    num_parallel: int,
    reports_dir: Path,
    on_update: Callable[[AnyEvent], None],
) -> None:
    on_update(RunStartedEvent(submission_files=submission_files))

    client = OpenAI()

    reference_content = extract_reference(reference_file)

    def run_submission(submission_idx: int, submission_file: Path) -> Report:
        try:
            submission_content = extract_submission(submission_file)

            parse_results = parse(reference_content, submission_content)

            on_update(
                SubmissionStartedEvent(
                    submission_idx=submission_idx, num_problems=len(parse_results)
                )
            )

            grade_results: list[GradeResult] = []
            for problem_idx, parsed in enumerate(parse_results):
                on_update(
                    ProblemStartedEvent(
                        submission_idx=submission_idx, problem_idx=problem_idx
                    )
                )

                grade_result = grade(client, model, parsed)
                grade_results.append(grade_result)

                on_update(
                    ProblemFinishedEvent(
                        submission_idx=submission_idx, problem_idx=problem_idx
                    )
                )

            on_update(
                SubmissionFinishedEvent(submission_idx=submission_idx, error=None)
            )

            return Report(
                reference_file=reference_file.name,
                submission_file=submission_file.name,
                grade_results=grade_results,
                error=None,
            )

        except Exception as e:
            logger.exception(
                f"Failed to grade submission {submission_file}", exc_info=e
            )

            on_update(
                SubmissionFinishedEvent(
                    submission_idx=submission_idx,
                    error=e,
                )
            )

            return Report(
                reference_file=reference_file.name,
                submission_file=submission_file.name,
                grade_results=[],
                error=str(e),
            )

    reports: list[Report] = []
    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        for report in executor.map(run_submission, *zip(*enumerate(submission_files))):
            reports.append(report)

    reports_dir.mkdir(parents=True, exist_ok=True)

    report_files: list[Path] = []
    for report in reports:
        report_file = reports_dir / f"{Path(report.submission_file).stem}.json"

        with open(report_file, "w") as f:
            json.dump(asdict(report), f)

        report_files.append(report_file)

    on_update(RunFinishedEvent(report_files=report_files))
