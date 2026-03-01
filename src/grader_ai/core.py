"""Core grading orchestration and data helpers."""

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

from grader_ai.extraction import Submission, extract_reference, extract_submission
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
class RunStartedEvent:
    submissions: list[str]


@dataclass(frozen=True)
class RunFinishedEvent:
    report_files: list[Path]


@dataclass(frozen=True)
class SubmissionStartedEvent:
    submission_idx: int
    submission: str
    num_problems: int


@dataclass(frozen=True)
class SubmissionFinishedEvent:
    submission_idx: int
    submission: str
    error: Exception | None


@dataclass(frozen=True)
class ProblemStartedEvent:
    submission_idx: int
    submission: str
    num_problems: int
    problem_idx: int


@dataclass(frozen=True)
class ProblemFinishedEvent:
    submission_idx: int
    submission: str
    num_problems: int
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
    on_update: Callable[[AnyEvent], None] | None = None,
) -> None:
    client = OpenAI()

    reference = extract_reference(reference_file)

    submissions = []
    for submission_file in submission_files:
        try:
            submission = extract_submission(submission_file)
            submissions.append(submission)

        except Exception as e:
            logger.exception(
                "Failed to extract submission from '%s'", submission_file, exc_info=e
            )

    logger.info("Extracted reference and %d submissions", len(submissions))

    def grade_submission(submission_idx: int, submission: Submission) -> Report:
        logger.info("Grading submission '%s'...", submission.name)

        try:
            parse_results = parse(reference, submission.content)

            if on_update is not None:
                on_update(
                    SubmissionStartedEvent(
                        submission_idx=submission_idx,
                        submission=submission.name,
                        num_problems=len(parse_results),
                    )
                )

            grade_results: list[GradeResult] = []
            for problem_idx, parsed in enumerate(parse_results, start=1):
                if on_update is not None:
                    on_update(
                        ProblemStartedEvent(
                            submission_idx=submission_idx,
                            submission=submission.name,
                            num_problems=len(parse_results),
                            problem_idx=problem_idx,
                        )
                    )

                grade_result = grade(client, model, parsed)
                grade_results.append(grade_result)

                if on_update is not None:
                    on_update(
                        ProblemFinishedEvent(
                            submission_idx=submission_idx,
                            submission=submission.name,
                            num_problems=len(parse_results),
                            problem_idx=problem_idx,
                        )
                    )

            if on_update is not None:
                on_update(
                    SubmissionFinishedEvent(
                        submission_idx=submission_idx,
                        submission=submission.name,
                        error=None,
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

            if on_update is not None:
                on_update(
                    SubmissionFinishedEvent(
                        submission_idx=submission_idx,
                        submission=submission.name,
                        error=e,
                    )
                )

            return Report(
                reference=reference_file.name,
                submission=submission.name,
                grade_results=[],
                error=str(e),
            )

    if on_update is not None:
        on_update(RunStartedEvent(submissions=[s.name for s in submissions]))

    reports: list[Report] = []
    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        for report in executor.map(grade_submission, *zip(*enumerate(submissions))):
            reports.append(report)

    logger.info("Grading complete, writing reports to '%s'...", reports_dir)

    reports_dir.mkdir(parents=True, exist_ok=True)

    report_files: list[Path] = []
    for report in reports:
        report_file = reports_dir / f"{report.submission}.json"

        with open(report_file, "w") as f:
            json.dump(asdict(report), f)

        report_files.append(report_file)

    if on_update is not None:
        on_update(RunFinishedEvent(report_files=report_files))
