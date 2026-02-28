"""Core grading orchestration logic (no file I/O)."""

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI

from grader_ai.extraction import Submission
from grader_ai.grading import GradeResult, grade
from grader_ai.parsing import parse

logger = logging.getLogger(__name__)


class ProgressStage(Enum):
    PROBLEM_GRADED = "problem_graded"
    SUBMISSION_COMPLETE = "submission_complete"
    ALL_DONE = "all_done"


@dataclass(frozen=True)
class ProgressEvent:
    stage: ProgressStage
    submission_name: str = ""
    submission_index: int = 0
    total_submissions: int = 0
    problem_index: int = 0
    total_problems: int = 0
    grade_result: GradeResult | None = None
    elapsed: float = 0.0
    error: Exception | None = None
    report: "Report | None" = None


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass(frozen=True)
class Report:
    reference: str
    submission: str
    grades: list[GradeResult]
    total_score: int
    max_score: int
    warnings: list[str]


def grade_all(
    client: OpenAI,
    model: str,
    reference_name: str,
    reference_content: str,
    submissions: list[Submission],
    parallel: int = 1,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[Report], dict[str, Exception]]:
    if parallel < 1:
        raise ValueError("parallel must be at least 1")

    reports: list[Report] = []
    errors: dict[str, Exception] = {}
    total = len(submissions)

    def run(index: int, submission: Submission) -> Report:
        return _grade_submission(
            client=client,
            model=model,
            reference_name=reference_name,
            reference_content=reference_content,
            submission=submission,
            submission_index=index,
            total_submissions=total,
            on_progress=on_progress,
        )

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(run, i, sub): (i, sub) for i, sub in enumerate(submissions)
        }
        for future in as_completed(futures):
            index, submission = futures[future]
            try:
                reports.append(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to grade submission '%s'.", submission.name)
                errors[submission.name] = exc
                if on_progress is not None:
                    on_progress(
                        ProgressEvent(
                            stage=ProgressStage.SUBMISSION_COMPLETE,
                            submission_name=submission.name,
                            submission_index=index,
                            total_submissions=total,
                            error=exc,
                        )
                    )

    if on_progress is not None:
        on_progress(ProgressEvent(stage=ProgressStage.ALL_DONE))
    return reports, errors


def _grade_submission(
    client: OpenAI,
    model: str,
    reference_name: str,
    reference_content: str,
    submission: Submission,
    *,
    submission_index: int = 0,
    total_submissions: int = 1,
    on_progress: ProgressCallback | None = None,
) -> Report:
    started_at = time.monotonic()
    parse_outcome = parse(reference_content, submission.content)
    results: list[GradeResult] = []
    n_problems = len(parse_outcome.results)

    for index, parsed in enumerate(parse_outcome.results):
        graded = grade(client=client, model=model, parsed=parsed)
        results.append(graded)
        if on_progress is not None:
            on_progress(
                ProgressEvent(
                    stage=ProgressStage.PROBLEM_GRADED,
                    submission_name=submission.name,
                    submission_index=submission_index,
                    total_submissions=total_submissions,
                    problem_index=index,
                    total_problems=n_problems,
                    grade_result=graded,
                    elapsed=time.monotonic() - started_at,
                )
            )

    report = Report(
        reference=reference_name,
        submission=submission.name,
        grades=results,
        total_score=sum(r.score for r in results),
        max_score=sum(r.credits for r in results),
        warnings=parse_outcome.warnings,
    )
    if on_progress is not None:
        on_progress(
            ProgressEvent(
                stage=ProgressStage.SUBMISSION_COMPLETE,
                submission_name=submission.name,
                submission_index=submission_index,
                total_submissions=total_submissions,
                elapsed=time.monotonic() - started_at,
                report=report,
            )
        )
    return report
