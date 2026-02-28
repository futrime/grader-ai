"""Core grading orchestration logic.

This module provides the shared business logic used by both the CLI and the
Gradio web interface.  It deliberately performs **no** file I/O — callers are
responsible for reading inputs and persisting outputs.
"""

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


# ---------------------------------------------------------------------------
# Progress event types
# ---------------------------------------------------------------------------


class ProgressStage(Enum):
    """Stages of the grading pipeline."""

    PARSING = "parsing"
    GRADING_PROBLEM = "grading_problem"
    GRADED_PROBLEM = "graded_problem"
    SUBMISSION_DONE = "submission_done"
    SUBMISSION_ERROR = "submission_error"
    ALL_DONE = "all_done"


@dataclass(frozen=True)
class ProgressEvent:
    """A progress event emitted during grading.

    Attributes:
        stage: The current pipeline stage.
        submission_name: Name of the submission being processed.
        submission_index: Zero-based index of the submission.
        total_submissions: Total number of submissions.
        problem_index: Zero-based index of the problem within the submission.
        total_problems: Total number of problems in this submission.
        grade_result: The grading result (set when ``stage`` is
            ``GRADED_PROBLEM`` or ``SUBMISSION_DONE``).
        elapsed: Seconds elapsed since the submission started processing.
        error: Exception that occurred (set when ``stage`` is
            ``SUBMISSION_ERROR``).
        report: The completed report (set when ``stage`` is
            ``SUBMISSION_DONE``).
    """

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


#: Type alias for progress callbacks.
ProgressCallback = Callable[[ProgressEvent], None]


@dataclass(frozen=True)
class Report:
    """Grading report for a single submission."""

    reference: str
    submission: str
    grades: list[GradeResult]
    total_score: int
    max_score: int
    warnings: list[str]


def grade_submission(
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
    """Parse, grade, and return a report for a single submission.

    This function performs no file I/O.

    Args:
        client: An initialised OpenAI client.
        model: The model identifier to use for grading.
        reference_name: Human-readable name for the reference (e.g. filename).
        reference_content: The full ``.tex`` content of the reference.
        submission: The submission to grade.
        submission_index: Zero-based position of this submission in the batch.
        total_submissions: Total number of submissions in the batch.
        on_progress: Optional callback invoked at each stage of processing.

    Returns:
        A ``Report`` containing per-problem grades and the total score.
    """
    t0 = time.monotonic()

    def _emit(
        stage: ProgressStage,
        *,
        problem_index: int = 0,
        total_problems: int = 0,
        grade_result: GradeResult | None = None,
        report: Report | None = None,
    ) -> None:
        if on_progress is not None:
            on_progress(
                ProgressEvent(
                    stage=stage,
                    submission_name=submission.name,
                    submission_index=submission_index,
                    total_submissions=total_submissions,
                    problem_index=problem_index,
                    total_problems=total_problems,
                    grade_result=grade_result,
                    elapsed=time.monotonic() - t0,
                    report=report,
                )
            )

    # -- Parse -----------------------------------------------------------
    _emit(ProgressStage.PARSING)
    parse_outcome = parse(reference_content, submission.content)

    logger.info(
        "Parsed %d problem(s) for submission '%s'.",
        len(parse_outcome.results),
        submission.name,
    )

    if parse_outcome.warnings:
        for warning in parse_outcome.warnings:
            logger.warning("Submission '%s': %s", submission.name, warning)

    total_problems = len(parse_outcome.results)

    # -- Grade each problem ----------------------------------------------
    grading_results: list[GradeResult] = []
    for idx, parsed in enumerate(parse_outcome.results):
        _emit(
            ProgressStage.GRADING_PROBLEM,
            problem_index=idx,
            total_problems=total_problems,
        )

        result = grade(
            client=client,
            model=model,
            parsed=parsed,
        )
        grading_results.append(result)

        _emit(
            ProgressStage.GRADED_PROBLEM,
            problem_index=idx,
            total_problems=total_problems,
            grade_result=result,
        )

    report = Report(
        reference=reference_name,
        submission=submission.name,
        grades=grading_results,
        total_score=sum(r.score for r in grading_results),
        max_score=sum(r.credits for r in grading_results),
        warnings=parse_outcome.warnings,
    )

    _emit(ProgressStage.SUBMISSION_DONE, report=report)

    return report


def grade_all(
    client: OpenAI,
    model: str,
    reference_name: str,
    reference_content: str,
    submissions: list[Submission],
    parallel: int = 1,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[Report], dict[str, Exception]]:
    """Grade every submission and return the results.

    This function performs no file I/O.

    Args:
        client: An initialised OpenAI client.
        model: The model identifier to use for grading.
        reference_name: Human-readable name for the reference (e.g. filename).
        reference_content: The full ``.tex`` content of the reference.
        submissions: List of submissions to grade.
        parallel: Maximum number of submissions to grade concurrently.
        on_progress: Optional callback invoked at each stage of processing.
            The callback may be invoked from multiple threads when
            ``parallel > 1``.

    Returns:
        A tuple of ``(reports, errors)`` where *reports* is a list of
        successful ``Report`` objects and *errors* maps submission names to
        the exceptions that occurred.
    """
    reports: list[Report] = []
    errors: dict[str, Exception] = {}
    total = len(submissions)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(
                grade_submission,
                client,
                model,
                reference_name,
                reference_content,
                submission,
                submission_index=idx,
                total_submissions=total,
                on_progress=on_progress,
            ): submission
            for idx, submission in enumerate(submissions)
        }
        for future in as_completed(futures):
            submission = futures[future]
            try:
                reports.append(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to grade submission '%s'.", submission.name)
                errors[submission.name] = exc
                if on_progress is not None:
                    on_progress(
                        ProgressEvent(
                            stage=ProgressStage.SUBMISSION_ERROR,
                            submission_name=submission.name,
                            total_submissions=total,
                            error=exc,
                        )
                    )

    if on_progress is not None:
        on_progress(ProgressEvent(stage=ProgressStage.ALL_DONE))

    return reports, errors
