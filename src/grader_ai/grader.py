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
class SubmissionCachedEvent:
    submission_idx: int
    num_problems: int
    report_file: Path


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
    | SubmissionCachedEvent
)


def run(
    *,
    reference_file: Path,
    submission_files: list[Path],
    model: str,
    num_parallel: int,
    reports_dir: Path,
    on_update: Callable[[AnyEvent], None],
    excel_path: Path | None = None,
    use_cache: bool = True,
) -> None:
    on_update(RunStartedEvent(submission_files=submission_files))

    reports_dir.mkdir(parents=True, exist_ok=True)

    cached_reports: dict[int, Report] = {}
    submissions_to_grade: list[tuple[int, Path]] = []
    report_files_by_idx = {
        submission_idx: reports_dir / f"{submission_file.stem}.json"
        for submission_idx, submission_file in enumerate(submission_files)
    }

    for submission_idx, submission_file in enumerate(submission_files):
        report_file = report_files_by_idx[submission_idx]
        cached_report = (
            _load_cached_report(
                report_file=report_file,
                reference_file=reference_file,
                submission_file=submission_file,
            )
            if use_cache
            else None
        )
        if cached_report is None:
            submissions_to_grade.append((submission_idx, submission_file))
            continue

        cached_reports[submission_idx] = cached_report
        on_update(
            SubmissionCachedEvent(
                submission_idx=submission_idx,
                num_problems=len(cached_report.grade_results),
                report_file=report_file,
            )
        )

    client = OpenAI() if submissions_to_grade else None
    reference_content = (
        extract_reference(reference_file) if submissions_to_grade else None
    )

    def run_submission(item: tuple[int, Path]) -> tuple[int, Report]:
        submission_idx, submission_file = item
        try:
            submission_content = extract_submission(submission_file)

            assert reference_content is not None
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

                assert client is not None
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

            return submission_idx, Report(
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

            return submission_idx, Report(
                reference_file=reference_file.name,
                submission_file=submission_file.name,
                grade_results=[],
                error=str(e),
            )

    reports_by_idx: dict[int, Report] = dict(cached_reports)
    if submissions_to_grade:
        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            for submission_idx, report in executor.map(
                run_submission, submissions_to_grade
            ):
                reports_by_idx[submission_idx] = report

    reports = [reports_by_idx[idx] for idx in range(len(submission_files))]
    report_files: list[Path] = []
    for submission_idx, report in enumerate(reports):
        report_file = report_files_by_idx[submission_idx]
        if submission_idx not in cached_reports:
            _write_report(report_file, report)

        report_files.append(report_file)

    on_update(RunFinishedEvent(report_files=report_files))

    if excel_path is not None:
        from grader_ai.excel_export import export_to_excel

        export_to_excel(excel_path, reports)


def _load_cached_report(
    *,
    report_file: Path,
    reference_file: Path,
    submission_file: Path,
) -> Report | None:
    if not report_file.exists():
        return None

    newest_input_mtime = max(
        reference_file.stat().st_mtime, submission_file.stat().st_mtime
    )
    if report_file.stat().st_mtime < newest_input_mtime:
        return None

    try:
        report = _load_report(report_file)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
        logger.warning("Ignoring invalid cached report '%s': %s", report_file, e)
        return None

    if report.error is not None:
        return None

    if report.reference_file != reference_file.name:
        return None

    if report.submission_file != submission_file.name:
        return None

    return report


def _load_report(report_file: Path) -> Report:
    with report_file.open(encoding="utf-8") as f:
        payload = json.load(f)

    return Report(
        reference_file=payload["reference_file"],
        submission_file=payload["submission_file"],
        grade_results=[GradeResult(**result) for result in payload["grade_results"]],
        error=payload["error"],
    )


def _write_report(report_file: Path, report: Report) -> None:
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f)
