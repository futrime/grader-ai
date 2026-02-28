"""Gradio web interface for grader-ai.

All file operations happen inside temporary directories so nothing is written
to the user's working directory.
"""

import json
import logging
import queue
import shutil
import tempfile
import threading
import time
from collections.abc import Generator
from dataclasses import asdict
from pathlib import Path

import gradio as gr
from openai import OpenAI

from grader_ai.core import ProgressEvent, ProgressStage, Report, grade_all
from grader_ai.extraction import extract_reference, extract_submissions

logger = logging.getLogger(__name__)


def _fetch_models() -> list[str]:
    """Fetch available model IDs from the OpenAI-compatible API.

    Returns a sorted list of model ID strings.  If the API is unreachable or
    returns an error, an empty list is returned so the UI can still accept
    manually typed model names.
    """
    try:
        client = OpenAI()
        models = client.models.list()
        return sorted(m.id for m in models)
    except Exception:
        logger.warning("Could not fetch model list from API.", exc_info=True)
        return []


def _fmt_seconds(value: float) -> str:
    """Format elapsed seconds into a compact string."""
    if value < 60:
        return f"{value:.1f}s"
    minutes = int(value // 60)
    seconds = int(value % 60)
    return f"{minutes}m {seconds:02d}s"


def _render_progress_table(
    states: dict[str, dict[str, int | float | str]],
) -> list[list[str]]:
    """Render progress rows for Gradio Dataframe."""
    rows: list[list[str]] = []
    for name in sorted(states):
        info = states[name]
        status = str(info.get("status", "pending"))
        score = int(info.get("score", 0))
        max_score = int(info.get("max_score", 0))
        graded = int(info.get("graded", 0))
        total = int(info.get("total", 0))
        elapsed_each = float(info.get("elapsed", 0.0))
        error = str(info.get("error", ""))
        rows.append(
            [
                name,
                status,
                f"{graded}/{total}" if total else "-",
                f"{score}/{max_score}",
                _fmt_seconds(elapsed_each),
                error,
            ]
        )
    return rows


def _render_progress_summary(
    *,
    done_count: int,
    error_count: int,
    total_submissions: int,
    graded_problems: int,
    total_problems: int,
    elapsed: float,
) -> str:
    """Render a concise progress summary using Markdown."""
    return (
        f"**Progress**  \\n"
        f"Submissions: {done_count}/{total_submissions} | "
        f"Problems: {graded_problems}/{total_problems} | "
        f"Errors: {error_count} | "
        f"Elapsed: {_fmt_seconds(elapsed)}"
    )


def _handle_grade(
    reference_file: str | None,
    submission_files: list[str] | None,
    model: str,
    parallel: int,
) -> Generator[tuple[list[dict], list[str], str, list[list[str]], str], None, None]:
    """Grade submissions and return results.

    All intermediate and output files are written to temporary directories.

    Args:
        reference_file: Path to the uploaded reference file (from Gradio).
        submission_files: List of paths to uploaded submission files.
        model: OpenAI model identifier.
        parallel: Number of parallel grading workers.

    Returns:
        A tuple of ``(report_dicts, report_file_paths, log_text,
        progress_rows, progress_summary)``.
    """
    # -- Validate inputs -------------------------------------------------
    if not reference_file:
        raise gr.Error("Please upload a reference file.")
    if not submission_files:
        raise gr.Error("Please upload at least one submission file.")
    if not model.strip():
        raise gr.Error("Please enter a model name.")

    log_lines: list[str] = []

    def _log(msg: str) -> None:
        logger.info(msg)
        log_lines.append(msg)

    # -- Set up temp directories -----------------------------------------
    work_dir = tempfile.mkdtemp(prefix="grader_ai_work_")
    output_dir = tempfile.mkdtemp(prefix="grader_ai_out_")

    started_at = time.monotonic()

    try:
        # Copy uploaded files into a controlled work directory.
        ref_dest = Path(work_dir) / Path(reference_file).name
        shutil.copy2(reference_file, ref_dest)
        _log(f"Reference: {ref_dest.name}")

        sub_dir = Path(work_dir) / "submissions"
        sub_dir.mkdir()
        for src_path in submission_files:
            shutil.copy2(src_path, sub_dir / Path(src_path).name)

        # -- Extract content ---------------------------------------------
        reference_content = extract_reference(ref_dest)
        submissions = extract_submissions(sub_dir)
        _log(f"Extracted {len(submissions)} submission(s).")

        if not submissions:
            raise gr.Error("No valid submissions found. Upload .tex or .zip files.")

        total_submissions = len(submissions)
        submission_states: dict[str, dict[str, int | float | str]] = {
            sub.name: {
                "status": "pending",
                "graded": 0,
                "total": 0,
                "score": 0,
                "max_score": 0,
                "last_feedback": "",
                "warning_count": 0,
                "elapsed": 0.0,
                "error": "",
            }
            for sub in submissions
        }

        event_queue: queue.Queue[ProgressEvent] = queue.Queue()
        reports_result: list[Report] = []
        errors_result: dict[str, Exception] = {}
        fatal_errors: list[Exception] = []
        done_event = threading.Event()

        def _on_progress(event: ProgressEvent) -> None:
            event_queue.put(event)

        def _run_grading() -> None:
            try:
                client = OpenAI()
                reports, errors = grade_all(
                    client=client,
                    model=model,
                    reference_name=ref_dest.name,
                    reference_content=reference_content,
                    submissions=submissions,
                    parallel=int(parallel),
                    on_progress=_on_progress,
                )
                reports_result.extend(reports)
                errors_result.update(errors)
            except Exception as exc:  # noqa: BLE001
                fatal_errors.append(exc)
            finally:
                done_event.set()

        grading_thread = threading.Thread(target=_run_grading, daemon=True)
        grading_thread.start()

        done_count = 0
        error_count = 0
        graded_problems = 0
        total_problems = 0

        while True:
            had_event = False
            while True:
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    break

                had_event = True
                state = submission_states.get(event.submission_name)
                if state is None:
                    continue

                state["elapsed"] = event.elapsed

                if event.stage is ProgressStage.PARSING:
                    state["status"] = "parsing"
                    _log(f"[{event.submission_name}] Parsing...")

                elif event.stage is ProgressStage.GRADING_PROBLEM:
                    if event.problem_index == 0:
                        total_problems += event.total_problems
                        state["total"] = event.total_problems
                        state["max_score"] = 0
                    state["status"] = "grading"
                    _log(
                        f"[{event.submission_name}] Grading "
                        f"Q{event.problem_index + 1}/{event.total_problems}..."
                    )

                elif event.stage is ProgressStage.GRADED_PROBLEM:
                    graded_problems += 1
                    state["status"] = "grading"
                    state["graded"] = event.problem_index + 1
                    if event.grade_result is not None:
                        grade_result = event.grade_result
                        state["score"] = int(state["score"]) + grade_result.score
                        state["max_score"] = (
                            int(state["max_score"]) + grade_result.credits
                        )
                        state["last_feedback"] = grade_result.feedback
                        _log(
                            f"[{event.submission_name}] "
                            f"Q{event.problem_index + 1}: "
                            f"{grade_result.score}/{grade_result.credits}"
                        )

                elif event.stage is ProgressStage.SUBMISSION_DONE:
                    done_count += 1
                    state["status"] = "done"
                    if event.report is not None:
                        state["score"] = event.report.total_score
                        state["max_score"] = event.report.max_score
                        state["total"] = len(event.report.grades)
                        state["graded"] = len(event.report.grades)
                        state["warning_count"] = len(event.report.warnings)
                    _log(
                        f"[{event.submission_name}] Done: "
                        f"{state['score']}/{state['max_score']}"
                    )

                elif event.stage is ProgressStage.SUBMISSION_ERROR:
                    done_count += 1
                    error_count += 1
                    state["status"] = "error"
                    state["error"] = str(event.error)
                    _log(f"ERROR [{event.submission_name}]: {event.error}")

            elapsed = time.monotonic() - started_at
            progress_rows = _render_progress_table(submission_states)
            progress_summary = _render_progress_summary(
                done_count=done_count,
                error_count=error_count,
                total_submissions=total_submissions,
                graded_problems=graded_problems,
                total_problems=total_problems,
                elapsed=elapsed,
            )

            if had_event or not done_event.is_set():
                # Keep outputs empty until finalization while still streaming
                # progress/log updates.
                yield [], [], "\n".join(log_lines), progress_rows, progress_summary

            if done_event.is_set() and event_queue.empty():
                break

            time.sleep(0.1)

        if fatal_errors:
            fatal_error = fatal_errors[0]
            raise gr.Error(str(fatal_error)) from fatal_error

        reports = reports_result
        errors = errors_result

        # -- Serialize reports to temp output dir ------------------------
        report_dicts: list[dict] = []
        report_files: list[str] = []

        for report in _sort_reports(reports):
            d = asdict(report)
            report_dicts.append(d)

            out_path = Path(output_dir) / f"{report.submission}.json"
            out_path.write_text(json.dumps(d, indent=2))
            report_files.append(str(out_path))

            if report.warnings:
                for warning in report.warnings:
                    _log(f"WARNING [{report.submission}]: {warning}")

        _log(f"Graded {len(reports)} submission(s) successfully.")

        if errors:
            for name, exc in sorted(errors.items()):
                _log(f"ERROR [{name}]: {exc}")
            _log(
                f"Failed to grade {len(errors)} submission(s): "
                + ", ".join(sorted(errors))
            )

        elapsed = time.monotonic() - started_at
        progress_rows = _render_progress_table(submission_states)
        progress_summary = _render_progress_summary(
            done_count=done_count,
            error_count=error_count,
            total_submissions=total_submissions,
            graded_problems=graded_problems,
            total_problems=total_problems,
            elapsed=elapsed,
        )

        yield (
            report_dicts,
            report_files,
            "\n".join(log_lines),
            progress_rows,
            progress_summary,
        )
        return

    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during grading.")
        log_lines.append(f"FATAL: {exc}")
        raise gr.Error(str(exc)) from exc


def _sort_reports(reports: list[Report]) -> list[Report]:
    """Sort reports by submission name for deterministic output."""
    return sorted(reports, key=lambda r: r.submission)


def create_app() -> gr.Blocks:
    """Build and return the Gradio ``Blocks`` application."""
    with gr.Blocks(title="grader-ai") as app:
        gr.Markdown("# grader-ai\nGrade assignments with LLMs.")

        with gr.Row():
            with gr.Column(scale=1):
                reference_input = gr.File(
                    label="Reference file (.tex or .zip)",
                    file_types=[".tex", ".zip"],
                    type="filepath",
                )
                submission_input = gr.File(
                    label="Submission file(s) (.tex or .zip)",
                    file_types=[".tex", ".zip"],
                    file_count="multiple",
                    type="filepath",
                )
                available_models = _fetch_models()
                model_input = gr.Dropdown(
                    label="Model",
                    choices=available_models,
                    allow_custom_value=True,
                )
                parallel_input = gr.Slider(
                    label="Parallel workers",
                    minimum=1,
                    maximum=16,
                    value=1,
                    step=1,
                )
                grade_button = gr.Button("Grade", variant="primary")

            with gr.Column(scale=2):
                progress_summary_output = gr.Markdown(
                    value=_render_progress_summary(
                        done_count=0,
                        error_count=0,
                        total_submissions=0,
                        graded_problems=0,
                        total_problems=0,
                        elapsed=0.0,
                    )
                )
                progress_table_output = gr.Dataframe(
                    headers=[
                        "Submission",
                        "Status",
                        "Problems",
                        "Score",
                        "Elapsed",
                        "Error",
                    ],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    value=[],
                    interactive=False,
                    label="Live progress",
                )
                log_output = gr.Textbox(
                    label="Log",
                    lines=8,
                    interactive=False,
                )
                report_output = gr.JSON(label="Reports")
                download_output = gr.File(
                    label="Download reports",
                    file_count="multiple",
                    type="filepath",
                    interactive=False,
                )

        grade_button.click(
            fn=_handle_grade,
            inputs=[reference_input, submission_input, model_input, parallel_input],
            outputs=[
                report_output,
                download_output,
                log_output,
                progress_table_output,
                progress_summary_output,
            ],
            show_progress="hidden",
        )

    return app


def launch(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Create the Gradio app and launch it.

    Args:
        host: Network interface to bind to.
        port: Port number.
        share: Whether to create a public Gradio link.
    """
    app = create_app()
    app.launch(server_name=host, server_port=port, share=share)
