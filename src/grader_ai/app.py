"""Gradio web interface for grader-ai."""

import json
import logging
import queue
import shutil
import tempfile
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import gradio as gr
from openai import OpenAI

from grader_ai.core import ProgressEvent, ProgressStage, Report, grade_all
from grader_ai.extraction import extract_reference, extract_submissions

logger = logging.getLogger(__name__)


def _fetch_models() -> list[str]:
    try:
        return sorted(model.id for model in OpenAI().models.list())
    except Exception:
        logger.warning("Could not fetch model list from API.", exc_info=True)
        return []


def _fmt_seconds(value: float) -> str:
    if value < 60:
        return f"{value:.1f}s"
    minutes = int(value // 60)
    return f"{minutes}m {int(value % 60):02d}s"


def _render_progress_table(
    states: dict[str, dict[str, int | float | str]],
) -> list[list[str]]:
    return [
        [
            name,
            str(s["status"]),
            f"{int(s['graded'])}/{int(s['total'])}" if int(s["total"]) else "-",
            f"{int(s['score'])}/{int(s['max_score'])}",
            _fmt_seconds(float(s["elapsed"])),
            str(s["error"]),
        ]
        for name, s in sorted(states.items())
    ]


def _render_summary(
    counters: dict[str, int], total_submissions: int, elapsed: float
) -> str:
    return (
        "**Progress**  \n"
        f"Submissions: {counters['done']}/{total_submissions} | "
        f"Problems: {counters['graded']}/{counters['total_problems']} | "
        f"Errors: {counters['errors']} | "
        f"Elapsed: {_fmt_seconds(elapsed)}"
    )


def _update_state(
    states: dict[str, dict[str, int | float | str]],
    event: ProgressEvent,
    counters: dict[str, int],
    logs: list[str],
) -> None:
    state = states.get(event.submission_name)
    if state is None:
        return
    state["elapsed"] = event.elapsed
    if event.stage is ProgressStage.PROBLEM_GRADED:
        state["status"] = "grading"
        if event.problem_index == 0:
            state["total"] = event.total_problems
            counters["total_problems"] += event.total_problems
        state["graded"] = event.problem_index + 1
        counters["graded"] += 1
        if event.grade_result is not None:
            state["score"] = int(state["score"]) + event.grade_result.score
            state["max_score"] = int(state["max_score"]) + event.grade_result.credits
    elif event.stage is ProgressStage.SUBMISSION_COMPLETE:
        counters["done"] += 1
        if event.error:
            counters["errors"] += 1
            state["status"] = "error"
            state["error"] = str(event.error)
            logs.append(f"ERROR [{event.submission_name}]: {event.error}")
        elif event.report is not None:
            state["status"] = "done"
            state["score"] = event.report.total_score
            state["max_score"] = event.report.max_score
            state["total"] = len(event.report.grades)
            state["graded"] = len(event.report.grades)
            logs.append(
                f"[{event.submission_name}] Done: "
                f"{event.report.total_score}/{event.report.max_score}"
            )


def _finalize_reports(
    reports: list[Report],
    errors: dict[str, Exception],
    output_dir: Path,
    logs: list[str],
) -> tuple[list[dict], list[str]]:
    report_dicts: list[dict] = []
    report_files: list[str] = []
    for report in sorted(reports, key=lambda r: r.submission):
        serialized = asdict(report)
        report_dicts.append(serialized)
        output_path = output_dir / f"{report.submission}.json"
        output_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
        report_files.append(str(output_path))
        for warning in report.warnings:
            logs.append(f"WARNING [{report.submission}]: {warning}")
    for name, exc in sorted(errors.items()):
        logs.append(f"ERROR [{name}]: {exc}")
    return report_dicts, report_files


def _handle_grade(
    reference_file: str | None,
    submission_files: list[str] | None,
    model: str,
    parallel: int,
) -> Generator[tuple[list[dict], list[str], str, list[list[str]], str], None, None]:
    if not reference_file:
        raise gr.Error("Please upload a reference file.")
    if not submission_files:
        raise gr.Error("Please upload at least one submission file.")
    if not model.strip():
        raise gr.Error("Please enter a model name.")

    logs: list[str] = []
    work_dir = Path(tempfile.mkdtemp(prefix="grader_ai_work_"))
    output_dir = Path(tempfile.mkdtemp(prefix="grader_ai_out_"))
    started_at = time.monotonic()

    try:
        reference_path = work_dir / Path(reference_file).name
        shutil.copy2(reference_file, reference_path)
        submission_dir = work_dir / "submissions"
        submission_dir.mkdir()
        for fp in submission_files:
            shutil.copy2(fp, submission_dir / Path(fp).name)

        reference_content = extract_reference(reference_path)
        submissions = extract_submissions(submission_dir)
        if not submissions:
            raise gr.Error("No valid submissions found. Upload .tex or .zip files.")

        logs.append(f"Extracted {len(submissions)} submission(s).")

        states: dict[str, dict[str, int | float | str]] = {
            sub.name: {
                "status": "pending",
                "graded": 0,
                "total": 0,
                "score": 0,
                "max_score": 0,
                "elapsed": 0.0,
                "error": "",
            }
            for sub in submissions
        }
        counters = {"done": 0, "errors": 0, "graded": 0, "total_problems": 0}
        event_queue: queue.Queue[ProgressEvent] = queue.Queue()

        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(
                grade_all,
                client=OpenAI(),
                model=model,
                reference_name=reference_path.name,
                reference_content=reference_content,
                submissions=submissions,
                parallel=int(parallel),
                on_progress=event_queue.put,
            )

            while not future.done() or not event_queue.empty():
                updated = False
                while not event_queue.empty():
                    try:
                        event = event_queue.get_nowait()
                    except queue.Empty:
                        break
                    updated = True
                    _update_state(states, event, counters, logs)

                summary = _render_summary(
                    counters, len(submissions), time.monotonic() - started_at
                )
                if updated or not future.done():
                    yield (
                        [],
                        [],
                        "\n".join(logs),
                        _render_progress_table(states),
                        summary,
                    )
                time.sleep(0.1)

        reports, grading_errors = future.result()
        report_dicts, report_files = _finalize_reports(
            reports, grading_errors, output_dir, logs
        )
        yield (
            report_dicts,
            report_files,
            "\n".join(logs),
            _render_progress_table(states),
            _render_summary(counters, len(submissions), time.monotonic() - started_at),
        )
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during grading.")
        raise gr.Error(str(exc)) from exc


def create_app() -> gr.Blocks:
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
                model_input = gr.Dropdown(
                    label="Model",
                    choices=_fetch_models(),
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
                    value="**Progress**  \nSubmissions: 0/0 | Problems: 0/0 | Errors: 0 | Elapsed: 0.0s"
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
                    datatype="str",
                    value=[],
                    interactive=False,
                    label="Live progress",
                )
                log_output = gr.Textbox(label="Log", lines=8, interactive=False)
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


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> None:
    create_app().launch(server_name=host, server_port=port, share=share)
