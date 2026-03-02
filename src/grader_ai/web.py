"""Gradio UI entrypoint for grader-ai."""

from __future__ import annotations

import argparse
import logging
import queue
import tempfile
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Final, Literal

import dotenv
import gradio as gr
import openai

import grader_ai.core

logger = logging.getLogger(__name__)

_STATUS_KEY_TO_IDX: Final[
    dict[Literal["Submission", "Status", "# Graded", "# Problems"], int]
] = {
    "Submission": 0,
    "Status": 1,
    "# Problems": 2,
    "# Graded": 3,
}


def main() -> None:
    dotenv.load_dotenv()

    args = _parse_args()

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as temp_dir:
        app = _build_app(Path(temp_dir))

        app.launch(server_name=args.host, server_port=args.port)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the grader-ai Gradio UI")
    parser.add_argument("--host", type=str, default=None, help="Server bind host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    return parser.parse_args()


def _build_app(temp_dir: Path) -> gr.Blocks:
    def fn(
        reference_file: str, submission_files: list[str], model: str, num_parallel: int
    ) -> Iterator[tuple[list[list[str]], list[str], Path | None]]:
        update_queue: queue.Queue[grader_ai.core.AnyEvent] = queue.Queue()

        def worker() -> None:
            try:
                grader_ai.core.run(
                    reference_file=Path(reference_file),
                    submission_files=[Path(s) for s in submission_files],
                    model=model,
                    num_parallel=num_parallel,
                    reports_dir=temp_dir / "reports",
                    on_update=lambda e: update_queue.put(e),
                )

            except Exception as e:
                logger.exception("Failed to grade submissions", exc_info=e)
                update_queue.put(grader_ai.core.RunFinishedEvent(report_files=[]))

        thread = threading.Thread(target=worker)
        thread.start()

        status: list[list[str]] = []
        report_files: list[str] = []

        while True:
            event = update_queue.get()

            if isinstance(event, grader_ai.core.RunStartedEvent):
                status = [
                    [str(f.name), "Pending", "", ""] for f in event.submission_files
                ]

            elif isinstance(event, grader_ai.core.RunFinishedEvent):
                report_files = [str(p) for p in event.report_files]

                break

            elif isinstance(event, grader_ai.core.SubmissionStartedEvent):
                status[event.submission_idx][_STATUS_KEY_TO_IDX["Status"]] = "Grading"
                status[event.submission_idx][_STATUS_KEY_TO_IDX["# Problems"]] = str(
                    event.num_problems
                )

            elif isinstance(event, grader_ai.core.SubmissionFinishedEvent):
                if event.error is None:
                    status[event.submission_idx][_STATUS_KEY_TO_IDX["Status"]] = (
                        "Finished"
                    )
                else:
                    status[event.submission_idx][_STATUS_KEY_TO_IDX["Status"]] = (
                        f"Error ({event.error})"
                    )

            elif isinstance(event, grader_ai.core.ProblemStartedEvent):
                status[event.submission_idx][_STATUS_KEY_TO_IDX["# Graded"]] = str(
                    event.problem_idx
                )

            elif isinstance(event, grader_ai.core.ProblemFinishedEvent):
                status[event.submission_idx][_STATUS_KEY_TO_IDX["# Graded"]] = str(
                    event.problem_idx + 1
                )

            yield status, report_files, None

        report_archive = temp_dir / "reports.zip"

        with zipfile.ZipFile(report_archive, "w") as zipf:
            for report_file in report_files:
                zipf.write(report_file, arcname=Path(report_file).name)

        yield status, report_files, report_archive

        thread.join()

    reference_file = gr.File(
        file_count="single", file_types=[".tex", ".zip"], label="Reference"
    )
    submission_files = gr.File(
        file_count="multiple", file_types=[".zip"], label="Submissions"
    )
    model = gr.Dropdown(_list_models(), label="Model")
    num_parallel = gr.Slider(minimum=1, maximum=16, value=1, step=1, label="# Parallel")

    status = gr.Dataframe(headers=list(_STATUS_KEY_TO_IDX.keys()), label="Status")
    report_files = gr.File(file_count="multiple", label="Reports")
    report_archive = gr.DownloadButton(label="Download All Reports")

    app = gr.Interface(
        fn=fn,
        inputs=[reference_file, submission_files, model, num_parallel],
        outputs=[status, report_files, report_archive],
        flagging_mode="never",
    )

    return app


def _list_models() -> list[str]:
    client = openai.OpenAI()

    models = [m.id for m in client.models.list().data]

    return models
