"""Gradio UI entrypoint for grader-ai."""

from __future__ import annotations

import argparse
import logging
import queue
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import dotenv
import gradio as gr
import openai
import pandas as pd

import grader_ai.core

logger = logging.getLogger(__name__)


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
    reports_dir = temp_dir / "reports"

    def fn(
        reference_file: str, submission_files: list[str], model: str, num_parallel: int
    ) -> Iterator[tuple[pd.DataFrame, list[str]]]:
        update_queue: queue.Queue[grader_ai.core.AnyEvent] = queue.Queue()

        def worker() -> None:
            try:
                grader_ai.core.run(
                    reference_file=Path(reference_file),
                    submission_files=[Path(s) for s in submission_files],
                    model=model,
                    num_parallel=num_parallel,
                    reports_dir=reports_dir,
                    on_update=lambda e: update_queue.put(e),
                )

            except Exception as e:
                logger.exception("Failed to grade submissions", exc_info=e)
                update_queue.put(grader_ai.core.RunFinishedEvent(report_files=[]))

        thread = threading.Thread(target=worker)
        thread.start()

        status = pd.DataFrame()
        report_files: list[str] = []

        while True:
            event = update_queue.get()

            if isinstance(event, grader_ai.core.RunStartedEvent):
                status = pd.DataFrame(
                    [(s, "Pending") for s in event.submissions],
                    columns=["Submission", "Progress"],
                )

            elif isinstance(event, grader_ai.core.RunFinishedEvent):
                break

            elif isinstance(event, grader_ai.core.SubmissionStartedEvent):
                status.iloc[event.submission_idx] = (
                    event.submission,
                    f"Running: 0 / {event.num_problems}",
                )

            elif isinstance(event, grader_ai.core.SubmissionFinishedEvent):
                if event.error is not None:
                    status.iloc[event.submission_idx] = (
                        event.submission,
                        f"{event.error}",
                    )
                else:
                    status.iloc[event.submission_idx] = (
                        event.submission,
                        "Done",
                    )

            elif isinstance(event, grader_ai.core.ProblemStartedEvent):
                status.iloc[event.submission_idx] = (
                    event.submission,
                    f"{event.problem_idx} / {event.num_problems}",
                )

            elif isinstance(event, grader_ai.core.ProblemFinishedEvent):
                status.iloc[event.submission_idx] = (
                    event.submission,
                    f"{event.problem_idx + 1} / {event.num_problems}",
                )

            yield status, report_files

        yield status, report_files

        thread.join()

    reference_file = gr.File(
        file_count="single", file_types=[".tex"], label="Reference"
    )
    submission_files = gr.File(
        file_count="multiple", file_types=[".zip"], label="Submissions"
    )
    model = gr.Dropdown(_list_models(), label="Model")
    num_parallel = gr.Slider(minimum=1, maximum=16, value=1, step=1, label="# Parallel")

    status = gr.Dataframe(
        value=pd.DataFrame(columns=["Submission", "Progress"]), label="Status"
    )
    report_files = gr.File(file_count="multiple", label="Reports")

    app = gr.Interface(
        fn=fn,
        inputs=[reference_file, submission_files, model, num_parallel],
        outputs=[status, report_files],
    )

    return app


def _list_models() -> list[str]:
    client = openai.OpenAI()

    models = [m.id for m in client.models.list().data]

    return models
