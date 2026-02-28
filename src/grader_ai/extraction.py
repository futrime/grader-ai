import logging
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Submission:
    name: str
    content: str


def extract_reference(reference_file: Path) -> str:
    return reference_file.read_text(encoding="utf-8")


def extract_submissions(submissions_dir: Path) -> list[Submission]:
    submissions = []

    for file in submissions_dir.iterdir():
        try:
            with ZipFile(file) as archive:
                content = archive.read("main.tex").decode("utf-8")

                submissions.append(Submission(name=file.stem, content=content))

        except Exception as e:
            logger.exception("Failed to extract submission '%s': %s", file.name, e)

    return submissions
