import logging
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Submission:
    name: str
    content: str


def extract_reference(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def extract_submission(file_path: Path) -> Submission:
    return Submission(
        name=file_path.stem, content=ZipFile(file_path).read("main.tex").decode("utf-8")
    )
