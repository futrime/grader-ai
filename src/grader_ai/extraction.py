import logging
from pathlib import Path
from zipfile import ZipFile

logger = logging.getLogger(__name__)


def extract_reference(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def extract_submission(file_path: Path) -> str:
    return ZipFile(file_path).read("main.tex").decode("utf-8")
