import zipfile
from pathlib import Path


def extract_reference(file_path: Path) -> str:
    if zipfile.is_zipfile(file_path):
        return zipfile.ZipFile(file_path).read("main.tex").decode("utf-8")

    else:
        return file_path.read_text(encoding="utf-8")


def extract_submission(file_path: Path) -> str:
    return zipfile.ZipFile(file_path).read("main.tex").decode("utf-8")
