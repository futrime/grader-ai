"""Extraction of .tex content from files, archives, and directories."""

import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Submission:
    """A single submission with a human-readable name and its .tex content."""

    name: str
    content: str


def extract_reference(path: Path) -> str:
    """Extract reference .tex content from a file or archive.

    Args:
        path: A ``.tex`` file or a ``.zip`` archive containing a ``.tex`` file.

    Returns:
        The decoded .tex content.

    Raises:
        ValueError: If the path is neither a .tex file nor a .zip archive.
        FileNotFoundError: If the path does not exist.
        RuntimeError: If no .tex file is found inside a .zip archive.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".tex":
        return path.read_text(encoding="utf-8")

    if suffix == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            content = _pick_tex_from_zip(archive)
            if content is None:
                raise RuntimeError(f"No .tex file found in reference archive: {path}")
            return content

    raise ValueError(f"Unsupported reference file type: {suffix}")


def extract_submissions(path: Path) -> list[Submission]:
    """Extract one or more submissions from a path.

    Supported input layouts:

    * A single ``.tex`` file.
    * A ``.zip`` archive containing a ``.tex`` file (single submission).
    * A ``.zip`` archive containing nested ``.zip`` archives and/or loose
      ``.tex`` files (multi-submission).
    * A directory containing ``.zip`` and/or ``.tex`` files.

    For each ``.zip`` that represents a single submission, the best ``.tex``
    file is chosen by preferring shallower paths and ``main.tex`` over other
    names.

    Args:
        path: A file or directory to extract submissions from.

    Returns:
        A list of :class:`Submission` objects, one per detected submission.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path type is not supported.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        return _extract_from_directory(path)

    suffix = path.suffix.lower()

    if suffix == ".tex":
        return [Submission(name=path.stem, content=path.read_text(encoding="utf-8"))]

    if suffix == ".zip":
        return _extract_from_zip_path(path)

    raise ValueError(f"Unsupported submission file type: {suffix}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _pick_tex_from_zip(archive: zipfile.ZipFile) -> str | None:
    """Pick the best .tex file from *archive* and return its content.

    Selection criteria (in order of priority):
    1. Prefer ``main.tex`` over other names.
    2. Prefer shallower directory depth.
    3. Break ties lexicographically by full path.

    Returns:
        The decoded UTF-8 content of the best .tex file, or ``None`` if no
        .tex file exists in the archive.
    """
    best_name: str | None = None
    best_key: tuple[bool, int, str] | None = None

    for name in archive.namelist():
        # Skip directories and non-.tex entries.
        if name.endswith("/") or not name.lower().endswith(".tex"):
            continue

        posix_path = PurePosixPath(name)
        candidate_key = (
            posix_path.name.lower() != "main.tex",  # False (0) for main.tex
            len(posix_path.parts),  # shallower first
            name,  # lexicographic tiebreaker
        )

        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_name = name

    if best_name is None:
        return None

    return archive.read(best_name).decode("utf-8")


def _extract_from_zip_path(path: Path) -> list[Submission]:
    """Extract submissions from a .zip file on disk.

    If the archive contains any nested ``.zip`` entries, it is treated as a
    multi-submission archive: each nested ``.zip`` and each loose ``.tex``
    file produces a separate :class:`Submission`.

    Otherwise the archive is treated as a single submission and the best
    ``.tex`` file is selected.
    """
    with zipfile.ZipFile(path, "r") as archive:
        nested_zips: list[str] = []
        loose_tex: list[str] = []

        for name in archive.namelist():
            if name.endswith("/"):
                continue
            lower = name.lower()
            if lower.endswith(".zip"):
                nested_zips.append(name)
            elif lower.endswith(".tex"):
                loose_tex.append(name)

        # Multi-submission mode: nested .zip files exist.
        if nested_zips:
            return _multi_submissions_from_zip(archive, nested_zips, loose_tex)

        # Single-submission mode.
        content = _pick_tex_from_zip(archive)
        if content is None:
            logger.warning("No .tex file found in submission archive: %s", path)
            return []
        return [Submission(name=path.stem, content=content)]


def _multi_submissions_from_zip(
    archive: zipfile.ZipFile,
    nested_zips: list[str],
    loose_tex: list[str],
) -> list[Submission]:
    """Build a list of submissions from a multi-submission .zip archive."""
    submissions: list[Submission] = []

    # Each nested .zip → one submission.
    for zip_name in nested_zips:
        stem = PurePosixPath(zip_name).stem
        data = archive.read(zip_name)
        try:
            with zipfile.ZipFile(io.BytesIO(data), "r") as inner:
                content = _pick_tex_from_zip(inner)
        except zipfile.BadZipFile:
            logger.warning("Skipping invalid nested zip: %s", zip_name)
            continue

        if content is None:
            logger.warning("No .tex found in nested zip: %s", zip_name)
            continue

        submissions.append(Submission(name=stem, content=content))

    # Each loose .tex → one submission.
    for tex_name in loose_tex:
        stem = PurePosixPath(tex_name).stem
        content = archive.read(tex_name).decode("utf-8")
        submissions.append(Submission(name=stem, content=content))

    return submissions


def _extract_from_directory(directory: Path) -> list[Submission]:
    """Extract submissions from all .tex and .zip files in *directory*.

    Only immediate children of *directory* are considered (non-recursive).
    """
    submissions: list[Submission] = []

    for child in sorted(directory.iterdir()):
        if child.is_file():
            suffix = child.suffix.lower()
            if suffix == ".tex":
                content = child.read_text(encoding="utf-8")
                submissions.append(Submission(name=child.stem, content=content))
            elif suffix == ".zip":
                submissions.extend(_extract_from_zip_path(child))

    return submissions
