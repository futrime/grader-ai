import zipfile
from pathlib import Path


def _find_main_tex_in_zip(zf: zipfile.ZipFile) -> str:
    """Find main.tex in zip, including inside subfolders."""
    for name in zf.namelist():
        if name.startswith("__MACOSX") or "/__MACOSX" in name:
            continue
        if name.endswith("/"):
            continue
        if Path(name).name == "main.tex":
            return name
    raise KeyError("main.tex not found in archive")


def extract_reference(file_path: Path) -> str:
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path) as zf:
            main_path = _find_main_tex_in_zip(zf)
            return zf.read(main_path).decode("utf-8")
    else:
        return file_path.read_text(encoding="utf-8")


def extract_submission(file_path: Path) -> str:
    if not zipfile.is_zipfile(file_path):
        raise ValueError(f"仅支持 .zip 格式，当前文件 {file_path.suffix} 不予批阅")
    with zipfile.ZipFile(file_path) as zf:
        main_path = _find_main_tex_in_zip(zf)
        return zf.read(main_path).decode("utf-8")
