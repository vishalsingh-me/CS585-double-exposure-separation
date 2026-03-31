from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def ensure_directory(path: Path, description: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{description} does not exist: {path}")
    if not path.is_dir():
        raise SystemExit(f"{description} is not a directory: {path}")
    return path


def ensure_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{description} does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"{description} is not a file: {path}")
    return path


def make_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_dataset_root(path: Path, required_children: list[str] | None = None) -> Path:
    required_children = required_children or []

    if path.exists():
        resolved = ensure_directory(path, "Dataset root")
    else:
        search_parent = path.parent
        if not search_parent.exists():
            raise SystemExit(f"Dataset root does not exist: {path}")

        candidates = []
        for candidate in sorted(search_parent.rglob(path.name)):
            if not candidate.is_dir():
                continue
            if required_children and not all((candidate / child).is_dir() for child in required_children):
                continue
            candidates.append(candidate)

        if not candidates:
            raise SystemExit(f"Dataset root does not exist: {path}")

        if len(candidates) > 1:
            rendered = ", ".join(candidate.as_posix() for candidate in candidates[:5])
            raise SystemExit(
                f"Dataset root does not exist at {path}, and multiple matching directories "
                f"were found under {search_parent}: {rendered}"
            )

        resolved = candidates[0]

    missing_children = [child for child in required_children if not (resolved / child).is_dir()]
    if missing_children:
        raise SystemExit(
            f"Dataset root {resolved} is missing required subdirectories: "
            + ", ".join(missing_children)
        )

    return resolved


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def top_level_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else "."


def extract_path_metadata(path: Path, root: Path) -> dict[str, str]:
    relative = path.relative_to(root)
    parts = relative.parts

    if len(parts) >= 3:
        source_split = parts[0]
        category_parts = parts[1:-1]
    elif len(parts) >= 2:
        source_split = ""
        category_parts = parts[:-1]
    else:
        source_split = ""
        category_parts = ()

    category_path = "/".join(category_parts)
    category = category_parts[-1] if category_parts else ""

    return {
        "relative_path": relative.as_posix(),
        "top_level_dir": parts[0] if len(parts) > 1 else ".",
        "source_split": source_split,
        "category": category,
        "category_path": category_path,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    make_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    ensure_file(path, "CSV file")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_json(path: Path, payload: dict[str, object]) -> None:
    make_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def materialize_file(source: Path, target: Path, mode: str) -> None:
    make_directory(target.parent)
    if target.exists() or target.is_symlink():
        target.unlink()

    if mode == "copy":
        shutil.copy2(source, target)
        return

    if mode == "symlink":
        target.symlink_to(source.resolve())
        return

    raise ValueError(f"Unsupported materialization mode: {mode}")
