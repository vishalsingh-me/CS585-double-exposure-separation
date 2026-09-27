from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from data_utils import (
    configure_logging,
    iter_image_files,
    resolve_dataset_root,
    top_level_name,
    write_json,
)

ARCHIVE_EXTENSIONS = {".tar", ".gz", ".tgz", ".zip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a downloaded image dataset directory and report image counts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/places365_standard"),
        help="Root directory that contains extracted dataset images.",
    )
    parser.add_argument(
        "--expected-top-level",
        nargs="*",
        default=[],
        help="Optional top-level directories that should exist under the dataset root.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for a JSON summary report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional logging information.",
    )
    return parser.parse_args()


def list_archive_files(root: Path) -> list[str]:
    archive_paths = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in ARCHIVE_EXTENSIONS:
            archive_paths.append(path.relative_to(root).as_posix())
    return archive_paths


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    root = resolve_dataset_root(args.root, required_children=args.expected_top_level)
    if root != args.root:
        logging.warning("Resolved dataset root %s to %s", args.root, root)

    image_paths = list(iter_image_files(root))

    if not image_paths:
        archive_files = list_archive_files(root)
        message = f"No image files were found under {root}."
        if archive_files:
            message += " Archive files are present, so the dataset may still need to be extracted."
        raise SystemExit(message)

    top_level_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()

    for image_path in image_paths:
        top_level_counts[top_level_name(image_path, root)] += 1
        extension_counts[image_path.suffix.lower()] += 1

    top_level_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
    missing_expected = sorted(
        expected for expected in args.expected_top_level if expected not in top_level_dirs
    )

    logging.info("Dataset root: %s", root)
    logging.info("Total images found: %d", len(image_paths))
    logging.info("Top-level directories found: %d", len(top_level_dirs))

    for name, count in top_level_counts.most_common(10):
        logging.info("Images under '%s': %d", name, count)

    for extension, count in sorted(extension_counts.items()):
        logging.info("Files with %s extension: %d", extension, count)

    if missing_expected:
        logging.error("Missing expected top-level directories: %s", ", ".join(missing_expected))
        raise SystemExit(1)

    if args.report_json is not None:
        report = {
            "dataset_root": str(root),
            "image_count": len(image_paths),
            "top_level_directory_count": len(top_level_dirs),
            "top_level_counts": dict(sorted(top_level_counts.items())),
            "extension_counts": dict(sorted(extension_counts.items())),
        }
        write_json(args.report_json, report)
        logging.info("Wrote summary report to %s", args.report_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
