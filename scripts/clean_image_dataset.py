from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_utils import (
    configure_logging,
    extract_path_metadata,
    iter_image_files,
    resolve_dataset_root,
    write_csv,
    write_json,
)


class ImageInspectionError(RuntimeError):
    """Raised when an image file cannot be decoded safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate image files and produce clean and rejected manifests."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/raw/places365_standard"),
        help="Root directory containing the downloaded source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/clean"),
        help="Directory where clean manifests should be written.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=64,
        help="Reject images narrower than this width.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=64,
        help="Reject images shorter than this height.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for processing only the first N images.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional logging information.",
    )
    return parser.parse_args()


def inspect_image(image_path: Path) -> tuple[int, int, str, str]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Pillow is required for image validation. Install dependencies with "
            "'pip install -r requirements.txt'."
        ) from exc

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image_format = image.format or ""
            mode = image.mode or ""
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageInspectionError(str(exc)) from exc

    return width, height, image_format, mode


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    input_root = resolve_dataset_root(args.input_root)
    if input_root != args.input_root:
        logging.warning("Resolved input root %s to %s", args.input_root, input_root)

    image_paths = list(iter_image_files(input_root))

    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        raise SystemExit(f"No image files were found under {input_root}.")

    valid_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []

    logging.info("Scanning %d image files under %s", len(image_paths), input_root)

    for image_path in image_paths:
        base_record = {
            **extract_path_metadata(image_path, input_root),
            "file_name": image_path.name,
            "extension": image_path.suffix.lower(),
            "file_size_bytes": image_path.stat().st_size,
        }

        try:
            width, height, image_format, mode = inspect_image(image_path)
        except ImageInspectionError as exc:
            rejected_rows.append(
                {
                    **base_record,
                    "width": "",
                    "height": "",
                    "format": "",
                    "mode": "",
                    "reject_reason": "unreadable",
                    "error_message": str(exc),
                }
            )
            continue

        if width < args.min_width or height < args.min_height:
            rejected_rows.append(
                {
                    **base_record,
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "mode": mode,
                    "reject_reason": "too_small",
                    "error_message": (
                        f"Image dimensions {width}x{height} are below the minimum "
                        f"{args.min_width}x{args.min_height}."
                    ),
                }
            )
            continue

        valid_rows.append(
            {
                **base_record,
                "width": width,
                "height": height,
                "format": image_format,
                "mode": mode,
            }
        )

    valid_path = args.output_dir / "valid_images.csv"
    rejected_path = args.output_dir / "rejected_images.csv"
    summary_path = args.output_dir / "clean_summary.json"

    valid_fields = [
        "relative_path",
        "top_level_dir",
        "source_split",
        "category",
        "category_path",
        "file_name",
        "extension",
        "file_size_bytes",
        "width",
        "height",
        "format",
        "mode",
    ]
    rejected_fields = valid_fields + ["reject_reason", "error_message"]

    write_csv(valid_path, valid_fields, valid_rows)
    write_csv(rejected_path, rejected_fields, rejected_rows)
    write_json(
        summary_path,
        {
            "input_root": str(input_root),
            "processed_images": len(image_paths),
            "valid_images": len(valid_rows),
            "rejected_images": len(rejected_rows),
            "min_width": args.min_width,
            "min_height": args.min_height,
        },
    )

    logging.info("Valid images: %d", len(valid_rows))
    logging.info("Rejected images: %d", len(rejected_rows))
    logging.info("Wrote valid manifest to %s", valid_path)
    logging.info("Wrote rejected manifest to %s", rejected_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
