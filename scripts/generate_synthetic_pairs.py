from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_utils import configure_logging, read_csv_rows, resolve_dataset_root, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate inputs for future synthetic mixture generation."
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("data/processed/splits/train.csv"),
        help="Split CSV that lists clean source images.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("data/raw/places365_standard"),
        help="Root directory that contains the clean source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/pairs/train"),
        help="Planned output directory for synthetic pair manifests and mixtures.",
    )
    parser.add_argument(
        "--pairs-per-image",
        type=int,
        default=1,
        help="Planned number of sampled partners per source image.",
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=None,
        help="Optional JSON file describing the planned synthetic pair interface.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional logging information.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    images_root = resolve_dataset_root(args.images_root)
    if images_root != args.images_root:
        logging.warning("Resolved images root %s to %s", args.images_root, images_root)

    split_rows = read_csv_rows(args.split_file)
    if not split_rows:
        raise SystemExit(f"No rows found in split file: {args.split_file}")

    missing_paths = []
    for row in split_rows:
        relative = row.get("relative_path")
        if not relative:
            raise SystemExit("Split file is missing the required 'relative_path' column.")
        source_path = images_root / relative
        if not source_path.exists():
            missing_paths.append(source_path.as_posix())
            if len(missing_paths) >= 10:
                break

    if missing_paths:
        raise SystemExit(
            "Some images referenced by the split file are missing under the image root. "
            f"Examples: {missing_paths}"
        )

    plan = {
        "status": "stub",
        "message": (
            "Synthetic pair generation is not implemented yet. "
            "This script currently validates the cleaned split interface for the next milestone."
        ),
        "split_file": str(args.split_file),
        "images_root": str(images_root),
        "output_dir": str(args.output_dir),
        "input_image_count": len(split_rows),
        "pairs_per_image": args.pairs_per_image,
        "expected_future_outputs": [
            "pair_manifest.csv",
            "mixture_images/",
            "targets/source_a/",
            "targets/source_b/",
        ],
        "expected_future_manifest_columns": [
            "pair_id",
            "split",
            "source_image_a",
            "source_image_b",
            "alpha",
            "mixture_path",
            "target_a_path",
            "target_b_path",
        ],
    }

    logging.info("Validated %d source images from %s", len(split_rows), args.split_file)
    logging.info("Synthetic pair generation remains a planned step.")

    if args.plan_json is not None:
        write_json(args.plan_json, plan)
        logging.info("Wrote synthetic generation plan to %s", args.plan_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
