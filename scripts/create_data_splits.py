from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from data_utils import configure_logging, read_csv_rows, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible train, validation, and test split manifests."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/interim/clean/valid_images.csv"),
        help="CSV manifest produced by clean_image_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/splits"),
        help="Directory where split CSV files should be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument(
        "--seed",
        type=int,
        default=585,
        help="Random seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional logging information.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(
            f"Split ratios must sum to 1.0, but received {train_ratio} + {val_ratio} + {test_ratio} = {total}."
        )


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    rows = read_csv_rows(args.manifest)
    if not rows:
        raise SystemExit(f"No rows found in manifest: {args.manifest}")

    required_columns = {"relative_path"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise SystemExit(
            "Manifest is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    rows = sorted(rows, key=lambda row: row["relative_path"])
    random.Random(args.seed).shuffle(rows)

    total_count = len(rows)
    train_count = int(total_count * args.train_ratio)
    val_count = int(total_count * args.val_ratio)
    test_count = total_count - train_count - val_count
    source_split_counts: dict[str, int] = {}

    if "source_split" in rows[0]:
        source_split_counts = {}
        for row in rows:
            source_split = row.get("source_split", "")
            if not source_split:
                continue
            source_split_counts[source_split] = source_split_counts.get(source_split, 0) + 1

    split_map = {
        "train": rows[:train_count],
        "val": rows[train_count : train_count + val_count],
        "test": rows[train_count + val_count :],
    }

    for split_name, split_rows in split_map.items():
        if not split_rows:
            logging.warning(
                "Project split '%s' is empty. Increase the subset size or adjust the split ratios.",
                split_name,
            )

    all_rows: list[dict[str, str]] = []
    for split_name, split_rows in split_map.items():
        split_with_labels = []
        for row in split_rows:
            labeled_row = dict(row)
            labeled_row["split"] = split_name
            split_with_labels.append(labeled_row)
            all_rows.append(labeled_row)

        fieldnames = list(split_with_labels[0].keys()) if split_with_labels else list(rows[0].keys()) + ["split"]
        write_csv(args.output_dir / f"{split_name}.csv", fieldnames, split_with_labels)
        logging.info("Wrote %s split with %d rows", split_name, len(split_with_labels))

    write_csv(args.output_dir / "all_images_with_splits.csv", list(all_rows[0].keys()), all_rows)
    write_json(
        args.output_dir / "split_summary.json",
        {
            "manifest": str(args.manifest),
            "seed": args.seed,
            "counts": {
                "train": train_count,
                "val": val_count,
                "test": test_count,
                "total": total_count,
            },
            "category_count": len({row["category"] for row in rows if row.get("category")}),
            "ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            "source_split_counts": source_split_counts,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
