from __future__ import annotations

import argparse
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

from data_utils import (
    configure_logging,
    ensure_directory,
    iter_image_files,
    materialize_file,
    read_csv_rows,
    resolve_dataset_root,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a small debug subset from a raw Places365-style dataset tree, "
            "or sample from existing split CSV files."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Optional raw dataset root, such as data/raw/places365_standard.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/interim/subsets/places365_small"),
        help="Output root for a raw-dataset subset with train and val category folders.",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=6,
        help="Maximum number of categories to sample in input-root mode.",
    )
    parser.add_argument(
        "--max-images-per-category",
        type=int,
        default=8,
        help="Maximum number of images per category and split in input-root mode.",
    )
    parser.add_argument(
        "--include-splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to include in input-root mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output-root before creating a new raw-dataset subset.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/processed/splits"),
        help="Directory containing train.csv, val.csv, and test.csv for split-based sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/subsets/debug"),
        help="Directory where split-based debug subset CSV files should be written.",
    )
    parser.add_argument("--train-count", type=int, default=100, help="Requested train subset size.")
    parser.add_argument("--val-count", type=int, default=20, help="Requested validation subset size.")
    parser.add_argument("--test-count", type=int, default=20, help="Requested test subset size.")
    parser.add_argument(
        "--seed",
        type=int,
        default=585,
        help="Random seed for deterministic subset selection.",
    )
    parser.add_argument(
        "--materialize-mode",
        choices=["none", "copy", "symlink"],
        default="none",
        help="Materialization mode. Raw subset mode defaults to copy when omitted.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Root directory of source images for split-based materialization.",
    )
    parser.add_argument(
        "--materialize-root",
        type=Path,
        default=Path("data/interim/debug_subset"),
        help="Root directory for copied or symlinked split-based debug subset images.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional logging information.",
    )
    return parser.parse_args()


def sample_rows(rows: list[dict[str, object]], sample_size: int, rng: random.Random) -> list[dict[str, object]]:
    rows_copy = list(rows)
    rng.shuffle(rows_copy)
    return rows_copy[: min(sample_size, len(rows_copy))]


def list_categories(dataset_root: Path, split_name: str) -> list[str]:
    split_root = ensure_directory(dataset_root / split_name, f"Split directory '{split_name}'")
    return sorted(path.name for path in split_root.iterdir() if path.is_dir())


def choose_categories(
    dataset_root: Path,
    include_splits: list[str],
    max_categories: int,
    rng: random.Random,
) -> list[str]:
    if max_categories <= 0:
        raise SystemExit("--max-categories must be greater than zero.")

    category_sets = [set(list_categories(dataset_root, split_name)) for split_name in include_splits]
    common_categories = sorted(set.intersection(*category_sets)) if category_sets else []

    if not common_categories:
        raise SystemExit(
            "No shared category directories were found across the requested splits: "
            + ", ".join(include_splits)
        )

    if max_categories >= len(common_categories):
        return common_categories

    return sorted(rng.sample(common_categories, max_categories))


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not output_root.is_dir():
            raise SystemExit(f"Output root is not a directory: {output_root}")
        if overwrite:
            shutil.rmtree(output_root)
        elif any(output_root.iterdir()):
            raise SystemExit(
                f"Output root already exists and is not empty: {output_root}. "
                "Use --overwrite or choose a different output path."
            )

    output_root.mkdir(parents=True, exist_ok=True)


def create_subset_from_raw(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    dataset_root = resolve_dataset_root(args.input_root, required_children=args.include_splits)
    if dataset_root != args.input_root:
        logging.warning("Resolved input root %s to %s", args.input_root, dataset_root)

    prepare_output_root(args.output_root, args.overwrite)
    selected_categories = choose_categories(
        dataset_root,
        args.include_splits,
        args.max_categories,
        rng,
    )

    effective_mode = args.materialize_mode if args.materialize_mode != "none" else "copy"
    rows: list[dict[str, object]] = []
    counts_by_split: dict[str, dict[str, int]] = defaultdict(dict)

    logging.info(
        "Creating subset from %s using %d categories and up to %d images per category.",
        dataset_root,
        len(selected_categories),
        args.max_images_per_category,
    )

    for split_name in args.include_splits:
        for category in selected_categories:
            category_root = ensure_directory(
                dataset_root / split_name / category,
                f"Category directory for {split_name}/{category}",
            )
            image_paths = list(iter_image_files(category_root))
            if not image_paths:
                logging.warning("No images found under %s", category_root)
                counts_by_split[split_name][category] = 0
                continue

            sampled_rows = sample_rows(
                [{"path": image_path} for image_path in image_paths],
                args.max_images_per_category,
                rng,
            )
            sampled_paths = sorted(row["path"] for row in sampled_rows)
            counts_by_split[split_name][category] = len(sampled_paths)

            for source_path in sampled_paths:
                relative = source_path.relative_to(dataset_root)
                target_path = args.output_root / relative
                materialize_file(source_path, target_path, effective_mode)
                rows.append(
                    {
                        "relative_path": relative.as_posix(),
                        "source_relative_path": relative.as_posix(),
                        "source_split": split_name,
                        "category": category,
                        "category_path": category,
                        "file_name": source_path.name,
                        "materialize_mode": effective_mode,
                        "seed": args.seed,
                    }
                )

    if not rows:
        raise SystemExit("The requested subset configuration produced zero images.")

    manifest_fields = [
        "relative_path",
        "source_relative_path",
        "source_split",
        "category",
        "category_path",
        "file_name",
        "materialize_mode",
        "seed",
    ]
    write_csv(args.output_root / "subset_manifest.csv", manifest_fields, rows)
    write_json(
        args.output_root / "subset_summary.json",
        {
            "requested_input_root": str(args.input_root),
            "resolved_input_root": str(dataset_root),
            "output_root": str(args.output_root),
            "include_splits": args.include_splits,
            "selected_categories": selected_categories,
            "max_categories": args.max_categories,
            "max_images_per_category": args.max_images_per_category,
            "materialize_mode": effective_mode,
            "seed": args.seed,
            "selected_image_count": len(rows),
            "counts_by_split": counts_by_split,
        },
    )

    logging.info("Created subset root at %s", args.output_root)
    logging.info("Selected %d total images", len(rows))
    return 0


def create_subset_from_split_manifests(args: argparse.Namespace) -> int:
    ensure_directory(args.splits_dir, "Splits directory")

    if args.materialize_mode != "none" and args.images_root is None:
        raise SystemExit("--images-root is required when materialize-mode is copy or symlink.")

    if args.images_root is not None:
        images_root = resolve_dataset_root(args.images_root)
        if images_root != args.images_root:
            logging.warning("Resolved images root %s to %s", args.images_root, images_root)
    else:
        images_root = None

    rng = random.Random(args.seed)
    requested_counts = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }
    summary_counts: dict[str, int] = {}

    for split_name, requested_count in requested_counts.items():
        split_rows = read_csv_rows(args.splits_dir / f"{split_name}.csv")
        subset_rows = sample_rows(split_rows, requested_count, rng)
        summary_counts[split_name] = len(subset_rows)

        if split_rows and not subset_rows and requested_count > 0:
            logging.warning("No rows were selected for split '%s'.", split_name)

        fieldnames = list(split_rows[0].keys()) if split_rows else ["relative_path", "split"]
        write_csv(args.output_dir / f"{split_name}.csv", fieldnames, subset_rows)
        logging.info(
            "Wrote debug subset for %s with %d rows (requested %d).",
            split_name,
            len(subset_rows),
            requested_count,
        )

        if args.materialize_mode == "none":
            continue

        for row in subset_rows:
            source_path = images_root / row["relative_path"]
            if not source_path.exists():
                raise SystemExit(
                    f"Source image referenced by split file does not exist: {source_path}"
                )

            target_path = args.materialize_root / split_name / row["relative_path"]
            materialize_file(source_path, target_path, args.materialize_mode)

    write_json(
        args.output_dir / "debug_subset_summary.json",
        {
            "splits_dir": str(args.splits_dir),
            "seed": args.seed,
            "requested_counts": requested_counts,
            "selected_counts": summary_counts,
            "materialize_mode": args.materialize_mode,
            "materialize_root": str(args.materialize_root),
        },
    )

    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    if args.input_root is not None:
        return create_subset_from_raw(args)

    return create_subset_from_split_manifests(args)


if __name__ == "__main__":
    raise SystemExit(main())
