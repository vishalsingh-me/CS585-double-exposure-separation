# Project Replication Handoff

Use this file as a short teammate checklist. The main operational guide is [docs/data_pipeline.md](docs/data_pipeline.md).

## What Is Already Done

- project scaffold is in place
- Places365 subset workflow works
- verification, cleaning, and split scripts work
- metadata is preserved in manifests
- synthetic pair generation is still only a stub

## What Was Tested

The tested workflow used:

- 4 categories
- 4 images per category per split
- source splits `train` and `val`
- seed `585`

Tested result:

- 32 total images
- 32 valid images
- 0 rejected images
- 25 train, 3 val, and 4 test project splits

## What A Teammate Should Do

1. Run all commands from the repository root.
2. Create the Python environment and install `requirements.txt`.
3. Download Places365 manually and place it under `data/raw/places365_standard/`.
4. Follow the exact command sequence in [docs/data_pipeline.md](docs/data_pipeline.md).
5. Inspect:
   - `data/interim/subsets/places365_small_clean/valid_images.csv`
   - `data/processed/subsets/places365_small_splits/train.csv`
   - `data/processed/subsets/places365_small_splits/split_summary.json`
6. Use the split CSV files as the starting point for the next modeling-stage data interface.

## Handoff for Modeling

Before modeling starts, the repository still needs:

- actual synthetic pair generation
- mixture and target file conventions
- model data-loader expectations

The main inputs prepared so far are:

- `data/interim/subsets/places365_small/`
- `data/processed/subsets/places365_small_splits/train.csv`
- `data/processed/subsets/places365_small_splits/val.csv`
- `data/processed/subsets/places365_small_splits/test.csv`

If you need the exact commands, expected outputs, and dataset placement rules, use [docs/data_pipeline.md](docs/data_pipeline.md).
