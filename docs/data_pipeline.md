# Data Pipeline

This is the main operational guide for the current repository state. It covers environment setup, where to place the dataset, the tested command sequence, the main outputs, and what remains before modeling.

## Current Scope

Implemented:

- dataset root verification
- small subset creation from Places365
- image cleaning and validation
- project train, val, and test split creation
- synthetic pair interface validation stub

Not implemented yet:

- actual synthetic pair generation
- model training
- model evaluation

## Prerequisites

Run all commands from the repository root.

Use Python 3 and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Raw data is downloaded manually and should not be committed.

## Dataset Location

Place Places365 under:

```text
data/raw/places365_standard/
├── train/
│   └── <category folders>
└── val/
    └── <category folders>
```

Important notes:

- Places365 is being used only as a clean-image source
- it is not a native double-exposure dataset
- the scripts default to `data/raw/places365_standard`
- if archive extraction creates a nested path such as `data/raw/data/raw/places365_standard`, the scripts try to resolve it automatically

## Tested Workflow

The following commands were tested successfully on a small subset:

- 4 categories
- 4 images per category per source split
- source splits `train` and `val`
- seed `585`

### 1. Create the subset

```bash
python3 scripts/create_debug_subset.py \
  --input-root data/raw/places365_standard \
  --output-root data/interim/subsets/places365_small \
  --max-categories 4 \
  --max-images-per-category 4 \
  --include-splits train val \
  --seed 585 \
  --overwrite
```

Outputs:

- `data/interim/subsets/places365_small/train/`
- `data/interim/subsets/places365_small/val/`
- `data/interim/subsets/places365_small/subset_manifest.csv`
- `data/interim/subsets/places365_small/subset_summary.json`

### 2. Verify the subset

```bash
python3 scripts/verify_dataset_structure.py \
  --root data/interim/subsets/places365_small \
  --expected-top-level train val \
  --report-json data/interim/subsets/places365_small_reports/verification.json
```

Outputs:

- `data/interim/subsets/places365_small_reports/verification.json`

### 3. Clean and validate images

```bash
python3 scripts/clean_image_dataset.py \
  --input-root data/interim/subsets/places365_small \
  --output-dir data/interim/subsets/places365_small_clean \
  --min-width 128 \
  --min-height 128
```

Outputs:

- `data/interim/subsets/places365_small_clean/valid_images.csv`
- `data/interim/subsets/places365_small_clean/rejected_images.csv`
- `data/interim/subsets/places365_small_clean/clean_summary.json`

### 4. Create project splits

```bash
python3 scripts/create_data_splits.py \
  --manifest data/interim/subsets/places365_small_clean/valid_images.csv \
  --output-dir data/processed/subsets/places365_small_splits \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 585
```

Outputs:

- `data/processed/subsets/places365_small_splits/train.csv`
- `data/processed/subsets/places365_small_splits/val.csv`
- `data/processed/subsets/places365_small_splits/test.csv`
- `data/processed/subsets/places365_small_splits/all_images_with_splits.csv`
- `data/processed/subsets/places365_small_splits/split_summary.json`

### 5. Validate the synthetic generation interface

```bash
python3 scripts/generate_synthetic_pairs.py \
  --split-file data/processed/subsets/places365_small_splits/train.csv \
  --images-root data/interim/subsets/places365_small \
  --plan-json data/processed/subsets/places365_small_pairs/train_plan.json
```

Outputs:

- `data/processed/subsets/places365_small_pairs/train_plan.json`

## Main Outputs

### Clean subset files

- `subset_manifest.csv`: records the selected subset files
- `subset_summary.json`: records chosen categories, split counts, seed, and resolved input root

### Verification and cleaning files

- `verification.json`: image counts by top-level directory and extension
- `valid_images.csv`: clean manifest for downstream use
- `rejected_images.csv`: rejected files with reasons
- `clean_summary.json`: counts of processed, valid, and rejected images

### Project split files

- `train.csv`, `val.csv`, `test.csv`: project-level split manifests
- `all_images_with_splits.csv`: combined manifest with a `split` column
- `split_summary.json`: split counts, seed, and source split counts

### Synthetic stub file

- `train_plan.json`: confirms that the train split and image root can be consumed by the future synthetic generation stage

Important manifest columns currently preserved:

- `relative_path`
- `top_level_dir`
- `source_split`
- `category`
- `category_path`
- `file_name`
- `extension`
- `file_size_bytes`
- `width`
- `height`
- `format`
- `mode`

## Expected Result from the Tested Run

The tested small subset run produced:

- 32 total images
- 32 valid images
- 0 rejected images
- project splits of 25 train, 3 val, and 4 test images

If your numbers are very different, first check the dataset path, subset parameters, and whether your local extraction layout differs.

## What Remains Before Modeling

The next stage still needs:

- real synthetic pair generation
- a blending operator and output format
- naming conventions for mixtures and targets
- model data-loader expectations

The current modeling inputs should come from:

- `data/interim/subsets/places365_small/`
- `data/processed/subsets/places365_small_splits/train.csv`
- `data/processed/subsets/places365_small_splits/val.csv`
- `data/processed/subsets/places365_small_splits/test.csv`

Synthetic generation is still a stub, so training cannot start until that stage is implemented.
