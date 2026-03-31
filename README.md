# CS585 Double Exposure Separation

Repository: `cs585-double-exposure-separation`

## Overview

This repository contains a CS585 course project on single-image double-exposure separation. The goal is to recover two latent source images from one observed image that contains a blended double exposure.

The current project direction is to create synthetic double-exposure training data from clean natural images and use deep learning models to predict the two hidden image layers. This repository is intended to support collaborative development, documentation, experiments, and milestone tracking for the course project.

## Problem Statement

In a double-exposure image, two underlying scenes are combined into a single observed image. Recovering the original component images from only one blended observation is an ill-posed inverse problem because multiple pairs of source images can produce similar mixtures.

This project studies how to model that ambiguity in a practical learning framework. We focus on synthetic data generation, explicit image formation modeling, and reconstruction methods that can separate the two latent layers from a single mixed image.

## Project Goals

- Build a clear and reproducible pipeline for synthetic double-exposure data generation.
- Prepare a clean image source dataset suitable for mixture synthesis.
- Establish baseline and advanced models for recovering two latent component images.
- Document assumptions, design choices, and milestone progress in a way that is easy for teammates and instructors to review.
- Create a repository structure that can support future experiments, evaluation, and reporting.

## Planned Approach

The project is currently organized around three modeling directions:

1. **Dual-head encoder-decoder baseline**  
   Train a shared encoder with two output heads that predict the two hidden image layers directly from a single blended input.

2. **Unrolled inference model with explicit forward operator**  
   Incorporate the image formation process into the model so that reconstruction is guided by an explicit mixture operator and iterative refinement.

3. **Conditional diffusion approach for ambiguous layer recovery**  
   Explore a generative approach for cases where the layer separation problem is highly ambiguous and multiple plausible decompositions may exist.

These directions are planned research paths. Model implementation and comparison are not yet complete.

## Repository Structure

The repository now includes a data preparation scaffold for the current milestone.

```text
cs585-double-exposure-separation/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/               # Downloaded source images, not committed
│   ├── interim/           # Cleaning reports and intermediate manifests
│   └── processed/         # Split files, debug subsets, future synthetic outputs
├── docs/
│   └── data_pipeline.md   # Data workflow documentation
└── scripts/
    ├── data_utils.py
    ├── verify_dataset_structure.py
    ├── clean_image_dataset.py
    ├── create_data_splits.py
    ├── create_debug_subset.py
    └── generate_synthetic_pairs.py
```

## Dataset Direction

The current dataset direction is to use **Places365** as a source of clean natural scene images for synthetic double-exposure generation.

Important note:

- Places365 is **not** a native double-exposure dataset.
- It is being considered as a source of clean scene images from which synthetic training pairs can be created.
- The synthetic pipeline would combine two clean images into one blended observation and retain the original pair as supervision targets.

This approach is useful because it gives controlled access to source layers during training. However, it also means that the realism of the synthetic mixtures and the gap to real double-exposure images must be considered carefully.

## Milestone-Oriented Workflow

The initial project workflow is organized around the following milestones:

1. **Data sourcing**  
   Identify and review candidate sources of clean natural images, starting with Places365.

2. **Data cleaning**  
   Filter unsuitable samples, organize metadata, and define selection criteria for synthetic pair creation.

3. **Preprocessing**  
   Standardize image format, resolution, and any normalization or cropping steps required for training.

4. **Synthetic data generation**  
   Create blended double-exposure inputs from pairs of clean images and store corresponding supervision targets.

5. **Documentation**  
   Record assumptions, file organization, data decisions, and progress after each milestone.

## Setup

The repository currently includes the data preparation pipeline for this milestone.

1. Clone the repository.
2. Create and activate a Python environment.
3. Install the required package with `pip install -r requirements.txt`.
4. Download and extract Places365 under `data/raw/places365_standard/`.
5. Run the data pipeline scripts described in [docs/data_pipeline.md](docs/data_pipeline.md).

Model training and evaluation setup will be added later.

## Data Pipeline

The repository now includes command-line scripts for:

- verifying a downloaded image dataset
- cleaning and filtering invalid image files
- creating train, validation, and test split manifests
- creating small debug subsets
- validating the future interface for synthetic mixture generation

Detailed usage instructions are in [docs/data_pipeline.md](docs/data_pipeline.md).
For a full teammate handoff and replication guide, see [docs/project_replication_guide.md](docs/project_replication_guide.md).

## Team Responsibilities

The following section is a placeholder and can be updated once roles are finalized.

| Team Member | Role | Current Responsibility |
| --- | --- | --- |
| Vishal Singh | Data pipeline | Data sourcing, cleaning, preprocessing |
| Angelina Sun | Modeling | Baseline implementation and training |
| Jenny Yang| Evaluation | Metrics, qualitative analysis, reporting |

## Current Status

- Repository initialized.
- Project scope and problem framing documented.
- Dataset direction identified, with Places365 under consideration as a source of clean scene images.
- Data preparation scripts for verification, cleaning, split generation, and debug subsetting have been added.
- Synthetic mixture generation remains a planned next step and is represented only by a validation stub.
- Model training code has not been implemented yet.
- No experimental results are reported yet.

## Next Steps

- Download and verify the clean source image dataset under `data/raw/places365_standard/`.
- Run the cleaning pipeline and review rejected files.
- Generate reproducible train, validation, and test manifests from the cleaned image set.
- Define the synthetic mixture generation procedure and pair sampling policy.
- Add the first implementation of synthetic double-exposure generation.
- Expand documentation as the pipeline is exercised on the real dataset.
