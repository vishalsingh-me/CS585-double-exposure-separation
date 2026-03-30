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

The repository is currently in an initialization stage. The following structure reflects the intended organization, including planned folders that may be added as the project develops.

```text
cs585-double-exposure-separation/
├── README.md
├── data/                  # Planned: dataset notes, metadata, preprocessing outputs
├── docs/                  # Planned: project notes, reports, references
├── notebooks/             # Planned: exploratory analysis and prototyping
├── scripts/               # Planned: data preparation and synthetic generation scripts
├── src/                   # Planned: model, training, and evaluation code
├── experiments/           # Planned: experiment configs and logs
└── results/               # Planned: qualitative outputs and evaluation summaries
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

Implementation is still being organized, so the setup process is not finalized yet.

Planned setup steps:

1. Clone the repository.
2. Create a project environment once dependencies are finalized.
3. Add dataset access instructions after the data pipeline is defined.
4. Add training and evaluation instructions after the first baseline is implemented.

Detailed commands will be added once the codebase structure and dependencies are stable.

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
- Data preparation, synthetic generation code, and model implementations are still in the planning stage.
- No experimental results are reported yet.

## Next Steps

- Confirm dataset access, usage constraints, and final image selection criteria.
- Define the synthetic mixture generation procedure and file format for training pairs.
- Add initial repository folders for data, scripts, and source code.
- Implement the first preprocessing utilities.
- Build a dual-head encoder-decoder baseline as the initial reference model.
- Expand the documentation as milestone work begins.
