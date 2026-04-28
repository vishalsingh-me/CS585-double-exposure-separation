import os
import unittest
from pathlib import Path
import csv
from PIL import Image

def test_pipeline():
    print("Running smoke test...")
    os.system("python scripts/generate_synthetic_pairs.py --input-csv data/processed/subsets/places365_small_splits/train.csv --image-root data/interim/subsets/places365_small --output-dir data/processed/synthetic --split train --num-pairs 2 --image-size 64 --mode linear_clean --save-masks")
    print("Checked output.")
    if os.path.exists("data/processed/synthetic/train/pair_manifest.csv"):
        print("Smoke test passed.")
    else:
        print("Smoke test failed.")

if __name__ == "__main__":
    test_pipeline()
