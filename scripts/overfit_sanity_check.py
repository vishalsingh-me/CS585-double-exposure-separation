import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Overfit Sanity Check Wrapper")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=585)
    args, unknown = parser.parse_known_args()
    
    cmd = [
        "python3", "src/train_baseline.py",
        "--train-manifest", args.manifest,
        "--val-manifest", args.manifest,
        "--output-dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--base-channels", str(args.base_channels),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--max-train-samples", str(args.max_samples),
        "--max-val-samples", str(args.max_samples)
    ] + unknown
    
    print(f"Running Overfit Sanity Check with {args.max_samples} samples for {args.epochs} epochs...")
    print("Command:", " ".join(cmd))
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("Overfit test failed during execution.")
    else:
        # Check logs
        log_file = os.path.join(args.output_dir, "train_log.csv")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                lines = f.readlines()
                last_line = lines[-1].strip().split(',')
                try:
                    final_train_loss = float(last_line[1])
                    if final_train_loss > 0.05:
                        print(f"WARNING: Overfit sanity test finished but final loss is somewhat high ({final_train_loss:.4f}). Expected it to memorize well.")
                    else:
                        print(f"SUCCESS: Overfit sanity test finished. Final loss: {final_train_loss:.4f}.")
                except:
                    pass
        print(f"Check results in {args.output_dir}")

if __name__ == "__main__":
    main()