import argparse
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.saved_synthetic_dataset import SavedSyntheticDataset, custom_collate_saved
from src.losses import get_reconstruction_loss, output_correlation_loss, permutation_invariant_l1_loss
from src.model_factory import build_model
from src.visualization_utils import match_predictions, save_separation_grid


def main():
    parser = argparse.ArgumentParser(description="Smoke test separator models.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=["dual_head_unet", "two_decoder_unet", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SavedSyntheticDataset(args.manifest, image_size=args.image_size)
    dataset.rows = dataset.rows[: args.batch_size]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_saved)
    batch = next(iter(loader))

    mix = batch["mixture"].to(device)
    s1 = batch["source_1"].to(device)
    s2 = batch["source_2"].to(device)

    model_names = ["dual_head_unet", "two_decoder_unet"] if args.model == "both" else [args.model]
    for model_name in model_names:
        model = build_model(model_name, base_channels=args.base_channels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        optimizer.zero_grad()
        p1, p2 = model(mix)
        pit_loss, _, _, is_a = permutation_invariant_l1_loss(p1, p2, s1, s2)
        alphas = []
        for item in batch["metadata"]:
            try:
                alphas.append(float(item.get("alpha", -1.0)))
            except (TypeError, ValueError):
                alphas.append(-1.0)
        recon_loss = get_reconstruction_loss(p1, p2, mix, torch.tensor(alphas, dtype=torch.float32, device=device), is_a)
        corr_loss = output_correlation_loss(p1, p2)
        total_loss = pit_loss + 0.1 * recon_loss + 0.01 * corr_loss
        total_loss.backward()
        optimizer.step()

        p1m, p2m = match_predictions(p1.detach(), p2.detach(), s1, s2, is_a)
        visual_path = args.output_dir / f"{model_name}_smoke_visualization.png"
        save_separation_grid(visual_path, s1.cpu(), s2.cpu(), mix.cpu(), p1m.cpu(), p2m.cpu(), max_rows=args.batch_size)

        print(f"model: {model_name}")
        print(f"device: {device}")
        print(f"pred_1 shape: {tuple(p1.shape)}")
        print(f"pred_2 shape: {tuple(p2.shape)}")
        print(f"pit_loss: {pit_loss.item():.6f}")
        print(f"recon_loss: {recon_loss.item():.6f}")
        print(f"corr_loss: {corr_loss.item():.6f}")
        print(f"total_loss: {total_loss.item():.6f}")
        print(f"saved visualization: {visual_path}")


if __name__ == "__main__":
    main()
