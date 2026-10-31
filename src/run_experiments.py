import argparse
import time
from pathlib import Path

import torch

from .data import make_loaders
from .model_mlp import UnconditionalMLPVAE
from .train import fit


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train the final latent-128 MLP VAE for 40 epochs.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; existing results are never overwritten.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    loaders = make_loaders(root / "data/chest_xray", crop_size=256,
                           B=64, num_workers=args.num_workers)
    model = UnconditionalMLPVAE(latent_dim=128)
    print(f"Starting 40 epochs: UnconditionalMLPVAE, resolution=256, B=64, "
          f"latent_dim=128, workers={args.num_workers}, device={args.device}", flush=True)
    start = time.perf_counter()
    fit(model, loaders["train"], loaders["val"], args.output_dir,
        epochs=40, learning_rate=1e-3, seed=42, device=args.device)
    print(f"Finished {args.output_dir} in {time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
