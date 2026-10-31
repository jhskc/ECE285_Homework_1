"""Build report figures from the selected run, without training or rescoring."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import make_transform
from src.model_mlp import UnconditionalMLPVAE

RUN = ROOT / "outputs/vae_256_mlp4_B64_Z128_lr0p001_run01"
OUT = ROOT / "final_figures/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})

history = pd.read_csv(RUN / "history.csv")
fig, axes = plt.subplots(1, 3, figsize=(8.4, 1.9))
for ax, key, title in zip(axes, ["loss", "mse", "kl"],
                          ["Total loss per image", "MSE per pixel", "KL per image"]):
    for split, color in [("train", "#2166ac"), ("val", "#c66018")]:
        ax.plot(history.epoch, history[f"{split}_{key}"], label=split, color=color, lw=1.2)
    selected = history.loc[history.epoch == 35].iloc[0]
    ax.plot(35, selected[f"val_{key}"], "o", color="#222222", ms=3)
    ax.axvline(35, color="#777777", ls=":", lw=0.7)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.15)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(pad=0.4, w_pad=1.3)
fig.savefig(OUT / "loss_curves.png", dpi=300)
plt.close(fig)

torch.set_num_threads(4)
checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)
assert checkpoint["epoch"] == 35
model = UnconditionalMLPVAE(latent_dim=128).eval()
model.load_state_dict(checkpoint["model"])
transform = make_transform(256)
originals = []
for label in ["NORMAL", "PNEUMONIA"]:
    path = sorted((ROOT / "data/chest_xray/val" / label).glob("*.jpeg"))[0]
    with Image.open(path) as image:
        originals.append(transform(image))
originals = torch.stack(originals)
with torch.no_grad():
    means, _ = model.encode(originals)
    reconstructed = model.decode(means).sigmoid()
fig, axes = plt.subplots(1, 4, figsize=(8, 2.3))
images = [originals[0, 0], reconstructed[0, 0], originals[1, 0], reconstructed[1, 0]]
titles = ["Normal: input", "Epoch 35: reconstruction", "Pneumonia: input", "Epoch 35: reconstruction"]
for ax, pixels, title in zip(axes, images, titles):
    ax.imshow(pixels, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
fig.tight_layout(pad=0.2, w_pad=0.5)
fig.savefig(OUT / "reconstructions_epoch35.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(8, 2.3))
for col, (epoch, code) in enumerate([(15, 0), (15, 1), (35, 0), (35, 1)]):
    with Image.open(RUN / f"samples_{epoch:02d}.png") as grid:
        assert grid.size == (1034, 1034)
        left = 2 + code * 258
        pixels = np.array(grid.crop((left, 2, left + 256, 258)))
    axes[col].imshow(pixels, cmap="gray", vmin=0, vmax=255)
    axes[col].set_title(f"Epoch {epoch}: latent {code + 1}", fontsize=9)
    axes[col].axis("off")
fig.tight_layout(pad=0.2, w_pad=0.5)
fig.savefig(OUT / "samples_epoch15_vs35.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("Report figures created from saved history and weights; no training or metric reruns.")
