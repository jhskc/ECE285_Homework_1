import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F
from torchvision.utils import save_image


def vae_loss(logits, x, means, logvar):
    """
    Input:  logits and x, shape (B, 1, H, W); x has values in [0, 1].
            means and logvar, shape (B, latent_dim).
    Output: total loss, reconstruction loss, and KL loss; each is scalar.

    Negative ELBO up to a constant, using fixed pixel variance 0.5.
    """
    reconstruction = F.mse_loss(logits.sigmoid(), x, reduction="sum") / len(x)
    kl = -0.5 * (1 + logvar - means.square() - logvar.exp()).sum() / len(x)
    return reconstruction + kl, reconstruction, kl


def run_epoch(model, loader, device, optimizer=None):
    """
    Input:  model, loader, device, and optimizer (omit for validation).
    Output: mean total, reconstruction, and KL losses per image.
    """
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    count = 0

    with torch.set_grad_enabled(training):
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()

            logits, means, logvar = model(images)
            losses = vae_loss(logits, images, means, logvar)
            if training:
                losses[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10,
                                               error_if_nonfinite=True)
                optimizer.step()

            mse = F.mse_loss(logits.detach().sigmoid(), images)
            totals += torch.stack([loss.detach() for loss in losses] + [mse]) * len(images)
            count += len(images)

    if count == 0:
        raise ValueError("Loader has no images")
    if not torch.isfinite(totals).all():
        raise FloatingPointError("Non-finite loss")
    values = (totals / count).tolist()
    return dict(zip(("loss", "reconstruction", "kl", "mse"), values))


def fit(model, train_loader, val_loader, output_dir, epochs=20,
        learning_rate=1e-4, seed=42, device=None):
    """
    Input:  model, train/val loaders, new output directory, and training settings.
    Output: history table; save config, losses, best/last checkpoints,
            and reconstruction/sample grids every five epochs.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    config = {
        "latent_dim": model.latent_dim, "B": train_loader.batch_size,
        "decoder_type": model.decoder_type,
        "epochs": epochs, "learning_rate": learning_rate, "seed": seed,
        "crop_size": model.image_size, "gradient_clip_norm": 10, "image_export_every": 5,
        "num_workers": train_loader.num_workers, "pin_memory": train_loader.pin_memory,
        "model_type": type(model).__name__,
        "hidden_dims": getattr(model, "hidden_dims", None),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    history = []
    best = float("inf")
    preview, _ = next(iter(val_loader))
    preview = preview[:8].to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    fixed_z = torch.randn(16, model.latent_dim, generator=generator, device=device)

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train = run_epoch(model, train_loader, device, optimizer)
        val = run_epoch(model, val_loader, device)

        history.append({
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train.items()},
            **{f"val_{name}": value for name, value in val.items()},
            "seconds": time.perf_counter() - start,
        })
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        checkpoint = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "config": config,
            "val_loss": val["loss"],
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val["loss"] < best:
            best = val["loss"]
            torch.save(checkpoint, output_dir / "best.pt")

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                means, _ = model.encode(preview)
                reconstructions = model.decode(means).sigmoid()
                samples = model.decode(fixed_z).sigmoid()
                save_image(torch.cat([preview, reconstructions]),
                           output_dir / f"reconstructions_{epoch:02d}.png",
                           nrow=len(preview))
                save_image(samples, output_dir / f"samples_{epoch:02d}.png", nrow=4)

        print(f"Epoch {epoch}/{epochs}: train={train['loss']:.2f}, "
              f"val={val['loss']:.2f}, val MSE/pixel={val['mse']:.6f}", flush=True)

    return pd.DataFrame(history)
