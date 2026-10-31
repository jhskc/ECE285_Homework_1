from pathlib import Path

import torch
from torch import nn
from ignite.metrics import FID, InceptionScore
from torchvision.models import Inception_V3_Weights, inception_v3


@torch.no_grad()
def evaluate(model, test_loader, seed=2026, device=None):
    """
    Input:  trained VAE, test loader, sampling seed, and optional device.
    Output: FID, IS, image counts, and evaluation settings.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    torch.hub.set_dir(str(Path(__file__).resolve().parents[1] / ".cache" / "torch"))
    weights = Inception_V3_Weights.DEFAULT
    inception = inception_v3(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    # Use pooled features for FID and the original classifier for IS.
    classifier = inception.fc
    inception.fc = nn.Identity()
    fid = FID(num_features=2048, device=device)
    inception_score = InceptionScore(num_features=1000, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    count = 0

    for real, _ in test_loader:
        real = real.to(device)
        z = torch.randn(len(real), model.latent_dim, generator=generator, device=device)
        generated = model.decode(z).sigmoid()

        # Both sources use the same RGB conversion and Inception preprocessing.
        real_features = inception(preprocess(real.repeat(1, 3, 1, 1)))
        generated_features = inception(preprocess(generated.repeat(1, 3, 1, 1)))
        probabilities = classifier(generated_features).softmax(dim=1)
        fid.update((generated_features, real_features))
        inception_score.update(probabilities)
        count += len(real)
        print(f"Evaluated {count}/{len(test_loader.dataset)} images", flush=True)

    return {
        "FID": float(fid.compute()),
        "IS": float(inception_score.compute()),
        "real_images": count, "generated_images": count, "seed": seed,
        "fid_features": "torchvision ImageNet Inception-v3 pooled 2048",
        "is_classes": 1000,
        "inception_weights": str(weights),
        "preprocessing": "grayscale to RGB; Inception weights.transforms()",
        "test_preprocessing": str(test_loader.dataset.transform),
        "test_reference": "official test split, including its within-split duplicate files",
        "torch": str(torch.__version__),
    }
