from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder


def make_transform(crop_size=256):
    """
    Input:  crop_size, the positive int for height and width of the square crop.
    Output: grayscale, proportional resize, center crop, and [0, 1] tensor transform.
    """
    assert crop_size > 0 and isinstance(crop_size, int)

    return v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize(crop_size),
        v2.CenterCrop(crop_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def is_image(path):
    path = Path(path)
    is_image_binary = path.suffix.lower() in (".jpeg", ".jpg", ".png")
    return is_image_binary


def make_loaders(root, crop_size=256, B=64, seed=42, num_workers=8, pin_memory=True):
    """
    Input:  root, the folder containing train/val/test; crop_size, square resolution.
            B, maximum images processed together; seed, training shuffle seed.
            num_workers, training loader workers; pin_memory, faster GPU transfers.
    Output: dictionary of train/val/test loaders yielding (images, labels).
            Images have shape (N, 1, crop_size, crop_size), where N <= B.
    """
    if not isinstance(B, int) or B <= 0:
        raise ValueError("B must be a positive integer")

    transform = make_transform(crop_size)
    loaders = {}
    for split in ("train", "val", "test"):
        workers = num_workers if split == "train" else 0
        dataset = ImageFolder(Path(root) / split, transform=transform, is_valid_file=is_image)
        loaders[split] = DataLoader(
            dataset,
            batch_size=B,
            shuffle=(split == "train"),
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=pin_memory,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed),
        )
    return loaders
