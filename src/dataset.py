"""Dataset and DataLoader setup for the fresh/rotten food images."""

from pathlib import Path
import random
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
DEFAULT_SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _stratified_split_indices(targets, seed: int):
    """Return disjoint indices split per class in 70/15/15 proportions."""
    indices_by_class = {}
    for index, target in enumerate(targets):
        indices_by_class.setdefault(target, []).append(index)

    rng = random.Random(seed)
    train_indices, validation_indices, test_indices = [], [], []

    for class_indices in indices_by_class.values():
        rng.shuffle(class_indices)
        count = len(class_indices)
        train_end = int(count * 0.70)
        validation_end = train_end + int(count * 0.15)

        train_indices.extend(class_indices[:train_end])
        validation_indices.extend(class_indices[train_end:validation_end])
        test_indices.extend(class_indices[validation_end:])

    # Shuffle within each split so batches do not group examples by class.
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)
    return train_indices, validation_indices, test_indices


def create_data_loaders(
    data_dir=None, batch_size: int = BATCH_SIZE, seed: int = DEFAULT_SEED
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Create train, validation, and test loaders and return the class mapping.

    Args:
        data_dir: ImageFolder root. Defaults to ``data/raw`` at the project root.
        batch_size: Number of images yielded by each loader batch.
        seed: Seed controlling the reproducible stratified split and train shuffle.

    Returns:
        ``(train_loader, validation_loader, test_loader, class_to_idx)``.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    data_dir = Path(data_dir)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # Use an untransformed ImageFolder to determine labels and stratified indices.
    source_dataset = datasets.ImageFolder(root=data_dir)
    train_indices, validation_indices, test_indices = _stratified_split_indices(
        source_dataset.targets, seed
    )

    # Each ImageFolder owns its transform, so subsets cannot accidentally share
    # a mutable transform setting.
    train_image_dataset = datasets.ImageFolder(
        root=data_dir, transform=train_transform
    )
    evaluation_image_dataset = datasets.ImageFolder(
        root=data_dir, transform=evaluation_transform
    )

    train_dataset = Subset(train_image_dataset, train_indices)
    validation_dataset = Subset(evaluation_image_dataset, validation_indices)
    test_dataset = Subset(evaluation_image_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        source_dataset.class_to_idx,
    )
