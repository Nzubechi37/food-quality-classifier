"""A small CNN for fresh versus rotten food image classification."""

import torch.nn as nn


class FoodQualityCNN(nn.Module):
    """Classify RGB images shaped ``[batch, 3, 224, 224]``.

    Returns raw logits shaped ``[batch, 2]`` in class order fresh (0), rotten (1).
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: learn basic edges and color patterns, then halve resolution.
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 2: detect more complex local features at reduced resolution.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 3: build higher-level features and downsample once more.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, images):
        """Return fresh/rotten logits for a batch of RGB images."""
        features = self.features(images)
        pooled_features = self.pool(features)
        return self.classifier(pooled_features)
