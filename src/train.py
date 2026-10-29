"""Train the food-quality CNN and save its best validation checkpoint."""

import argparse
from pathlib import Path
import random

import torch
from torch import nn
from torch.optim import Adam

from src.dataset import create_data_loaders
from src.model import FoodQualityCNN


DEFAULT_SEED = 42
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-3
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "model" / "food_quality_cnn.pth"


def train_model(
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Train the CNN, returning the path to its lowest-validation-loss checkpoint."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    random.seed(seed)
    torch.manual_seed(seed)

    train_loader, validation_loader, _test_loader, class_to_idx = (
        create_data_loaders(seed=seed)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FoodQualityCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss_sum = 0.0
        training_sample_count = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            training_loss_sum += loss.item() * batch_size
            training_sample_count += batch_size

        training_loss = training_loss_sum / training_sample_count

        model.eval()
        validation_loss_sum = 0.0
        validation_sample_count = 0
        correct_predictions = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                validation_loss_sum += loss.item() * batch_size
                validation_sample_count += batch_size
                correct_predictions += (logits.argmax(dim=1) == labels).sum().item()

        validation_loss = validation_loss_sum / validation_sample_count
        validation_accuracy = correct_predictions / validation_sample_count

        print(
            f"Epoch {epoch}/{epochs} | "
            f"training loss: {training_loss:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.2%}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
                CHECKPOINT_PATH,
            )

    print(f"Best checkpoint saved to: {CHECKPOINT_PATH}")
    return CHECKPOINT_PATH


def main():
    parser = argparse.ArgumentParser(description="Train the food-quality CNN.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    train_model(epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed)


if __name__ == "__main__":
    main()
