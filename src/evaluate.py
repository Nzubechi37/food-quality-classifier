"""Evaluate the saved food-quality CNN on the held-out test split."""

import argparse
from pathlib import Path

import torch
from torch import nn

from src.dataset import create_data_loaders
from src.model import FoodQualityCNN


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent / "model" / "food_quality_cnn.pth"
)


def evaluate_model(checkpoint_path: Path = CHECKPOINT_PATH) -> dict:
    """Load a checkpoint and report loss and per-class test metrics."""
    _train_loader, _validation_loader, test_loader, dataset_class_to_idx = (
        create_data_loaders()
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    class_to_idx = checkpoint.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("Checkpoint does not contain a valid class_to_idx mapping")

    # Model outputs are ordered by the checkpoint mapping. Convert test labels
    # from ImageFolder's mapping in case its numeric indices differ.
    idx_to_checkpoint_idx = {}
    for class_name, dataset_idx in dataset_class_to_idx.items():
        if class_name not in class_to_idx:
            raise ValueError(f"Test data class {class_name!r} is missing from checkpoint")
        idx_to_checkpoint_idx[dataset_idx] = class_to_idx[class_name]
    if set(dataset_class_to_idx) != set(class_to_idx):
        raise ValueError("Test data classes do not match the checkpoint classes")

    model = FoodQualityCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    class_count = len(class_to_idx)
    confusion = torch.zeros(class_count, class_count, dtype=torch.int64)
    loss_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            targets = torch.tensor(
                [idx_to_checkpoint_idx[int(label)] for label in labels],
                dtype=torch.long,
                device=device,
            )
            logits = model(images)
            loss_sum += criterion(logits, targets).item()
            predictions = logits.argmax(dim=1)
            sample_count += targets.size(0)
            for actual, predicted in zip(targets.cpu(), predictions.cpu()):
                confusion[actual, predicted] += 1

    if sample_count == 0:
        raise ValueError("The test DataLoader contains no samples")

    accuracy = confusion.diag().sum().item() / sample_count
    metrics = {}
    for class_name, class_idx in sorted(class_to_idx.items(), key=lambda item: item[1]):
        true_positive = confusion[class_idx, class_idx].item()
        false_positive = confusion[:, class_idx].sum().item() - true_positive
        false_negative = confusion[class_idx, :].sum().item() - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    results = {
        "loss": loss_sum / sample_count,
        "accuracy": accuracy,
        "per_class": metrics,
        "sample_count": sample_count,
    }
    print(f"Test loss: {results['loss']:.4f}")
    print(f"Test accuracy: {accuracy:.2%}")
    for class_name, values in metrics.items():
        print(
            f"{class_name}: precision {values['precision']:.2%} | "
            f"recall {values['recall']:.2%} | F1-score {values['f1_score']:.2%}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved food-quality CNN.")
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT_PATH,
        help="Path to the saved model checkpoint.",
    )
    args = parser.parse_args()
    evaluate_model(args.checkpoint)


if __name__ == "__main__":
    main()
