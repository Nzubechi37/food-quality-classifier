"""Run single-image inference with the saved food-quality CNN."""

import argparse
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

from src.model import FoodQualityCNN


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent / "model" / "food_quality_cnn.pth"
)
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def predict_image(image_path):
    """Return ``(predicted_class, confidence)`` for a local image file.

    Confidence is the predicted class's softmax probability, in the range
    0.0–1.0. The checkpoint's class mapping defines the output labels.
    """
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
    if not isinstance(checkpoint, dict) or not {
        "model_state_dict",
        "class_to_idx",
    }.issubset(checkpoint):
        raise ValueError(
            "Checkpoint must contain 'model_state_dict' and 'class_to_idx'"
        )

    class_to_idx = checkpoint["class_to_idx"]
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("Checkpoint contains an invalid class_to_idx mapping")
    idx_to_class = {index: name for name, index in class_to_idx.items()}

    model = FoodQualityCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with Image.open(image_path) as image:
        image_tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]
        predicted_idx = int(probabilities.argmax().item())
        if predicted_idx not in idx_to_class:
            raise ValueError(
                f"Predicted index {predicted_idx} is missing from class_to_idx"
            )
        predicted_class = idx_to_class[predicted_idx]
        confidence = float(probabilities[predicted_idx].item())

    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict food quality from an image.")
    parser.add_argument("image_path", type=Path, help="Path to a local image file.")
    args = parser.parse_args()
    predicted_class, confidence = predict_image(args.image_path)
    print(f"Prediction: {predicted_class} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
