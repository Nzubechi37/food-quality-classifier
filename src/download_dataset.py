"""Download the raw fresh/rotten fruit images from Hugging Face."""

from pathlib import Path

from datasets import ClassLabel, load_dataset
from PIL import Image


DATASET_ID = "Project-AgML/fresh_rotten_fruit_classification"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    # The "raw" config contains the original 3,200 images; augmented is separate.
    dataset = load_dataset(DATASET_ID, "raw", split="train")
    label_feature = dataset.features["label"]

    for label in ("fresh", "rotten"):
        (OUTPUT_DIR / label).mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(dataset):
        label_value = row["label"]
        if isinstance(label_feature, ClassLabel):
            label = label_feature.int2str(label_value)
        else:
            label = str(label_value)

        if label not in {"fresh", "rotten"}:
            raise ValueError(f"Unexpected dataset label at row {index}: {label!r}")

        image = row["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a decoded image at row {index}, got {type(image).__name__}")

        # JPEG output gives every sample a standard image extension, independent
        # of its encoding in the Hugging Face dataset.
        destination = OUTPUT_DIR / label / f"{label}_{index:05d}.jpg"
        image.convert("RGB").save(destination, format="JPEG", quality=95)

    print(f"Saved {len(dataset)} raw images under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
