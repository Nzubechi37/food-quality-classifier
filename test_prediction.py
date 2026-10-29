from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from src.predict import predict_image


def test_predict_image_returns_valid_result():
    with TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "sample.jpg"
        Image.new("RGB", (224, 224), color=(100, 150, 100)).save(image_path)

        predicted_class, confidence = predict_image(image_path)

    assert predicted_class in {"fresh", "rotten"}
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
