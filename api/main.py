"""FastAPI endpoints for the food-quality image classifier."""

from io import BytesIO
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.predict import predict_image


app = FastAPI(title="Food Quality API")


@app.get("/health")
def health():
    """Report that the API process is running."""
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Classify one uploaded image as fresh or rotten."""
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Check the file contents instead of trusting the filename or MIME type.
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from None

    try:
        # predict_image accepts a path, so use a temporary file and remove it
        # as soon as inference finishes.
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_image = Path(temporary_directory) / "upload"
            temporary_image.write_bytes(image_bytes)
            predicted_class, confidence = predict_image(temporary_image)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the uploaded image: {error}",
        ) from error

    return {
        "prediction": predicted_class,
        "confidence": confidence,
    }
