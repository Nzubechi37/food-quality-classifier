"""Streamlit interface for the AI Food Quality Inspector API."""

import json
import uuid
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict",
)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def predict_image(file_name: str, file_bytes: bytes, content_type: str) -> dict:
    """Send an image to the FastAPI backend as a multipart upload."""
    boundary = f"----FoodInspector{uuid.uuid4().hex}"
    body = b"\r\n".join(
        [
            f"--{boundary}".encode(),
            (
                f'Content-Disposition: form-data; name="file"; '
                f'filename="{file_name}"'
            ).encode(),
            f"Content-Type: {content_type}".encode(),
            b"",
            file_bytes,
            f"--{boundary}--".encode(),
            b"",
        ]
    )
    request = Request(
        API_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


st.set_page_config(page_title="AI Food Quality Inspector", layout="centered")
st.title("AI Food Quality Inspector")
st.write(
    "Upload a fruit photo to check whether it looks fresh or rotten. "
    "The image is analyzed by the Food Quality API."
)

uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, and PNG.",
)

if uploaded_file is not None:
    extension = (uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else "")
    if extension not in ALLOWED_EXTENSIONS:
        st.error("Please upload an image in JPG, JPEG, or PNG format.")
    else:
        image_bytes = uploaded_file.getvalue()
        if not image_bytes:
            st.error("That file is empty. Please choose another image.")
        else:
            st.image(image_bytes, caption=uploaded_file.name, use_container_width=True)
            if st.button("Inspect image", type="primary", use_container_width=True):
                content_type = uploaded_file.type or "application/octet-stream"
                try:
                    with st.spinner("Inspecting your image..."):
                        result = predict_image(uploaded_file.name, image_bytes, content_type)

                    prediction = result.get("prediction")
                    confidence = float(result.get("confidence"))
                    if prediction not in {"fresh", "rotten"} or not 0 <= confidence <= 1:
                        raise ValueError("The API returned an unexpected prediction result.")

                    st.subheader("Result")
                    st.success(f"Predicted class: **{prediction.capitalize()}**")
                    st.metric("Confidence", f"{confidence:.1%}")
                except HTTPError as error:
                    try:
                        detail = json.loads(error.read().decode("utf-8")).get("detail")
                    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                        detail = None
                    st.error(detail or f"The API could not process this image (HTTP {error.code}).")
                except URLError:
                    st.error(
                        "Could not connect to the Food Quality API. Make sure the backend "
                        "is running at http://127.0.0.1:8000, then try again."
                    )
                except (TimeoutError, json.JSONDecodeError, TypeError, ValueError) as error:
                    st.error(f"The prediction request failed: {error}")
                except OSError as error:
                    st.error(f"The prediction request failed: {error}")
