from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("1.h5")  # or "1.keras"
model=tf.keras.models.load_model("autoencoder_256.keras")


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image1(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
def read_file_as_image(data) -> np.ndarray:
    """Load image exactly like during training."""
    image = (
        Image.open(BytesIO(data))
        .convert("RGB")          # 3 channels
        .resize((256, 256))      # make shape match the model
    )
    image = np.asarray(image, dtype=np.float32) / 255.0  # 0‑1 range
    return image



@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    contents = await file.read()  # Read once

    # For autoencoder (normalize + resize)
    image_auto = (
        Image.open(BytesIO(contents))
        .convert("RGB")
        .resize((256, 256))
    )
    image_auto = np.asarray(image_auto, dtype=np.float32) / 255.0
    img_batch_auto = np.expand_dims(image_auto, 0)

    # For classifier (resize only if that's what you did during training)
    image_clf = Image.open(BytesIO(contents))
    img_batch_clf = np.expand_dims(image_clf, 0)

    # Step 1: Autoencoder anomaly detection
    reconstructed = model.predict(img_batch_auto)
    reconstruction_error = np.mean((img_batch_auto - reconstructed) ** 2)

    THRESHOLD = 0.025  # Tune this based on validation

    if reconstruction_error > THRESHOLD:
        return {
            "class": "Unknown",
            "confidence": 0.0,
            "warning": "This is likely NOT a potato leaf.",
            "reconstruction_error": float(reconstruction_error)
        }

    predictions = MODEL.predict(img_batch_clf)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
@app.get("/")
async def root():
    return {"message": "Welcome to Potato Disease Classifier API"}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
