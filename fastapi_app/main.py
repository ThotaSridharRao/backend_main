from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://potato-disease-classification-streamlit.onrender.com/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Load the saved model
MODEL = tf.keras.models.load_model("./saved_models/1")

@app.get("/ping")
async def ping():
    return "Hello, I am alive!"

def read_file_as_image(data) -> np.ndarray:
    """Convert uploaded file bytes to a NumPy image array."""
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))  # Resize to match model input size
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image prediction using the loaded model."""
    try:
        # Read and preprocess the image
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)  # Add batch dimension

        # Perform prediction
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Return the prediction result
        return {
            "class": predicted_class,
            "confidence": float(confidence)
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)