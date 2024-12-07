from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import os

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

class_names = ["Early Blight", "Late Blight", "Healthy"]

# Model path for Heroku (using /tmp to store model temporarily)
MODEL_PATH = "/tmp/potatoes.h5" 

# Check if the model is loaded; if not, load it
model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            # You can include the model in the repo or download it from a remote location
            # For now, ensure the model file is in your repository or upload it manually to Heroku
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")

def read_files_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    load_model()  # Load the model if not already loaded
    
    # Read and preprocess the image
    image = read_files_as_image(await file.read())
    image = image.resize((256, 256))  # Resize the image as per the model's input size
    image = np.array(image) / 255  # Normalize the image to 0-1 range
    img_array = np.expand_dims(image, 0)  # Expand dimensions to match model input

    # Predict using the loaded model
    predictions = model.predict(img_array)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}
