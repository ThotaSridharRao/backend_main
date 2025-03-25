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

@app.get("/ping")
async def ping():
    return "Hello I am alive"

def read_files_as_image(data) -> np.ndarray:
    """ Convert uploaded file bytes to a NumPy image array """
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """ Handle image prediction directly """
    try:
        # Read image and preprocess
        image = read_files_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        # Simulating model prediction for testing (replace with your model's inference)
        prediction = np.random.rand(1, 3)  # Replace with your actual model's output
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        # ✅ Return in the expected format
        return {
            "predictions": prediction.tolist()
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
