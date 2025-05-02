import os
import shutil
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uvicorn

# Load model
model = load_model('model/potatoes.h5')  # Adjust path if needed

# Config
UPLOAD_FOLDER = 'D:\\Potato-disease-classification\\fastapi_app\\uploaded_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()

def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
async def root():
    return {"message": "Potato Disease Classifier is Running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        return JSONResponse(status_code=400, content={"error": "Invalid file type"})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = preprocess_image(file_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    class_names = ['Early Blight', 'Late Blight', 'Healthy']
    label = class_names[predicted_class]
    confidence = float(np.max(prediction)) * 100

    return {
        "filename": file.filename,
        "prediction": label,
        "confidence": confidence
    }

# Add this to run with `python main.py`
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
