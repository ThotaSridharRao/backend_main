import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import requests

endpoint = "http://localhost:8502/v1/models/potato_model:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_files_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def predict(image: np.ndarray) -> np.ndarray:
    img_batch = np.expand_dims(image, 0)
    json_data = {
        "instances": img_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    response_json = response.json()
    if "predictions" not in response_json:
        raise KeyError("The response does not contain 'predictions'. Response: " + str(response_json))
    prediction = np.array(response_json["predictions"][0])
    return prediction

st.title("Image Prediction App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = read_files_as_image(uploaded_file.read())
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = predict(image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence * 100:.2f}%")