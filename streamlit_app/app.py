import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import requests

# Use Docker container name for internal communication
endpoint = "https://potato-disease-classification-n7ab.onrender.com/predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def predict(image: np.ndarray):
    # Convert the image to bytes
    image_bytes = BytesIO()
    Image.fromarray(image).save(image_bytes, format='JPEG')

    # Send the image as a file to FastAPI
    files = {"file": ("image.jpg", image_bytes.getvalue(), "image/jpeg")}
    
    response = requests.post(endpoint, files=files)
    
    # Handle potential errors
    try:
        response_json = response.json()
        if "predictions" not in response_json:
            raise KeyError("The response does not contain 'predictions'. Response: " + str(response_json))
        
        prediction = np.array(response_json["predictions"][0])
        return prediction

    except Exception as e:
        st.error(f"Failed to get prediction: {str(e)}")
        return np.array([0.0, 0.0, 0.0])  # Return empty prediction in case of failure

st.title("Potato Disease Prediction App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    st.write("Classifying...")
    
    prediction = predict(image)
    
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
