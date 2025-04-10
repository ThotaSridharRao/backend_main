import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model('..\\potatoes.h5')  # Adjust the path as needed

# Configuration
UPLOAD_FOLDER = 'D:\\Potato-disease-classification\\flask_app\\uploaded_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Adjust based on model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Check file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict
        img = preprocess_image(filepath)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]

        class_names = ['Early Blight', 'Late Blight', 'Healthy']
        label = class_names[predicted_class]
        confidence = float(np.max(prediction))

        return jsonify({
            'filename': filename,
            'prediction': label,
            'confidence': confidence
        })

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == "__main__":
    app.run(debug=True)
