from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the saved classification model
classification_model = tf.keras.models.load_model('CC.h5')

def classify_image(model, img):
    prediction = model.predict(img)
    return "Coral" if prediction[0] < 0.5 else "Non-coral"

# Load the saved health detection model
health_model = tf.keras.models.load_model('PP.h5')

def health_detect_image(model, img):
    prediction = model.predict(img)
    class_names = ['Healthy', 'Unhealthy']
    scores = prediction[0]
    predicted_class = class_names[np.argmax(scores)]
    confidence = np.max(scores)
    return predicted_class, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file:
        img = Image.open(file)
        img = img.resize((150, 150))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = classify_image(classification_model, img)
        return jsonify({'prediction': prediction})
    return jsonify({'error': 'No file provided'})

@app.route('/health_detect', methods=['POST'])
def health_detect():
    file = request.files.get('file')
    if file:
        img = Image.open(file)
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction, confidence = health_detect_image(health_model, img)
        return jsonify({'prediction': prediction, 'confidence': float(confidence)})
    return jsonify({'error': 'No file provided'})

if __name__ == '__main__':
    app.run(debug=True)
