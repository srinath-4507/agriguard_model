import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

MODEL_PATH = 'efficientnetb0_plant_damage_detector.h5'
IMG_SIZE = 224
PREDICTION_THRESHOLD = 0.5

# Load model
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return "Welcome to the Plant Damage Detector API!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)

        score = float(prediction[0][0])
        damage_type = "damaged" if score >= PREDICTION_THRESHOLD else "not_damaged"
        confidence = score if damage_type == "damaged" else 1 - score

        return jsonify({
            "damage_type": damage_type,
            "damage_score": round(score, 4),
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
