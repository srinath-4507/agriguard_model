import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Define model path and image size (consistent with training)
MODEL_PATH = 'efficientnetb0_plant_damage_detector.h5'
IMG_SIZE = 224
PREDICTION_THRESHOLD = 0.5 # Threshold for binary classification

# Load the model
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

@app.route('/')
def home():
    return "Welcome to the Plant Damage Detector API! Upload an image to /predict to get a prediction."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded. Please check server logs."
        }), 500

    if 'file' not in request.files:
        return jsonify({
            "error": "No file part in the request"
        }), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "error": "No selected file"
        }), 400

    if file:
        try:
            # Read the image file
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert('RGB') # Ensure image is in RGB format

            # Preprocess the image
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Create a batch
            img_array /= 255.0 # Normalize pixels to [0, 1]

            # Make prediction
            prediction = model.predict(img_array)[0][0] # Get the single prediction value

            # Interpret prediction
            damage_score = float(prediction)
            severity = damage_score # In a binary context, score can be severity

            if prediction >= PREDICTION_THRESHOLD:
                damage_type = 'damaged'
                # Confidence for 'damaged' is how far above 0.5 it is, scaled to 0-1
                confidence = (prediction - PREDICTION_THRESHOLD) / (1.0 - PREDICTION_THRESHOLD) if prediction > PREDICTION_THRESHOLD else 0.0
            else:
                damage_type = 'not_damaged'
                # Confidence for 'not_damaged' is how far below 0.5 it is, scaled to 0-1
                confidence = (PREDICTION_THRESHOLD - prediction) / PREDICTION_THRESHOLD if prediction < PREDICTION_THRESHOLD else 0.0

            # Ensure confidence is within [0, 1]
            confidence = np.clip(confidence, 0.0, 1.0)

            response = {
                "damage_type": damage_type,
                "severity": round(severity, 4),
                "damage_score": round(damage_score, 4),
                "confidence": round(confidence, 4)
            }
            return jsonify(response)

        except Exception as e:
            return jsonify({
                "error": f"Error processing image or making prediction: {e}"
            }), 500

    return jsonify({
        "error": "Unknown error during file upload"
    }), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible from outside the container in Render
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
