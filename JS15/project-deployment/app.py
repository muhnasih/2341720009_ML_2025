import pickle
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model dan scaler
model = load_model("model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return "Model Deployment is Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Contoh input (SESUIKAN)
        # Misal model Anda pakai 4 fitur
        features = [
            data["feature1"],
            data["feature2"],
            data["feature3"],
            data["feature4"]
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        # Contoh klasifikasi biner
        result = int(prediction[0][0] > 0.5)

        return jsonify({
            "prediction": result,
            "confidence": float(prediction[0][0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
