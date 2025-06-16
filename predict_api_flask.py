from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
from flask_cors import CORS
import os

# Load model
MODEL_PATH = "best_model_ever.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file 'best_model_ever.pkl' not found.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)

# Optional: Serve HTML (index and predictor form)
@app.route('/')
def serve_home():
    return send_from_directory('.', 'index.html')

@app.route('/predictor')
def serve_predictor():
    return send_from_directory('.', 'price_predictor.html')

# API route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app for platform compatibility (Render)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
