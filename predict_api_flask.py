
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS
from flask import send_from_directory

# Load model
with open("../model/best_model_ever.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)  # Allow all origins

@app.route('/')
def serve_home():
    return send_from_directory('.', 'index.html')

@app.route('/predictor')
def serve_predictor():
    return send_from_directory('.', 'price_predictor_final_linked.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
