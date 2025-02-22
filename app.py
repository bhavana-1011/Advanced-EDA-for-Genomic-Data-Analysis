from flask import Flask, request, jsonify, render_template
import torch
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os
import sys


# Ensure `src/` is included in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from src.autoencoder import load_trained_autoencoder

app = Flask(__name__)

# ✅ Load dataset & fit the scaler
data_path = "outputs/processed_vcf_data.csv"
df = pd.read_csv(data_path).select_dtypes(include=["number"])

scaler = StandardScaler()
scaler.fit(df)
print(f"✅ Scaler fitted on {df.shape[1]} features")  # Debugging log

# ✅ Load trained Autoencoder (Must be retrained if dimensions mismatch)
best_model_path = "outputs/best_autoencoder.pth"
input_dim = df.shape[1]  # Ensure it matches processed data
autoencoder = load_trained_autoencoder(best_model_path, input_dim=input_dim, encoding_dim=2)
print("✅ Autoencoder Model Loaded!")

# ✅ Load KMeans model
kmeans_model_path = "outputs/kmeans_model.pkl"
with open(kmeans_model_path, "rb") as f:
    kmeans = pickle.load(f)
print("✅ KMeans Model Loaded!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        print(f"📥 Received Input: {data}")

        # Convert input into NumPy array & check dimensions
        data_array = np.array([data], dtype=np.float32)

        if data_array.shape[1] != input_dim:
            return jsonify({"error": f"Expected {input_dim} features, but got {data_array.shape[1]}"}), 400

        # Standardize input
        data_scaled = scaler.transform(data_array)
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

        # Encode input using Autoencoder
        encoded_features = autoencoder.encode(data_tensor).detach().numpy()
        print(f"🧬 Encoded Features: {encoded_features}")

        # Predict cluster using KMeans
        predicted_cluster = kmeans.predict(encoded_features)[0]
        print(f"✅ Predicted Cluster: {predicted_cluster}")

        return jsonify({"Cluster": int(predicted_cluster)})

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
