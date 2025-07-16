# src/predict.py

import mlflow
import mlflow.pyfunc
from src.utils import clean_text

# 🎯 Set MLflow URI
mlflow.set_tracking_uri("sqlite:///mlruns_store/mlflow.db")

# 🔍 Load latest version of the registered model
client = mlflow.tracking.MlflowClient()
latest = client.get_latest_versions("SpamClassifierModel")[0].version
model = mlflow.pyfunc.load_model(f"models:/SpamClassifierModel/{latest}")

def predict_message(message, threshold=0.3):
    cleaned = clean_text(message)

    # ⚙️ Model handles vectorization internally
    prob = model.predict([cleaned])[0][1]

    label = "Spam 🚫" if prob >= threshold else "Not Spam ✅"
    return label, round(prob * 100, 2)
