import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlruns_store/mlflow.db")

client = MlflowClient()
models = client.get_registered_models()

for model in models:
    print(f"✅ Found registered model: {model.name}")
