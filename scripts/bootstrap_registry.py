import os
from mlflow import MlflowClient

MODEL_NAME = "xgboost_text_tfidf"
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

client = MlflowClient(tracking_uri=tracking_uri)

versions = client.search_model_versions(filter_string=f"name = '{MODEL_NAME}'")

if not versions:
    versions = client.get_latest_versions(MODEL_NAME)

if not versions:
    raise RuntimeError(f"Aucune version trouvée pour {MODEL_NAME}")

latest = max(versions, key=lambda v: int(v.version))

client.set_model_version_tag(MODEL_NAME, latest.version, "status", "prod")
print(f"Version {latest.version} définie en prod")