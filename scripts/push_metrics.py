"""
Ce script récupère les métriques de la dernière version du modèle depuis MLflow et les pousse vers Prometheus Pushgateway pour être visualisées dans Grafana.
Il est conçu pour être exécuté sans avoir à passer par l'entraînement du modèle.
"""

import os
from mlflow import MlflowClient
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
PROMETHEUS_PUSHGATEWAY_URL = os.getenv("PROMETHEUS_PUSHGATEWAY_URL", "pushgateway:9091")
MODEL_NAME = "xgboost_text_tfidf"

def main():
    """
    Récupère les métriques de la dernière version du modèle depuis MLflow et les pousse vers Prometheus Pushgateway.
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = list(client.search_model_versions(f"name = '{MODEL_NAME}'"))
    if not versions:
        raise RuntimeError("Aucune version MLflow trouvée.")

    latest = max(versions, key=lambda v: int(v.version))
    tags = getattr(latest, "tags", {}) or {}

    metrics = {
        "accuracy": float(tags.get("accuracy", -1)),
        "f1_macro": float(tags.get("f1_macro", -1)),
        "f1_weighted": float(tags.get("f1_weighted", -1)),
    }

    registry = CollectorRegistry()
    for name, value in metrics.items():
        g = Gauge(name, f"Recovered {name}", registry=registry)
        g.set(value)

    push_to_gateway(PROMETHEUS_PUSHGATEWAY_URL, job="rakuten_train", registry=registry)
    print("Métriques repoussées avec succès.")

if __name__ == "__main__":
    main()