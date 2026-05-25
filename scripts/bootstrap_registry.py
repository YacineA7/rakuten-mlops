import os
from mlflow import MlflowClient

MODEL_NAME = "xgboost_text_tfidf"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

def get_all_versions(client: MlflowClient, model_name: str):
    """Récupère toutes les versions d'un modèle donné dans le registre MLflow."""
    versions = list(client.search_model_versions(f"name = '{model_name}'"))
    if not versions:
        raise RuntimeError(f"Aucune version trouvée pour {model_name}")
    return versions


def pick_target_version(versions):
    """Sélectionne la version à passer en prod : la version avec le tag 'status=test' et la meilleure d'accuracy."""
    return max(versions, key=lambda v: int(v.version))


def get_status(version):
    """Récupère le tag 'status' d'une version de modèle."""
    return (getattr(version, "tags", {}) or {}).get("status")


def main():
    """Script pour promouvoir une version de modèle dans MLflow Registry en prod"""
    client = MlflowClient(tracking_uri=TRACKING_URI)

    versions = get_all_versions(client, MODEL_NAME)
    target = pick_target_version(versions)

    current_prod_versions = [
        v for v in versions
        if get_status(v) == "prod" and str(v.version) != str(target.version)
    ]

    for v in current_prod_versions:
        client.set_model_version_tag(MODEL_NAME, str(v.version), "status", "archive")

    client.set_model_version_tag(MODEL_NAME, str(target.version), "status", "prod")

    print(f"Version cible : {target.version}")
    if current_prod_versions:
        archived = ", ".join(str(v.version) for v in current_prod_versions)
        print(f"Anciennes versions prod archivées : {archived}")
    else:
        print("Aucune ancienne version prod à archiver.")
    print(f"Version {target.version} définie comme unique prod")


if __name__ == "__main__":
    main()