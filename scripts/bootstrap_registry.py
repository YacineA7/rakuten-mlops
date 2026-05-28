import os
from mlflow import MlflowClient

MODEL_NAME = "xgboost_text_tfidf"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
SELECTION_METRIC = "f1_weighted" # Métrique utilisée pour selectionner la meilleure version pour notre problème de classe déséquilibrée.


def get_all_versions(client: MlflowClient, model_name: str):
    """Récupère toutes les versions d'un modèle donné dans le registre MLflow."""
    versions = list(client.search_model_versions(f"name = '{model_name}'"))
    if not versions:
        raise RuntimeError(f"Aucune version trouvée pour {model_name}")
    return versions


def get_status(version) -> str | None:
    """Récupère le tag 'status' d'une version de modèle."""
    return (getattr(version, "tags", {}) or {}).get("status")


def get_metric(version, metric_name: str) -> float:
    """Récupère une métrique depuis les tags MLflow, sinon renvoie -1."""
    raw_value = (getattr(version, "tags", {}) or {}).get(metric_name)
    if raw_value is None:
        return -1.0
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return -1.0


def pick_target_version(versions, metric_name: str):
    """
    Sélectionne la version à passer en prod :
    parmi les versions avec status='test', prend celle avec la meilleure métrique.
    En cas d'égalité, prend la version la plus récente.
    """
    test_versions = [v for v in versions if get_status(v) == "test"]

    if not test_versions:
        raise RuntimeError("Aucune version avec status='test' disponible pour promotion.")

    return max(
        test_versions,
        key=lambda v: (get_metric(v, metric_name), int(v.version))
    )


def main():
    """Promeut en prod la meilleure version test et archive les anciennes prod."""
    client = MlflowClient(tracking_uri=TRACKING_URI)

    versions = get_all_versions(client, MODEL_NAME)
    target = pick_target_version(versions, SELECTION_METRIC)

    current_prod_versions = [
        v for v in versions
        if get_status(v) == "prod" and str(v.version) != str(target.version)
    ]

    for v in current_prod_versions:
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=str(v.version),
            key="status",
            value="archive"
        )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=str(target.version),
        key="status",
        value="prod"
    )

    target_score = get_metric(target, SELECTION_METRIC)

    print(f"Version cible : {target.version}")
    print(f"Métrique de sélection : {SELECTION_METRIC}={target_score}")

    if current_prod_versions:
        archived = ", ".join(str(v.version) for v in current_prod_versions)
        print(f"Anciennes versions prod archivées : {archived}")
    else:
        print("Aucune ancienne version prod à archiver.")

    print(f"Version {target.version} définie comme unique prod")


if __name__ == "__main__":
    main()