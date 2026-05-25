import os
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient

from utils.preprocessing import preprocess_product_text

ARTIFACTS_DIR = Path("artifacts")
TFIDF_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
MODEL_NAME = "xgboost_text_tfidf"


class RakutenPredictor:
    def __init__(self):
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        self.tfidf = joblib.load(TFIDF_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)

        self.model = None
        self.model_name = None
        self.model_version = None

        self.reload_prod_model()


    def _list_versions(self):
        """Récupère toutes les versions du modèle dans MLflow Registry."""
        return list(self.client.search_model_versions(f"name = '{MODEL_NAME}'"))


    def _get_status(self, version):
        """Récupère le tag 'status' d'une version de modèle."""
        return (getattr(version, "tags", {}) or {}).get("status")


    def _get_latest_version(self):
        """Récupère la version la plus récente du modèle, indépendamment de son statut."""
        versions = self._list_versions()
        if not versions:
            return None
        return max(versions, key=lambda v: int(v.version))


    def _get_latest_version_by_status(self, status_value: str):
        """Récupère la version la plus récente du modèle avec un statut donné (ex: 'prod' ou 'test')."""
        versions = self._list_versions()
        matching_versions = [
            v for v in versions
            if self._get_status(v) == status_value
        ]

        if not matching_versions:
            return None

        return max(matching_versions, key=lambda v: int(v.version))


    def reload_prod_model(self):
        """Charge le modèle actuellement en production depuis MLflow Registry. Si aucun modèle n'est en prod, charge la version la plus récente disponible."""
        selected_version = self._get_latest_version_by_status("prod")

        if selected_version is None:
            selected_version = self._get_latest_version()

        if selected_version is None:
            raise RuntimeError(f"Aucune version trouvée pour le modèle {MODEL_NAME} dans MLflow Registry.")

        model_uri = f"models:/{MODEL_NAME}/{selected_version.version}"
        self.model = mlflow.sklearn.load_model(model_uri)
        self.model_name = MODEL_NAME
        self.model_version = selected_version.version


    def predict(self, designation: str, description: str) -> int:
        """
        Prédit le prdtypecode d'un produit à partir de sa designation et description. 
        Applique les mêmes étapes de preprocessing que pour l'entraînement, puis utilise le modèle chargé pour faire la prédiction.
        """
        processed_text = preprocess_product_text(designation, description)
        X = self.tfidf.transform([processed_text])
        y_pred_encoded = self.model.predict(X)
        y_pred_label = self.label_encoder.inverse_transform(y_pred_encoded.astype(int))
        return int(y_pred_label[0])


    def promote_test_if_better(self):
        """Compare les performances du modèle test avec le modèle prod actuel, et promeut le modèle test en prod si il a une meilleure performance."""
        test_version = self._get_latest_version_by_status("test")
        prod_version = self._get_latest_version_by_status("prod")

        if test_version is None:
            return {"message": "Aucun modèle test disponible"}

        test_score = float((getattr(test_version, "tags", {}) or {}).get("f1_weighted", "-1"))

        if prod_version is None:
            self.client.set_model_version_tag(MODEL_NAME, test_version.version, "status", "prod")
            self.reload_prod_model()
            return {
                "action": "promu",
                "new_prod_version": test_version.version,
                "reason": "Aucun modèle prod existant"
            }

        prod_score = float((getattr(prod_version, "tags", {}) or {}).get("f1_weighted", "-1"))

        if test_score > prod_score:
            self.client.set_model_version_tag(MODEL_NAME, prod_version.version, "status", "archive")
            self.client.set_model_version_tag(MODEL_NAME, test_version.version, "status", "prod")
            self.reload_prod_model()
            return {
                "action": "promu",
                "archived_version": prod_version.version,
                "new_prod_version": test_version.version,
                "test_score": test_score,
                "prod_score": prod_score
            }

        return {
            "action": "kept_prod",
            "test_version": test_version.version,
            "prod_version": prod_version.version,
            "test_score": test_score,
            "prod_score": prod_score
        }


    def get_model_info(self):
        """Récupère le nom et la version du modèle actuellement chargé."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
        }