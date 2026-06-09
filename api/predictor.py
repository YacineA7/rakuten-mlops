"""
Module de prédiction pour le projet Rakuten MLOps.
Ce module définit la classe RakutenPredictor, qui contient la logique de chargement du modèle et de prédiction.
Il inclut également les fonctions pour interagir avec MLflow Registry, telles que :
- Récupérer les versions du modèle
- Récupérer les métriques et tags associés à chaque version
- Sélectionner la meilleure version test pour la promotion en prod
- Promouvoir une version test en prod et archiver les anciennes versions prod
"""

from datetime import time
import os
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient, MlflowException

from utils.preprocessing import preprocess_product_text

ARTIFACTS_DIR = Path("artifacts")
TFIDF_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
MODEL_NAME = "xgboost_text_tfidf"
SELECTION_METRIC = "f1_weighted"



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

        try:
            self.reload_prod_model()
        except Exception as e:
            print(f"[API] Predictor initialisé sans modèle chargé: {e}")


    def _retry(self, fn, retries=10, delay=2):
        last_err = None
        for _ in range(retries):
            try:
                return fn()
            except (ConnectionError, MlflowException, OSError) as e:
                last_err = e
                time.sleep(delay)
        raise last_err


    def _list_versions(self):
        """Récupère toutes les versions du modèle dans MLflow Registry."""
        try:
            return list(self.client.search_model_versions(f"name = '{MODEL_NAME}'"))
        except Exception as e:
            print(f"[API] Impossible de lire le registry MLflow: {e}")
            return []


    def _get_status(self, version):
        """Récupère le tag 'status' d'une version de modèle."""
        return (getattr(version, "tags", {}) or {}).get("status")


    def _get_latest_version(self):
        """Récupère la version la plus récente du modèle, indépendamment de son statut."""
        versions = self._list_versions()
        if not versions:
            return None
        return max(versions, key=lambda v: int(v.version))


    def _get_metric(self, version, metric_name: str) -> float:
        """Récupère une métrique depuis les tags MLflow, sinon renvoie -1."""
        if version is None:
            return -1.0
        raw_value = (getattr(version, "tags", {}) or {}).get(metric_name)
        if raw_value is None:
            return -1.0
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return -1.0
        

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


    def _pick_best_test_version(self, versions):
        """
        Sélectionne la meilleure version test selon f1_weighted.
        En cas d'égalité, prend la plus récente.
        """
        test_versions = [v for v in versions if self._get_status(v) == "test"]

        if not test_versions:
            return None

        return max(
            test_versions,
            key=lambda v: (self._get_metric(v, SELECTION_METRIC), int(v.version))
        )


    def reload_prod_model(self):
        """Charge le modèle actuellement en production depuis MLflow Registry. Si aucun modèle n'est en prod, charge la version la plus récente disponible."""
        selected_version = self._get_latest_version_by_status("prod")
        if selected_version is None:
            selected_version = self._get_latest_version()
        if selected_version is None:
            # Pas de modèle disponible — on ne crashe pas, l'API démarre quand même
            self.model = None
            self.model_name = None
            self.model_version = None
            return

        model_uri = f"models:/{MODEL_NAME}/{selected_version.version}"
        self.model = mlflow.sklearn.load_model(model_uri)
        self.model_name = MODEL_NAME
        self.model_version = selected_version.version
        
        self.tfidf = joblib.load(TFIDF_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)


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
        """
        Promeut en prod la meilleure version test selon f1_weighted,
        uniquement si elle est strictement meilleure que le modèle prod actuel.
        Archive les anciennes versions prod si promotion effectuée.
        """
        versions = self._list_versions()
        if not versions:
            raise RuntimeError(f"Aucune version trouvée pour le modèle {MODEL_NAME}")

        target_version = self._pick_best_test_version(versions)
        if target_version is None:
            return {"action": "no_test_available", "message": "Aucun modèle test disponible"}

        target_score = self._get_metric(target_version, SELECTION_METRIC)

        current_prod_version = self._get_latest_version_by_status("prod")
        current_prod_score = self._get_metric(current_prod_version, SELECTION_METRIC)

        # Ne promouvoir que si le test est strictement meilleur
        if current_prod_version is not None and target_score <= current_prod_score:
            self.reload_prod_model() # Assure que le modèle en prod est bien chargé dans l'API même si pas de promotion
            return {
                "action": "kept_prod",
                "selection_metric": SELECTION_METRIC,
                "reason": "Le modèle prod actuel est meilleur ou égal au meilleur modèle test",
                "prod_version": str(current_prod_version.version),
                "prod_score": current_prod_score,
                "test_version": str(target_version.version),
                "test_score": target_score,
            }

        current_prod_versions = [
            v for v in versions
            if self._get_status(v) == "prod" and str(v.version) != str(target_version.version)
        ]
        current_prod_version = (
            max(current_prod_versions, key=lambda v: int(v.version))
            if current_prod_versions else None
        )
        current_prod_score = self._get_metric(current_prod_version, SELECTION_METRIC)

        for v in current_prod_versions:
            self.client.set_model_version_tag(
                name=MODEL_NAME, version=str(v.version), key="status", value="archive"
            )
        self.client.set_model_version_tag(
            name=MODEL_NAME, version=str(target_version.version), key="status", value="prod"
        )
        self.reload_prod_model()

        result = {
            "action": "promu",
            "selection_metric": SELECTION_METRIC,
            "new_prod_version": str(target_version.version),
            "test_version": str(target_version.version),
            "test_score": target_score,
            "reason": f"Meilleure version test sélectionnée selon {SELECTION_METRIC}",
        }
        if current_prod_version is not None:
            result["prod_version"] = str(current_prod_version.version)
            result["prod_score"] = current_prod_score
            result["archived_version"] = str(current_prod_version.version)
        if current_prod_versions:
            result["archived_versions"] = [str(v.version) for v in current_prod_versions]

        return result


    def get_model_info(self):
        """Récupère le nom et la version du modèle actuellement chargé."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
        }