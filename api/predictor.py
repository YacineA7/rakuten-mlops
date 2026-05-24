import os
from pathlib import Path
import joblib
from mlflow import MlflowClient
import mlflow

from utils.preprocessing import preprocess_product_text

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")

TFIDF_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
MODEL_NAME = "xgboost_text_tfidf"


class RakutenPredictor:
    def __init__(self):
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000") # Récupère l'URI de suivi de MLflow à partir des variables d'environnement
        self.client = MlflowClient(tracking_uri=self.tracking_uri) # Initialise le client MLflow pour interagir avec le serveur MLflow
        self.tfidf = joblib.load(TFIDF_PATH) # Charge le TF-IDF vectorizer utilisé pour l'entraînement
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH) # Charge le label encoder utilisé pour l'entraînement
        self.model = None # Initialise le modèle à None, il sera chargé dans la méthode reload_prod_model
        self.model_name = None # Initialise le nom du modèle à None, il sera défini dans la méthode reload_prod_model
        self.model_version = None # Initialise la version du modèle à None, elle sera définie dans la méthode reload_prod_model
        self.reload_prod_model() # Charge le modèle de production au moment de l'initialisation du prédicteur


    def _get_version_by_status(self, status_value: str):
        versions = self.client.search_model_versions(f"name='{MODEL_NAME}'") # Recherche les versions du modèle dans le registre de modèles MLflow
        for v in versions:
            if v.tags.get("Status") == status_value: # Vérifie si la version a le tag "Status" avec la valeur spécifiée
                return v.version # Si oui, retourne le numéro de version
        return None # Si aucune version n'a le tag "Status" avec la valeur spécifiée, retourne None 


    def reload_prod_model(self):
        prod_version = self._get_version_by_status("prod")
        if prod_version is None:
            raise RuntimeError("Aucun modèle avec le tag status=prod dans MLflow Registry.")

        model_uri = f"models:/{MODEL_NAME}/{prod_version.version}"
        self.model = mlflow.sklearn.load_model(model_uri)
        self.model_name = MODEL_NAME
        self.model_version = prod_version.version


    def predict(self, designation: str, description: str) -> int:
        """Prédit le prdtypecode à partir de la designation et description du produit"""
        processed_text = preprocess_product_text(designation, description) # Utilise la même fonction de preprocessing que pour l'entraînement
        X = self.tfidf.transform([processed_text]) # Vectorisation du texte avec le même TF-IDF que pour l'entraînement
        y_pred_encoded = self.model.predict(X) # Prédit la classe encodée (0 à 26)
        y_pred_label = self.label_encoder.inverse_transform(y_pred_encoded.astype(int)) # Convertit les classes encodées en labels originaux
        return int(y_pred_label[0])
    

    def promote_test_if_better(self):
        test_version = self._get_version_by_status("test")
        prod_version = self._get_version_by_status("prod")

        if test_version is None:
            return {"Aucun modèle test disponible"}

        test_score = float(test_version.tags.get("f1_weighted", "-1"))

        if prod_version is None:
            self.client.set_model_version_tag(MODEL_NAME, test_version.version, "status", "prod")
            self.reload_prod_model()
            return {
                "action": "promu",
                "new_prod_version": test_version.version,
                "reason": "Aucun modèle prod existant"
            }

        prod_score = float(prod_version.tags.get("f1_weighted", "-1"))

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
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
        }
