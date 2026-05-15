from pathlib import Path
import joblib

from utils.preprocessing import preprocess_product_text

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")

TFIDF_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
MODEL_PATH = MODELS_DIR / "xgb_model.joblib"


class RakutenPredictor:
    def __init__(self):
        self.tfidf = joblib.load(TFIDF_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.model = joblib.load(MODEL_PATH)
        self.model_name = "xgboost_text_tfidf"

    def predict(self, designation: str, description: str) -> int:
        """Prédit le prdtypecode à partir de la designation et description du produit"""
        processed_text = preprocess_product_text(designation, description) # Utilise la même fonction de preprocessing que pour l'entraînement
        X = self.tfidf.transform([processed_text]) # Vectorisation du texte avec le même TF-IDF que pour l'entraînement
        y_pred_encoded = self.model.predict(X) # Prédit la classe encodée (0 à 26)
        y_pred_label = self.label_encoder.inverse_transform(y_pred_encoded.astype(int)) # Convertit les classes encodées en labels originaux
        return int(y_pred_label[0])
