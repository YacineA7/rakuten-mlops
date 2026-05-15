from pathlib import Path
import json
from pyexpat import model
import joblib
import numpy as np
from scipy import sparse
from xgboost import XGBClassifier


ARTIFACTS_DIR = Path("artifacts") # Dossier contenant les artéfacts (données, encoder etc...)
MODEL_DIR = Path("model") # Dossier où est enregistré le modèle entraîné
MODEL_DIR.mkdir(parents=True, exist_ok=True) # Création du dossier si il n'existe pas


def load_data():
    """Charge les données d'entraînement et de validation à partir des fichiers .npz"""
    X_train = sparse.load_npz(ARTIFACTS_DIR / "X_train.npz")
    y_train = np.load(ARTIFACTS_DIR / "y_train.npy")

    X_valid_path = ARTIFACTS_DIR / "X_valid.npz"
    y_valid_path = ARTIFACTS_DIR / "y_valid.npy"

    # Vérifie si les fichiers de validation existent avant de les charger
    X_valid = sparse.load_npz(ARTIFACTS_DIR / "X_valid.npz") if X_valid_path.exists() else None
    y_valid = np.load(ARTIFACTS_DIR / "y_valid.npy") if y_valid_path.exists() else None

    return X_train, y_train, X_valid, y_valid


def build_model(num_classes: int) -> XGBClassifier: # type hint pour indiquer que la fonction retourne un objet de type XGBClassifier
    """Construit le modèle XGBoost avec les hyperparamètres spécifiés"""
    model = XGBClassifier(
        objective="multi:softprob", # Objectif pour la classification multi-classes : probabilités pour chaque classe
        num_class=num_classes, # Nombre de classes
        learning_rate=0.1,
        max_depth=8, # Profondeur maximale de l'arbre : plus la profondeur est haute, plus le modele est complexe, 
        n_estimators=600, # Nombre d'arbres
        subsample=0.8, # Utilisation de 80% des échantillons pour chaque arbre, pour réduire le surapprentissage
        colsample_bytree=0.8, # Utilisation de 80% des caractéristiques pour chaque arbre, pour réduire le surapprentissage
        reg_lambda=1.0,
        tree_method="hist", 
        eval_metric="mlogloss", # Utilisation de la log-loss pour évaluer les performances du modèle : plus la log-loss est petite, plus le modele est performant
        n_jobs=-1,
        random_state=42,
    )
    return model


def train_model(model, X_train, y_train, X_valid=None, y_valid=None):
    if X_valid is not None and y_valid is not None:
       model.fit(
           X_train,
           y_train,
           eval_set=[(X_valid, y_valid)],
           verbose=40, # Affiche les résultats tous les 40 arbres
       )
    else:
        model.fit(X_train, y_train)
    
    return model


def save_model(model, model_dir: Path, num_classes: int):
    """Enregistre le modèle entraîné dans le dossier spécifié"""
    joblib.dump(model, model_dir / "xgb_model.joblib") # Enregistre le modèle dans un fichier .joblib
    # Enregistre les métadonnées du modèle dans un fichier JSON
    train_metadata = {
        "model_type": "XGBClassifier",
        "objective": "multi:softprob",
        "num_class": num_classes,
        "learning_rate": 0.1,
        "max_depth": 8,
        "n_estimators": 600,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "n_jobs": -1,
        "random_state": 42,
    }

    with open(model_dir / "train_metadata.json", "w") as f:
        json.dump(train_metadata, f)

    print(f"Modèle entrainé enregistré dans {model_dir}")


def main():
    """Fonction principale pour l'entrainement du modèle"""
    X_train, y_train, X_valid, y_valid = load_data() # Chargement des données d'entraînement et de validation prétraitées
    num_classes = len(np.unique(y_train)) # Détermine le nombre de classes à partir des étiquettes d'entraînement
    model = build_model(num_classes)
    model = train_model(model, X_train, y_train, X_valid, y_valid)
    
    save_model(model, MODEL_DIR, num_classes)

if __name__ == "__main__":
    main()