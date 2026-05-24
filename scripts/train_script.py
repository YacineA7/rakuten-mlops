from pathlib import Path
import json
import joblib
from mlflow import client
import numpy as np
from scipy import sparse
from xgboost import XGBClassifier
import warnings
import mlflow
import mlflow.sklearn
import os
from mlflow import MlflowClient, active_run, run
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')


ARTIFACTS_DIR = Path("artifacts") # Dossier contenant les artéfacts (données, encoder etc...)
MODEL_DIR = Path("models") # Dossier où est enregistré le modèle entraîné
MLRUNS_DIR = Path("mlruns") # Dossier où MLflow stocke les expériences et les runs : contient tous les logs, métriques, modèles enregistrés par MLflow
EXPERIMENT_NAME = "rakuten_text_xgboost" # Nom de l'expérience MLflow : tous les runs seront organisés sous ce nom d'expérience
MODEL_DIR.mkdir(parents=True, exist_ok=True) # Création du dossier si il n'existe pas

mlflow.set_experiment(EXPERIMENT_NAME) # Définit l'expérience MLflow : tous les runs seront organisés sous ce nom d'expérience

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


def get_model_params(num_classes: int) -> dict:
    """Retourne les hyperparamètres du modèle sous forme de dictionnaire, pour faciliter l'enregistrement dans MLflow."""
    return {
        "model_type": "XGBClassifier",
        "objective": "multi:softprob", # Objectif pour la classification multi-classes : probabilités pour chaque classe
        "num_class": num_classes, # Nombre de classes
        "learning_rate": 0.1,
        "max_depth": 8, # Profondeur maximale de l'arbre : plus la profondeur est haute, plus le modele est complexe
        "n_estimators": 600, # Nombre d'arbres
        "subsample": 0.8, # Utilisation de 80% des échantillons pour chaque arbre, pour réduire le surapprentissage
        "colsample_bytree": 0.8, # Utilisation de 80% des caractéristiques pour chaque arbre, pour réduire le surapprentissage
        "reg_lambda": 1.0, 
        "tree_method": "hist",
        "eval_metric": "mlogloss", # Utilisation de la log-loss pour évaluer les performances du modèle : plus la log-loss est petite, plus le modele est performant
        "n_jobs": -1,
        "random_state": 42,
    }


def build_model(params: dict) -> XGBClassifier:
    model_params = params.copy() # Copie des paramètres pour éviter de modifier le dictionnaire original
    model_params.pop("model_type", None) # Supprime la clé "model_type" qui n'est pas un hyperparamètre de XGBClassifier
    return XGBClassifier(**model_params)


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


def evaluate_model(model, X_valid, y_valid):
    """Évalue le modèle sur l'ensemble de validation et retourne les métriques."""
    if X_valid is None or y_valid is None:
        return None

    y_pred = model.predict(X_valid)

    metrics = {
        "accuracy": float(accuracy_score(y_valid, y_pred)),
        "f1_macro": float(f1_score(y_valid, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_valid, y_pred, average="weighted")),
    }
    return metrics


def log_to_mlflow(model, params: dict, metrics: dict | None):
    """Enregistre les paramètres, les métriques et le modèle dans MLflow pour le suivi des expériences."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MODEL_NAME = "xgboost_text_tfidf"

    with mlflow.start_run(run_name="xgboost_text_training") as active_run:
        mlflow.log_params(params)

        if metrics is not None:
            mlflow.log_metrics(metrics)

        mlflow.log_artifact(str(ARTIFACTS_DIR / "ingestion_metadata.json"))

        if (MODEL_DIR / "train_metadata.json").exists():
            mlflow.log_artifact(str(MODEL_DIR / "train_metadata.json"))

        mlflow.sklearn.log_model(model, name="xgb_model", registered_model_name=MODEL_NAME) # Enregistre le modèle dans MLflow et l'enregistre aussi dans le registre de modèles MLflow sous le nom spécifié

        run_id = active_run.info.run_id # Récupère l'ID du run pour pouvoir le référencer dans l'API
        
    client = MlflowClient(tracking_uri=tracking_uri)

    latest_versions = client.search_model_versions(
        filter_string=f"name = '{MODEL_NAME}'"
    )

    if not latest_versions:
        raise RuntimeError(f"Aucune version trouvée pour {MODEL_NAME}")
    
    latest_version = max(int(v.version) for v in latest_versions) # Détermine le numéro de version à attribuer au modèle
    
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=str(latest_version),
        key="Status",
        value="Test"
    )

    if metrics is not None:
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=str(latest_version),
            key="accuracy",
            value=str(metrics["accuracy"])
        )
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=str(latest_version),
            key="f1_macro",
            value=str(metrics["f1_macro"])
        )
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=str(latest_version),
            key="f1_weighted",
            value=str(metrics["f1_weighted"])
        )
    
    client.set_model_version_tag( 
        name=MODEL_NAME,
        version=str(latest_version),
        key="run_id", # C'est le run_id qui va nous permettre de trouver le modele en prod
        value=run_id 
    )

    print(f"Modèle enregistré dans MLflow Registry avec run_id : {run_id} et version : {latest_version}")


def main():
    """Fonction principale pour l'entrainement du modèle"""
    X_train, y_train, X_valid, y_valid = load_data() # Chargement des données d'entraînement et de validation prétraitées
    num_classes = len(np.unique(y_train)) # Détermine le nombre de classes à partir des étiquettes d'entraînement

    params = get_model_params(num_classes) # Récupère les hyperparamètres du modèle
    model = build_model(params) # Construit le modèle XGBoost avec les hyperparamètres spécifiés
    model = train_model(model, X_train, y_train, X_valid, y_valid) # Entraîne le modèle sur les données d'entraînement, avec validation
    
    save_model(model, MODEL_DIR, num_classes) # Enregistre le modèle entraîné et les métadonnées d'entraînement

    metrics = evaluate_model(model, X_valid, y_valid) # Évaluation du modèle sur l'ensemble de validation
    log_to_mlflow(model, params, metrics) # Enregistrement dans MLflow

if __name__ == "__main__":
    main()