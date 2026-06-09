"""
Script d'ingestion des rawdata et du preprocessing.
Ce script charge les données brutes, applique le nettoyage et le prétraitement du texte, 
vectorise le texte avec TfidfVectorizer, encode les labels de la variable cible, 
et enregistre tous les artéfacts nécessaires pour l'entraînement du modèle dans le dossier "artifacts".
"""


import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

import json
import joblib
import warnings
import os

from utils.preprocessing import build_corpus

warnings.filterwarnings('ignore')

DATA_DIR = Path("data/raw")
ARTIFACTS_DIR = Path("artifacts")  # Dossier contenant les artéfacts (données, encoder etc...)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "X_train_update.csv"
Y_TRAIN_PATH = DATA_DIR / "Y_train_CVw08PX.csv"


# Fonction de chargement des données brutes

def load_raw_data(x_path: Path, y_path: Path) -> pd.DataFrame:
    """Charge les données d'entraînement et de validation à partir des fichiers CSV."""
    x_train = pd.read_csv(x_path, nrows=1000) # Limite à 1000 lignes pour un chargement plus rapide pendant le développement
    y_train = pd.read_csv(y_path, nrows=1000) # Limite à 1000 lignes pour un chargement plus rapide pendant le développement

    raw_data = pd.merge(
        x_train,
        y_train,
        left_index=True,
        right_index=True,
    )
    raw_data = raw_data.drop(["Unnamed: 0_y"], axis=1)
    raw_data.rename(columns={"Unnamed: 0_x": "id"}, inplace=True)
    raw_data.set_index(["id"], inplace=True)

    return raw_data


# Encodage des labels de la variable cible avec LabelEncoder
def label_encoder(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encodage des 27 labels de la variable cible avec LabelEncoder en entiers de 0 à 26, 
    Retourne à la fois les labels encodés et l'objet LabelEncoder pour pouvoir faire l'inverse_transform plus tard
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def split_data(X: pd.Series, y: np.ndarray) -> tuple:
    """
    Séparation des données en un ensemble d'entraînement (80%) et de validation (20%) 
    avec stratification pour conserver la même distribution de classes dans les deux ensembles.
    random_state fixé pour la reproductibilité.
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    return X_train, X_valid, y_train, y_valid


def vectorize_text(X_train: pd.Series, X_valid: pd.Series) -> tuple:
    """
    Vectorisation du texte avec TfidfVectorizer, 
    Retourne les matrices TF-IDF pour l'entraînement et la validation.
    Retourne aussi le TfidfVectorizer pour pouvoir faire la même transformation sur les données de test et dans l'API plus tard.
    """
    # Initialisation de TfidfVectorizer avec des paramètres pour limiter le nombre de features et les n-grams
    tfidf = TfidfVectorizer(
        max_features=500, # Limite le nombre de features à 500 avec TF-IDF pour un entraînement plus rapide
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    # Vectorisation du texte avec TF-IDF
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_valid_tfidf = tfidf.transform(X_valid)

    return X_train_tfidf, X_valid_tfidf, tfidf


def save_artifacts(X_train, y_train, X_valid, y_valid, tfidf, label_encoder, artifacts_dir: Path):
    """
    Enregistre les artéfacts de l'ingestion et du préprocessing :
    - Matrices TF-IDF d'entraînement et de validation
    - Labels encodés d'entraînement et de validation
    - TfidfVectorizer pour la vectorisation 
    - LabelEncoder pour l'encodage  des labels
    """
    print("[INGEST] Lancement de la sauvegarde des artefacts")

    sparse.save_npz(artifacts_dir / "X_train.npz", X_train) # Enregistre la matrice TF-IDF d'entraînement
    np.save(artifacts_dir / "y_train.npy", y_train) # Enregistre les labels encodés d'entraînement

    sparse.save_npz(artifacts_dir / "X_valid.npz", X_valid) # Enregistre la matrice TF-IDF de validation
    np.save(artifacts_dir / "y_valid.npy", y_valid) # Enregistre les labels encodés de validation

    joblib.dump(tfidf, artifacts_dir / "tfidf_vectorizer.pkl") # Enregistre le TfidfVectorizer pour pouvoir faire la même transformation sur les données de test et dans l'API plus tard
    joblib.dump(label_encoder, artifacts_dir / "label_encoder.pkl") # Enregistre le LabelEncoder pour pouvoir faire l'inverse_transform plus tard (déchiffrage des classes encodées en labels originaux)

    ingestion_metadata = {
        "ingestion_date": pd.Timestamp.now().isoformat(),
        "X_train_shape": list(X_train.shape), # Dimensions de la matrices d'entraînement
        "X_valid_shape": list(X_valid.shape), # Dimensions de la matrices de validation
        "n_train_samples": int(X_train.shape[0]), # Nombre d'exemples d'entraînement
        "n_valid_samples": int(X_valid.shape[0]), # Nombre d'exemples de validation
        "n_features_tfidf": int(X_train.shape[1]), # Nombre de features après vectorisation
        "n_classes": int(len(label_encoder.classes_)), # Nombre de classes cibles
        "classes": label_encoder.classes_.tolist(), # Liste des classes cibles
        "split": {
            "test_size" : 0.2,
            "random_state": 42,
            "stratify": True
        },
        "tfidf_params": {
            "max_features": 500,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95
        }
    }

    metadata_path = artifacts_dir / "ingestion_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(ingestion_metadata, f, indent=2, ensure_ascii=False)

    print(f"[INGEST] Artefacts créés et enregistrés dans le dossier ./{artifacts_dir}")


def main():
    print("[INGEST] Démarrage de l'ingestion")

    # Chargement des données brutes
    df = load_raw_data(X_TRAIN_PATH, Y_TRAIN_PATH) 
    print(f"[INGEST] Dataset chargé : {df.shape}")

    # Nettoyage du texte et création de la colonne "text"
    corpus = build_corpus(df)
    print("[INGEST] Corpus construit et prétraité")
    
    # Encodage des labels de la variable cible
    y_enc, le = label_encoder(df["prdtypecode"]) 
    print("[INGEST] Labels encodés")

    # Séparation des données en un ensemble d'entraînement et de validation
    X_train, X_valid, y_train, y_valid = split_data(corpus, y_enc)
    print(f"[INGEST] Split Train : {X_train.shape[0]} échantillons")
    print(f"[INGEST] Split Valid : {X_valid.shape[0]} échantillons")

    # Vectorisation du texte avec TfidfVectorizer
    X_train_tfidf, X_valid_tfidf, tfidf = vectorize_text(X_train, X_valid)
    print(f"[INGEST] TF-IDF train shape : {X_train_tfidf.shape}")
    print(f"[INGEST] TF-IDF valid shape : {X_valid_tfidf.shape}")
    print(f"[INGEST] Textes vectorisés avec TF-IDF : {X_train_tfidf.shape[1]} features")

    # Enregistrement des artéfacts de l'ingestion et du préprocessing
    save_artifacts(
        X_train_tfidf, y_train,
        X_valid_tfidf, y_valid,
        tfidf, le,
        ARTIFACTS_DIR 
    )

    print("[INGEST] Ingestion terminée avec succès !")


if __name__ == "__main__":
    main()