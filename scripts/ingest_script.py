"""
Script de démonstration conforme à PEP8.

Ce module illustre une structure claire :
- imports regroupés
- constantes en majuscules
- fonctions bien nommées
- point d'entrée principal
- gestion des arguments
- journalisation
"""


import nltk
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer

import json
import joblib
import os
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

warnings.filterwarnings('ignore')

DATA_DIR = Path("data/raw")
ARTIFACTS_DIR = Path("artifacts")  # Dossier contenant les artéfacts (données, encoder etc...)

X_TRAIN_PATH = DATA_DIR / "X_train_update.csv"
Y_TRAIN_PATH = DATA_DIR / "Y_train_CVw08PX.csv"


# Fonction de chargement des données brutes

def load_rawdata(x_path: Path, y_path: Path) -> pd.DataFrame:
    """Charge les données d'entraînement et de validation à partir des fichiers CSV."""
    x_train = pd.read_csv(x_path)
    y_train = pd.read_csv(y_path)

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


# Le reste du module est temporairement désactivé pour permettre l'importation
# et l'exécution des tests unitaires sur load_rawdata.
# Ajoutez ici vos fonctions de nettoyage de texte et le pipeline lorsque le module
# sera stabilisé.


def clean_text(text) -> str:
    """
    Nettoyage de base des raw_data  : suppression des balises HTML, des URLs, conversion en minuscules, suppression de la ponctuation et des chiffres.
    """
    if pd.isnull(text):
        return ""

    # Suppression des balises HTML
    text = re.sub(r'<.*?>', '', text)

    # Remplacement des <br /> par un espace
    text = text.replace(r'<br />', ' ')

    # Remplacement des référence de caractère HTML
    text = text.replace(r'&amp;', '&')
    text = text.replace(r'&nbsp;', ' ')
    text = text.replace(r'&lt', '<')
    text = text.replace(r'&gt', '>')
    text = text.replace(r'&quot', '"')
    text = text.replace(r'&#39', "'")
    text = text.replace(r'&eacute', 'e')
    text = text.replace(r'&egrave', 'e')
    text = text.replace(r'&ecirc', 'e')

    # Suppression des URLs et des liens  
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Conversion en minuscules
    text = text.lower()

    # Suppression des espaces supplémentaires
    text = re.sub(r'\s+', ' ', text).strip()

    # Suppression de la ponctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Suppression des chiffres
    text = re.sub(r'\d+', '', text)

    return text 


def built_text(df: pd.DataFrame) -> pd.Series:
    
    """"
    Nettoyage du texte et création de la colonne "text" en concaténant les colonnes "designation" et "description" 
    Après nettoyage du corpus avec la fonction clean_text
    """

     # Nettoyage du texte simple pour les colonnes de texte
    df['clean_designation'] = df['designation'].apply(clean_text)
    df['clean_description'] = df['description'].apply(clean_text)

    # Concatenation designation + description dans une nouvelle colonne "text" pour l'analyse de texte
    df['text'] = df['clean_designation'] + ' ' + df['clean_description']

    return df["text"]


def build_stopword_set() -> set:
    """
    Construit l'ensemble des stopwords français et anglais, ainsi que le mot "generique" (omniprésent dans les textes).
    """
    nltk.download("stopwords", quiet=True)

    stop_fr = set(stopwords.words("french"))
    stop_en = set(stopwords.words("english"))

    stop_set = stop_fr.union(stop_en) # Combine les stopwords français et anglais
    stop_set.add("generique") # Ajoute le mot "generique" à l'ensemble des stopwords

    return stop_set


def delete_stopwords(text: str, stop_set: set):
    """
    Suppression des mots vides (stopwords)
    """
    tokens = [
        w for w in text.split()
        if w not in stop_set and len(w) > 1  # Garde mots > 1 caractère
    ]
    return " ".join(tokens) 


def stem_text(text: str, stemmer: SnowballStemmer) -> str:
    """
    Application du stemming français sur chaque mot
    """
    tokens = [stemmer.stem(w) for w in text.split()] # Stemming mot par mot 
    return " ".join(tokens)  


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
        max_features=50000,
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
            "max_features": 50000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95
        }
    }

    with open(artifacts_dir / "ingestion_metadata.json", "w") as f:
        json.dump(ingestion_metadata, f)

    print(f"Artifacts enregistrés dans {artifacts_dir}")



    def main():
        df = load_rawdata(X_TRAIN_PATH, Y_TRAIN_PATH) # Chargement des données brutes
        corpus = built_text(df) # Nettoyage du texte et création de la colonne "text"
        stop_set = build_stopword_set() # Construction de l'ensemble de stopwords
        stemmer = SnowballStemmer("french") # Initialisation du stemmer français
        corpus_cleaned = corpus.apply(lambda x: delete_stopwords(x, stop_set)) # Suppression des stopwords
        corpus_stemmed = corpus_cleaned.apply(lambda x: stem_text(x, stemmer)) # Application du stemming sur le texte nettoyé
        y_enc, le = label_encoder(df["prdtypecode"]) # Encodage des labels de la variable cible
        X_train, X_valid, y_train, y_valid = split_data(corpus_stemmed, y_enc) # Séparation des données en un ensemble d'entraînement et de validation
        X_train_tfidf, X_valid_tfidf, tfidf = vectorize_text(X_train, X_valid) # Vectorisation du texte avec TfidfVectorizer

        save_artifacts( # Enregistrement des artéfacts de l'ingestion et du préprocessing
            X_train_tfidf, y_train,
            X_valid_tfidf, y_valid,
            tfidf, le,
            ARTIFACTS_DIR 
        )

    if __name__ == "__main__":
        main()