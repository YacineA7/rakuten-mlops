"""
Script de prediction  du modele .

Ce module illustre une structure claire :
- Récuperer le model valide et sauvegardé 
- définition des metriques d'évaluation de performance du model  
- Affichage des metriques  
- Supprission des stopwords (EN & FR )
- Encodage de l'ensemble du set 
- Véctorisation du steeming avec TF-IDF 
- Ajouter étape de la sauvegarde de l'ensembledes fonctions 
- Faire appel à la fonction main à la toute fin 
"""


import re
import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/raw")
ARTIFACTS_DIR = Path("artifacts")  # Dossier contenant les artéfacts (données, encoder etc...)
MODEL_DIR = Path("model")
PREDICTION_DIR = Path("predictions")  # Dossier pour enregistrer les prédictions
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)  # Création du


X_VALIDE_PATH = DATA_DIR / "X_test_update.csv"
TFIDF_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
MODEL_PATH = MODEL_DIR / "xgb_model.joblib"


def load_data(X_VALIDE_PATH: Path) -> pd.DataFrame:
    """Charge les données de validation à partir du fichier .csv."""
    X_valid = pd.read_csv(X_VALIDE_PATH, index_col="id")
    return X_valid

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

    # Suppression de la ponctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Suppression des chiffres
    text = re.sub(r'\d+', '', text)

    # Normalisation des espaces après nettoyage
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def built_text(df: pd.DataFrame) -> pd.Series:
    """
    Nettoyage du texte et création de la colonne "text" en concaténant les colonnes "designation" et "description".
    Après nettoyage du corpus avec la fonction clean_text.
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














