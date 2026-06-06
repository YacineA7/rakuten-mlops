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
import joblib
import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from sklearn.pipeline import Pipeline
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
from sklearn.metrics import ConfusionMatrixDisplay
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/raw")
ARTIFACTS_DIR = Path("artifacts")  # Dossier contenant les artéfacts (données, encoder etc...)
MODEL_DIR = Path("models")
PREDICTION_DIR = Path("predictions")  # Dossier pour enregistrer les prédictions
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)  # Création du

X_VALID_PATH = DATA_DIR / "X_test_update.csv"
TFIDF_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
MODEL_PATH = MODEL_DIR / "xgb_model.joblib"


def load_test_data(X_VALID_PATH: Path) -> pd.DataFrame:
    """Charge les données de validation à partir du fichier .csv."""
    df = pd.read_csv(X_VALID_PATH)

    if "id" in df.columns:
        df = df.set_index("id")
    elif "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "id"})
        df = df.set_index("id")
    else:
        raise ValueError("Le fichier de validation doit contenir une colonne 'id' ou 'Unnamed: 0' pour l'index.")
    
    return df

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


def load_prediction_artifacts():
    """
    Charge :
    - le vectorizer TF-IDF entraîné sur train
    - le label encoder appris sur y_train
    - le modèle XGBoost entraîné
    """
    tfidf = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    model = joblib.load(MODEL_PATH)

    return tfidf, label_encoder, model


def vectorize_text(corpus: pd.Series, tfidf):
    """
    Vectorisation du texte avec TfidfVectorizer, 
    Retourne les matrices TF-IDF pour la validation.
    Retourne aussi le TfidfVectorizer pour pouvoir faire la même transformation sur les données de test et dans l'API plus tard.
    """

    # Vectorisation du texte avec TF-IDF
    X_valid_tfidf = tfidf.transform(corpus)

    return  X_valid_tfidf


def predict_classes(model, X_valid_tfidf):
    """
    Applique le modèle XGBoost sur les données test vectorisées.
    Avec multi:softmax, le résultat est directement un vecteur
    de classes encodées (0 à 26).
    """
    y_pred_encoded = model.predict(X_valid_tfidf)

    print(f"[PREDICT] Prédictions générées : {len(y_pred_encoded)}")
    return y_pred_encoded


def decode_predictions(y_pred_encoded, label_encoder):
    """
    Décodage des classes prédites de leur format encodé (0 à 26) à leur format original (noms de catégories).
    """
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    print(f"[PREDICT] Prédictions décodées : {len(y_pred_labels)}")
    return y_pred_labels


def save_predictions(df_test: pd.DataFrame, y_pred_labels: np.ndarray):
    """
    Enregistre les prédictions dans un fichier CSV.
    Le fichier contiendra les colonnes "id" et "prdtypecode".
    """
    predictions_df = pd.DataFrame({
        "id": df_test.index,
        "prdtypecode": y_pred_labels
    })

    output_path = PREDICTION_DIR / "predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"[PREDICT] Prédictions enregistrées dans : {output_path}")


def main():
    # Charger les données de validation
    df_test = load_test_data(X_VALID_PATH)

    # Construire l'ensemble des stopwords
    stop_set = build_stopword_set()

    # Nettoyer le texte et construire la colonne "text"
    corpus = built_text(df_test)

    # Supprimer les stopwords du corpus
    corpus_no_stop = corpus.apply(lambda x: delete_stopwords(x, stop_set))

    # Appliquer le stemming français sur le corpus nettoyé
    stemmer = SnowballStemmer("french")
    corpus_stemmed = corpus_no_stop.apply(lambda x: stem_text(x, stemmer))

    # Charger les artéfacts nécessaires à la prédiction
    tfidf, label_encoder, model = load_prediction_artifacts()

    # Vectoriser le texte de validation avec le TF-IDF chargé
    X_valid_tfidf = vectorize_text(corpus_stemmed, tfidf)

    # Générer les prédictions encodées avec le modèle XGBoost
    y_pred_encoded = predict_classes(model, X_valid_tfidf)

    # Décoder les prédictions pour obtenir les noms de catégories
    y_pred_labels = decode_predictions(y_pred_encoded, label_encoder)

    # Enregistrer les prédictions dans un fichier CSV
    save_predictions(df_test, y_pred_labels)


if __name__ == "__main__":
    main()