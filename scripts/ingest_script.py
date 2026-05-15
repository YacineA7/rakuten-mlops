"""
Script d'ingestion des rawdata qui sera conforme à PEP8.

Ce module illustre une structure claire :
- imports des rawdatas
- Préparation du text avant son nettoyyage 
- Etape de nettoyage du set 
- Ensuite vient l'assemblage de deux colonnes  de texte 
- Supprission des stopwords (EN & FR )
- Encodage de l'ensemble du set 
- Véctorisation du steeming avec TF-IDF 
- Ajouter étape de la sauvegarde de l'ensembledes fonctions 
- Faire appel à la fonction main à la toute fin 
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os
import warnings
import re
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


def clean_text(text):
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

def built_text(df):
    
    '''
    assemblage de deux colonnes: les colonnes description et désignations sont jointes l'uneà l'autres '''

     # Nettoyage du texte simple pour les colonnes de texte
    df['clean_designation'] = df['designation'].apply(clean_text)
    df['clean_description'] = df['description'].apply(clean_text)

    df['text'] = df['clean_designation'] + ' ' + df['clean_description']

    return df["text"]





# "Générique" est un terme générique et non discriminant pour différencier les classes de produits, on l'ajoute à la liste des stopwords
stop_all.add("générique")


def delete_stopwords(text):
    """
    Suppression des mots vides (stopwords)
    """
    return " ".join([w for w in text.split() if w not in stop_all and len(w) > 1])  # Garde mots > 1 caractère

# Application de la suppression des stopwords
clean_data["text_nostop"] = clean_data['text'].apply(delete_stopwords)  # Texte sans stopwords

# Stemming français (réduction des mots à leur racine)
stemmer = SnowballStemmer("french")  # Stemmer optimisé pour le français
def stem_text(text):
    """
    Application du stemming sur chaque mot
    """
    return " ".join([stemmer.stem(w) for w in text.split()])  # Stemming mot par mot 


    

# Préparation des données pour l'analyse de texte
X = clean_data['text_nostop']
y = clean_data['prdtypecode']

# Encodage des labels de la variable cible avec LabelEncoder

def label_encoder(text): 
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

return label_encoder(text)



# Création des ensembles d'entraînement et de test pour l'analyse de texte (20% pour le test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Initialisation de TfidfVectorizer avec des paramètres pour limiter le nombre de features et les n-grams
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

# Vectorisation du texte avec TF-IDF
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)