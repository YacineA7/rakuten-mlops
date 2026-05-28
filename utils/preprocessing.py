"""
Module de prétraitement du texte.
Ce module contient les fonctions nécessaires pour nettoyer, prétraiter et construire le corpus de texte 
à partir des champs "designation" et "description" du DataFrame source.
Il inclut des étapes de nettoyage du texte, de suppression des stopwords et de stemming, 
pour préparer les données textuelles avant la vectorisation et l'entraînement du modèle.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords", quiet=True)

STOP_FR = set(stopwords.words("french"))
STOP_EN = set(stopwords.words("english"))
STOP_SET = STOP_FR.union(STOP_EN)
STOP_SET.add("generique")

STEMMER = SnowballStemmer("french")


def clean_text(text) -> str:
    if pd.isnull(text):
        return ""

    text = re.sub(r"<.*?>", " ", str(text))
    text = text.replace("<br />", " ")

    text = text.replace("&amp;", " ")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", " ")
    text = text.replace("&gt;", " ")
    text = text.replace("&quot;", " ")
    text = text.replace("&#39;", " ")
    text = text.replace("&eacute;", "e")
    text = text.replace("&egrave;", "e")
    text = text.replace("&ecirc;", "e")

    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str) -> str:
    """
    Suppression des mots vides (stopwords)
    """
    tokens = [w for w in text.split() if w not in STOP_SET and len(w) > 1]
    return " ".join(tokens)


def stem_text(text: str) -> str:
    """
    Applique le stemming français.
    """
    tokens = [STEMMER.stem(w) for w in text.split()]
    return " ".join(tokens)


def preprocess_product_text(designation: str, description: str) -> str:
    """
    Nettoyage du texte pour le traitement NLP,
    Création d'un champ "full_text" combinant la désignation et la description, après nettoyage, suppression des stopwords et stemming.
    """
    clean_designation = clean_text(designation)
    clean_description = clean_text(description)

    full_text = f"{clean_designation} {clean_description}".strip()
    full_text = remove_stopwords(full_text)
    full_text = stem_text(full_text)

    return full_text


def build_corpus(df: pd.DataFrame) -> pd.Series:
    """"
    Construit le corpus préprocessé à partir du DataFrame source.
    """
    return df.apply(
        lambda row: preprocess_product_text(
            row.get("designation", ""),
            row.get("description", "")
        ),
        axis=1
    )