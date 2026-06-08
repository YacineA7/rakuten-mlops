"""
Tests unitaires pour les fonctions des différents scripts du projet
pour les fonctions de nettoyage, de vectorisation, de prédiction et de décodage des classes.
"""


import pandas as pd
from pathlib import Path
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pytest
import scripts.predict_script as ps


from scripts.predict_script import (
    clean_text,
    built_text,
    delete_stopwords,
    stem_text,
    load_test_data,
    vectorize_text,
    predict_classes,
    decode_predictions,
)


def test_clean_text() -> None:
    """Vérifie la normalisation de base du texte dans clean_text."""
    text = "<p>Hello &amp; world! 123</p>"
    expected = "hello world"
    result = ps.clean_text(text)
    assert result == expected


def test_clean_text_null() -> None:
    """Vérifie que clean_text renvoie une chaîne vide pour une valeur nulle."""
    result = ps.clean_text(None)
    assert result == ""


def test_built_text() -> None:
    """Vérifie que built_text concatène correctement designation et description."""
    df = pd.DataFrame({
        "designation": ["Produit A", "Produit B"],
        "description": ["Description A", "Description B"],
    })

    result = ps.built_text(df)
    expected = pd.Series(["produit a description a", "produit b description b"], name='text')
    pd.testing.assert_series_equal(result, expected)


def test_delete_stopwords_custom_set() -> None:
    """Vérifie la suppression des stopwords à partir d'un ensemble personnalisé."""
    stop_set = {"le", "la", "un", "generique"}
    text = "le produit generique est bon"
    expected = "produit est bon"
    result = ps.delete_stopwords(text, stop_set)
    assert result == expected


def test_stem_text() -> None:
    """Vérifie que stem_text applique correctement le stemming."""
    stemmer = SnowballStemmer("french")
    text = "manger maisons"
    expected = "mang maison"
    result = ps.stem_text(text, stemmer)
    assert result == expected


def test_load_test_data_with_id(tmp_path) -> None:
    """Vérifie que load_test_data lit un fichier CSV et utilise la colonne id comme index."""
    csv_path = tmp_path / "X_test_update.csv"

    df = pd.DataFrame({"id": [1, 2], "text": ["A", "B"]})
    df.to_csv(csv_path, index=False)

    result = ps.load_test_data(csv_path)

    expected = pd.DataFrame({"text": ["A", "B"]}, index=pd.Index([1, 2], name="id"))
    pd.testing.assert_frame_equal(result, expected)


def test_vectorize_text() -> None:
    """Vérifie que vectorize_text retourne une matrice de la bonne forme."""
    tfidf = TfidfVectorizer()
    tfidf.fit(["produit bon", "produit mauvais"])
    corpus = pd.Series(["produit bon"])
    X = vectorize_text(corpus, tfidf)
    assert X.shape[0] == 1


def test_predict_classes(): 
    """Vérifie la fonction predict_classes avec un modèle factice."""
    class DummyModel:
        def predict(self, X):
            return [1]

    X = [[0, 1, 0]]
    y = predict_classes(DummyModel(), X)
    assert list(y) == [1]


def test_decode_predictions():
    """Vérifie que decode_predictions décode correctement les classes prédites."""
    le = LabelEncoder()
    le.fit([10, 20, 30])
    y = decode_predictions([1], le)
    assert list(y) == [20]
