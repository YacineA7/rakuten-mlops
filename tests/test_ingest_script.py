import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from predict_script import (
    clean_text,
    built_text,
    delete_stopwords,
    stem_text,
    load_test_data,
    vectorize_text,
    predict_classes,
    decode_predictions,
    save_predictions,
)

def test_clean_text():
    text = "<p>Hello</p> https://site.com 123 !"
    assert clean_text(text) == "hello"

def test_built_text():
    df = pd.DataFrame({
        "designation": ["Mon Produit"],
        "description": ["Très Bien"]
    })
    result = built_text(df)
    assert len(result) == 1
    assert "mon produit" in result.iloc[0]
    assert "très bien" in result.iloc[0]

def test_delete_stopwords():
    stop_set = {"le", "de", "generique"}
    assert delete_stopwords("le produit generique bon", stop_set) == "produit bon"

def test_stem_text():
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer("french")
    result = stem_text("manger maisons", stemmer)
    assert isinstance(result, str)
    assert len(result.split()) == 2

def test_load_test_data_with_id(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({
        "id": [1, 2],
        "designation": ["a", "b"],
        "description": ["c", "d"]
    }).to_csv(csv_path, index=False)

    df = load_test_data(csv_path)
    assert list(df.index) == [1, 2]

def test_load_test_data_with_unnamed_index(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({
        "Unnamed: 0": [10, 20],
        "designation": ["a", "b"],
        "description": ["c", "d"]
    }).to_csv(csv_path, index=False)

    df = load_test_data(csv_path)
    assert list(df.index) == [10, 20]

def test_vectorize_text():
    """"""
    tfidf = TfidfVectorizer()
    tfidf.fit(["produit bon", "produit mauvais"])
    corpus = pd.Series(["produit bon"])
    X = vectorize_text(corpus, tfidf)
    assert X.shape[0] == 1

def test_predict_classes():
    class DummyModel:
        def predict(self, X):
            return [1]

    X = [[0, 1, 0]]
    y = predict_classes(DummyModel(), X)
    assert list(y) == [1]

def test_decode_predictions():
    
    le = LabelEncoder()
    le.fit([10, 20, 30])
    y = decode_predictions([1], le)
    assert list(y) == [20]

def test_save_predictions(tmp_path, monkeypatch):
    import predict_script
    monkeypatch = None