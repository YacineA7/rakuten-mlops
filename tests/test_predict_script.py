import tempfile
from pathlib import Path

import pandas as pd

from scripts import predict_script


def test_load_data_reads_csv_and_sets_index() -> None:
    """Vérifie que load_data lit un fichier CSV et utilise la colonne id comme index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "X_test_update.csv"

        df = pd.DataFrame({"id": [1, 2], "text": ["A", "B"]})
        df.to_csv(csv_path, index=False)

        result = predict_script.load_data(csv_path)

        expected = pd.DataFrame({"text": ["A", "B"]}, index=pd.Index([1, 2], name="id"))
        pd.testing.assert_frame_equal(result, expected)


def test_clean_text_basic() -> None:
    """Vérifie la normalisation de base du texte dans clean_text."""
    text = "<p>Hello &amp; world! 123</p>"
    expected = "hello world"
    result = predict_script.clean_text(text)
    assert result == expected


def test_clean_text_null() -> None:
    """Vérifie que clean_text renvoie une chaîne vide pour une valeur nulle."""
    result = predict_script.clean_text(None)
    assert result == ""


def test_built_text() -> None:
    """Vérifie que built_text concatène correctement designation et description."""
    df = pd.DataFrame({
        "designation": ["Produit A", "Produit B"],
        "description": ["Description A", "Description B"],
    })

    result = predict_script.built_text(df)
    expected = pd.Series(["produit a description a", "produit b description b"], name='text')
    pd.testing.assert_series_equal(result, expected)


def test_delete_stopwords_custom_set() -> None:
    """Vérifie la suppression des stopwords à partir d'un ensemble personnalisé."""
    stop_set = {"le", "la", "un", "generique"}
    text = "le produit generique est bon"
    expected = "produit est bon"
    result = predict_script.delete_stopwords(text, stop_set)
    assert result == expected
