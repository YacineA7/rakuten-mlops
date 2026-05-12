import tempfile
from pathlib import Path

import pandas as pd

from scripts import ingest_script


def test_load_rawdata_merges_and_sets_index() -> None:
    """Vérifie que load_rawdata fusionne les fichiers et définit l'index sur la colonne id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        x_path = tmpdir_path / "X_train.csv"
        y_path = tmpdir_path / "Y_train.csv"

        x_df = pd.DataFrame({"feature": [1, 2, 3]}, index=[10, 20, 30])
        y_df = pd.DataFrame({"label": ["a", "b", "c"]}, index=[10, 20, 30])

        x_df.to_csv(x_path)
        y_df.to_csv(y_path)

        result = ingest_script.load_rawdata(x_path, y_path)

        expected = pd.DataFrame(
            {"feature": [1, 2, 3], "label": ["a", "b", "c"]},
            index=pd.Index([10, 20, 30], name="id"),
        )

        pd.testing.assert_frame_equal(result, expected)
