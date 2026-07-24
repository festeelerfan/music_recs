import pandas as pd

from src.index.faiss_index import build_index, load_index, query_by_mbid, save_index
from src.index.knn import load_features


def _synthetic_df():
    # 4 tracks on a line: "a" and "b" are close, "c" and "d" are far away.
    return pd.DataFrame(
        {
            "mbid": ["a", "b", "c", "d"],
            "tonal.key_key": ["C", "C", "D", "D"],
            "feat1": [0.0, 0.1, 10.0, 10.1],
            "feat2": [0.0, 0.1, 10.0, 10.1],
        }
    )


def test_query_by_mbid_finds_nearest_and_excludes_self(tmp_path):
    csv_path = tmp_path / "features.csv"
    _synthetic_df().to_csv(csv_path, index=False)

    df, feature_columns = load_features(csv_path)
    scaler, index = build_index(df, feature_columns)

    results = query_by_mbid(df, feature_columns, scaler, index, "a", k=3)
    result_mbids = [mbid for mbid, _ in results]

    assert "a" not in result_mbids
    assert result_mbids[0] == "b"


def test_save_and_load_index_roundtrip(tmp_path):
    csv_path = tmp_path / "features.csv"
    _synthetic_df().to_csv(csv_path, index=False)

    df, feature_columns = load_features(csv_path)
    scaler, index = build_index(df, feature_columns)

    index_path = str(tmp_path / "index.faiss")
    scaler_path = str(tmp_path / "scaler.joblib")
    save_index(index, scaler, index_path, scaler_path)

    loaded_scaler, loaded_index = load_index(index_path, scaler_path)
    results = query_by_mbid(df, feature_columns, loaded_scaler, loaded_index, "a", k=3)

    assert [mbid for mbid, _ in results][0] == "b"