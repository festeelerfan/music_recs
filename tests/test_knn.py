import pandas as pd

from src.index.knn import NON_FEATURE_COLUMNS, build_index, load_features, query_by_mbid


def _synthetic_df():
    # 4 tracks on a line: "a" and "b" are close, "c" and "d" are far away.
    return pd.DataFrame(
        {
            "mbid": ["a", "b", "c", "d"],
            "tonal.key_key": ["C", "C", "D", "D"],
            "tonal.key_scale": ["major"] * 4,
            "feat1": [0.0, 0.1, 10.0, 10.1],
            "feat2": [0.0, 0.1, 10.0, 10.1],
        }
    )


def test_non_feature_columns_excluded(tmp_path):
    csv_path = tmp_path / "features.csv"
    _synthetic_df().to_csv(csv_path, index=False)

    df, feature_columns = load_features(csv_path)
    assert set(feature_columns) == {"feat1", "feat2"}
    assert "mbid" not in feature_columns
    assert NON_FEATURE_COLUMNS.isdisjoint(feature_columns)


def test_query_by_mbid_finds_nearest_and_excludes_self(tmp_path):
    csv_path = tmp_path / "features.csv"
    _synthetic_df().to_csv(csv_path, index=False)

    df, feature_columns = load_features(csv_path)
    scaler, nn = build_index(df, feature_columns)

    results = query_by_mbid(df, feature_columns, scaler, nn, "a", k=3)
    result_mbids = [mbid for mbid, _ in results]

    assert "a" not in result_mbids  # never recommend the query track itself
    assert result_mbids[0] == "b"  # closest point on the synthetic line


def test_query_by_mbid_missing_mbid_raises(tmp_path):
    csv_path = tmp_path / "features.csv"
    _synthetic_df().to_csv(csv_path, index=False)

    df, feature_columns = load_features(csv_path)
    scaler, nn = build_index(df, feature_columns)

    try:
        query_by_mbid(df, feature_columns, scaler, nn, "nonexistent", k=3)
        assert False, "expected ValueError"
    except ValueError:
        pass