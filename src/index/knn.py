"""Exact brute-force kNN over AcousticBrainz low-level features. 
Will update to an approximate method later to deal with 
more data.
"""

import argparse

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

NON_FEATURE_COLUMNS = {"mbid"}


def load_features(csv_path):
    """Use every numeric column as a feature - this schema has grown to
    include categorical fields (e.g. tonal.key_key, highlevel.*.value)
    which aren't part of the distance metric, so select by dtype rather
    than maintaining a name-based exclude list."""
    df = pd.read_csv(csv_path)
    feature_columns = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    df = df.dropna(subset=feature_columns).reset_index(drop=True)
    return df, feature_columns


def build_index(df, feature_columns):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_columns])
    nn = NearestNeighbors(metric="euclidean")
    nn.fit(X)
    return scaler, nn


def query_by_mbid(df, feature_columns, scaler, nn, mbid, k=10):
    row = df.index[df["mbid"] == mbid]
    if row.empty:
        raise ValueError(f"mbid {mbid} not found in loaded data")
    X = scaler.transform(df[feature_columns])
    distances, indices = nn.kneighbors(X[row], n_neighbors=k + 1)
    results = [
        (df.iloc[i]["mbid"], dist)
        for dist, i in zip(distances[0], indices[0])
        if df.iloc[i]["mbid"] != mbid
    ]
    return results[:k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/acousticbrainz_sample.csv")
    parser.add_argument("--mbid", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    df, feature_columns = load_features(args.csv)
    scaler, nn = build_index(df, feature_columns)
    for mbid, dist in query_by_mbid(df, feature_columns, scaler, nn, args.mbid, args.k):
        print(f"{dist:.3f}  {mbid}")