"""Approximate nearest-neighbor search over AcousticBrainz features via
FAISS, for corpus sizes where brute-force kNN (src/index/knn.py) becomes
too slow. HNSW needs no training step (unlike IVF-family indexes), which
keeps this a straightforward drop-in swap at this dataset's scale.
"""

import argparse

import faiss
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.index.knn import load_features

HNSW_M = 32  # graph connectivity - standard default, good recall/speed tradeoff


def build_index(df, feature_columns, m=HNSW_M):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_columns]).astype(np.float32)
    index = faiss.IndexHNSWFlat(X.shape[1], m)
    index.add(X)
    return scaler, index


def query_by_mbid(df, feature_columns, scaler, index, mbid, k=10):
    row = df.index[df["mbid"] == mbid]
    if row.empty:
        raise ValueError(f"mbid {mbid} not found in loaded data")
    X = scaler.transform(df[feature_columns]).astype(np.float32)
    distances, indices = index.search(X[row.to_numpy()], k + 1)
    results = [
        (df.iloc[i]["mbid"], dist)
        for dist, i in zip(distances[0], indices[0])
        if df.iloc[i]["mbid"] != mbid
    ]
    return results[:k]


def save_index(index, scaler, index_path, scaler_path):
    faiss.write_index(index, index_path)
    joblib.dump(scaler, scaler_path)


def load_index(index_path, scaler_path):
    return joblib.load(scaler_path), faiss.read_index(index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/acousticbrainz_bulk_1m.csv")
    parser.add_argument("--mbid", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--save-index", default=None)
    parser.add_argument("--save-scaler", default=None)
    args = parser.parse_args()

    df, feature_columns = load_features(args.csv)
    print(f"loaded {len(df)} rows, {len(feature_columns)} feature columns")
    scaler, index = build_index(df, feature_columns)

    if args.save_index and args.save_scaler:
        save_index(index, scaler, args.save_index, args.save_scaler)

    for mbid, dist in query_by_mbid(df, feature_columns, scaler, index, args.mbid, args.k):
        print(f"{dist:.3f}  {mbid}")