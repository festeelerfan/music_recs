"""Embed a random sample of the local library with MuQ-MuLan and save the
result (embeddings + aligned track metadata) for free-text querying."""

import argparse
import random

import pandas as pd
import torch

from src.embed.muq_mulan import embed_audio_files, load_model
from src.ingest.library import iter_library_tracks

DEFAULT_EMBEDDINGS_PATH = "data/processed/library_embeddings.pt"
DEFAULT_TRACKS_PATH = "data/processed/library_tracks.csv"


def build_index(library_root, n_tracks, seed=0):
    tracks = list(iter_library_tracks(library_root))
    random.Random(seed).shuffle(tracks)
    tracks = tracks[:n_tracks]

    model = load_model()
    embeds = embed_audio_files(model, [t["path"] for t in tracks])
    return embeds, pd.DataFrame(tracks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embeddings-out", default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--tracks-out", default=DEFAULT_TRACKS_PATH)
    args = parser.parse_args()

    embeds, tracks_df = build_index(args.root, args.n, args.seed)
    torch.save(embeds, args.embeddings_out)
    tracks_df.to_csv(args.tracks_out, index=False)
    print(f"wrote {len(tracks_df)} embeddings to {args.embeddings_out} / {args.tracks_out}")