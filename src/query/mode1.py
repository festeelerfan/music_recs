"""Mode 1 (segment/descriptor mode), whole-track slice: free-text query
against a MuQ-MuLan-embedded library sample. Segment-level windowing 
to be done later - this ranks whole tracks only."""

import argparse

import pandas as pd
import torch

from src.embed.muq_mulan import embed_text, load_model

DEFAULT_EMBEDDINGS_PATH = "data/processed/library_embeddings.pt"
DEFAULT_TRACKS_PATH = "data/processed/library_tracks.csv"


def query(model, embeds, tracks_df, text, k=10):
    text_embed = embed_text(model, [text])[0]
    scores = model.calc_similarity(embeds, text_embed)
    top = torch.topk(scores, k=min(k, len(tracks_df)))
    return [
        (tracks_df.iloc[int(i)]["artist"], tracks_df.iloc[int(i)]["title"], score.item())
        for score, i in zip(top.values, top.indices)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--tracks", default=DEFAULT_TRACKS_PATH)
    args = parser.parse_args()

    embeds = torch.load(args.embeddings)
    tracks_df = pd.read_csv(args.tracks)
    model = load_model()
    for artist, title, score in query(model, embeds, tracks_df, args.text, args.k):
        print(f"{score:.3f}  {artist} - {title}")