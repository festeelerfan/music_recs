"""Mode 1 (segment/descriptor mode), segment-level: free-text query against
individual 10s clips rather than whole-track averages, so a match can point
at a specific passage ("this part sounds like X") instead of a diluted
whole-track vibe. See src/query/mode1.py for the whole-track version.
"""

import argparse

import pandas as pd
import torch

from src.embed.muq_mulan import embed_text, load_model

DEFAULT_EMBEDDINGS_PATH = "data/processed/segment_embeddings.pt"
DEFAULT_SEGMENTS_PATH = "data/processed/segment_tracks.csv"


def _format_timestamp(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def query(model, embeds, segments_df, text, k=10, style="moment", top_n=5):
    """Rank distinct songs by their segments, not raw segments - otherwise
    one song's own timestamps can crowd out the rest of the top-k.

    style="moment": score each song by its single best-matching segment -
    for queries about a specific event ("guitar solo", "breakdown").
    style="sustained": score each song by the mean of its top `top_n`
    segments - for queries about a quality that should hold up across a
    meaningful chunk of the song ("melancholic piano"), not just one
    passage. Which style fits a given query is a judgment call left to the
    caller for now (see project notes on query-intent routing).
    """
    text_embed = embed_text(model, [text])[0]
    scores = model.calc_similarity(embeds, text_embed)

    df = segments_df.copy()
    df["score"] = scores.tolist()
    df = df.sort_values("score", ascending=False)

    if style == "moment":
        per_track = df.groupby("path", sort=False).head(1)
    elif style == "sustained":
        top_segments = df.groupby("path", sort=False).head(top_n)
        per_track = top_segments.groupby("path", sort=False, as_index=False).agg(
            {"artist": "first", "title": "first", "segment_start_sec": "first", "score": "mean"}
        )
    else:
        raise ValueError(f"unknown style: {style!r} (expected 'moment' or 'sustained')")

    per_track = per_track.sort_values("score", ascending=False).head(k)
    return [
        (row["artist"], row["title"], row["segment_start_sec"], row["score"])
        for _, row in per_track.iterrows()
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--style", choices=["moment", "sustained"], default="moment")
    parser.add_argument("--top-n", type=int, default=5, help="only used with --style sustained")
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--segments", default=DEFAULT_SEGMENTS_PATH)
    args = parser.parse_args()

    embeds = torch.load(args.embeddings)
    segments_df = pd.read_csv(args.segments)
    model = load_model()
    results = query(model, embeds, segments_df, args.text, args.k, args.style, args.top_n)
    for artist, title, start_sec, score in results:
        print(f"{score:.3f}  {artist} - {title}  @ {_format_timestamp(start_sec)}")