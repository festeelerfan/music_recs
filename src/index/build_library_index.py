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


def extend_index(
    library_root,
    additional_n,
    seed=0,
    embeddings_path=DEFAULT_EMBEDDINGS_PATH,
    tracks_path=DEFAULT_TRACKS_PATH,
):
    """Embed `additional_n` tracks not already in the saved index and append
    them, rather than re-embedding everything from scratch. Matches on file
    path (not position) since the underlying filtered track list can change
    between runs (e.g. filter tweaks), which would silently break any
    position-based "resume" scheme."""
    existing_tracks_df = pd.read_csv(tracks_path)
    existing_embeds = torch.load(embeddings_path)
    existing_paths = set(existing_tracks_df["path"])

    candidates = [t for t in iter_library_tracks(library_root) if t["path"] not in existing_paths]
    random.Random(seed).shuffle(candidates)
    new_tracks = candidates[:additional_n]

    model = load_model()
    new_embeds = embed_audio_files(model, [t["path"] for t in new_tracks])

    combined_embeds = torch.cat([existing_embeds, new_embeds], dim=0)
    combined_tracks_df = pd.concat(
        [existing_tracks_df, pd.DataFrame(new_tracks)], ignore_index=True
    )
    return combined_embeds, combined_tracks_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embeddings-out", default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--tracks-out", default=DEFAULT_TRACKS_PATH)
    parser.add_argument(
        "--extend",
        action="store_true",
        help="add --n new tracks to the existing saved index instead of rebuilding from scratch",
    )
    args = parser.parse_args()

    if args.extend:
        embeds, tracks_df = extend_index(
            args.root, args.n, args.seed, args.embeddings_out, args.tracks_out
        )
    else:
        embeds, tracks_df = build_index(args.root, args.n, args.seed)

    torch.save(embeds, args.embeddings_out)
    tracks_df.to_csv(args.tracks_out, index=False)
    print(f"wrote {len(tracks_df)} embeddings to {args.embeddings_out} / {args.tracks_out}")