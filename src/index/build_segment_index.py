"""Embed a sample of the local library at segment level (10s clips, MuQ-
MuLan's native window) rather than averaging each track into one vector.
Lets queries match a specific passage instead of a diluted whole-track
average - the actual point of Mode 1 in the project brief.
"""

import argparse
import random

import pandas as pd
import torch

from src.embed.muq_mulan import embed_audio_segments, load_model
from src.ingest.library import iter_library_tracks

DEFAULT_EMBEDDINGS_PATH = "data/processed/segment_embeddings.pt"
DEFAULT_SEGMENTS_PATH = "data/processed/segment_tracks.csv"


def _embed_tracks(model, tracks):
    all_embeds = []
    rows = []
    for track in tracks:
        embeds, starts_sec = embed_audio_segments(model, track["path"])
        all_embeds.append(embeds)
        for start_sec in starts_sec:
            rows.append(
                {
                    "path": track["path"],
                    "artist": track["artist"],
                    "title": track["title"],
                    "segment_start_sec": start_sec,
                }
            )
    return torch.cat(all_embeds, dim=0), pd.DataFrame(rows)


def build_segment_index(library_root, n_tracks, seed=0):
    tracks = list(iter_library_tracks(library_root))
    random.Random(seed).shuffle(tracks)
    tracks = tracks[:n_tracks]

    model = load_model()
    return _embed_tracks(model, tracks)


def extend_segment_index(
    library_root,
    additional_n,
    seed=0,
    embeddings_path=DEFAULT_EMBEDDINGS_PATH,
    segments_path=DEFAULT_SEGMENTS_PATH,
    checkpoint_every=20,
):
    """Embed `additional_n` tracks not already in the saved segment index
    and append them, rather than re-embedding everything from scratch.
    Matches on file path (not position), same reasoning as
    build_library_index.extend_index. Saves to disk every
    `checkpoint_every` tracks (not just once at the end) so a crash or
    interruption during a long run doesn't lose everything - re-running
    with the same arguments picks up where it left off, since already-saved
    paths are excluded from the candidate list."""
    existing_segments_df = pd.read_csv(segments_path)
    existing_embeds = torch.load(embeddings_path)
    existing_paths = set(existing_segments_df["path"])

    candidates = [t for t in iter_library_tracks(library_root) if t["path"] not in existing_paths]
    random.Random(seed).shuffle(candidates)
    new_tracks = candidates[:additional_n]

    model = load_model()
    combined_embeds = existing_embeds
    combined_segments_df = existing_segments_df
    batch = []
    for i, track in enumerate(new_tracks, start=1):
        batch.append(track)
        if len(batch) >= checkpoint_every or i == len(new_tracks):
            new_embeds, new_rows = _embed_tracks(model, batch)
            combined_embeds = torch.cat([combined_embeds, new_embeds], dim=0)
            combined_segments_df = pd.concat([combined_segments_df, new_rows], ignore_index=True)
            torch.save(combined_embeds, embeddings_path)
            combined_segments_df.to_csv(segments_path, index=False)
            print(f"checkpoint: {combined_segments_df['path'].nunique()} tracks total")
            batch = []

    return combined_embeds, combined_segments_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embeddings-out", default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--segments-out", default=DEFAULT_SEGMENTS_PATH)
    parser.add_argument(
        "--extend",
        action="store_true",
        help="add --n new tracks to the existing saved segment index instead of rebuilding from scratch",
    )
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()

    if args.extend:
        embeds, segments_df = extend_segment_index(
            args.root, args.n, args.seed, args.embeddings_out, args.segments_out, args.checkpoint_every
        )
    else:
        embeds, segments_df = build_segment_index(args.root, args.n, args.seed)
    torch.save(embeds, args.embeddings_out)
    segments_df.to_csv(args.segments_out, index=False)
    print(
        f"wrote {len(segments_df)} segments from {segments_df['path'].nunique()} "
        f"tracks to {args.embeddings_out} / {args.segments_out}"
    )