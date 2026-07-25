"""Mode 2 (artist + favorites), single-anchor version: "find me songs like
<title> by <artist>" - query-by-example over your own library, since we
have no legal source for full audio of songs outside it (see project
audio-sourcing constraints).

Similarity here is intentionally whole-song, not moment-based (unlike Mode
1's segment styles) - but "whole-song" is computed via Dynamic Time Warping
over each track's full segment sequence, not a single mean-pooled vector.
Mean-pooling discards how a song's character evolves over its duration (its
"shape"); DTW compares the shapes directly. Uses the same segment-level
corpus as mode1_segments.py.
"""

import argparse
import difflib

import pandas as pd
import torch

from src.embed.muq_mulan import embed_audio_segments, load_model
from src.index.dtw import dtw_distance
from src.ingest.library import iter_library_tracks

DEFAULT_EMBEDDINGS_PATH = "data/processed/segment_embeddings.pt"
DEFAULT_SEGMENTS_PATH = "data/processed/segment_tracks.csv"


def _clean(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip().lower()


def _ratio(a, b):
    return difflib.SequenceMatcher(None, _clean(a), _clean(b)).ratio()


def _best_match(candidates, title, artist=None, threshold=0.6):
    """Title carries most of the weight - artist is a secondary signal,
    not blended in equally. Otherwise a wrong-title/same-artist candidate
    (e.g. a different song by the same artist) can outscore the real match
    purely on the strength of the artist name matching."""
    best, best_score = None, 0.0
    for c in candidates:
        title_score = _ratio(title, c.get("title"))
        score = 0.7 * title_score + 0.3 * _ratio(artist, c.get("artist")) if artist else title_score
        if score > best_score:
            best_score, best = score, c
    return (best, best_score) if best_score >= threshold else (None, best_score)


def find_anchor(title, artist, library_root, segments_path=DEFAULT_SEGMENTS_PATH):
    """Find the requested track by scanning the full library, not just the
    already-embedded subset - a mediocre match that happens to already be
    embedded must never shadow a much better match sitting elsewhere in the
    library. Returns (track_dict_or_None, already_embedded)."""
    all_candidates = list(iter_library_tracks(library_root))
    match, _ = _best_match(all_candidates, title, artist)
    if match is None:
        return None, False

    embedded_paths = set(pd.read_csv(segments_path)["path"])
    return match, match["path"] in embedded_paths


def _track_segments(embeds, segments_df, path):
    """A track's segment embeddings, in chronological order."""
    rows = segments_df[segments_df["path"] == path].sort_values("segment_start_sec")
    return embeds[rows.index.to_numpy()]


def query_by_example(embeds, segments_df, anchor_path, k=5):
    anchor_segments = _track_segments(embeds, segments_df, anchor_path)
    if len(anchor_segments) == 0:
        raise ValueError(f"{anchor_path} not found among embedded tracks")

    results = []
    for path in segments_df["path"].unique():
        if path == anchor_path:
            continue
        candidate_segments = _track_segments(embeds, segments_df, path)
        distance = dtw_distance(anchor_segments, candidate_segments)
        row = segments_df[segments_df["path"] == path].iloc[0]
        results.append((row["artist"], row["title"], distance))

    results.sort(key=lambda r: r[2])  # lower distance = more similar
    return results[:k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--artist", default=None)
    parser.add_argument("--root", default="/Users/john/Music/iTunes/iTunes Media/Music")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--segments", default=DEFAULT_SEGMENTS_PATH)
    args = parser.parse_args()

    match, already_embedded = find_anchor(args.title, args.artist, args.root, args.segments)
    if match is None:
        label = f"'{args.title}'" + (f" by {args.artist}" if args.artist else "")
        print(f"No match found in your library for {label}")
        raise SystemExit(1)

    print(f"anchor: {match['artist']} - {match['title']}", "(already embedded)" if already_embedded else "(embedding now)")

    segments_df = pd.read_csv(args.segments)
    embeds = torch.load(args.embeddings)

    if not already_embedded:
        model = load_model()
        new_segment_embeds, starts_sec = embed_audio_segments(model, match["path"])
        embeds = torch.cat([embeds, new_segment_embeds], dim=0)
        new_rows = pd.DataFrame(
            [
                {
                    "path": match["path"],
                    "artist": match["artist"],
                    "title": match["title"],
                    "segment_start_sec": s,
                }
                for s in starts_sec
            ]
        )
        segments_df = pd.concat([segments_df, new_rows], ignore_index=True)
        torch.save(embeds, args.embeddings)
        segments_df.to_csv(args.segments, index=False)

    for artist, title, distance in query_by_example(embeds, segments_df, match["path"], args.k):
        print(f"{distance:.3f}  {artist} - {title}")