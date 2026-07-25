import math

import pandas as pd
import torch

from src.query.mode2 import _best_match, find_anchor, query_by_example


def test_best_match_finds_close_title_and_artist():
    candidates = [
        {"artist": "Queen", "title": "Bohemian Rhapsody", "path": "a.mp3"},
        {"artist": "Queen", "title": "Don't Stop Me Now", "path": "b.mp3"},
        {"artist": "Someone Else", "title": "Unrelated Song", "path": "c.mp3"},
    ]
    match, score = _best_match(candidates, "bohemian rhapsody", "queen")
    assert match["path"] == "a.mp3"
    assert score > 0.6


def test_best_match_below_threshold_returns_none():
    candidates = [{"artist": "Someone", "title": "Completely Different", "path": "a.mp3"}]
    match, score = _best_match(candidates, "xyzzy plugh qux", "nobody")
    assert match is None


def test_best_match_does_not_prefer_wrong_title_same_artist():
    # regression test: a different song by the right artist previously
    # outscored the correct title because artist/title were blended equally.
    candidates = [
        {"artist": "Justin Timberlake", "title": "Rock Your Body", "path": "a.mp3"},
        {"artist": "Justin Timberlake", "title": "Suit & Tie", "path": "b.mp3"},
    ]
    match, score = _best_match(candidates, "Suit & Tie", "Justin Timberlake")
    assert match["path"] == "b.mp3"


def test_best_match_handles_nan_artist():
    # regression test: a CSV round-trip turns a blank artist field into a
    # float NaN, not an empty string, which crashed the ratio computation.
    candidates = [{"artist": math.nan, "title": "Untitled Track", "path": "a.mp3"}]
    match, score = _best_match(candidates, "Untitled Track", "Some Artist")
    assert match["path"] == "a.mp3"


def test_best_match_works_with_title_only():
    candidates = [
        {"artist": "Queen", "title": "Bohemian Rhapsody", "path": "a.mp3"},
        {"artist": "Other", "title": "Something Else Entirely", "path": "b.mp3"},
    ]
    match, score = _best_match(candidates, "bohemian rhapsody", None)
    assert match["path"] == "a.mp3"


def test_find_anchor_does_not_let_embedded_mediocre_match_shadow_better_full_library_match(
    monkeypatch, tmp_path
):
    # regression test: "Android52 - I Love You" (an exact match, sitting in
    # the full library but not yet embedded) used to get shadowed by
    # "Hercelot - Love You" (a mediocre ~0.69 match that only won because it
    # happened to already be in the smaller embedded subset, which the old
    # tiered lookup checked first and short-circuited on).
    embedded_tracks = pd.DataFrame(
        [{"artist": "Hercelot", "title": "Love You", "path": "hercelot.mp3"}]
    )
    tracks_path = tmp_path / "tracks.csv"
    embedded_tracks.to_csv(tracks_path, index=False)

    full_library = [
        {"artist": "Hercelot", "title": "Love You", "path": "hercelot.mp3"},
        {"artist": "Android52", "title": "I Love You", "path": "android52.mp3"},
    ]
    monkeypatch.setattr(
        "src.query.mode2.iter_library_tracks", lambda root: iter(full_library)
    )

    match, already_embedded = find_anchor(
        "I Love You", "Android52", "fake_root", str(tracks_path)
    )
    assert match["path"] == "android52.mp3"
    assert already_embedded is False


def _shape_fixture():
    anchor = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])  # rises then falls
    similar = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    )  # same shape, stretched
    different = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])  # different order

    embeds = torch.cat([anchor, similar, different], dim=0)
    segments_df = pd.DataFrame(
        {
            "path": ["anchor.mp3"] * 3 + ["similar.mp3"] * 4 + ["different.mp3"] * 3,
            "artist": ["A"] * 3 + ["B"] * 4 + ["C"] * 3,
            "title": ["Anchor"] * 3 + ["Similar"] * 4 + ["Different"] * 3,
            "segment_start_sec": [0, 10, 20] + [0, 10, 20, 30] + [0, 10, 20],
        }
    )
    return embeds, segments_df


def test_query_by_example_excludes_anchor_and_ranks_by_shape():
    embeds, segments_df = _shape_fixture()
    results = query_by_example(embeds, segments_df, "anchor.mp3", k=2)
    titles = [title for _, title, _ in results]
    assert "Anchor" not in titles
    assert titles[0] == "Similar"  # closer DTW shape match than "Different"


def test_track_segments_returns_chronological_order_regardless_of_csv_row_order():
    from src.query.mode2 import _track_segments

    embeds = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    # rows deliberately out of chronological order
    segments_df = pd.DataFrame(
        {
            "path": ["a.mp3", "a.mp3", "a.mp3"],
            "segment_start_sec": [20, 0, 10],
        }
    )
    ordered = _track_segments(embeds, segments_df, "a.mp3")
    # segment_start_sec 0 -> row index 1 -> [1.0, 0.0]; 10 -> index 2 -> [0.5,0.5]; 20 -> index 0 -> [0,1]
    assert torch.equal(ordered, torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]))