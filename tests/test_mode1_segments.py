import pandas as pd
import torch

from src.query.mode1_segments import _format_timestamp, query


class _FakeModel:
    def calc_similarity(self, audio_latents, text_latents):
        return torch.einsum("i d, d -> i", audio_latents, text_latents)


def test_format_timestamp():
    assert _format_timestamp(0) == "0:00"
    assert _format_timestamp(65) == "1:05"
    assert _format_timestamp(260) == "4:20"


def _fixture():
    # Song1: one huge spike (a "moment") among otherwise low segments.
    # Song2: consistently decent scores throughout (a "sustained" quality).
    embeds = torch.tensor(
        [
            [1.0, 0.0],  # Song1 seg 0 - spike
            [0.1, 0.0],  # Song1 seg 10
            [0.1, 0.0],  # Song1 seg 20
            [0.6, 0.0],  # Song2 seg 0
            [0.6, 0.0],  # Song2 seg 10
            [0.6, 0.0],  # Song2 seg 20
        ]
    )
    segments_df = pd.DataFrame(
        {
            "path": ["song1.mp3"] * 3 + ["song2.mp3"] * 3,
            "artist": ["A"] * 3 + ["B"] * 3,
            "title": ["Song1"] * 3 + ["Song2"] * 3,
            "segment_start_sec": [0, 10, 20, 0, 10, 20],
        }
    )
    return embeds, segments_df


def test_moment_style_ranks_by_single_best_segment(monkeypatch):
    embeds, segments_df = _fixture()
    monkeypatch.setattr(
        "src.query.mode1_segments.embed_text", lambda model, texts: torch.tensor([[1.0, 0.0]])
    )
    results = query(_FakeModel(), embeds, segments_df, "q", k=2, style="moment")
    # Song1's spike (1.0) beats Song2's best single segment (0.6)
    assert results[0][:3] == ("A", "Song1", 0)
    assert results[1][:3] == ("B", "Song2", 0)


def test_sustained_style_prefers_consistent_song(monkeypatch):
    embeds, segments_df = _fixture()
    monkeypatch.setattr(
        "src.query.mode1_segments.embed_text", lambda model, texts: torch.tensor([[1.0, 0.0]])
    )
    results = query(_FakeModel(), embeds, segments_df, "q", k=2, style="sustained", top_n=3)
    # mean(1.0, 0.1, 0.1) = 0.4 for Song1 vs mean(0.6, 0.6, 0.6) = 0.6 for Song2
    assert results[0][:2] == ("B", "Song2")
    assert results[0][3] > results[1][3]


def test_query_does_not_duplicate_same_track(monkeypatch):
    embeds, segments_df = _fixture()
    monkeypatch.setattr(
        "src.query.mode1_segments.embed_text", lambda model, texts: torch.tensor([[1.0, 0.0]])
    )
    results = query(_FakeModel(), embeds, segments_df, "q", k=10, style="moment")
    titles = [title for _, title, _, _ in results]
    assert len(titles) == len(set(titles))