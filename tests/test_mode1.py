import pandas as pd
import torch

from src.query.mode1 import query


class _FakeModel:
    def calc_similarity(self, audio_latents, text_latents):
        return torch.einsum("i d, d -> i", audio_latents, text_latents)


def test_query_ranks_by_similarity_and_handles_tensor_indices(monkeypatch):
    # regression test: torch.topk indices are 0-d tensors, not plain ints -
    # tracks_df.iloc[] needs an explicit int() conversion (previously a bug).
    embeds = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [-1.0, 0.0],
        ]
    )
    tracks_df = pd.DataFrame(
        {"artist": ["A", "B", "C", "D"], "title": ["t1", "t2", "t3", "t4"]}
    )

    monkeypatch.setattr(
        "src.query.mode1.embed_text", lambda model, texts: torch.tensor([[1.0, 0.0]])
    )

    results = query(_FakeModel(), embeds, tracks_df, "some query", k=2)

    assert [artist for artist, _, _ in results] == ["A", "C"]
    assert results[0][2] > results[1][2]  # scores sorted descending