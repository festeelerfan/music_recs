import pandas as pd
import torch

from src.index.build_segment_index import extend_segment_index


def _fake_embed_audio_segments(model, path):
    # 2 fixed segments per track, embedding value derived from path so
    # different tracks are distinguishable.
    val = float(len(path))
    return torch.tensor([[val, 0.0], [val, 0.0]]), [0, 10]


def _patch(monkeypatch, candidates):
    monkeypatch.setattr(
        "src.index.build_segment_index.iter_library_tracks", lambda root: iter(candidates)
    )
    monkeypatch.setattr("src.index.build_segment_index.load_model", lambda: object())
    monkeypatch.setattr(
        "src.index.build_segment_index.embed_audio_segments", _fake_embed_audio_segments
    )


def test_extend_segment_index_checkpoints_correctly(tmp_path, monkeypatch):
    embeddings_path = tmp_path / "embeds.pt"
    segments_path = tmp_path / "segments.csv"
    torch.save(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), embeddings_path)
    pd.DataFrame(
        [
            {"path": "existing.mp3", "artist": "A", "title": "Existing", "segment_start_sec": 0},
            {"path": "existing.mp3", "artist": "A", "title": "Existing", "segment_start_sec": 10},
        ]
    ).to_csv(segments_path, index=False)

    candidates = [{"path": f"track{i}.mp3", "artist": "X", "title": f"Track {i}"} for i in range(5)]
    _patch(monkeypatch, candidates)

    embeds, segments_df = extend_segment_index(
        "fake_root",
        additional_n=5,
        embeddings_path=str(embeddings_path),
        segments_path=str(segments_path),
        checkpoint_every=2,
    )

    assert segments_df["path"].nunique() == 6  # existing + 5 new
    assert embeds.shape[0] == 12  # 6 tracks * 2 segments each

    # the checkpointed on-disk state should match the in-memory result -
    # a crash right after this call would still leave everything saved.
    reloaded_df = pd.read_csv(segments_path)
    reloaded_embeds = torch.load(embeddings_path)
    assert reloaded_df["path"].nunique() == 6
    assert reloaded_embeds.shape[0] == 12


def test_extend_segment_index_resume_does_not_reprocess_or_duplicate(tmp_path, monkeypatch):
    embeddings_path = tmp_path / "embeds.pt"
    segments_path = tmp_path / "segments.csv"
    torch.save(torch.empty(0, 2), embeddings_path)
    pd.DataFrame(columns=["path", "artist", "title", "segment_start_sec"]).to_csv(
        segments_path, index=False
    )

    candidates = [{"path": f"track{i}.mp3", "artist": "X", "title": f"Track {i}"} for i in range(3)]
    _patch(monkeypatch, candidates)

    # simulate a run that only got through 2 of 3 tracks before "crashing"
    extend_segment_index(
        "fake_root", additional_n=2, embeddings_path=str(embeddings_path),
        segments_path=str(segments_path), checkpoint_every=1,
    )
    assert pd.read_csv(segments_path)["path"].nunique() == 2

    # "resume": should pick up exactly the remaining track, no duplicates
    extend_segment_index(
        "fake_root", additional_n=5, embeddings_path=str(embeddings_path),
        segments_path=str(segments_path), checkpoint_every=1,
    )
    final_df = pd.read_csv(segments_path)
    assert final_df["path"].nunique() == 3
    assert set(final_df["path"].unique()) == {"track0.mp3", "track1.mp3", "track2.mp3"}