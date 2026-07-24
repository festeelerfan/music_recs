from src.ingest import library


class _FakeInfo:
    def __init__(self, length):
        self.length = length


class _FakeAudio:
    def __init__(self, length, tags):
        self.info = _FakeInfo(length)
        self.tags = tags


def _patch_tracks(monkeypatch, tmp_path, fake_tracks):
    """fake_tracks: dict of filename -> _FakeAudio (or None for unreadable)."""
    for name in fake_tracks:
        (tmp_path / name).write_bytes(b"")

    def fake_read_tags(path):
        return fake_tracks.get(path.name)

    monkeypatch.setattr(library, "_read_tags", fake_read_tags)


def test_filters_too_short_and_too_long(monkeypatch, tmp_path):
    fake_tracks = {
        "short.mp3": _FakeAudio(10, {"artist": ["A"], "title": ["Short"]}),
        "ok.mp3": _FakeAudio(180, {"artist": ["A"], "title": ["Ok"]}),
        "long.mp3": _FakeAudio(1300, {"artist": ["A"], "title": ["Long"]}),
    }
    _patch_tracks(monkeypatch, tmp_path, fake_tracks)

    titles = [t["title"] for t in library.iter_library_tracks(tmp_path)]
    assert titles == ["Ok"]


def test_filters_excluded_artist(monkeypatch, tmp_path):
    fake_tracks = {
        "prank.mp3": _FakeAudio(60, {"artist": ["Longmont Potion Castle"], "title": ["Call"]}),
        "music.mp3": _FakeAudio(60, {"artist": ["Real Artist"], "title": ["Song"]}),
    }
    _patch_tracks(monkeypatch, tmp_path, fake_tracks)

    titles = [t["title"] for t in library.iter_library_tracks(tmp_path)]
    assert titles == ["Song"]


def test_skips_blank_title_but_allows_blank_artist(monkeypatch, tmp_path):
    fake_tracks = {
        "no_title.mp3": _FakeAudio(60, {"artist": ["Someone"], "title": [""]}),
        "no_artist.mp3": _FakeAudio(60, {"artist": [""], "title": ["Untitled Track"]}),
    }
    _patch_tracks(monkeypatch, tmp_path, fake_tracks)

    results = list(library.iter_library_tracks(tmp_path))
    assert len(results) == 1
    assert results[0]["title"] == "Untitled Track"
    assert results[0]["artist"] == ""


def test_skips_unreadable_files(monkeypatch, tmp_path):
    fake_tracks = {"broken.mp3": None}
    _patch_tracks(monkeypatch, tmp_path, fake_tracks)

    assert list(library.iter_library_tracks(tmp_path)) == []