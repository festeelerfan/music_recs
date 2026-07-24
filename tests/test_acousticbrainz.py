from src.ingest.acousticbrainz import (
    _flatten,
    _get,
    iter_bulk_records,
    iter_bulk_records_with_highlevel,
)


def test_get_nested_path():
    doc = {"a": {"b": {"c": 42}}}
    assert _get(doc, ("a", "b", "c")) == 42


def test_get_missing_key_returns_none():
    doc = {"a": {"b": 1}}
    assert _get(doc, ("a", "x", "c")) is None
    assert _get(doc, ("z",)) is None


def test_flatten_extracts_mbid_and_scalars():
    doc = {
        "metadata": {"tags": {"musicbrainz_recordingid": ["mbid-123"]}},
        "lowlevel": {"average_loudness": 0.5, "mfcc": {"mean": [1, 2, 3]}},
        "rhythm": {"bpm": 120.0},
        "tonal": {"key_key": "C", "key_scale": "major", "hpcp": {"mean": [0.1, 0.2]}},
    }
    row = _flatten(doc)
    assert row["mbid"] == "mbid-123"
    assert row["lowlevel.average_loudness"] == 0.5
    assert row["rhythm.bpm"] == 120.0
    assert row["tonal.key_key"] == "C"
    assert row["lowlevel.mfcc.mean.0"] == 1
    assert row["lowlevel.mfcc.mean.2"] == 3
    assert row["tonal.hpcp.mean.1"] == 0.2


def test_flatten_handles_missing_sections():
    row = _flatten({})
    assert row["mbid"] is None
    assert row["rhythm.bpm"] is None


def test_iter_bulk_records_dedups_across_archives(monkeypatch):
    archive_a = [{"mbid": "1"}, {"mbid": "2"}]
    archive_b = [{"mbid": "2"}, {"mbid": "3"}]  # "2" repeated across archives

    def fake_iter_archive_records(url):
        return iter(archive_a if url == "urlA" else archive_b)

    monkeypatch.setattr(
        "src.ingest.acousticbrainz._iter_archive_records", fake_iter_archive_records
    )

    rows = list(iter_bulk_records(["urlA", "urlB"], limit=10))
    mbids = [r["mbid"] for r in rows]
    assert mbids == ["1", "2", "3"]


def test_iter_bulk_records_respects_limit(monkeypatch):
    archive_a = [{"mbid": str(i)} for i in range(10)]
    monkeypatch.setattr(
        "src.ingest.acousticbrainz._iter_archive_records", lambda url: iter(archive_a)
    )

    rows = list(iter_bulk_records(["urlA"], limit=3))
    assert len(rows) == 3


def _ll_doc(mbid):
    return {"metadata": {"tags": {"musicbrainz_recordingid": [mbid]}}}


def _hl_doc(mbid, happy_value=None):
    doc = {"metadata": {"tags": {"musicbrainz_recordingid": [mbid]}}}
    if happy_value is not None:
        doc["highlevel"] = {"mood_happy": {"value": happy_value, "probability": 0.9}}
    return doc


def test_iter_bulk_records_with_highlevel_merges_aligned_archives(monkeypatch):
    ll_docs = [_ll_doc("1"), _ll_doc("2")]
    hl_docs = [_hl_doc("1", "happy"), _hl_doc("2", "not_happy")]

    def fake_iter_archive_docs(url):
        return iter(ll_docs if url == "llA" else hl_docs)

    monkeypatch.setattr(
        "src.ingest.acousticbrainz._iter_archive_docs", fake_iter_archive_docs
    )

    rows = list(iter_bulk_records_with_highlevel(["llA"], ["hlA"], limit=10))
    assert [r["highlevel.mood_happy.value"] for r in rows] == ["happy", "not_happy"]


def test_iter_bulk_records_with_highlevel_keeps_columns_on_mismatch(monkeypatch):
    # low-level and high-level archives out of alignment for one record.
    ll_docs = [_ll_doc("1"), _ll_doc("2")]
    hl_docs = [_hl_doc("mismatched-mbid", "happy"), _hl_doc("2", "not_happy")]

    def fake_iter_archive_docs(url):
        return iter(ll_docs if url == "llA" else hl_docs)

    monkeypatch.setattr(
        "src.ingest.acousticbrainz._iter_archive_docs", fake_iter_archive_docs
    )

    rows = list(iter_bulk_records_with_highlevel(["llA"], ["hlA"], limit=10))
    # mismatched row still has the column, just unset - keeps CSV columns consistent
    assert rows[0]["highlevel.mood_happy.value"] is None
    assert rows[1]["highlevel.mood_happy.value"] == "not_happy"
    assert set(rows[0].keys()) == set(rows[1].keys())