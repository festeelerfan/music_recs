"""Stream AcousticBrainz low-level data without downloading full archives.

Supports both the small curated sample dumps (for quick iteration)
and the full dump (30 archives, ~1M recordings each) for scaling up.
Bulk streaming is hardened for long unattended runs: malformed entries are
skipped rather than crashing the whole pull, results are checkpointed to
disk periodically, and a partial output file can be resumed.
"""

import argparse
import json
import tarfile
from pathlib import Path

import pandas as pd
import requests
import zstandard

DUMPS_BASE = "https://data.metabrainz.org/pub/musicbrainz/acousticbrainz/dumps"

SAMPLE_URL = f"{DUMPS_BASE}/acousticbrainz-sample-json-20220623/acousticbrainz-lowlevel-sample-json-20220623-0.tar.zst"

BULK_LOWLEVEL_URLS = [
    f"{DUMPS_BASE}/acousticbrainz-lowlevel-json-20220623/acousticbrainz-lowlevel-json-20220623-{i}.tar.zst"
    for i in range(30)
]

BULK_HIGHLEVEL_URLS = [
    f"{DUMPS_BASE}/acousticbrainz-highlevel-json-20220623/acousticbrainz-highlevel-json-20220623-{i}.tar.zst"
    for i in range(30)
]

# Curated subset of AcousticBrainz's high-level classifier outputs (mood/
# genre/danceability). Archive N of the high-level dump lines up record-for-
# record with archive N of the low-level dump (verified empirically), so the
# two can be streamed in lockstep and merged without a separate lookup pass.
HIGHLEVEL_FIELDS = [
    "danceability",
    "mood_acoustic",
    "mood_aggressive",
    "mood_electronic",
    "mood_happy",
    "mood_party",
    "mood_relaxed",
    "mood_sad",
    "genre_dortmund",
    "voice_instrumental",
    "tonal_atonal",
]

# Curated subset of AcousticBrainz's ~120 low-level descriptors: enough to
# do a first-pass similarity comparison without hand-flattening every field.
SCALAR_FIELDS = {
    "lowlevel.average_loudness": ("lowlevel", "average_loudness"),
    "lowlevel.dynamic_complexity": ("lowlevel", "dynamic_complexity"),
    "lowlevel.spectral_centroid.mean": ("lowlevel", "spectral_centroid", "mean"),
    "lowlevel.spectral_energy.mean": ("lowlevel", "spectral_energy", "mean"),
    "lowlevel.spectral_rolloff.mean": ("lowlevel", "spectral_rolloff", "mean"),
    "lowlevel.spectral_flux.mean": ("lowlevel", "spectral_flux", "mean"),
    "lowlevel.dissonance.mean": ("lowlevel", "dissonance", "mean"),
    "rhythm.bpm": ("rhythm", "bpm"),
    "rhythm.danceability": ("rhythm", "danceability"),
    "rhythm.beats_loudness.mean": ("rhythm", "beats_loudness", "mean"),
    "tonal.chords_changes_rate": ("tonal", "chords_changes_rate"),
    "tonal.key_key": ("tonal", "key_key"),
    "tonal.key_scale": ("tonal", "key_scale"),
}

VECTOR_FIELDS = {
    "lowlevel.mfcc.mean": ("lowlevel", "mfcc", "mean"),
    "tonal.hpcp.mean": ("tonal", "hpcp", "mean"),
}


def _get(doc, path):
    for key in path:
        if not isinstance(doc, dict) or key not in doc:
            return None
        doc = doc[key]
    return doc


def _flatten(doc):
    row = {}
    tags = _get(doc, ("metadata", "tags")) or {}
    row["mbid"] = (tags.get("musicbrainz_recordingid") or [None])[0]
    for name, path in SCALAR_FIELDS.items():
        row[name] = _get(doc, path)
    for name, path in VECTOR_FIELDS.items():
        vec = _get(doc, path) or []
        for i, v in enumerate(vec):
            row[f"{name}.{i}"] = v
    return row


def _flatten_highlevel(doc):
    row = {}
    hl = doc.get("highlevel") or {}
    for field in HIGHLEVEL_FIELDS:
        entry = hl.get(field) or {}
        row[f"highlevel.{field}.value"] = entry.get("value")
        row[f"highlevel.{field}.probability"] = entry.get("probability")
    return row


def _mbid_of(doc):
    tags = _get(doc, ("metadata", "tags")) or {}
    return (tags.get("musicbrainz_recordingid") or [None])[0]


def _iter_archive_docs(url):
    """Yield parsed JSON docs from one archive, skipping any entry that
    fails to parse rather than aborting the whole stream."""
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        decompressor = zstandard.ZstdDecompressor().stream_reader(response.raw)
        with tarfile.open(fileobj=decompressor, mode="r|") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                try:
                    yield json.load(tar.extractfile(member))
                except Exception:
                    continue


def _iter_archive_records(url):
    for doc in _iter_archive_docs(url):
        yield _flatten(doc)


def iter_lowlevel_records(url=SAMPLE_URL, limit=2000):
    """Yield up to `limit` flattened feature rows from a single archive,
    closing the connection as soon as the limit is reached."""
    count = 0
    for row in _iter_archive_records(url):
        yield row
        count += 1
        if count >= limit:
            return


def iter_bulk_records(urls, limit, seen_mbids=None):
    """Yield up to `limit` deduplicated (by mbid) rows across multiple
    archives, moving to the next archive once one is exhausted."""
    seen_mbids = set() if seen_mbids is None else seen_mbids
    count = 0
    for url in urls:
        for row in _iter_archive_records(url):
            mbid = row.get("mbid")
            if not mbid or mbid in seen_mbids:
                continue
            seen_mbids.add(mbid)
            yield row
            count += 1
            if count >= limit:
                return


def _empty_highlevel_row():
    row = {}
    for field in HIGHLEVEL_FIELDS:
        row[f"highlevel.{field}.value"] = None
        row[f"highlevel.{field}.probability"] = None
    return row


def iter_bulk_records_with_highlevel(
    lowlevel_urls, highlevel_urls, limit, seen_mbids=None
):
    """Like iter_bulk_records, but streams the matching high-level archive
    alongside each low-level archive and merges mood/genre tags in. Every
    yielded row has the same highlevel.* columns whether or not a match was
    found, so checkpointed CSV writes stay column-consistent."""
    seen_mbids = set() if seen_mbids is None else seen_mbids
    count = 0
    for ll_url, hl_url in zip(lowlevel_urls, highlevel_urls):
        ll_docs = _iter_archive_docs(ll_url)
        hl_docs = _iter_archive_docs(hl_url)
        for ll_doc, hl_doc in zip(ll_docs, hl_docs):
            row = _flatten(ll_doc)
            mbid = row.get("mbid")
            if not mbid or mbid in seen_mbids:
                continue
            seen_mbids.add(mbid)

            row.update(_empty_highlevel_row())
            if _mbid_of(hl_doc) == mbid:
                row.update(_flatten_highlevel(hl_doc))

            yield row
            count += 1
            if count >= limit:
                return


def download_sample(out_path, limit=2000):
    rows = list(iter_lowlevel_records(limit=limit))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def download_bulk(
    out_path,
    limit,
    urls=BULK_LOWLEVEL_URLS,
    highlevel_urls=None,
    checkpoint_every=5000,
    resume=True,
):
    """Stream `limit` deduplicated records from the real bulk dump, writing
    to `out_path` every `checkpoint_every` records so a crash or interruption
    doesn't lose everything already pulled. Re-running with resume=True picks
    up where a previous partial run left off. Pass highlevel_urls to also
    merge in mood/genre tags from the matching high-level archives."""
    out_path = Path(out_path)
    seen_mbids = set()
    already = 0

    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        seen_mbids.update(existing["mbid"].dropna().tolist())
        already = len(existing)
        print(f"resuming: {already} rows already in {out_path}")

    remaining = limit - already
    if remaining <= 0:
        return already

    if highlevel_urls:
        records = iter_bulk_records_with_highlevel(
            urls, highlevel_urls, limit=remaining, seen_mbids=seen_mbids
        )
    else:
        records = iter_bulk_records(urls, limit=remaining, seen_mbids=seen_mbids)

    write_header = not (out_path.exists() and already > 0)
    buffer = []
    total = already
    for row in records:
        buffer.append(row)
        total += 1
        if len(buffer) >= checkpoint_every:
            pd.DataFrame(buffer).to_csv(out_path, mode="a", header=write_header, index=False)
            write_header = False
            buffer.clear()
            print(f"checkpoint: {total} rows written")

    if buffer:
        pd.DataFrame(buffer).to_csv(out_path, mode="a", header=write_header, index=False)

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--out", default="data/processed/acousticbrainz_sample.csv")
    parser.add_argument("--bulk", action="store_true", help="stream the real bulk dump instead of the sample")
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument(
        "--no-highlevel",
        action="store_true",
        help="skip merging in the high-level mood/genre dump (bulk mode only)",
    )
    args = parser.parse_args()

    if args.bulk:
        highlevel_urls = None if args.no_highlevel else BULK_HIGHLEVEL_URLS
        total = download_bulk(
            args.out,
            args.limit,
            highlevel_urls=highlevel_urls,
            checkpoint_every=args.checkpoint_every,
        )
        print(f"wrote {total} rows to {args.out}")
    else:
        df = download_sample(args.out, limit=args.limit)
        print(f"wrote {len(df)} rows to {args.out}")