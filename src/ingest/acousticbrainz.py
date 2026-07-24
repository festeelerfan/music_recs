"""Stream a subset of the AcousticBrainz low-level sample dump without
downloading or extracting the full ~2GB/100k-file archive."""

import argparse
import json
import tarfile

import pandas as pd
import requests
import zstandard

SAMPLE_URL = (
    "https://data.metabrainz.org/pub/musicbrainz/acousticbrainz/dumps/"
    "acousticbrainz-sample-json-20220623/"
    "acousticbrainz-lowlevel-sample-json-20220623-0.tar.zst"
)

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


def iter_lowlevel_records(url=SAMPLE_URL, limit=2000):
    """Yield up to `limit` flattened feature rows, closing the connection
    as soon as the limit is reached rather than downloading the full archive."""
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        decompressor = zstandard.ZstdDecompressor().stream_reader(response.raw)
        with tarfile.open(fileobj=decompressor, mode="r|") as tar:
            count = 0
            for member in tar:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                doc = json.load(tar.extractfile(member))
                yield _flatten(doc)
                count += 1
                if count >= limit:
                    return


def download_sample(out_path, limit=2000):
    rows = list(iter_lowlevel_records(limit=limit))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--out", default="data/processed/acousticbrainz_sample.csv")
    args = parser.parse_args()
    df = download_sample(args.out, limit=args.limit)
    print(f"wrote {len(df)} rows to {args.out}")