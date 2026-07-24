"""Song Describer Dataset (MTG, 2023): human-written captions + audio drawn
from MTG-Jamendo, used only for a quantitative text->audio retrieval eval
(recall@k) - not part of the similarity index itself.
https://zenodo.org/records/10072001

The audio.zip on Zenodo is ~3.3GB for all 706 tracks. Since eval only needs
a small subset, individual entries are fetched via HTTP range requests
against the zip's central directory, same "don't download more than you
need" approach used for the AcousticBrainz bulk dump.

Central directory offsets are parsed by hand rather than via `zipfile`:
zipfile applies a "self-extracting stub" correction to header offsets that
assumes the byte stream it was given is the whole archive starting at file
offset 0. Since we only ever hand it the tail of the archive, that
correction misfires and produces bogus (negative) offsets.
"""

import struct
import zlib
from pathlib import Path

import pandas as pd
import requests

METADATA_URL = "https://zenodo.org/api/records/10072001/files/song_describer.csv/content"
AUDIO_ZIP_URL = "https://zenodo.org/api/records/10072001/files/audio.zip/content"

_LOCAL_HEADER_SIZE = 30  # fixed part of a zip local file header
_EOCD_SIGNATURE = b"PK\x05\x06"
_EOCD_FORMAT = "<4sHHHHIIH"  # fixed 22-byte EOCD record
_CD_HEADER_FORMAT = "<4sHHHHHHIIIHHHHHII"  # fixed 46-byte central directory file header
_CD_HEADER_SIZE = 46


def download_metadata(out_path):
    response = requests.get(METADATA_URL)
    response.raise_for_status()
    Path(out_path).write_bytes(response.content)
    return pd.read_csv(out_path)


def _zip_member_name(track_id):
    return f"audio/{track_id % 100:02d}/{track_id}.2min.mp3"


def _parse_central_directory(buf):
    """Return {name: {compress_size, compress_type, header_offset}} parsed
    directly from a buffer containing the zip's tail (EOCD + central dir)."""
    eocd_idx = buf.rfind(_EOCD_SIGNATURE)
    if eocd_idx == -1:
        raise ValueError("EOCD not found in tail buffer; increase tail_bytes")

    _, _, _, _, total_entries, cd_size, _, _ = struct.unpack_from(
        _EOCD_FORMAT, buf, eocd_idx
    )
    cd_start = eocd_idx - cd_size
    if cd_start < 0:
        raise ValueError("central directory not fully within tail buffer; increase tail_bytes")

    entries = {}
    pos = cd_start
    for _ in range(total_entries):
        fields = struct.unpack_from(_CD_HEADER_FORMAT, buf, pos)
        comp_size, name_len, extra_len, comment_len = fields[8], fields[10], fields[11], fields[12]
        method = fields[4]
        header_offset = fields[16]

        name_start = pos + _CD_HEADER_SIZE
        name = buf[name_start : name_start + name_len].decode("utf-8", errors="replace")
        entries[name] = {
            "compress_size": comp_size,
            "compress_type": method,
            "header_offset": header_offset,
        }
        pos = name_start + name_len + extra_len + comment_len

    return entries


def _fetch_central_directory(zip_url, tail_bytes=2_000_000):
    head = requests.head(zip_url, allow_redirects=True)
    total_size = int(head.headers["Content-Length"])
    start = max(0, total_size - tail_bytes)
    resp = requests.get(zip_url, headers={"Range": f"bytes={start}-{total_size - 1}"})
    resp.raise_for_status()
    return _parse_central_directory(resp.content)


def _fetch_entry_bytes(zip_url, entry):
    """Fetch and decompress a single zip member via ranged GETs, without
    downloading the rest of the archive."""
    start = entry["header_offset"]
    header = requests.get(
        zip_url, headers={"Range": f"bytes={start}-{start + _LOCAL_HEADER_SIZE - 1}"}
    ).content
    name_len = int.from_bytes(header[26:28], "little")
    extra_len = int.from_bytes(header[28:30], "little")
    data_start = start + _LOCAL_HEADER_SIZE + name_len + extra_len
    data_end = data_start + entry["compress_size"] - 1

    data = requests.get(
        zip_url, headers={"Range": f"bytes={data_start}-{data_end}"}
    ).content

    if entry["compress_type"] == 0:  # stored, uncompressed
        return data
    return zlib.decompressobj(-15).decompress(data)


def download_audio_subset(track_ids, out_dir, zip_url=AUDIO_ZIP_URL):
    """Fetch only the audio for the given track_ids, keyed by track_id."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_name = _fetch_central_directory(zip_url)

    paths = {}
    for track_id in track_ids:
        out_path = out_dir / f"{track_id}.mp3"
        if not out_path.exists():
            entry = by_name.get(_zip_member_name(track_id))
            if entry is None:
                continue
            out_path.write_bytes(_fetch_entry_bytes(zip_url, entry))
        paths[track_id] = out_path
    return paths