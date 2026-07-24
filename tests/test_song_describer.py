import io
import zipfile

from src.ingest.song_describer import _parse_central_directory, _zip_member_name


def test_zip_member_name_formula():
    assert _zip_member_name(156061) == "audio/61/156061.2min.mp3"
    assert _zip_member_name(1051203) == "audio/03/1051203.2min.mp3"


def test_parse_central_directory_finds_correct_local_header_offsets():
    # Regression test: zipfile's own self-extracting-stub offset correction
    # gives bogus offsets when only handed a truncated tail of a zip, which
    # is exactly what happens when streaming just the central directory.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("audio/61/156061.2min.mp3", b"fake mp3 bytes" * 100)
        zf.writestr("audio/03/1051203.2min.mp3", b"other fake bytes" * 50)

    data = buf.getvalue()
    entries = _parse_central_directory(data)

    assert set(entries) == {"audio/61/156061.2min.mp3", "audio/03/1051203.2min.mp3"}
    for entry in entries.values():
        offset = entry["header_offset"]
        assert offset >= 0
        assert data[offset : offset + 4] == b"PK\x03\x04"