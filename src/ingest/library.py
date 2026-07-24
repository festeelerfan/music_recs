"""Walk my local music folder into a track list, filtering out
non-music noise (short clips, oversized files, non-music artists)."""

from pathlib import Path

import mutagen

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".aif", ".aiff"}
MIN_DURATION_S = 30
MAX_DURATION_S = 20 * 60
EXCLUDED_ARTISTS = {"longmont potion castle"} # shoutout LPC


def _read_tags(path):
    try:
        audio = mutagen.File(path, easy=True)
    except Exception:
        return None
    if audio is None or audio.info is None:
        return None
    return audio


def iter_library_tracks(root):
    root = Path(root)
    for path in root.rglob("*"):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        audio = _read_tags(path)
        if audio is None:
            continue

        duration = audio.info.length
        if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
            continue

        tags = audio.tags or {}
        artist = (tags.get("artist", [""])[0] if tags else "") or ""
        if artist.strip().lower() in EXCLUDED_ARTISTS:
            continue

        title = (tags.get("title", [""])[0] if tags else "") or ""
        if not title.strip():
            continue

        yield {
            "path": str(path),
            "artist": artist,
            "title": title,
            "duration_s": duration,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    count = 0
    for _ in iter_library_tracks(args.root):
        count += 1
    print(f"{count} tracks pass filters")