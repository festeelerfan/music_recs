Music similarity engine — content-based (not collaborative-filtering) music
recommendation.

## Setup

```
pip install -r requirements.txt
brew install ffmpeg  # required for librosa to decode mp3/m4a via audioread
```

## Layout

- `src/ingest/` — one loader per data source (AcousticBrainz, FMA, own library)
- `src/features/` — classical/interpretable feature extraction (AcousticBrainz
  parsing now; chord/key detection later)
- `src/embed/` — joint audio-text embedding via MuQ-MuLan (Tencent, 2025;
  https://github.com/tencent-ailab/MuQ), whole-track and segment-level
  (native 10s clips, order preserved - see embed_audio_segments)
- `src/index/` — nearest-neighbor index build/query
- `src/query/` — one handler per query mode: `mode1.py` (free text,
  whole-track), `mode1_segments.py` (free text, segment-level - ranks
  distinct songs by either their single best-matching moment (`--style
  moment`, e.g. "guitar solo") or the mean of their top segments (`--style
  sustained`, e.g. "melancholic piano"); which style fits a query is a
  judgment call left to the caller until real query-intent routing exists),
  `mode2.py` (query-by-example, "songs like X by Y" - library-only, no
  external audio source), mood and conversational modes still to come
- `src/eval/` — quantitative retrieval eval (recall@k against the Song
  Describer Dataset) — separate from the similarity index itself
- `data/raw/` — untouched source dumps (gitignored)
- `data/processed/` — parsed feature tables, embeddings, index files (gitignored)
- `notebooks/` — exploration
