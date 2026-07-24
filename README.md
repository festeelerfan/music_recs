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
  https://github.com/tencent-ailab/MuQ) — whole-track first, segment
  windowing later
- `src/index/` — nearest-neighbor index build/query
- `src/query/` — one handler per query mode (segment/descriptor, artist+
  favorites, mood, conversational)
- `src/eval/` — quantitative retrieval eval (recall@k against the Song
  Describer Dataset) — separate from the similarity index itself
- `data/raw/` — untouched source dumps (gitignored)
- `data/processed/` — parsed feature tables, embeddings, index files (gitignored)
- `notebooks/` — exploration

## Build order

1. AcousticBrainz ingest + features + a plain kNN query (no audio, no ML
   inference) — validates the scaffolding end to end.
2. Audio-text embedding over owned library, whole-track, free-text query.
3. Segment-level embedding.
4. Demucs/instrument separation and everything after — not yet.
