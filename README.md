Music similarity engine — content-based (not collaborative-filtering) music
recommendation. Project brief lives outside this folder (see the top-level
CLAUDE.md at the repo root) — this directory is the implementation only.

## Layout

- `src/ingest/` — one loader per data source (AcousticBrainz, FMA, own library)
- `src/features/` — classical/interpretable feature extraction (AcousticBrainz
  parsing now; chord/key detection later)
- `src/embed/` — joint audio-text embedding (backbone TBD — CLAP vs. MuLan;
  whole-track first, segment windowing later)
- `src/index/` — nearest-neighbor index build/query
- `src/query/` — one handler per query mode (segment/descriptor, artist+
  favorites, mood, conversational)
- `data/raw/` — untouched source dumps (gitignored)
- `data/processed/` — parsed feature tables, embeddings, index files (gitignored)
- `notebooks/` — exploration and retrieval eval

## Build order

1. AcousticBrainz ingest + features + a plain kNN query (no audio, no ML
   inference) — validates the scaffolding end to end.
2. Audio-text embedding over owned library, whole-track, free-text query.
3. Segment-level embedding.
4. Demucs/instrument separation and everything after — not yet.
