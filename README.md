Music similarity engine — content-based (not collaborative-filtering) music
recommendation. 

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
