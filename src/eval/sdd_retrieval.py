"""Quantitative text->audio retrieval eval (recall@k) against the Song
Describer Dataset. Prior validation of MuQ-MuLan quality was purely by eye
(eyeballing top-5 results) - this gives an actual number. Evaluation only,
not part of the similarity index itself.
"""

import argparse
import random

import pandas as pd
import torch

from src.embed.muq_mulan import embed_audio_files, embed_text, load_model
from src.ingest.song_describer import download_audio_subset, download_metadata

DEFAULT_METADATA_PATH = "data/processed/song_describer.csv"
DEFAULT_AUDIO_DIR = "data/processed/song_describer_audio"


def build_eval_set(n_tracks, seed=0, metadata_path=DEFAULT_METADATA_PATH, audio_dir=DEFAULT_AUDIO_DIR):
    try:
        captions_df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        captions_df = download_metadata(metadata_path)

    track_ids = captions_df["track_id"].unique().tolist()
    random.Random(seed).shuffle(track_ids)
    track_ids = track_ids[:n_tracks]

    audio_paths = download_audio_subset(track_ids, audio_dir)
    captions_df = captions_df[captions_df["track_id"].isin(audio_paths)].reset_index(drop=True)
    return captions_df, audio_paths


def run_eval(captions_df, audio_paths, ks=(1, 5, 10)):
    model = load_model()

    ordered_track_ids = list(audio_paths.keys())
    track_embeds = embed_audio_files(model, [str(audio_paths[tid]) for tid in ordered_track_ids])
    track_index = {tid: i for i, tid in enumerate(ordered_track_ids)}

    hits = {k: 0 for k in ks}
    total = 0
    for _, row in captions_df.iterrows():
        text_embed = embed_text(model, [row["caption"]])[0]
        scores = model.calc_similarity(track_embeds, text_embed)
        ranking = torch.argsort(scores, descending=True)
        true_pos = (ranking == track_index[row["track_id"]]).nonzero(as_tuple=True)[0].item()
        for k in ks:
            if true_pos < k:
                hits[k] += 1
        total += 1

    return {f"recall@{k}": hits[k] / total for k in ks}, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tracks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    captions_df, audio_paths = build_eval_set(args.n_tracks, args.seed)
    print(f"eval set: {len(audio_paths)} tracks, {len(captions_df)} captions")

    metrics, total = run_eval(captions_df, audio_paths)
    print(f"over {total} captions:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")