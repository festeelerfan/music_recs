"""Joint audio-text embedding via MuQ-MuLan (Tencent, 2025).
https://github.com/tencent-ailab/MuQ
"""

import librosa
import torch
from muq import MuQMuLan

MODEL_ID = "OpenMuQ/MuQ-MuLan-large"
SAMPLE_RATE = 24000


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    return MuQMuLan.from_pretrained(MODEL_ID).to(_device()).eval()


def embed_audio_files(model, paths):
    """Whole-track embeddings, one per path. Derived from the same
    per-segment computation as embed_audio_segments (just averaged) rather
    than a separate model pass, so embedding a track never happens twice."""
    embeds = []
    for path in paths:
        segment_embeds, _ = embed_audio_segments(model, path)
        embeds.append(segment_embeds.mean(dim=0, keepdim=True))
    return torch.cat(embeds, dim=0)


def embed_audio_segments(model, path):
    """Return per-10s-clip embeddings for one track without averaging them
    away into a single whole-track vector. Segment i covers
    [i * model.clip_secs, (i+1) * model.clip_secs); order is preserved (see
    MuQMuLan._get_all_clips - a plain sequential walk from the start of the
    track). This reuses the model's own internal clip-splitting/embedding
    step rather than reimplementing it, but relies on muq's non-public
    _get_all_clips/mulan_module internals, so it could break on a muq
    version bump."""
    device = next(model.parameters()).device
    wav, _ = librosa.load(path, sr=SAMPLE_RATE)
    wav_tensor = torch.tensor(wav).to(device)
    with torch.no_grad():
        clips = model._get_all_clips(wav_tensor)  # (num_clips, samples)
        # one clip at a time - batching all clips at once (as MuQMuLan's own
        # parallel_processing=True path does) can exhaust MPS/GPU memory on
        # longer tracks.
        embeds = torch.stack(
            [
                model.mulan_module.get_audio_latents(clip.unsqueeze(0)).squeeze(0)
                for clip in clips
            ]
        )
    starts_sec = [i * model.clip_secs for i in range(clips.shape[0])]
    return embeds, starts_sec


def embed_text(model, texts):
    with torch.no_grad():
        return model(texts=texts)


def similarity(model, audio_embeds, text_embeds):
    return model.calc_similarity(audio_embeds, text_embeds)