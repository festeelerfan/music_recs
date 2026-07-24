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
    device = next(model.parameters()).device
    embeds = []
    with torch.no_grad():
        for path in paths:
            wav, _ = librosa.load(path, sr=SAMPLE_RATE)
            wavs = torch.tensor(wav).unsqueeze(0).to(device)
            embeds.append(model(wavs=wavs))
    return torch.cat(embeds, dim=0)


def embed_text(model, texts):
    with torch.no_grad():
        return model(texts=texts)


def similarity(model, audio_embeds, text_embeds):
    return model.calc_similarity(audio_embeds, text_embeds)