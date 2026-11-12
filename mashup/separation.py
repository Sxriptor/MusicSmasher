"""Vocal / instrumental stem separation with Demucs (htdemucs)."""
from __future__ import annotations

import threading

import numpy as np

_model = None
_model_name = None
_lock = threading.Lock()


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(name: str):
    global _model, _model_name
    if _model is None or _model_name != name:
        from demucs.pretrained import get_model
        m = get_model(name)
        m.eval()
        _model, _model_name = m, name
    return _model


def separate(audio: np.ndarray, model_name: str = "htdemucs", progress=None) -> tuple[np.ndarray, np.ndarray]:
    """audio: (2, n) float32 at 44.1 kHz. Returns (vocals, instrumental), same shape."""
    import torch
    from demucs.apply import apply_model

    with _lock:  # one separation at a time (GPU memory)
        model = _get_model(model_name)
        if progress:
            progress(f"Separating stems on {_device().upper()}...")
        wav = torch.from_numpy(audio)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav = (wav - mean) / std
        with torch.no_grad():
            out = apply_model(model, wav[None], device=_device(), shifts=1,
                              split=True, overlap=0.25, progress=False)[0]
        out = out * std + mean
        names = list(model.sources)
        vi = names.index("vocals")
        vocals = out[vi].cpu().numpy()
        instrumental = sum(out[i] for i in range(len(names)) if i != vi).cpu().numpy()
        if _device() == "cuda":
            torch.cuda.empty_cache()
    return vocals.astype(np.float32), instrumental.astype(np.float32)
