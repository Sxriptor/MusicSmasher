"""Decoding / encoding through ffmpeg, plus the rubberband stretch/shift step."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100  # working sample rate for all stems / renders

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def decode_to_wav(src: Path, dst: Path) -> None:
    """Decode any ffmpeg-readable file to 44.1 kHz stereo float WAV."""
    _run([FFMPEG, "-y", "-v", "error", "-i", str(src), "-vn",
          "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(dst)])


def read_wav(path: Path) -> np.ndarray:
    """Returns float32 array shaped (channels, samples)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR, f"unexpected sample rate {sr} in {path}"
    return data.T.copy()


def write_wav(path: Path, audio: np.ndarray) -> None:
    sf.write(str(path), audio.T, SR, subtype="FLOAT")


def stretch_shift(src: Path, dst: Path, tempo: float, semitones: float, kind: str) -> None:
    """Time-stretch (tempo factor, >1 = faster) and pitch-shift via ffmpeg's rubberband.

    kind = "vocals" uses formant preservation (avoids chipmunk sound) and smooth
    transients; kind = "beat" keeps drum transients crisp.
    """
    if abs(tempo - 1.0) < 1e-4 and abs(semitones) < 1e-3:
        shutil.copyfile(src, dst)
        return
    pitch = 2.0 ** (semitones / 12.0)
    if kind == "vocals":
        opts = "transients=mixed:detector=soft:formant=preserved:pitchq=quality:window=standard"
    else:
        opts = "transients=crisp:detector=compound:pitchq=quality:window=standard"
    af = f"rubberband=tempo={tempo:.6f}:pitch={pitch:.6f}:{opts}"
    _run([FFMPEG, "-y", "-v", "error", "-i", str(src), "-af", af,
          "-ar", str(SR), "-c:a", "pcm_f32le", str(dst)])


def encode_mp3(src_wav: Path, dst_mp3: Path) -> None:
    _run([FFMPEG, "-y", "-v", "error", "-i", str(src_wav),
          "-c:a", "libmp3lame", "-b:a", "320k", str(dst_mp3)])
