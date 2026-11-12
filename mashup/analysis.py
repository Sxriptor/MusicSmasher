"""BPM, beat grid, downbeat, musical key and vocal-presence detection."""
from __future__ import annotations

import numpy as np
import librosa

AN_SR = 22050
HOP = 512

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Key profiles from Sha'ath's KeyFinder (tuned on pop/electronic music).
_MAJOR = np.array([7.2390, 3.5035, 3.5845, 2.8451, 5.8190, 4.5587,
                   2.4478, 6.9946, 3.3911, 4.5561, 4.0717, 4.4594])
_MINOR = np.array([7.0011, 3.1445, 4.3623, 5.4054, 3.6748, 4.0842,
                   3.9112, 6.2017, 3.6368, 2.8795, 5.3503, 3.8348])

# Camelot wheel numbers, indexed by tonic pitch class.
_CAMELOT_MAJOR = {0: 8, 7: 9, 2: 10, 9: 11, 4: 12, 11: 1, 6: 2, 1: 3, 8: 4, 3: 5, 10: 6, 5: 7}
_CAMELOT_MINOR = {9: 8, 4: 9, 11: 10, 6: 11, 1: 12, 8: 1, 3: 2, 10: 3, 5: 4, 0: 5, 7: 6, 2: 7}


def key_name(tonic: int, mode: str) -> str:
    return f"{NOTES[tonic % 12]} {'major' if mode == 'major' else 'minor'}"


def camelot(tonic: int, mode: str) -> str:
    if mode == "major":
        return f"{_CAMELOT_MAJOR[tonic % 12]}B"
    return f"{_CAMELOT_MINOR[tonic % 12]}A"


def to_mono(audio: np.ndarray) -> np.ndarray:
    """(2, n) @44.1k -> mono @22.05k for analysis."""
    mono = audio.mean(axis=0)
    return librosa.resample(mono, orig_sr=44100, target_sr=AN_SR)


# --------------------------------------------------------------------------- tempo

def detect_tempo(y: np.ndarray) -> dict:
    """Return bpm, beat times, first downbeat and confidence.

    The BPM is refined by a least-squares line through the tracked beats, which is
    far more precise than the raw estimate; precision matters because a 0.5 BPM
    error drifts ~0.75 s over a 3-minute song.
    """
    _, y_perc = librosa.effects.hpss(y)
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=AN_SR, hop_length=HOP, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=AN_SR, hop_length=HOP,
                                           start_bpm=120, tightness=400)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=AN_SR, hop_length=HOP)

    if len(beat_times) < 8:
        return {"bpm": round(tempo, 2) if tempo > 0 else 120.0, "beats": beat_times.tolist(),
                "downbeat": float(beat_times[0]) if len(beat_times) else 0.0,
                "period": 60.0 / (tempo or 120.0), "confidence": 0.0}

    period, t0, conf = _fit_grid(beat_times)
    bpm = 60.0 / period
    while bpm < 70:
        bpm *= 2
    while bpm >= 180:
        bpm /= 2
    # Most produced music sits on an integer BPM; snap when we're very close.
    if abs(bpm - round(bpm)) < 0.2:
        bpm = float(round(bpm))
    period = 60.0 / bpm
    # Re-fit the grid phase for the final period.
    t0 = _grid_phase(beat_times, period)

    downbeat = _find_downbeat(y, t0, period)
    return {"bpm": round(bpm, 2), "beats": beat_times.tolist(), "downbeat": downbeat,
            "period": period, "confidence": round(conf, 2)}


def _fit_grid(beat_times: np.ndarray) -> tuple[float, float, float]:
    """Robust linear fit time = t0 + k*period over the tracked beats."""
    d = np.diff(beat_times)
    med = np.median(d)
    # Assign integer beat indices allowing for skipped / doubled beats.
    idx = np.concatenate([[0], np.cumsum(np.maximum(1, np.round(d / med)))])
    keep = np.ones(len(idx), bool)
    for _ in range(3):
        A = np.vstack([idx[keep], np.ones(keep.sum())]).T
        (period, t0), *_ = np.linalg.lstsq(A, beat_times[keep], rcond=None)
        resid = beat_times - (t0 + period * idx)
        keep = np.abs(resid) < max(0.03, 2.5 * np.std(resid[keep]))
    conf = float(np.clip(1.0 - np.std(resid[keep]) / (0.1 * period), 0, 1)) * min(1.0, keep.mean() * 1.1)
    return float(period), float(t0), conf


def _grid_phase(beat_times: np.ndarray, period: float) -> float:
    phases = np.angle(np.mean(np.exp(2j * np.pi * beat_times / period)))
    t0 = (phases / (2 * np.pi)) * period
    return float(t0 % period)


def _find_downbeat(y: np.ndarray, t0: float, period: float) -> float:
    """Pick which of 4 beat phases is beat 1, using low-frequency (kick/bass) onsets."""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=AN_SR, n_fft=2048)
    low = S[freqs < 150].sum(axis=0)
    low_on = np.maximum(0, np.diff(low, prepend=low[0]))
    full_on = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=AN_SR)
    n = min(len(low_on), len(full_on))
    strength = low_on[:n] / (low_on.max() + 1e-9) + 0.5 * full_on[:n] / (full_on.max() + 1e-9)
    dur = n * HOP / AN_SR
    scores = []
    for p in range(4):
        ts = np.arange(t0 + p * period, dur, 4 * period)
        fr = np.clip(np.round(ts * AN_SR / HOP).astype(int), 0, n - 1)
        # take a small window max to tolerate slight misalignment
        vals = [strength[max(0, f - 2):f + 3].max() for f in fr]
        scores.append(np.mean(vals) if vals else 0)
    p = int(np.argmax(scores))
    return float(t0 + p * period)


# ---------------------------------------------------------------------------- key

def detect_key(y: np.ndarray) -> dict:
    """Key from a chroma profile correlated against major/minor templates."""
    y_harm = librosa.effects.harmonic(y, margin=3.0)
    tuning = librosa.estimate_tuning(y=y_harm, sr=AN_SR)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=AN_SR, hop_length=HOP, tuning=tuning,
                                        n_chroma=12, bins_per_octave=36)
    # Weight frames by energy so silence / noise doesn't vote.
    rms = librosa.feature.rms(y=y_harm, hop_length=HOP)[0]
    n = min(chroma.shape[1], len(rms))
    w = rms[:n] ** 2
    profile = (chroma[:, :n] * w).sum(axis=1)
    profile = profile / (profile.sum() + 1e-9)

    scores = []
    for tonic in range(12):
        for mode, tmpl in (("major", _MAJOR), ("minor", _MINOR)):
            t = np.roll(tmpl, tonic)
            r = np.corrcoef(profile, t)[0, 1]
            scores.append((r, tonic, mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    conf = float(np.clip((best[0] - second[0]) * 10, 0, 1))
    alts = [{"key": key_name(t, m), "tonic": t, "mode": m, "score": round(float(r), 3)}
            for r, t, m in scores[1:4]]
    return {"tonic": best[1], "mode": best[2], "key": key_name(best[1], best[2]),
            "camelot": camelot(best[1], best[2]), "confidence": round(conf, 2),
            "tuning_cents": round(float(tuning) * 100, 1), "alternatives": alts}


# ------------------------------------------------------------------------- vocals

def vocal_presence(vocals: np.ndarray, mix: np.ndarray) -> dict:
    """How much real singing is in the separated vocal stem.

    Separation always leaks a little, so we look at the fraction of 0.5 s windows
    where the vocal stem is loud relative to the mix.
    """
    v = vocals.mean(axis=0)
    m = mix.mean(axis=0)
    win = 22050
    n = len(v) // win
    if n == 0:
        return {"has_vocals": False, "vocal_ratio": 0.0, "active_fraction": 0.0}
    v_rms = np.sqrt((v[: n * win].reshape(n, win) ** 2).mean(axis=1))
    m_rms = np.sqrt((m[: n * win].reshape(n, win) ** 2).mean(axis=1)) + 1e-9
    loud = m_rms > (m_rms.max() * 0.03)
    rel_db = 20 * np.log10(v_rms / m_rms + 1e-9)
    active = (rel_db > -12) & loud
    frac = float(active.sum() / max(1, loud.sum()))
    ratio = float(np.sqrt((v ** 2).mean()) / (np.sqrt((m ** 2).mean()) + 1e-9))
    return {"has_vocals": frac > 0.12, "vocal_ratio": round(ratio, 3),
            "active_fraction": round(frac, 3)}


def first_vocal_time(vocals: np.ndarray) -> float:
    v = vocals.mean(axis=0)
    win = 2205
    n = len(v) // win
    if n == 0:
        return 0.0
    rms = np.sqrt((v[: n * win].reshape(n, win) ** 2).mean(axis=1))
    thr = rms.max() * 0.1
    idx = np.argmax(rms > thr)
    return float(idx * win / 44100)
