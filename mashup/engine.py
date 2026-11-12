"""Track analysis (with on-disk cache), mashup preparation and export."""
from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from . import analysis, audio_io
from .audio_io import SR
from .planner import make_plan

DATA = Path(__file__).resolve().parent.parent / "data"
TRACKS = DATA / "tracks"
SESSIONS = DATA / "sessions"

Progress = Callable[[str], None]
ANALYSIS_VERSION = 3


def _noop(_: str) -> None:
    pass


def track_dir(track_id: str) -> Path:
    return TRACKS / track_id


def load_track(track_id: str) -> dict:
    return json.loads((track_dir(track_id) / "analysis.json").read_text())


def save_track_meta(track_id: str, meta: dict) -> None:
    (track_dir(track_id) / "analysis.json").write_text(json.dumps(meta, indent=2))


# ------------------------------------------------------------------- analysis

def analyze_track(src: Path, original_name: str | None = None, progress: Progress = _noop) -> dict:
    """Decode, separate, and analyze a file. Results are cached by content hash."""
    track_id = audio_io.file_hash(src)
    d = track_dir(track_id)
    meta_path = d / "analysis.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("version") == ANALYSIS_VERSION:
            progress("Loaded cached analysis.")
            if original_name:
                meta["name"] = original_name
            return meta
    d.mkdir(parents=True, exist_ok=True)

    progress("Decoding audio...")
    mix_path = d / "mix.wav"
    if not mix_path.exists():
        audio_io.decode_to_wav(src, mix_path)
    mix = audio_io.read_wav(mix_path)

    voc_path, inst_path = d / "vocals.wav", d / "instrumental.wav"
    if not (voc_path.exists() and inst_path.exists()):
        from .separation import separate
        vocals, inst = separate(mix, progress=progress)
        audio_io.write_wav(voc_path, vocals)
        audio_io.write_wav(inst_path, inst)
    else:
        vocals, inst = audio_io.read_wav(voc_path), audio_io.read_wav(inst_path)

    progress("Checking for vocals...")
    vp = analysis.vocal_presence(vocals, mix)

    progress("Detecting tempo and beat grid...")
    y_mix = analysis.to_mono(mix)
    tempo = analysis.detect_tempo(y_mix)

    progress("Detecting key...")
    y_key = analysis.to_mono(inst) if vp["has_vocals"] else y_mix
    key = analysis.detect_key(y_key)

    meta = {
        "version": ANALYSIS_VERSION,
        "id": track_id,
        "name": original_name or src.name,
        "duration": round(mix.shape[1] / SR, 2),
        "bpm": tempo["bpm"],
        "bpm_confidence": tempo["confidence"],
        "downbeat": round(tempo["downbeat"], 4),
        "beat_count": len(tempo["beats"]),
        "tonic": key["tonic"],
        "mode": key["mode"],
        "key": key["key"],
        "camelot": key["camelot"],
        "key_confidence": key["confidence"],
        "key_alternatives": key["alternatives"],
        "tuning_cents": key["tuning_cents"],
        "first_vocal": round(analysis.first_vocal_time(vocals), 2) if vp["has_vocals"] else None,
        **vp,
    }
    save_track_meta(track_id, meta)
    progress("Analysis done.")
    return meta


# --------------------------------------------------------------------- helpers

def _processed(track_id: str, stem: str, tempo: float, semis: int, kind: str) -> Path:
    d = track_dir(track_id)
    src = d / f"{stem}.wav"
    if abs(tempo - 1) < 1e-4 and semis == 0:
        return src
    dst = d / f"proc_{stem}_{tempo:.6f}_{semis:+d}.wav"
    if not dst.exists():
        tmp = dst.with_suffix(".tmp.wav")
        audio_io.stretch_shift(src, tmp, tempo, semis, kind)
        tmp.replace(dst)
    return dst


def _active_rms(x: np.ndarray) -> float:
    """RMS over the louder half of 0.4 s windows (ignores silence / breaks)."""
    m = x.mean(axis=0)
    win = int(0.4 * SR)
    n = len(m) // win
    if n == 0:
        return float(np.sqrt((m ** 2).mean()) + 1e-9)
    r = np.sqrt((m[: n * win].reshape(n, win) ** 2).mean(axis=1))
    r = np.sort(r)[n // 2:]
    return float(np.sqrt((r ** 2).mean()) + 1e-9)


def _place(x: np.ndarray, delay: int, length: int) -> np.ndarray:
    """Put x on a timeline of `length` samples starting at `delay` (may be negative)."""
    out = np.zeros((x.shape[0], length), dtype=np.float32)
    if delay >= length or -delay >= x.shape[1]:
        return out
    if delay >= 0:
        seg = x[:, : length - delay]
        out[:, delay: delay + seg.shape[1]] = seg
    else:
        seg = x[:, -delay: -delay + length]
        out[:, : seg.shape[1]] = seg
    return out


def _write_preview(path: Path, x: np.ndarray) -> float:
    """16-bit WAV for the browser, peak-normalized. Returns gain to undo the scaling."""
    peak = float(np.abs(x).max()) or 1.0
    scale = 0.99 / peak
    sf.write(str(path), (x * scale).T, SR, subtype="PCM_16")
    return 1.0 / scale


def _safe_name(s: str) -> str:
    s = re.sub(r"\.[A-Za-z0-9]{2,4}$", "", s)
    return re.sub(r"[^\w\- ()&]+", "", s).strip()[:60] or "track"


# ------------------------------------------------------------------- prepare

def prepare(params: dict, progress: Progress = _noop) -> dict:
    """Stretch/shift both stems, compute the automatic alignment and levels, and
    write them as a session the editor can play.  Nothing is mixed yet.

    params: vocal_track, beat_track, mode, semitones, tempo_relation
    """
    v = dict(load_track(params["vocal_track"]))
    b = dict(load_track(params["beat_track"]))
    plan = make_plan(v, b, mode=params.get("mode", "vocals_to_beat"),
                     semitone_override=params.get("semitones"),
                     tempo_relation=params.get("tempo_relation", "auto"))

    # Instrumental-only files are used whole (no separation artifacts).
    beat_stem = "instrumental" if b.get("has_vocals") else "mix"

    progress(f"Stretching/shifting vocals (x{plan.vocal_tempo:.3f}, {plan.vocal_semitones:+d} st)...")
    vpath = _processed(v["id"], "vocals", plan.vocal_tempo, plan.vocal_semitones, "vocals")
    progress(f"Stretching/shifting instrumental (x{plan.beat_tempo:.3f}, {plan.beat_semitones:+d} st)...")
    bpath = _processed(b["id"], beat_stem, plan.beat_tempo, plan.beat_semitones, "beat")

    progress("Aligning and leveling...")
    vocals = audio_io.read_wav(vpath)
    beat = audio_io.read_wav(bpath)

    # Beat grids after stretching: vocal-song downbeat lands on a beat-track downbeat.
    beat_period = 60.0 / plan.final_bpm
    bar = 4 * beat_period
    v_down = v["downbeat"] / plan.vocal_tempo
    b_down = b["downbeat"] / plan.beat_tempo
    auto_offset = (b_down - v_down) % bar

    # High-pass the vocal stem to drop bass/kick bleed from separation.
    sos = butter(2, 100, btype="highpass", fs=SR, output="sos")
    vocals = sosfiltfilt(sos, vocals, axis=1).astype(np.float32)

    # Baseline levels: keep the vocal-to-music balance of the original song,
    # then scale both so the default mix sits around -14 dBFS (active RMS).
    length = beat.shape[1]
    placed = _place(vocals, int(round(auto_offset * SR)), length)
    ratio = float(np.clip((v.get("vocal_ratio") or 0.7) * 1.15, 0.45, 1.3))
    v_gain = ratio * _active_rms(beat) / _active_rms(placed)
    loud = 10 ** (-14 / 20) / _active_rms(placed * v_gain + beat)
    vocals *= v_gain * loud
    beat *= loud

    session_id = uuid.uuid4().hex[:12]
    d = SESSIONS / session_id
    d.mkdir(parents=True, exist_ok=True)
    progress("Writing stems for the editor...")
    audio_io.write_wav(d / "vocals_f.wav", vocals)
    audio_io.write_wav(d / "beat_f.wav", beat)
    vocal_preview_gain = _write_preview(d / "vocals.wav", vocals)
    beat_preview_gain = _write_preview(d / "beat.wav", beat)

    session = {
        "session_id": session_id,
        "params": params,
        "plan": plan.to_dict(),
        "vocal_track": v["name"],
        "beat_track": b["name"],
        "beat_source": beat_stem,
        "duration": round(length / SR, 3),
        "vocal_duration": round(vocals.shape[1] / SR, 3),
        "auto_offset": round(auto_offset, 4),
        "beat_period": beat_period,
        "grid_origin": b_down % beat_period,
        "downbeat": b_down,
        "vocals_url": f"/sessions/{session_id}/vocals.wav",
        "beat_url": f"/sessions/{session_id}/beat.wav",
        "vocal_preview_gain": vocal_preview_gain,
        "beat_preview_gain": beat_preview_gain,
    }
    (d / "session.json").write_text(json.dumps(session, indent=2))
    progress("Ready.")
    return session


# -------------------------------------------------------------------- export

def export(session_id: str, offset_s: float, vocal_gain_db: float = 0.0,
           beat_gain_db: float = 0.0, vocal_muted: bool = False, beat_muted: bool = False,
           progress: Progress = _noop) -> dict:
    """Mix the session's stems with the editor's offset and levels, limit, encode."""
    d = SESSIONS / session_id
    session = json.loads((d / "session.json").read_text())
    progress("Mixing...")
    vocals = audio_io.read_wav(d / "vocals_f.wav")
    beat = audio_io.read_wav(d / "beat_f.wav")
    length = beat.shape[1]

    vg = 0.0 if vocal_muted else 10 ** (vocal_gain_db / 20)
    bg = 0.0 if beat_muted else 10 ** (beat_gain_db / 20)
    mix = _place(vocals, int(round(offset_s * SR)), length) * vg + beat * bg
    fade = min(length, int(2.0 * SR))
    mix[:, length - fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

    n = len(list(d.glob("export_*.mp3"))) + 1
    stem = f"export_{n}"
    raw = d / f"{stem}_raw.wav"
    audio_io.write_wav(raw, mix)
    progress("Limiting and encoding MP3...")
    final_wav = d / f"{stem}.wav"
    audio_io._run([audio_io.FFMPEG, "-y", "-v", "error", "-i", str(raw),
                   "-af", "alimiter=limit=0.95:attack=3:release=60:level=0",
                   "-c:a", "pcm_s16le", str(final_wav)])
    raw.unlink()
    audio_io.encode_mp3(final_wav, d / f"{stem}.mp3")

    nice = f"{_safe_name(session['vocal_track'])} x {_safe_name(session['beat_track'])}"
    result = {
        "session_id": session_id,
        "plan": session["plan"],
        "vocal_track": session["vocal_track"],
        "beat_track": session["beat_track"],
        "offset_s": round(offset_s, 4),
        "nudge_ms": round((offset_s - session["auto_offset"]) * 1000, 1),
        "vocal_gain_db": vocal_gain_db,
        "beat_gain_db": beat_gain_db,
        "duration": round(length / SR, 2),
        "mp3": f"/sessions/{session_id}/{stem}.mp3",
        "wav": f"/sessions/{session_id}/{stem}.wav",
        "filename": nice,
    }
    (d / f"{stem}.json").write_text(json.dumps(result, indent=2))
    progress("Done.")
    return result


def render(params: dict, progress: Progress = _noop) -> dict:
    """One-shot prepare + export using the automatic alignment (CLI / scripting)."""
    s = prepare(params, progress)
    return export(s["session_id"], s["auto_offset"], progress=progress)


def clear_cache() -> None:
    shutil.rmtree(DATA, ignore_errors=True)
