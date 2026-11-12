"""Decide how far to stretch / shift each side so the vocals sit on the beat in key.

Conventional mashup practice is to treat the instrumental as the "bed" and conform
the acapella to it (``vocals_to_beat``): drums smear when stretched, while a
vocal stretched with formant preservation holds up well.  When the instrumental
is the one far from the vocal's natural tempo/key, moving the beat
(``beat_to_vocals``) or splitting the difference (``meet``) can sound better,
so all three are offered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

from .analysis import NOTES, camelot, key_name

MODES = ("vocals_to_beat", "beat_to_vocals", "meet")


@dataclass
class Plan:
    mode: str
    vocal_tempo: float          # rubberband tempo factor for the vocal stem (>1 = faster)
    beat_tempo: float           # same for the instrumental
    vocal_semitones: int
    beat_semitones: int
    tempo_relation: str         # "same", "half-time" or "double-time"
    vocal_bpm_equiv: float      # vocal BPM after half/double interpretation, before stretching
    final_bpm: float
    final_key: str
    final_camelot: str
    key_relation: str
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _relative_major_pc(tonic: int, mode: str) -> int:
    return tonic % 12 if mode == "major" else (tonic + 3) % 12


def _wrap_semitones(d: int) -> int:
    d %= 12
    return d - 12 if d > 6 else (-6 if d == 6 else d)


def make_plan(vocal: dict, beat: dict, mode: str = "vocals_to_beat",
              semitone_override: int | None = None, tempo_relation: str = "auto") -> Plan:
    """vocal / beat: dicts with bpm, tonic, mode (from track analysis)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    warnings: list[str] = []

    # ---- tempo: allow the vocal to ride a half-time / double-time feel
    bpm_v, bpm_b = float(vocal["bpm"]), float(beat["bpm"])
    options = {"same": 1.0, "half-time": 0.5, "double-time": 2.0}
    if tempo_relation == "auto":
        tempo_relation = min(options, key=lambda k: abs(math.log(bpm_b / (bpm_v * options[k]))))
    eq = bpm_v * options[tempo_relation]
    R = bpm_b / eq  # speed factor the vocal needs to land on the beat's tempo

    if mode == "vocals_to_beat":
        vt, bt, final_bpm = R, 1.0, bpm_b
    elif mode == "beat_to_vocals":
        vt, bt, final_bpm = 1.0, 1.0 / R, eq
    else:
        vt, bt = math.sqrt(R), 1.0 / math.sqrt(R)
        final_bpm = bpm_b / math.sqrt(R)

    for name, t in (("Vocals", vt), ("Instrumental", bt)):
        if abs(t - 1) > 0.15:
            warnings.append(f"{name} stretched {abs(t - 1) * 100:.0f}% - expect audible artifacts.")

    # ---- key: compare via relative major, so A minor over C major needs no shift
    pc_v = _relative_major_pc(vocal["tonic"], vocal["mode"])
    pc_b = _relative_major_pc(beat["tonic"], beat["mode"])
    diff = _wrap_semitones(pc_b - pc_v) if semitone_override is None else int(semitone_override)

    if mode == "vocals_to_beat":
        vs, bs = diff, 0
    elif mode == "beat_to_vocals":
        vs, bs = 0, -diff
    else:
        vs = math.ceil(diff / 2) if diff > 0 else math.floor(diff / 2)
        bs = vs - diff

    for name, s in (("Vocals", vs), ("Instrumental", bs)):
        if abs(s) >= 4:
            warnings.append(f"{name} pitched {s:+d} semitones - large shifts can sound unnatural.")

    # Final key = the beat's key moved by bs.
    fk_tonic = (beat["tonic"] + bs) % 12
    final_key = key_name(fk_tonic, beat["mode"])
    if vocal["mode"] == beat["mode"]:
        rel = "same mode"
    else:
        rel = "relative major/minor (compatible)"
    if semitone_override is not None:
        rel = "manual override"

    return Plan(mode=mode, vocal_tempo=round(vt, 6), beat_tempo=round(bt, 6),
                vocal_semitones=vs, beat_semitones=bs, tempo_relation=tempo_relation,
                vocal_bpm_equiv=round(eq, 2), final_bpm=round(final_bpm, 2),
                final_key=final_key, final_camelot=camelot(fk_tonic, beat["mode"]),
                key_relation=rel, warnings=warnings)


__all__ = ["Plan", "make_plan", "MODES", "NOTES"]
