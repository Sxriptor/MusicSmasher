"""Local web app: load two tracks, analyze, and render a key/tempo-matched mashup.

Run:  .venv\\Scripts\\python app.py   then open http://127.0.0.1:8765
"""
from __future__ import annotations

import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mashup import engine
from mashup.planner import make_plan

ROOT = Path(__file__).resolve().parent
INPUT_TMP = ROOT / "data" / "tmp"
INPUT_TMP.mkdir(parents=True, exist_ok=True)
engine.SESSIONS.mkdir(parents=True, exist_ok=True)
engine.TRACKS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Mashup Studio")
pool = ThreadPoolExecutor(max_workers=2)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def _start_job(fn, *args) -> str:
    job_id = uuid.uuid4().hex[:10]
    job = {"id": job_id, "status": "running", "log": [], "result": None, "error": None}
    with jobs_lock:
        jobs[job_id] = job

    def progress(msg: str) -> None:
        job["log"].append(msg)

    def run():
        try:
            job["result"] = fn(*args, progress=progress)
            job["status"] = "done"
        except Exception as e:  # surface the error to the UI
            traceback.print_exc()
            job["error"] = str(e)
            job["status"] = "error"

    pool.submit(run)
    return job_id


def _input_dir() -> Path:
    path = INPUT_TMP / uuid.uuid4().hex
    path.mkdir()
    return path


def _youtube_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if (parsed.scheme not in ("http", "https") or parsed.username or parsed.password
            or host not in ("youtube.com", "www.youtube.com", "m.youtube.com",
                            "music.youtube.com", "youtu.be", "www.youtu.be")):
        raise HTTPException(400, "Enter a valid YouTube video URL")
    return url


class YouTubeInput(BaseModel):
    url: str


@app.post("/api/analyze-url")
def analyze_url(source: YouTubeInput):
    url = _youtube_url(source.url)

    def work(video_url: str, progress):
        import yt_dlp

        tmpdir = _input_dir()
        try:
            options = {
                "format": "bestaudio/best",
                "outtmpl": str(tmpdir / "source.%(ext)s"),
                "noplaylist": True,
                "max_filesize": 150 * 1024 * 1024,
                "quiet": True,
                "no_warnings": True,
                "cachedir": False,
            }
            with yt_dlp.YoutubeDL(options) as ydl:
                progress("Checking YouTube video...")
                info = ydl.extract_info(video_url, download=False)
                if not info or info.get("_type") != "video":
                    raise ValueError("URL must point to a single YouTube video")
                if info.get("duration") and info["duration"] > 3600:
                    raise ValueError("YouTube video must be one hour or shorter")
                progress("Downloading audio from YouTube...")
                ydl.extract_info(video_url, download=True)
            files = [p for p in tmpdir.glob("source.*") if p.is_file() and not p.name.endswith(".part")]
            if len(files) != 1:
                raise RuntimeError("YouTube audio download did not produce a usable file")
            progress("Audio downloaded. Analyzing track...")
            return engine.analyze_track(files[0], info.get("title") or "YouTube track", progress)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return {"job_id": _start_job(work, url)}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "track.mp3").suffix or ".mp3"
    tmpdir = _input_dir()
    dst = tmpdir / f"upload{suffix}"
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)

    def work(path: Path, name: str, progress):
        try:
            return engine.analyze_track(path, name, progress)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return {"job_id": _start_job(work, dst, file.filename or dst.name)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job


class TrackPatch(BaseModel):
    bpm: float | None = None
    tonic: int | None = None
    mode: str | None = None


@app.patch("/api/tracks/{track_id}")
def patch_track(track_id: str, patch: TrackPatch):
    """Manual correction of detected BPM / key (saved to the cache)."""
    meta = engine.load_track(track_id)
    if patch.bpm:
        meta["bpm"] = round(float(patch.bpm), 2)
        meta["bpm_manual"] = True
    if patch.tonic is not None and patch.mode in ("major", "minor"):
        from mashup.analysis import camelot, key_name
        meta.update(tonic=patch.tonic % 12, mode=patch.mode,
                    key=key_name(patch.tonic, patch.mode), camelot=camelot(patch.tonic, patch.mode),
                    key_manual=True)
    engine.save_track_meta(track_id, meta)
    return meta


class PlanParams(BaseModel):
    vocal_track: str
    beat_track: str
    mode: str = "vocals_to_beat"
    semitones: int | None = None
    tempo_relation: str = "auto"


class ExportParams(BaseModel):
    session_id: str
    offset_s: float
    vocal_gain_db: float = 0
    beat_gain_db: float = 0
    vocal_muted: bool = False
    beat_muted: bool = False


@app.post("/api/plan")
def plan(p: PlanParams):
    v, b = engine.load_track(p.vocal_track), engine.load_track(p.beat_track)
    return make_plan(v, b, p.mode, p.semitones, p.tempo_relation).to_dict()


@app.post("/api/prepare")
def prepare(p: PlanParams):
    """Stretch/align stems and open them in the editor (no mixing yet)."""
    return {"job_id": _start_job(engine.prepare, p.model_dump())}


@app.post("/api/export")
def export(p: ExportParams):
    if not (engine.SESSIONS / p.session_id / "session.json").exists():
        raise HTTPException(404, "unknown session - prepare again")
    d = p.model_dump()
    sid = d.pop("session_id")
    return {"job_id": _start_job(lambda progress: engine.export(sid, progress=progress, **d))}


@app.get("/api/tracks/{track_id}/{stem}")
def stem(track_id: str, stem: str):
    if stem not in ("vocals", "instrumental", "mix"):
        raise HTTPException(404)
    path = engine.track_dir(track_id) / f"{stem}.wav"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


app.mount("/sessions", StaticFiles(directory=engine.SESSIONS), name="sessions")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


if __name__ == "__main__":
    print("Mashup Studio -> http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
