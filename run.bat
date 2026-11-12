@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Creating environment - first run takes a few minutes...
  py -3.13 -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  .venv\Scripts\python -m pip install librosa soundfile numpy scipy fastapi uvicorn python-multipart julius einops pyyaml tqdm huggingface-hub lameenc safetensors
  .venv\Scripts\python -m pip install --no-deps demucs dora-search
)
.venv\Scripts\python -m pip show yt-dlp >nul 2>&1 || .venv\Scripts\python -m pip install yt-dlp
start "" http://127.0.0.1:8765
.venv\Scripts\python app.py
