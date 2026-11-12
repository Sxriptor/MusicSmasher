# Mashup Studio

Add two tracks (full songs or instrumentals) by uploading audio files or pasting YouTube video URLs into either track slot. The app splits each into vocals and instrumental, detects BPM and key, then puts the vocals from one over the instrumental from the other, time-stretched and pitch-shifted so they match.

**Run:** double-click `run.bat`. The first run sets up `.venv`. Then open http://127.0.0.1:8765.

## Pipeline
1. **Input and decode**: YouTube URLs are downloaded with `yt-dlp` (one video, up to one hour); ffmpeg decodes each file to 44.1 kHz stereo.
2. **Separate**: Demucs `htdemucs` splits it into vocals and instrumental. It runs on CUDA when available, and the weights come from Hugging Face.
3. **Analyze** (`mashup/analysis.py`):
   - BPM comes from librosa beat tracking. A least-squares fit of the beat grid refines it, and values close to a whole number snap to it.
   - The downbeat is found from low-frequency (kick) onsets.
   - The key comes from CQT chroma matched against KeyFinder major/minor profiles, and the result is also shown in Camelot notation.
   - Vocal presence decides whether a file is a song or an instrumental.
4. **Plan** (`mashup/planner.py`):
   - Half-time and double-time are handled automatically.
   - Keys are compared through their relative major, so A minor over C major needs no shift.
   - There are three modes: vocals→instrumental (default), instrumental→vocals, or meet in the middle.
5. **Render** (`mashup/engine.py`):
   - ffmpeg rubberband does the stretching. Vocals use formant preservation, and the instrumental keeps crisp transients.
   - The vocal's downbeat is aligned to the instrumental's bar grid.
   - The vocal is high-passed, and the levels follow the vocal/music balance of the original song.
   - A limiter is applied, and the result is saved as MP3 and WAV in `data/renders/`.

Detected BPM and key can be corrected by hand in the UI. If a song was recorded "between keys" or the detector picks the relative key, fix it there. Analyses and stems are cached in `data/tracks/` by file hash.
