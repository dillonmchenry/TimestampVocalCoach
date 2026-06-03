"""FastAPI app for the Sprint-2 vocal-coach demo.

Endpoints:

    GET  /                                         -> index.html (static)
    GET  /api/songs                                -> list of available songs
    GET  /api/songs/{song_id}/manifest             -> manifest.json
    GET  /api/songs/{song_id}/reference_annotation -> reference_annotation.json
    GET  /api/songs/{song_id}/audio/instrumental   -> stream the instrumental
    GET  /api/songs/{song_id}/audio/reference      -> stream the reference vocal
    POST /api/songs/{song_id}/analyze              -> upload performance.wav, run pipeline
    GET  /api/songs/{song_id}/performances/{perf_id}/analysis -> PerformanceAnalysis
    GET  /api/songs/{song_id}/performances/{perf_id}/audio    -> user wav

Run locally::

    uvicorn web.api.main:app --reload

The analyze endpoint runs the pipeline *inline* on the request thread, which
is fine for short clips during the demo. For longer songs, swap this for the
async task pattern (job_id + polling endpoint).
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vocal_coach.align_v2 import measure_song
from vocal_coach.coaching_config import CoachingConfig, DEFAULT_CONFIG_RELPATH
from vocal_coach.highlights import select_highlights
from vocal_coach.loudness import compute_loudness, write_loudness_track
from vocal_coach.overview import compute_overview
from vocal_coach.pitch import extract_f0, write_pitch_track
from vocal_coach.reference import load_reference
from vocal_coach.schemas import (
    LoudnessTrack,
    PerformanceAnalysis,
    PitchTrack,
    StarsMetadataEntry,
    StarsTrack,
)
from vocal_coach.song import load_manifest
from vocal_coach.trends import compute_section_trends
from vocal_coach.stars_runner import (
    DEFAULT_STARS_DIR,
    STARS_PROFILE_FAST,
    STARS_PROFILE_FULL,
    STARS_PROFILES,
    run_stars_with_profile,
    write_stars_track,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SONGS_ROOT = REPO_ROOT / "data" / "songs"
STATIC_ROOT = REPO_ROOT / "web" / "static"
CONFIG_PATH = REPO_ROOT / DEFAULT_CONFIG_RELPATH

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}


app = FastAPI(title="Vocal Coach (Sprint 2)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _song_dir(song_id: str) -> Path:
    """Return the song bundle directory or 404."""
    candidate = (SONGS_ROOT / song_id).resolve()
    try:
        candidate.relative_to(SONGS_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid song id")
    if not candidate.is_dir() or not (candidate / "manifest.json").is_file():
        raise HTTPException(status_code=404, detail=f"Song {song_id!r} not found")
    return candidate


def _perf_dir(song_id: str, perf_id: str) -> Path:
    candidate = (SONGS_ROOT / song_id / "performances" / perf_id).resolve()
    try:
        candidate.relative_to(SONGS_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid perf id")
    if not candidate.is_dir():
        raise HTTPException(status_code=404, detail="Performance not found")
    return candidate


def _audio_response(path: Path) -> FileResponse:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio not found: {path.name}")
    media_type = AUDIO_MIME.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(str(path), media_type=media_type, filename=path.name)


def _stars_metadata_for_perf(song_dir: Path, perf_dir: Path, user_audio: Path) -> Path:
    """Build a perf-specific stars_metadata.json reusing the song's word/phone lists."""
    import json

    meta_in = song_dir / "stars_metadata.json"
    raw = json.loads(meta_in.read_text(encoding="utf-8"))
    if not raw:
        raise HTTPException(status_code=500, detail="Reference stars metadata is empty")
    entry = raw[0]
    user_entry = StarsMetadataEntry(
        item_name=f"{song_dir.name}__perf",
        wav_fn=str(user_audio.resolve()).replace("\\", "/"),
        word=entry["word"],
        ph=entry["ph"],
        ph2words=entry["ph2words"],
        ph_durs=entry.get("ph_durs"),
        word_durs=entry.get("word_durs"),
    )
    out = perf_dir / "stars_metadata.json"
    out.write_text(
        json.dumps([user_entry.model_dump(exclude_none=True)], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def _maybe_load(path: Path, model):
    if path.is_file():
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    return None


def _resolve_torch_device(requested: str = "cuda") -> str:
    if requested.lower() == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return requested
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/songs")
def list_songs():
    """Return a list of installed songs."""
    if not SONGS_ROOT.is_dir():
        return JSONResponse({"songs": []})
    out = []
    for song_dir in sorted(SONGS_ROOT.iterdir()):
        if not (song_dir / "manifest.json").is_file():
            continue
        try:
            manifest = load_manifest(song_dir)
        except Exception:
            continue
        out.append(
            {
                "song_id": manifest.song_id,
                "title": manifest.title,
                "artist": manifest.artist,
                "language": manifest.language,
                "duration_s": manifest.duration_s,
                "has_instrumental": manifest.instrumental_path is not None,
                "has_reference_pitch": manifest.reference_pitch_path is not None,
                "has_reference_stars": manifest.reference_stars_path is not None,
            }
        )
    return {"songs": out}


@app.get("/api/songs/{song_id}/manifest")
def get_manifest(song_id: str):
    song_dir = _song_dir(song_id)
    return JSONResponse(content=load_manifest(song_dir).model_dump())


@app.get("/api/songs/{song_id}/reference_annotation")
def get_reference_annotation(song_id: str):
    song_dir = _song_dir(song_id)
    annotation = load_reference(song_dir)
    return JSONResponse(content=annotation.model_dump())


@app.get("/api/songs/{song_id}/audio/instrumental")
def get_instrumental(song_id: str):
    song_dir = _song_dir(song_id)
    manifest = load_manifest(song_dir)
    if not manifest.instrumental_path:
        raise HTTPException(status_code=404, detail="Instrumental not available")
    return _audio_response(song_dir / manifest.instrumental_path)


@app.get("/api/songs/{song_id}/audio/reference")
def get_reference_audio(song_id: str):
    song_dir = _song_dir(song_id)
    manifest = load_manifest(song_dir)
    return _audio_response(song_dir / manifest.reference_vocal_path)


@app.post("/api/songs/{song_id}/analyze")
async def analyze(
    song_id: str,
    file: UploadFile = File(...),
    perf_id: Optional[str] = Form(None),
    skip_user_stars: bool = Form(False),
    device: str = Form("cuda"),
    # Interim: default to the (correct) teacher until the student is retrained
    # on NanoPitch F0. The UI checkbox can still request "fast".
    stars_profile: str = Form(STARS_PROFILE_FAST),
):
    """Run the dual-track analysis on an uploaded performance."""
    if stars_profile not in STARS_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"stars_profile must be one of {STARS_PROFILES}",
        )
    song_dir = _song_dir(song_id)
    manifest = load_manifest(song_dir)
    reference = load_reference(song_dir)

    perf_id = perf_id or uuid.uuid4().hex[:8]
    perf_dir = song_dir / "performances" / perf_id
    perf_dir.mkdir(parents=True, exist_ok=True)

    # Persist the upload (preserve extension if it's a known audio type).
    src_name = file.filename or "performance.wav"
    suffix = Path(src_name).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        suffix = ".wav"
    user_audio = perf_dir / f"performance{suffix}"
    with user_audio.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)

    # 1. NanoPitch
    pitch_path = perf_dir / "pitch.json"
    torch_device = _resolve_torch_device(device)
    pitch_user = extract_f0(user_audio, device=torch_device)
    pitch_user.sample_id = f"{manifest.song_id}__{perf_id}"
    write_pitch_track(pitch_user, pitch_path)

    # 2. STARS on user vocal (unless explicitly skipped)
    stars_user: StarsTrack | None = None
    stars_path = perf_dir / "stars.json"
    if not skip_user_stars:
        try:
            meta_path = _stars_metadata_for_perf(song_dir, perf_dir, user_audio)
            stars_user = run_stars_with_profile(
                profile=stars_profile,
                metadata_path=meta_path,
                save_dir=perf_dir / "stars_out",
                sample_id=f"{manifest.song_id}__{perf_id}",
                stars_dir=DEFAULT_STARS_DIR,
            )
            write_stars_track(stars_user, stars_path)
        except Exception as exc:  # pragma: no cover - depends on STARS env
            # Fall back to pitch-only analysis when STARS isn't usable.
            stars_user = None
            (perf_dir / "stars_error.txt").write_text(str(exc), encoding="utf-8")

    # 3. Reference STARS + reference pitch + reference loudness (precomputed by build_song.py)
    stars_ref_path = song_dir / (manifest.reference_stars_path or "reference/stars.json")
    stars_ref = _maybe_load(stars_ref_path, StarsTrack)
    pitch_ref_path = song_dir / (manifest.reference_pitch_path or "reference/pitch.json")
    pitch_ref = _maybe_load(pitch_ref_path, PitchTrack)
    loudness_ref_path = song_dir / (manifest.reference_loudness_path or "reference/loudness.json")
    loudness_ref = _maybe_load(loudness_ref_path, LoudnessTrack)

    # 4. Loudness on the user vocal (best-effort); load back for measure_song
    loudness_user: LoudnessTrack | None = None
    loudness_user_path = perf_dir / "loudness.json"
    try:
        loudness_user = compute_loudness(user_audio, sample_id=pitch_user.sample_id)
        write_loudness_track(loudness_user, loudness_user_path)
    except Exception:
        pass

    # 5. Align + section trends + highlights + overview
    cfg = CoachingConfig.load(CONFIG_PATH)
    notes, techniques, offset, octave_shift = measure_song(
        reference,
        pitch_user=pitch_user,
        pitch_ref=pitch_ref,
        stars_ref=stars_ref,
        stars_user=stars_user,
        loudness_user=loudness_user,
        loudness_ref=loudness_ref,
        config=cfg,
    )
    section_trends = compute_section_trends(reference, notes, techniques)
    highlights = select_highlights(
        reference,
        notes,
        techniques,
        config=cfg,
        sections=section_trends,
    )
    overview = compute_overview(
        notes,
        techniques,
        sections=section_trends,
        section_best_overall_min_notes=cfg.highlights.section_best_overall_min_notes,
        octave_shift_semitones=octave_shift,
        arrival_late_ms=cfg.arrival.late_ms,
    )

    analysis = PerformanceAnalysis(
        song_id=manifest.song_id,
        perf_id=perf_id,
        reference_sample_id=reference.sample_id,
        duration_s=manifest.duration_s,
        global_offset_s=offset,
        octave_shift_semitones=octave_shift,
        pitch_user_path=str(pitch_path.relative_to(song_dir)).replace("\\", "/"),
        pitch_ref_path=(
            str(pitch_ref_path.relative_to(song_dir)).replace("\\", "/")
            if pitch_ref_path.is_file()
            else None
        ),
        stars_user_path=(
            str(stars_path.relative_to(song_dir)).replace("\\", "/")
            if stars_path.is_file()
            else None
        ),
        stars_ref_path=(
            str(stars_ref_path.relative_to(song_dir)).replace("\\", "/")
            if stars_ref_path.is_file()
            else None
        ),
        notes=notes,
        techniques=techniques,
        highlights=highlights,
        sections=section_trends,
        overview=overview,
    )
    out_path = perf_dir / "analysis.json"
    out_path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    return JSONResponse(content=analysis.model_dump())


@app.get("/api/songs/{song_id}/performances/{perf_id}/analysis")
def get_analysis(song_id: str, perf_id: str):
    perf_dir = _perf_dir(song_id, perf_id)
    path = perf_dir / "analysis.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Analysis not found")
    return JSONResponse(
        content=PerformanceAnalysis.model_validate_json(
            path.read_text(encoding="utf-8")
        ).model_dump()
    )


@app.get("/api/songs/{song_id}/performances/{perf_id}/audio")
def get_perf_audio(song_id: str, perf_id: str):
    perf_dir = _perf_dir(song_id, perf_id)
    candidates = [p for p in perf_dir.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES]
    if not candidates:
        raise HTTPException(status_code=404, detail="Performance audio missing")
    return _audio_response(candidates[0])


@app.get("/api/songs/{song_id}/performances/{perf_id}/pitch")
def get_perf_pitch(song_id: str, perf_id: str):
    perf_dir = _perf_dir(song_id, perf_id)
    path = perf_dir / "pitch.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Pitch track not found")
    return JSONResponse(
        content=PitchTrack.model_validate_json(path.read_text(encoding="utf-8")).model_dump()
    )


@app.get("/api/songs/{song_id}/performances/{perf_id}/loudness")
def get_perf_loudness(song_id: str, perf_id: str):
    """Per-frame RMS (dB) track; the UI turns this into a waveform envelope.

    Serving precomputed peaks keeps the waveform render independent of the
    browser decoding the full (potentially 40 MB+) performance WAV.
    """
    perf_dir = _perf_dir(song_id, perf_id)
    path = perf_dir / "loudness.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Loudness track not found")
    return JSONResponse(
        content=LoudnessTrack.model_validate_json(
            path.read_text(encoding="utf-8")
        ).model_dump()
    )


# ---------------------------------------------------------------------------
# Static frontend (mounted last so /api routes take priority)
# ---------------------------------------------------------------------------


if STATIC_ROOT.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_ROOT), html=True), name="static")
