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

import logging
import shutil
import threading
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import time

import numpy as np

from vocal_coach.align_v2 import (
    alignment_sanity_check,
    estimate_global_offset_s,
    estimate_global_offset_upload,
    measure_song,
)
from vocal_coach.coaching_config import CoachingConfig, DEFAULT_CONFIG_RELPATH
from vocal_coach.feedback import generate_performance_summary, rewrite_card_summaries
from vocal_coach.highlights import select_highlights
from vocal_coach.llm import LLMClient
from vocal_coach.loudness import compute_loudness, write_loudness_track
from vocal_coach.overview import compute_overview
from vocal_coach.pitch import extract_f0, write_pitch_track
from vocal_coach.rag import RAGStore, DEFAULT_DB_PATH
from vocal_coach.reference import load_reference
from vocal_coach.schemas import (
    LoudnessTrack,
    PerformanceAnalysis,
    PitchTrack,
    StarsMetadataEntry,
    StarsTrack,
    VocalProfile,
)
from vocal_coach.song import load_manifest
from vocal_coach.trends import compute_section_trends
from vocal_coach.vocal_profile import load_vocal_profile
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

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
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

_SERVER_START_TIME = time.time()

# ---------------------------------------------------------------------------
# In-process job store for async analysis
# ---------------------------------------------------------------------------
# Maps job_id -> {"status": "pending"|"running"|"done"|"error",
#                 "result": dict|None, "error": str|None}
# Simple dict is safe for single-user concurrency (no race on write since
# each job is owned by exactly one background thread).
_jobs: Dict[str, Dict[str, Any]] = {}


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
    import json as _json

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

        # Load genre_tags from vocal_profile.json if available.
        genre_tags: list[str] = []
        vp_path = song_dir / manifest.vocal_profile_path if manifest.vocal_profile_path else None
        if vp_path and vp_path.is_file():
            try:
                vp_data = _json.loads(vp_path.read_text(encoding="utf-8"))
                genre_tags = vp_data.get("genre_tags", [])
            except Exception:
                pass

        # Check for a cover image (jpg or png).
        cover_url: str | None = None
        for ext in ("jpg", "jpeg", "png", "webp"):
            if (song_dir / f"cover.{ext}").is_file():
                cover_url = f"/api/songs/{manifest.song_id}/cover"
                break

        out.append(
            {
                "song_id": manifest.song_id,
                "title": manifest.title,
                "artist": manifest.artist,
                "language": manifest.language,
                "duration_s": manifest.duration_s,
                "genre_tags": genre_tags,
                "cover_url": cover_url,
                "has_instrumental": manifest.instrumental_path is not None,
                "has_reference_pitch": manifest.reference_pitch_path is not None,
                "has_reference_stars": manifest.reference_stars_path is not None,
                "has_vocal_profile": manifest.vocal_profile_path is not None,
            }
        )
    return {"songs": out}


@app.get("/api/songs/{song_id}/cover")
def get_song_cover(song_id: str):
    """Serve the cover image for a song (cover.jpg / cover.png / etc.)."""
    song_dir = _song_dir(song_id)
    for ext in ("jpg", "jpeg", "png", "webp"):
        cover_path = song_dir / f"cover.{ext}"
        if cover_path.is_file():
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
            return FileResponse(str(cover_path), media_type=mime)
    raise HTTPException(status_code=404, detail="No cover image found for this song.")


@app.get("/api/songs/{song_id}/manifest")
def get_manifest(song_id: str):
    song_dir = _song_dir(song_id)
    return JSONResponse(content=load_manifest(song_dir).model_dump())


@app.get("/api/songs/{song_id}/reference_annotation")
def get_reference_annotation(song_id: str):
    song_dir = _song_dir(song_id)
    annotation = load_reference(song_dir)
    return JSONResponse(content=annotation.model_dump())


@app.get("/api/songs/{song_id}/vocal_profile")
def get_vocal_profile(song_id: str):
    """Return the LLM-generated vocal profile for a song, or 404 if not yet generated."""
    song_dir = _song_dir(song_id)
    manifest = load_manifest(song_dir)
    if not manifest.vocal_profile_path:
        raise HTTPException(
            status_code=404,
            detail="Vocal profile not yet generated. Run scripts/build_song.py.",
        )
    profile_path = song_dir / manifest.vocal_profile_path
    profile = load_vocal_profile(profile_path)
    if profile is None:
        raise HTTPException(status_code=404, detail="Vocal profile file missing or unreadable.")
    return JSONResponse(content=profile.model_dump())


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


def _run_analysis_job(
    job_id: str,
    song_dir: Path,
    perf_id: str,
    perf_dir: Path,
    user_audio: Path,
    stars_profile: str,
    skip_user_stars: bool,
    torch_device: str,
    recording_duration_s: Optional[float] = None,
) -> None:
    """Background worker: runs the full pipeline and writes results to _jobs."""
    _jobs[job_id]["status"] = "running"
    try:
        # Step 0 — Preparing audio
        _jobs[job_id]["step"] = 0
        manifest = load_manifest(song_dir)
        reference = load_reference(song_dir)

        # Step 1 — Tracking pitch and volume
        _jobs[job_id]["step"] = 1
        pitch_path = perf_dir / "pitch.json"
        pitch_user = extract_f0(user_audio, device=torch_device)
        pitch_user.sample_id = f"{manifest.song_id}__{perf_id}"
        write_pitch_track(pitch_user, pitch_path)

        # Loudness on the user vocal (best-effort, grouped with pitch under step 1)
        loudness_user: LoudnessTrack | None = None
        loudness_user_path = perf_dir / "loudness.json"
        try:
            loudness_user = compute_loudness(user_audio, sample_id=pitch_user.sample_id)
            write_loudness_track(loudness_user, loudness_user_path)
        except Exception:
            pass

        # Step 2 — Comparing notes and techniques
        _jobs[job_id]["step"] = 2

        # Reference artifacts (precomputed by build_song.py)
        stars_ref_path = song_dir / (manifest.reference_stars_path or "reference/stars.json")
        stars_ref = _maybe_load(stars_ref_path, StarsTrack)
        pitch_ref_path = song_dir / (manifest.reference_pitch_path or "reference/pitch.json")
        pitch_ref = _maybe_load(pitch_ref_path, PitchTrack)
        loudness_ref_path = song_dir / (manifest.reference_loudness_path or "reference/loudness.json")
        loudness_ref = _maybe_load(loudness_ref_path, LoudnessTrack)

        # STARS on user vocal (unless explicitly skipped)
        stars_user: StarsTrack | None = None
        stars_path = perf_dir / "stars.json"
        if not skip_user_stars:
            try:
                meta_path = _stars_metadata_for_perf(song_dir, perf_dir, user_audio)
                # Reuse the NanoPitch F0 already computed in Step 1 so the
                # student model skips its own RMVPE pass (~190 MB model load).
                _nanopitch_f0 = np.array(
                    [f.f0_hz for f in pitch_user.frames], dtype=np.float32
                )
                stars_user = run_stars_with_profile(
                    profile=stars_profile,
                    metadata_path=meta_path,
                    save_dir=perf_dir / "stars_out",
                    sample_id=f"{manifest.song_id}__{perf_id}",
                    stars_dir=DEFAULT_STARS_DIR,
                    nanopitch_f0=_nanopitch_f0,
                )
                write_stars_track(stars_user, stars_path)
            except Exception as exc:
                stars_user = None
                (perf_dir / "stars_error.txt").write_text(str(exc), encoding="utf-8")

        cfg = CoachingConfig.load(CONFIG_PATH)

        # Distinguish uploaded files from live karaoke recordings.
        # The frontend sends recording_duration_s only for karaoke; uploads omit it.
        is_upload = recording_duration_s is None

        # For karaoke partial recordings, estimate the global offset first so we
        # can convert recording_duration_s (user time) into song time and filter
        # the reference to only the notes the user actually sang.
        segment_end_song_s: Optional[float] = None
        analysis_reference = reference
        if (
            recording_duration_s is not None
            and recording_duration_s < reference.duration_s - 2.0
        ):
            early_offset = estimate_global_offset_s(reference, pitch_user, config=cfg)
            # song_time = user_time - offset  →  segment_end = duration - offset
            seg_end = recording_duration_s - early_offset
            if seg_end > 5.0:
                segment_end_song_s = round(seg_end, 3)
                # Shallow-copy the reference with filtered notes and sections.
                filtered_notes = [n for n in reference.notes if n.start_s < seg_end]
                filtered_sections = [s for s in reference.sections if s.start_s < seg_end]
                analysis_reference = reference.model_copy(update={
                    "notes": filtered_notes,
                    "sections": filtered_sections,
                })

        ref_vocal_path = song_dir / manifest.reference_vocal_path

        if is_upload:
            # Uploads have an unknown start time — use the chroma-assisted wide search.
            pre_offset = estimate_global_offset_upload(
                analysis_reference,
                pitch_user,
                ref_audio_path=ref_vocal_path,
                user_audio_path=user_audio,
                config=cfg,
            )
            notes, techniques, offset, octave_shift = measure_song(
                analysis_reference,
                pitch_user=pitch_user,
                pitch_ref=pitch_ref,
                stars_ref=stars_ref,
                stars_user=stars_user,
                loudness_user=loudness_user,
                loudness_ref=loudness_ref,
                config=cfg,
                global_offset_s=pre_offset,
            )
        else:
            # Karaoke: use the existing fast voicing-only estimator (+/-1.5 s).
            notes, techniques, offset, octave_shift = measure_song(
                analysis_reference,
                pitch_user=pitch_user,
                pitch_ref=pitch_ref,
                stars_ref=stars_ref,
                stars_user=stars_user,
                loudness_user=loudness_user,
                loudness_ref=loudness_ref,
                config=cfg,
            )

        # Post-alignment sanity check — detect and optionally retry misalignment.
        alignment_ok = alignment_sanity_check(notes, config=cfg)
        alignment_warning_flag = False

        if not alignment_ok and cfg.global_offset.retry_on_sanity_fail:
            # Retry with the chroma-assisted wider search regardless of source.
            retry_offset = estimate_global_offset_upload(
                analysis_reference,
                pitch_user,
                ref_audio_path=ref_vocal_path,
                user_audio_path=user_audio,
                config=cfg,
            )
            if not np.isclose(retry_offset, offset, atol=cfg.global_offset.step_s):
                notes, techniques, offset, octave_shift = measure_song(
                    analysis_reference,
                    pitch_user=pitch_user,
                    pitch_ref=pitch_ref,
                    stars_ref=stars_ref,
                    stars_user=stars_user,
                    loudness_user=loudness_user,
                    loudness_ref=loudness_ref,
                    config=cfg,
                    global_offset_s=retry_offset,
                )
                alignment_ok = alignment_sanity_check(notes, config=cfg)

        if not alignment_ok:
            alignment_warning_flag = True
            logger.warning(
                "[analyze] alignment_warning for perf_id=%s: "
                "low pct_in_tune with reasonable voiced_coverage; offset=%.3fs",
                perf_id, offset,
            )

        section_trends = compute_section_trends(analysis_reference, notes, techniques)

        # Step 3 — Selecting coaching highlights
        _jobs[job_id]["step"] = 3
        _vp_path = song_dir / (manifest.vocal_profile_path or "vocal_profile.json")
        vocal_profile = load_vocal_profile(_vp_path)

        highlights = select_highlights(
            analysis_reference,
            notes,
            techniques,
            config=cfg,
            sections=section_trends,
            vocal_profile=vocal_profile,
        )

        # Step 4 — Preparing your feedback
        _jobs[job_id]["step"] = 4
        overview = compute_overview(
            notes,
            techniques,
            sections=section_trends,
            section_best_overall_min_notes=cfg.highlights.section_best_overall_min_notes,
            octave_shift_semitones=octave_shift,
            arrival_late_ms=cfg.highlights.overview_arrival_late_ms,
            arrival_edge_margin_ms=cfg.highlights.overview_arrival_edge_margin_ms,
            alignment_warning=alignment_warning_flag,
        )

        # Sprint 3 Phase B: LLM feedback (card rewriting + performance summary)
        _llm_client = LLMClient.from_config(cfg.llm)
        _rag_store: RAGStore | None = None
        if _llm_client is not None:
            _db_path = REPO_ROOT / DEFAULT_DB_PATH
            if _db_path.is_dir():
                _rag_store = RAGStore(db_path=_db_path)
            rewrite_card_summaries(
                highlights.moments,
                llm=_llm_client,
                rag=_rag_store,
                vocal_profile=vocal_profile,
                reference=analysis_reference,
                song_title=manifest.title,
                artist=manifest.artist,
                max_tokens=cfg.llm.card_rewrite_max_tokens,
            )
            _perf_summary = generate_performance_summary(
                highlights.moments,
                overview,
                section_trends,
                llm=_llm_client,
                rag=_rag_store,
                vocal_profile=vocal_profile,
                song_title=manifest.title,
                artist=manifest.artist,
                max_tokens=cfg.llm.summary_max_tokens,
            )
        else:
            _perf_summary = None

        # For partial recordings, duration_s reflects the recording length so
        # the frontend timeline scales to the recorded portion only.
        perf_duration_s = recording_duration_s if recording_duration_s is not None else manifest.duration_s

        analysis = PerformanceAnalysis(
            song_id=manifest.song_id,
            perf_id=perf_id,
            reference_sample_id=reference.sample_id,
            duration_s=perf_duration_s,
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
            performance_summary=_perf_summary,
            segment_end_song_s=segment_end_song_s,
        )
        out_path = perf_dir / "analysis.json"
        out_path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = analysis.model_dump()
        _jobs[job_id]["perf_id"] = perf_id
        _jobs[job_id]["song_id"] = manifest.song_id

    except Exception as exc:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)


def _sniff_extension(path: Path) -> str:
    """Return the correct file extension based on magic bytes.

    Windows Media Foundation (used by audioread on Windows) is extension-
    sensitive: feeding it WebM bytes in a .wav file causes it to pick the
    wrong decoder and fail.  Reading the first 12 bytes lets us detect the
    real container format so we can rename before decoding.
    """
    try:
        with path.open("rb") as f:
            header = f.read(12)
    except OSError:
        return path.suffix.lower()

    if header[:4] == b"RIFF":
        return ".wav"
    if header[:4] == b"fLaC":
        return ".flac"
    if header[:4] == b"OggS":
        return ".ogg"
    # WebM / Matroska EBML header
    if header[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    # MP4/M4A: "ftyp" box starts at byte 4
    if header[4:8] == b"ftyp":
        return ".m4a"
    return path.suffix.lower()


def _ensure_wav(src: Path) -> Path:
    """Convert browser-recorded audio (webm, ogg, m4a) to WAV.

    Chrome's MediaRecorder produces audio/webm (Opus) and Firefox produces
    audio/ogg.  Windows audioread (WMF) is extension-sensitive: it picks the
    wrong decoder when WebM bytes are stored in a .wav file, causing the load
    to fail entirely.  This function:

      1. Sniffs the real container format from magic bytes.
      2. Renames to the correct extension if needed so decoders get the right hint.
      3. Converts to a clean PCM-16 WAV via librosa + soundfile.

    Downstream loaders (NanoPitch, loudness, STARS) only ever see a valid WAV.
    Returns the path to the WAV (same as ``src`` if it was already valid WAV).
    """
    import librosa
    import soundfile as sf

    real_ext = _sniff_extension(src)

    # Already a real WAV — nothing to do.
    if real_ext == ".wav":
        return src

    # Rename so the OS/audioread decoder gets the correct extension hint.
    correctly_named = src.with_suffix(real_ext)
    if correctly_named != src:
        src.rename(correctly_named)
        src = correctly_named

    logger.info("[analyze] converting %s -> WAV", src.name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wav, sr = librosa.load(str(src), sr=None, mono=True)
    out = src.with_suffix(".wav")
    sf.write(str(out), wav, sr, subtype="PCM_16")
    return out


@app.post("/api/songs/{song_id}/analyze")
async def analyze(
    song_id: str,
    file: UploadFile = File(...),
    perf_id: Optional[str] = Form(None),
    skip_user_stars: bool = Form(False),
    device: str = Form("cuda"),
    stars_profile: str = Form(STARS_PROFILE_FAST),
    recording_duration_s: Optional[float] = Form(None),
):
    """Upload a performance and start analysis in the background.

    Returns a job_id immediately. Poll GET /api/jobs/{job_id} for status.
    When status == "done", the response includes the full analysis result.
    This async pattern avoids HTTP proxy timeouts on long-running inference.
    """
    if stars_profile not in STARS_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"stars_profile must be one of {STARS_PROFILES}",
        )
    song_dir = _song_dir(song_id)

    perf_id = perf_id or uuid.uuid4().hex[:8]
    perf_dir = song_dir / "performances" / perf_id
    perf_dir.mkdir(parents=True, exist_ok=True)

    # Save the upload synchronously before returning — file object closes after
    # the request ends so the background thread can't read it.
    src_name = file.filename or "performance.wav"
    suffix = Path(src_name).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        suffix = ".wav"
    user_audio = perf_dir / f"performance{suffix}"
    with user_audio.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)

    # Convert webm/ogg/m4a from MediaRecorder to WAV so all downstream
    # processing (NanoPitch, STARS, loudness) uses soundfile cleanly.
    user_audio = _ensure_wav(user_audio)

    torch_device = _resolve_torch_device(device)
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "pending", "step": None, "result": None, "error": None}

    thread = threading.Thread(
        target=_run_analysis_job,
        kwargs=dict(
            job_id=job_id,
            song_dir=song_dir,
            perf_id=perf_id,
            perf_dir=perf_dir,
            user_audio=user_audio,
            stars_profile=stars_profile,
            skip_user_stars=skip_user_stars,
            torch_device=torch_device,
            recording_duration_s=recording_duration_s,
        ),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job_id, "perf_id": perf_id, "status": "pending"})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """Poll the status of a background analysis job.

    Response shape:
      {"status": "pending"|"running"|"done"|"error",
       "step": int|null,       -- 0-4 progress index, null until running
       "job_id": str,
       "perf_id": str|null,
       "song_id": str|null,
       "result": PerformanceAnalysis|null,
       "error": str|null}
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JSONResponse({
        "job_id": job_id,
        "status": job["status"],
        "step": job.get("step"),
        "perf_id": job.get("perf_id"),
        "song_id": job.get("song_id"),
        "result": job.get("result"),
        "error": job.get("error"),
    })


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


@app.get("/api/songs/{song_id}/reference/pitch")
def get_reference_pitch(song_id: str):
    """Return the precomputed reference pitch track for a song."""
    song_dir = _song_dir(song_id)
    manifest = load_manifest(song_dir)
    rel = manifest.reference_pitch_path or "reference/pitch.json"
    path = song_dir / rel
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Reference pitch track not found")
    return JSONResponse(
        content=PitchTrack.model_validate_json(path.read_text(encoding="utf-8")).model_dump()
    )


@app.get("/api/songs/{song_id}/reference/loudness")
def get_reference_loudness(song_id: str):
    """Return the precomputed reference loudness track for a song."""
    song_dir = _song_dir(song_id)
    manifest = load_manifest(song_dir)
    rel = manifest.reference_loudness_path or "reference/loudness.json"
    path = song_dir / rel
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Reference loudness track not found")
    return JSONResponse(
        content=LoudnessTrack.model_validate_json(path.read_text(encoding="utf-8")).model_dump()
    )


@app.get("/api/songs/{song_id}/reference/stars")
def get_reference_stars(song_id: str):
    """Return the precomputed reference STARS track for a song."""
    song_dir = _song_dir(song_id)
    manifest = load_manifest(song_dir)
    rel = manifest.reference_stars_path or "reference/stars.json"
    path = song_dir / rel
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Reference STARS track not found")
    return JSONResponse(
        content=StarsTrack.model_validate_json(path.read_text(encoding="utf-8")).model_dump()
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
# Health check (mounted before static so it's always reachable)
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    """Lightweight liveness probe used by Cloudflare Tunnel monitoring and testers."""
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_ok else None
    except Exception:
        cuda_ok = False
        cuda_device = None

    songs = []
    if SONGS_ROOT.is_dir():
        songs = [d.name for d in sorted(SONGS_ROOT.iterdir()) if (d / "manifest.json").is_file()]

    return {
        "status": "ok",
        "uptime_s": round(time.time() - _SERVER_START_TIME),
        "cuda": cuda_ok,
        "cuda_device": cuda_device,
        "songs_available": len(songs),
        "song_ids": songs,
    }


# ---------------------------------------------------------------------------
# Static frontend (mounted last so /api routes take priority)
# ---------------------------------------------------------------------------


if STATIC_ROOT.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_ROOT), html=True), name="static")
