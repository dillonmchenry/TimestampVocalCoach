"""Vocal Coach pipeline package.

Sprint 1 modules:
    - schemas: Pydantic models for every JSON artifact in the pipeline
    - reference: parse a GTSinger sample directory into reference_annotation.json
    - pitch: NanoPitch wrapper -> pitch.json
    - stars_runner: STARS subprocess wrapper -> stars.json
    - loudness: per-frame RMS / dBFS -> loudness.json
    - align: aggregate one ReferenceNote into a NoteCard

Sprint 2 modules:
    - ultrastar: UltraStar (.txt) chart parser
    - song: build a song bundle (manifest.json + reference_annotation.json)
            from an UltraStar chart
    - align_v2: dual-track (reference vs user) note measurements
    - highlights: deterministic coaching-moment detectors
    - coaching_config: tunable thresholds for align_v2 + highlights
"""

__version__ = "0.2.0"
