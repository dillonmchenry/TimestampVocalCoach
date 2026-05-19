"""Vocal Coach pipeline package.

Sprint 1 modules:
    - schemas: Pydantic models for every JSON artifact in the pipeline
    - reference: parse a GTSinger sample directory into reference_annotation.json
    - pitch: NanoPitch wrapper -> pitch.json
    - stars_runner: STARS subprocess wrapper -> stars.json
    - loudness: per-frame RMS / dBFS -> loudness.json
"""

__version__ = "0.1.0"
