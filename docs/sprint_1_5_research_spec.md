# Sprint 1.5 Research Spec: Continuous Signals and Cross-Dimensional Detection

This document is the primary deliverable for Sprint 1.5.
It grounds the plan document in published vocal pedagogy and MIR research,
and provides concrete algorithmic and schema specifications for Sprint 2 implementation.

---

## Part 1: Vocal Pedagogy Catalogue

### Sources

- Yamamoto et al. (ISMIR 2022) — "Analysis and Detection of Singing Techniques in Repertoires of J-POP Solo Singers"
- Loscos et al. (AES 2003) — "Performance Analysis and Scoring of the Singing Voice" (MTG Barcelona)
- Nix / Baston Training (2024) — "Vibrato: A Guide for Singers and Singing Teachers"
- Larrouy-Maestri & Pfordresher (2018, J. Experimental Psychology) — "Pitch Perception in Music: Do Scoops Matter?"
- voicescience.org lexicon — "Vibrato Rate", "Breath Support"
- Roubeau et al. (2004) / UNSW acoustics — "The Mechanics and Acoustics of the Singing Voice: Registers"
- SingWise (2024) — "How to Eliminate Register Breaks"

---

### 1a. Continuous Pitch Features

#### Vibrato Characterisation

Vibrato is a periodic oscillation of F0 around a centre pitch, perceived as a natural release of breath pressure at the larynx. It has three measurable parameters:

| Parameter | Definition | Healthy classical range | Pathological range |
|---|---|---|---|
| Rate (Hz) | Oscillation cycles per second | 5.0–6.5 Hz | < 4 Hz (wobble), > 7 Hz (bleat) |
| Extent (cents peak-to-peak) | Total span of oscillation | 50–120 cents | < 20 cents (straight tone), > 150 cents (excessive) |
| Jitter (%) | Cycle-to-cycle rate irregularity | < 3% | > 10% (tremor) |

Pop/CCM norms differ — many pop singers use straight tone intentionally; vibrato is a stylistic choice, not a baseline requirement. STARS already outputs a binary `vibrato` flag per phoneme. The new `vibrato_rate_hz` and `vibrato_extent_cents` fields are **complementary**: they answer *how* the vibrato was performed, not *whether* it was present.

**Detection algorithm** for a voiced note window with N >= 30 frames (~300ms at 10ms hop):

1. Compute `cents[voiced]` relative to target MIDI (already done in `_measure_pitch()`)
2. Subtract the linear trend (detrend): `cents_detrended = cents - linreg_fit(cents)`
3. Compute autocorrelation: `acf = np.correlate(cents_detrended, cents_detrended, mode='full')`
4. Take the positive-lag half; find first peak after lag >= 3 frames (30ms minimum half-period = ~16 Hz max rate)
5. If a peak exists at lag `k` with correlation > 0.4, vibrato rate = `1 / (k * hop_s)`, extent = `2 * std(cents_detrended)`
6. Cross-reference: if STARS `vibrato` score < 0.1 and computed rate falls outside 4–7 Hz, treat as noise, return None

**Minimum note duration:** 300ms of voiced frames.

---

#### Scoop / Pitch Approach (Portamento On-attack)

A scoop is an upward glide into a note from below the target, completing during the attack window. Research shows:
- Typical scoop extent: 30–100 cents below target at the start of the attack, settling within ~220ms (Larrouy-Maestri & Pfordresher 2018)
- The most common singing technique in J-POP (>29 occurrences per singer, Yamamoto et al. 2022)
- Magnitudes depend on melodic context: upward-motion scoops average ~74 cents; downward-motion scoops ~30 cents

Pedagogically: scoops are acceptable as a stylistic device in pop/R&B but are discouraged in classical training as they obscure the target pitch. Identifying and quantifying them is valuable regardless of style.

**Detection algorithm** using `core_start_s` as the boundary between attack and sustain windows:

```
attack_window = pitch frames in [note.start_s, core_start_s]  # already trimmed by CoreWindowConfig
sustain_window = pitch frames in [core_start_s, core_end_s]

if len(attack_voiced) < 5 or len(sustain_voiced) < 5:
    return None  # not enough frames

attack_median_cents = median(cents[attack_voiced])
sustain_median_cents = median(cents[sustain_voiced])  # already computed as median_cents
scoop_cents = sustain_median_cents - attack_median_cents
# Positive = started below target (scoop from below)
# Negative = started above target (fall/overshoot)
```

Threshold: `|scoop_cents| > 25 cents` to distinguish from noise. Minimum attack window: ~150ms (15 frames).

---

#### Release Trajectory

How pitch exits the note: clean cutoff, downward fall (drop), or vibrato fade-out. This is pedagogically relevant to legato and phrasing.

**Detection algorithm**:

```
release_window = pitch frames in [core_end_s, note.end_s]
if len(release_voiced) < 5:
    return None
release_slope_cents_per_s = _slope(times[release_voiced], cents[release_voiced])
```

Interpretation:
- `release_slope < -150 cents/s`: falling release / "drop"
- `-150 to +150 cents/s`: clean release
- `> +150 cents/s`: rising release

---

#### Sustain Stability (Jitter)

Standard deviation of cents in the core window after removing vibrato oscillation. Distinguishes a steady straight tone from a trembling one with the same median pitch.

```
sustain_jitter_cents = std(cents[voiced_inside])
```

Low priority for Sprint 2. Vibrato extent (`2 * std`) already captures this; a separate field adds value only when STARS indicates no vibrato but jitter is high.

---

### 1b. Continuous Loudness Features

#### Dynamic Envelope Shape

Current detection reduces a note's loudness to a single linear slope (`rms_fade_db_per_s`), missing non-linear shapes. The MTG Barcelona system (Loscos et al. 2003) identifies five envelope archetypes: attack, sustain, vibrato-sustain, release, and transition. For our purposes a simpler 5-way classification is sufficient:

| Label | Acoustic signature |
|---|---|
| `sustain` | Flat slope, low RMS variance throughout |
| `crescendo` | Monotonically rising RMS |
| `decrescendo` | Monotonically falling RMS (same as `fade_db_per_s < 0` but explicitly classified) |
| `swell` | Rising in first half, falling in second half |
| `sforzando` | Sharp RMS peak in first ~100ms, then decays |

**Classification algorithm** (simple heuristic, no ML needed):

```
rms = rms_db[inside]
half = len(rms) // 2
slope_first  = _slope(times[:half], rms[:half])
slope_second = _slope(times[half:], rms[half:])

if abs(slope_first) < 1.0 and abs(slope_second) < 1.0:
    envelope_shape = "sustain"
elif slope_first > 2.0 and slope_second > 2.0:
    envelope_shape = "crescendo"
elif slope_first < -2.0 and slope_second < -2.0:
    envelope_shape = "decrescendo"
elif slope_first > 2.0 and slope_second < -2.0:
    envelope_shape = "swell"
elif (rms[:max(1, len(rms)//5)].mean() - rms[len(rms)//5:].mean()) > 3.0:
    envelope_shape = "sforzando"
else:
    envelope_shape = "sustain"  # default
```

Slope thresholds (dB/s) should be added to `HighlightsConfig` or a new `ShapeConfig` section.

---

#### Onset Ramp

How quickly voiced frames reach full amplitude at note start. A fast ramp = clean onset; slow ramp = breathy attack. Combines with low early `voicing_confidence` to diagnose a breathy onset.

```
onset_window = frames in [note.start_s, note.start_s + 0.08]  # 80ms
onset_ramp_db_per_s = _slope(times[onset_window], rms[onset_window])
```

Interpretation: `onset_ramp_db_per_s < 5 dB/s` for a 80ms window = slow/breathy onset.

---

#### Support Consistency (RMS Std)

Standard deviation of RMS in the core window. A note that starts well-supported but fades will have high variance even if `rms_fade_db_per_s` is moderate.

```
sustain_rms_std = float(np.std(rms[core_inside]))
```

Deferred alongside `sustain_jitter_cents`.

---

### 1c. Cross-Dimensional Techniques

#### Flat-with-Fade: Breath Support Diagnosis

Voice science literature (voicescience.org, 2024) describes breath support failure as unmanaged passive lung recoil driving inconsistent subglottal pressure. The acoustic signature is **co-occurring pitch flatness and amplitude decay**:

- `median_cents < -20` (flat)
- `rms_fade_db_per_s < -3.0 dB/s` (fading)

When both appear on the same note the most probable root cause is insufficient support rather than independent pitch and dynamics issues. Coaching copy should address the mechanism, not both symptoms separately.

**Pedagogical grounding**: Breath support is defined as active resistance to passive recoil; a singer losing support creates both falling pressure (volume drops) and destabilised vocal fold vibration (pitch sags). (voicescience.org lexicon; hvsconservatory.com, 2024).

---

#### Registration Strain: Sharp-When-Loud on High Notes

When a singer pushes chest voice above the passaggio, F1 tracks H2 beyond its natural boundary, creating forced phonation that drives pitch sharp under high subglottal pressure (Roubeau et al. 2004; UNSW acoustics). The acoustic signature:

- `midi_pitch >= voice_passaggio_midi` (note is in the upper register transition range)
- `median_cents > +20` (sharp)
- `user_rms_db > loudness_median_db + 3` (louder than the singer's own typical level)

For a genre-agnostic system, using `midi_pitch >= 69` (A4) as a rough proxy for the upper transition area is reasonable, though the exact pitch depends on voice type.

---

#### Controlled Crescendo

A phrase where the singer grows in volume while maintaining pitch accuracy. This is harder than either dimension alone and worth affirming explicitly.

- Phrase-level: `mean(rms_delta_db) > +2 dB` across a window
- AND `mean(pct_in_tune) > 0.65` across the same window

More precise than the current `dynamic_surge`, which fires purely on volume.

---

#### Technique Density vs. Pitch Accuracy

Sections or windows where the user uses many STARS techniques but pitch accuracy drops. Suggests attention to expression at the expense of fundamentals.

- `technique_density = count(notes_with_technique > 0) / window_size`
- `mean(pct_in_tune)` in the same window

If `technique_density > 0.5` AND `mean(pct_in_tune) < 0.5`, surface a moment: "Lots of expression here, but pitch suffers — try nailing the notes first, then adding texture."

---

## Part 2: Pipeline Architecture for Continuous Signal Extraction

### Design principle

Frame arrays live and die in `align_v2.py`. The `highlights.py` layer must never receive raw frames — only scalars on `NoteMeasurementV2`. Sprint 2 adds two new extraction functions (`_measure_pitch_shape`, `_measure_loudness_shape`) called from `measure_note()`.

### New fields on `NoteMeasurementV2` (schemas.py)

All fields are `Optional` with `None` default, maintaining backward compatibility with v4 `analysis.json` files.

```python
# Sub-note pitch shape (computed in _measure_pitch_shape)
scoop_cents: Optional[float] = Field(
    None,
    description=(
        "Signed pitch offset at attack onset relative to sustain median (cents). "
        "Positive = scooped from below; negative = fell from above. "
        "None when attack window is too short (< 150ms) or note is unvoiced."
    ),
)
release_slope_cents_per_s: Optional[float] = Field(
    None,
    description=(
        "Linear pitch slope in the release window (post core_end_s, cents/s). "
        "Negative = falling release; positive = rising."
    ),
)
vibrato_rate_hz: Optional[float] = Field(
    None,
    description=(
        "Dominant vibrato oscillation rate (Hz) detected from autocorrelation of "
        "detrended cents curve. Only populated when STARS vibrato score >= 0.1 "
        "and note duration >= 300ms."
    ),
)
vibrato_extent_cents: Optional[float] = Field(
    None,
    description=(
        "Peak-to-peak vibrato extent in cents. "
        "Computed as 2 * std(detrended_cents). "
        "Populated alongside vibrato_rate_hz."
    ),
)

# Sub-note loudness shape (computed in _measure_loudness_shape)
envelope_shape: Optional[str] = Field(
    None,
    description=(
        "Loudness envelope archetype for the note: "
        "one of 'sustain', 'crescendo', 'decrescendo', 'swell', 'sforzando'. "
        "None when loudness track is unavailable or note is too short."
    ),
)
onset_ramp_db_per_s: Optional[float] = Field(
    None,
    description=(
        "Mean RMS slope in the first 80ms of the note (dB/s). "
        "Low values (< 5 dB/s) indicate a breathy or slow attack onset."
    ),
)
```

### New extraction functions in `align_v2.py`

**`_measure_pitch_shape(note, pitch, core_start, core_end, *, config) -> dict`**

Called from `measure_note()` after `_measure_pitch()`. Receives the same `PitchArrays` slice.

Pseudocode:

```python
def _measure_pitch_shape(
    note: ReferenceNote,
    pitch: PitchArrays,
    core_start: float,
    core_end: float,
    *,
    config: CoachingConfig,
    stars_vibrato_score: Optional[float] = None,
    octave_shift_semitones: int = 0,
) -> dict:
    result = {
        "scoop_cents": None,
        "release_slope_cents_per_s": None,
        "vibrato_rate_hz": None,
        "vibrato_extent_cents": None,
    }
    target_midi = note.midi_pitch + octave_shift_semitones
    voiced = (pitch.voicing >= config.pitch.voicing_threshold) & (pitch.f0_hz > 0)

    # Scoop: compare attack cents to sustain median
    attack_mask = voiced & (pitch.times >= note.start_s) & (pitch.times < core_start)
    sustain_mask = voiced & (pitch.times >= core_start) & (pitch.times < core_end)
    attack_dur = float(core_start - note.start_s)
    if attack_mask.sum() >= 5 and attack_dur >= 0.15 and sustain_mask.sum() >= 2:
        attack_cents = np.nanmedian(_hz_to_cents(pitch.f0_hz[attack_mask], target_midi))
        sustain_cents = np.nanmedian(_hz_to_cents(pitch.f0_hz[sustain_mask], target_midi))
        result["scoop_cents"] = float(sustain_cents - attack_cents)

    # Release slope
    release_mask = voiced & (pitch.times >= core_end) & (pitch.times < note.end_s)
    if release_mask.sum() >= 5:
        release_cents = _hz_to_cents(pitch.f0_hz[release_mask], target_midi)
        result["release_slope_cents_per_s"] = _slope(pitch.times[release_mask], release_cents)

    # Vibrato: only run when STARS has flagged vibrato and note is long enough
    sustain_dur = float(core_end - core_start)
    if sustain_mask.sum() >= 30 and sustain_dur >= 0.30:
        if stars_vibrato_score is None or stars_vibrato_score >= 0.1:
            cents = _hz_to_cents(pitch.f0_hz[sustain_mask], target_midi)
            cents_detrended = cents - np.polyval(np.polyfit(np.arange(len(cents)), cents, 1), np.arange(len(cents)))
            acf = np.correlate(cents_detrended, cents_detrended, mode='full')
            acf = acf[len(acf)//2:]  # positive lags only
            acf /= (acf[0] + 1e-9)
            min_lag = max(3, int(round(1.0 / (7.0 * config.pitch.hop_seconds))))  # 7Hz max
            max_lag = int(round(1.0 / (4.0 * config.pitch.hop_seconds)))          # 4Hz min
            search = acf[min_lag:max_lag]
            if len(search) > 0 and search.max() > 0.4:
                peak_lag = min_lag + int(search.argmax())
                result["vibrato_rate_hz"] = 1.0 / (peak_lag * pitch.hop_seconds)
                result["vibrato_extent_cents"] = 2.0 * float(np.std(cents_detrended))
    return result
```

Note: `PitchArrays` does not currently expose `hop_seconds` to `_measure_pitch()`. The `hop_seconds` attribute already exists on the dataclass (line 64 of `align_v2.py`) — it just needs to be passed through, or read from `pitch.hop_seconds`.

**`_measure_loudness_shape(note, loudness, start_s, end_s, *, config) -> dict`**

```python
def _measure_loudness_shape(
    note: ReferenceNote,
    loudness: LoudnessArrays,
    start_s: float,
    end_s: float,
    *,
    config: CoachingConfig,
) -> dict:
    result = {"envelope_shape": None, "onset_ramp_db_per_s": None}
    inside = (loudness.times >= start_s) & (loudness.times < end_s)
    rms = loudness.rms_db[inside]
    t   = loudness.times[inside]
    if rms.size < 6:
        return result

    # Envelope shape
    half = len(rms) // 2
    slope_first  = _slope(t[:half], rms[:half].astype(np.float64))
    slope_second = _slope(t[half:], rms[half:].astype(np.float64))
    THRESH = 2.0  # dB/s threshold for "meaningfully sloped"
    if abs(slope_first) < THRESH and abs(slope_second) < THRESH:
        shape = "sustain"
    elif slope_first > THRESH and slope_second > THRESH:
        shape = "crescendo"
    elif slope_first < -THRESH and slope_second < -THRESH:
        shape = "decrescendo"
    elif slope_first > THRESH and slope_second < -THRESH:
        shape = "swell"
    elif rms[:max(1, len(rms)//5)].mean() - rms[len(rms)//5:].mean() > 3.0:
        shape = "sforzando"
    else:
        shape = "sustain"
    result["envelope_shape"] = shape

    # Onset ramp
    onset_end = start_s + 0.08
    onset_mask = (loudness.times >= start_s) & (loudness.times < onset_end)
    t_on = loudness.times[onset_mask]
    r_on = loudness.rms_db[onset_mask]
    if t_on.size >= 3:
        result["onset_ramp_db_per_s"] = _slope(t_on, r_on.astype(np.float64))
    return result
```

### Integration in `measure_note()`

```python
# After existing pitch_stats and loud are computed:
pitch_shape = _measure_pitch_shape(
    note, pitch_user, cs, ce,
    config=config,
    stars_vibrato_score=_max_vibrato_score(user_phones),  # from NoteTechniqueComparison
    octave_shift_semitones=octave_shift_semitones,
)
loudness_shape = (
    _measure_loudness_shape(note, loudness_user, note.start_s, note.end_s, config=config)
    if loudness_user is not None else {}
)

return NoteMeasurementV2(
    ...  # existing fields unchanged
    scoop_cents=pitch_shape.get("scoop_cents"),
    release_slope_cents_per_s=pitch_shape.get("release_slope_cents_per_s"),
    vibrato_rate_hz=pitch_shape.get("vibrato_rate_hz"),
    vibrato_extent_cents=pitch_shape.get("vibrato_extent_cents"),
    envelope_shape=loudness_shape.get("envelope_shape"),
    onset_ramp_db_per_s=loudness_shape.get("onset_ramp_db_per_s"),
)
```

`_max_vibrato_score()` is a small helper that takes the note's user phone list and returns the max `technique_scores["vibrato"]` across phonemes, or `None` if technique scores aren't available.

### Config additions (`coaching_config.py` + `coaching.yaml`)

New `ShapeConfig` dataclass:

```python
@dataclass
class ShapeConfig:
    scoop_min_cents: float = 25.0
    release_slope_drop_threshold: float = -150.0
    release_slope_rise_threshold: float = 150.0
    vibrato_acf_min_correlation: float = 0.4
    vibrato_min_rate_hz: float = 4.0
    vibrato_max_rate_hz: float = 7.0
    vibrato_min_duration_s: float = 0.30
    scoop_min_attack_s: float = 0.15
    envelope_slope_threshold_db_per_s: float = 2.0
    onset_ramp_breathy_threshold_db_per_s: float = 5.0
    sforzando_peak_advantage_db: float = 3.0
```

Added to `CoachingConfig` as `shape: ShapeConfig = field(default_factory=ShapeConfig)`.

### Schema version

`analysis_version` bumps `"v4"` to `"v5"` in `PerformanceAnalysis`.

---

## Part 3: Cross-Dimensional Detectors

### Design principle

Composite detectors in `highlights.py` read multiple fields from `NoteMeasurementV2` on the same note or window. They are structurally identical to existing detectors: they return `list[CoachingMoment]` and are appended to `candidates` in `select_highlights()`.

### New moment types and MOMENT_CATEGORY entries

| moment.type | MOMENT_CATEGORY | MOMENT_FEEDBACK_BASIS |
|---|---|---|
| `breath_support_issue` | `pitch` | `absolute` |
| `registration_strain` | `pitch` | `absolute` |
| `controlled_crescendo` | `dynamics` | `absolute` |
| `scoop_habit` | `pitch` | `absolute` |
| `vibrato_characterisation` | `technique` | `absolute` |

### `detect_breath_support_issues()`

```
Scan per-note: median_cents < -flat_threshold AND rms_fade_db_per_s < -fade_threshold
Cluster consecutive qualifying notes into runs.
Surface the longest run as a CoachingMoment.

Title: "Breath support dropping here"
Summary: "This phrase goes {N}c flat as your volume fades — focus on sustaining
          breath pressure through the end of each note."
```

Threshold candidates (tunable via config): `flat_threshold = 20.0 cents`, `fade_threshold = 3.0 dB/s`.
Requires at least 2 consecutive notes to qualify.

### `detect_registration_strain()`

```
Scan per-note: midi_pitch >= config.shape.passaggio_midi (default 69, A4)
               AND median_cents > +sharp_threshold
               AND user_rms_db > loudness.median_db + loud_threshold

Surface the worst note (highest median_cents) as a CoachingMoment.

Title: "Pushing on '{lyric_word}'"
Summary: "This {note_name} sits {N}c sharp while you're singing loud —
          try easing back the pressure or shifting into a lighter registration."
```

Threshold candidates: `sharp_threshold = 25 cents`, `loud_threshold = 3 dB`.

Note: `loudness.median_db` is the track-level anchor computed in `LoudnessArrays.from_track()` and already attached to the `LoudnessArrays` object. It is not currently stored on `NoteMeasurementV2`. Two options: (a) pass it through as a config/context value in `detect_registration_strain()`, or (b) add `user_rms_relative_db` = `user_rms_db - track_median_db` as a new field on `NoteMeasurementV2`.
Option (b) is cleaner and consistent with how `rms_delta_db` already normalises against both tracks.

### `detect_controlled_crescendo()`

```
Phrase-window scan (window_min..window_max notes, same as dynamic detectors).
For each window:
    mean_delta = mean(rms_delta_db[i] for i in window if not None)
    mean_pct = mean(pct_in_tune[i] for i in window if not None)
    if mean_delta > dynamic_threshold AND mean_pct > pct_in_tune_threshold:
        record as candidate; keep best-scoring window.

Title: "Controlled volume push"
Summary: "You built {delta:.1f} dB more energy here while staying {pct:.0f}% in tune —
          real dynamic control."
```

Threshold candidates: `dynamic_threshold = 2.0 dB`, `pct_in_tune_threshold = 0.60`.

### `detect_scoop_habit()`

```
Scan per-note: |scoop_cents| > config.shape.scoop_min_cents (25 cents default)
Collect all qualifying notes.
If count >= config.highlights.scoop_min_notes (default 3):
    Surface as a single CoachingMoment spanning the earliest to latest note.

Title: "Scooping into notes"
Summary: "You approach {count} notes with a {mean_scoop:.0f}c upward glide into the pitch —
          try landing directly on the target for a cleaner attack."
```

For stylistic/comparative framing (when the reference does it too), basis becomes `comparative`.

### `detect_vibrato_characterisation()`

```
Scan notes where vibrato_rate_hz is not None.
Compute mean and std of vibrato_rate_hz and vibrato_extent_cents across the performance.
Surface two moment types:
  (a) If std(rate) > 0.8 Hz: "Vibrato rate varies a lot" (coaching)
  (b) If mean(extent) > 120 cents: "Vibrato is quite wide" (coaching)
  (c) If mean(rate) in 5.0-6.5 Hz and std(rate) < 0.5 Hz: "Consistent vibrato" (affirming)
```

### Integration into `select_highlights()`

```python
# Continuous pitch
candidates.extend(detect_scoop_habit(reference, notes, config=cfg))
vib = detect_vibrato_characterisation(reference, notes, config=cfg)
if vib: candidates.append(vib)

# Cross-dimensional
breath = detect_breath_support_issues(reference, notes, config=cfg)
if breath: candidates.append(breath)
strain = detect_registration_strain(reference, notes, config=cfg)
if strain: candidates.append(strain)
candidates.extend(detect_controlled_crescendo(reference, notes, config=cfg))
```

No changes to `_select_diverse()` — the new types participate in the existing round-robin using their `MOMENT_CATEGORY` entries.

### Section-scope composite moments (deferred to Sprint 3 of Sprint 2)

The plan notes a future "technique density vs. pitch accuracy" section moment. This requires `SectionTrend` to track `technique_density` across expressive techniques (it already does, from Sprint 1) and `pct_in_tune`. Deferred because the section-level data is already rich enough to implement without sub-note features.

---

## Summary of Schema Changes for Sprint 2

| File | Change |
|---|---|
| `schemas.py` — `NoteMeasurementV2` | Add 6 Optional fields: `scoop_cents`, `release_slope_cents_per_s`, `vibrato_rate_hz`, `vibrato_extent_cents`, `envelope_shape`, `onset_ramp_db_per_s` |
| `schemas.py` — `PerformanceAnalysis` | `analysis_version` "v4" -> "v5" |
| `schemas.py` — `CoachingMoment` | No schema change needed; new types fit the existing model |
| `coaching_config.py` | Add `ShapeConfig` dataclass; add to `CoachingConfig` |
| `coaching.yaml` | Add `shape:` section |
| `align_v2.py` | Add `_measure_pitch_shape()`, `_measure_loudness_shape()`, update `measure_note()` |
| `highlights.py` | Add 5 new detector functions, update `MOMENT_CATEGORY`, `MOMENT_FEEDBACK_BASIS`, `select_highlights()` |
| `web/static/app.js` | Add new moment types to `MOMENT_TYPE_CATEGORY`, `MOMENT_FEEDBACK_BASIS`, `TYPE_LABEL` |
| `web/static/style.css` | Add `card.breath_support_issue`, `card.registration_strain`, etc. colour rules |
