/**
 * analysis-graph.js
 *
 * Canvas-based analysis graph that renders above the WaveSurfer waveform.
 * Switches between five visualization modes based on the active feedback filter:
 *
 *   null         -> Per-note mimic score line graph  (Reference Match)
 *   "pitch"      -> Continuous pitch curve + reference MIDI blocks
 *   "timing"     -> Note alignment overlay (ref vs. user, misalignment highlighted)
 *   "expression" -> Technique comparison two-row view with dropdown
 *   "volume"     -> Volume envelope (user line + reference filled area)
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PLAYHEAD_COLOR = "#3b6fd4";
const PITCH_LINE_COLOR = "#3b6fd4";
const REF_MIDI_BLOCK_COLOR = "rgba(120, 113, 108, 0.22)";
const REF_PITCH_LINE_COLOR = "rgba(120, 113, 108, 0.7)";

const GOOD_COLOR   = "#15803d";   // --good
const BAD_COLOR    = "#b91c1c";   // --bad
const NEUTRAL_COLOR = "#a8a29e";  // grey

const TIMING_REF_COLOR  = "rgba(120, 113, 108, 0.55)";
const TIMING_USER_COLOR = "rgba(59, 111, 212, 0.75)";
const TIMING_GAP_COLOR  = "rgba(59, 111, 212, 0.22)";

const TECH_NOTE_DEFAULT = "rgba(200, 195, 190, 0.6)";
const TECH_NOTE_HIT     = "#7c4dff";   // --expr purple

const VOL_REF_FILL   = "rgba(0, 0, 0, 0.08)";
const VOL_REF_STROKE = "rgba(120, 113, 108, 0.55)";
const VOL_USER_STROKE = "#3b6fd4";

// Stars technique names (same order as backend STARS_TECH_NAMES)
const STARS_TECH_NAMES = [
  "bubble", "breathe", "pharyngeal", "vibrato",
  "glissando", "mixed", "falsetto", "weak", "strong",
];

const TECH_DISPLAY = {
  bubble: "Bubble", breathe: "Breathe", pharyngeal: "Pharyngeal",
  vibrato: "Vibrato", glissando: "Glissando", mixed: "Mixed",
  falsetto: "Falsetto", weak: "Weak", strong: "Strong",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Convert f0 in Hz to MIDI pitch number.
 * Returns null for unvoiced (f0_hz <= 0).
 */
function hzToMidi(f0_hz) {
  if (f0_hz <= 0) return null;
  return 12 * Math.log2(f0_hz / 440) + 69;
}

/**
 * Get note name string for a MIDI number (e.g. 69 -> "A4").
 */
function midiToName(midi) {
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const octave = Math.floor(midi / 12) - 1;
  return names[midi % 12] + octave;
}

/**
 * Compute mean and standard deviation of an array of numbers.
 */
function meanStd(arr) {
  if (!arr.length) return { mean: 0.5, std: 0.2 };
  const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
  const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
  return { mean, std: Math.sqrt(variance) };
}

/**
 * Compute per-note mimic score using the same weights as the backend.
 * Returns a value in [0, 1], or null if no data is available.
 */
function computeNoteScore(note, techniqueMap) {
  const W_PITCH   = 0.60;
  const W_TECH    = 0.25;
  const W_ARRIVAL = 0.15;
  const LATE_MS   = 150;

  const components = [];

  if (note.pct_in_tune != null) {
    components.push([W_PITCH, note.pct_in_tune]);
  }

  const tech = techniqueMap.get(note.note_index);
  if (tech) {
    const refCount = tech.reference_techniques?.length ?? 0;
    const matchCount = tech.matched?.length ?? 0;
    const ratio = refCount > 0 ? matchCount / refCount : null;
    if (ratio != null) {
      components.push([W_TECH, ratio]);
    }
  }

  if (note.arrival_offset_ms != null) {
    const consistency = Math.max(0, 1 - Math.abs(note.arrival_offset_ms) / LATE_MS);
    components.push([W_ARRIVAL, consistency]);
  }

  if (!components.length) return null;

  // Redistribute weights proportionally when a component is missing
  const totalWeight = components.reduce((s, [w]) => s + w, 0);
  return components.reduce((s, [w, v]) => s + (w / totalWeight) * v, 0);
}

// ---------------------------------------------------------------------------
// AnalysisGraph class
// ---------------------------------------------------------------------------

export class AnalysisGraph {
  /**
   * @param {HTMLCanvasElement} canvas - The <canvas> element (sits directly in timeline-grid)
   * @param {HTMLSpanElement} label   - The label span
   * @param {HTMLSelectElement} techniqueSelect - The technique dropdown
   * @param {Function} apiUrl         - Function to prefix paths with API base
   * @param {Function} seekAndPlay    - Callback(userTimeSec) to seek + play
   */
  constructor(canvas, label, techniqueSelect, apiUrl, seekAndPlay) {
    this._canvas         = canvas;
    this._ctx            = canvas.getContext("2d");
    this._label          = label;
    this._techniqueSelect = techniqueSelect;
    this._apiUrl         = apiUrl;
    this._seekAndPlay    = seekAndPlay;

    this._analysis    = null;
    this._reference   = null;
    this._filter      = null;           // null | "pitch" | "timing" | "expression" | "volume"
    this._technique   = "vibrato";      // currently selected technique
    this._playheadFraction = 0;
    this._raf         = null;

    // Lazy-loaded data cache (cleared on new analysis)
    this._cache = {
      userPitch:    null,
      refPitch:     null,
      userLoudness: null,
      refLoudness:  null,
      refStars:     null,
    };

    // Bind event listeners
    this._onCanvasClick  = this._handleCanvasClick.bind(this);
    this._onTechChange   = this._handleTechChange.bind(this);
    this._resizeObserver = new ResizeObserver(() => this._onResize());

    canvas.addEventListener("click", this._onCanvasClick);
    techniqueSelect.addEventListener("change", this._onTechChange);
    // Observe the canvas itself — its size changes with the grid column
    this._resizeObserver.observe(canvas);
  }

  // ---- Public API ----

  /**
   * Bind new analysis + reference data. Resets the cache.
   */
  setAnalysis(analysis, reference) {
    this._analysis  = analysis;
    this._reference = reference;
    this._cache     = { userPitch: null, refPitch: null, userLoudness: null, refLoudness: null, refStars: null };
    this._playheadFraction = 0;
    this._syncCanvas();
    this._render();
  }

  /**
   * Switch the graph mode. filter is null | "pitch" | "timing" | "expression" | "volume".
   */
  setFilter(filter) {
    this._filter = filter;
    this._updateLabel();
    this._techniqueSelect.hidden = filter !== "expression";
    this._syncCanvas();
    this._renderWithFetch();
  }

  /**
   * Update the playhead position (fraction 0..1 through the song).
   */
  setPlayheadPosition(fraction) {
    this._playheadFraction = Math.max(0, Math.min(1, fraction));
    // Defer to rAF loop if already running; otherwise do a lightweight redraw
    if (!this._raf) {
      this._drawPlayhead();
    }
  }

  /**
   * Start a requestAnimationFrame loop that keeps the playhead moving smoothly.
   * Call when playback starts; stopPlayheadLoop() when paused.
   */
  startPlayheadLoop(getCurrentTime) {
    this._stopPlayheadLoop();
    const ctx = this._ctx;
    const loop = () => {
      const dur = this._analysis?.duration_s ?? 0;
      if (dur > 0) {
        this._playheadFraction = Math.max(0, Math.min(1, getCurrentTime() / dur));
      }
      const W = this._canvas.clientWidth;
      const H = this._canvas.clientHeight;
      if (W && H) {
        this._renderScene(ctx, W, H);
        this._overlayPlayhead(ctx, W, H);
      }
      this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  stopPlayheadLoop() {
    this._stopPlayheadLoop();
    this._drawPlayhead();
  }

  resize() {
    this._syncCanvas();
    this._render();
  }

  destroy() {
    this._stopPlayheadLoop();
    this._resizeObserver.disconnect();
    this._canvas.removeEventListener("click", this._onCanvasClick);
    this._techniqueSelect.removeEventListener("change", this._onTechChange);
  }

  // ---- Private: setup ----

  _syncCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w   = this._canvas.clientWidth;
    const h   = this._canvas.clientHeight;
    if (w <= 0 || h <= 0) return;
    this._canvas.width  = Math.round(w * dpr);
    this._canvas.height = Math.round(h * dpr);
    this._ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  _updateLabel() {
    const labels = {
      null:         "Reference\nMatch",
      pitch:        "Pitch",
      timing:       "Timing",
      expression:   "Technique",
      volume:       "Volume",
    };
    const key = this._filter === null ? "null" : this._filter;
    this._label.textContent = labels[key] ?? "Reference\nMatch";
  }

  _stopPlayheadLoop() {
    if (this._raf != null) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
  }

  // ---- Private: event handlers ----

  _handleCanvasClick(e) {
    if (!this._analysis) return;
    const rect = this._canvas.getBoundingClientRect();
    const fraction = (e.clientX - rect.left) / rect.width;
    const songTime = fraction * (this._analysis.duration_s ?? 0);
    // Convert song time -> user time using global_offset_s
    const userTime = songTime + (this._analysis.global_offset_s ?? 0);
    this._seekAndPlay(Math.max(0, userTime));
  }

  _handleTechChange() {
    this._technique = this._techniqueSelect.value;
    this._render();
  }

  _onResize() {
    this._syncCanvas();
    this._render();
  }

  // ---- Private: fetch helpers ----

  async _ensureUserPitch() {
    if (this._cache.userPitch) return this._cache.userPitch;
    const { song_id, perf_id } = this._analysis;
    try {
      const r = await fetch(this._apiUrl(
        `/api/songs/${encodeURIComponent(song_id)}/performances/${encodeURIComponent(perf_id)}/pitch`
      ));
      if (r.ok) this._cache.userPitch = await r.json();
    } catch { /* skip */ }
    return this._cache.userPitch;
  }

  async _ensureRefPitch() {
    if (this._cache.refPitch) return this._cache.refPitch;
    const { song_id } = this._analysis;
    try {
      const r = await fetch(this._apiUrl(`/api/songs/${encodeURIComponent(song_id)}/reference/pitch`));
      if (r.ok) this._cache.refPitch = await r.json();
    } catch { /* skip */ }
    return this._cache.refPitch;
  }

  async _ensureUserLoudness() {
    if (this._cache.userLoudness) return this._cache.userLoudness;
    const { song_id, perf_id } = this._analysis;
    try {
      const r = await fetch(this._apiUrl(
        `/api/songs/${encodeURIComponent(song_id)}/performances/${encodeURIComponent(perf_id)}/loudness`
      ));
      if (r.ok) this._cache.userLoudness = await r.json();
    } catch { /* skip */ }
    return this._cache.userLoudness;
  }

  async _ensureRefLoudness() {
    if (this._cache.refLoudness) return this._cache.refLoudness;
    const { song_id } = this._analysis;
    try {
      const r = await fetch(this._apiUrl(`/api/songs/${encodeURIComponent(song_id)}/reference/loudness`));
      if (r.ok) this._cache.refLoudness = await r.json();
    } catch { /* skip */ }
    return this._cache.refLoudness;
  }

  async _ensureRefStars() {
    if (this._cache.refStars) return this._cache.refStars;
    const { song_id } = this._analysis;
    try {
      const r = await fetch(this._apiUrl(`/api/songs/${encodeURIComponent(song_id)}/reference/stars`));
      if (r.ok) this._cache.refStars = await r.json();
    } catch { /* skip */ }
    return this._cache.refStars;
  }

  // ---- Private: render orchestration ----

  /**
   * Draw only the visualization scene (no playhead).
   * Called by _redraw() and by the rAF loop.
   */
  _renderScene(ctx, W, H) {
    ctx.clearRect(0, 0, W, H);
    if (!this._analysis) return;

    switch (this._filter) {
      case "pitch":      this._drawPitchGraph(ctx, W, H);      break;
      case "timing":     this._drawTimingGraph(ctx, W, H);     break;
      case "expression": this._drawTechniqueGraph(ctx, W, H);  break;
      case "volume":     this._drawVolumeGraph(ctx, W, H);     break;
      default:           this._drawMimicGraph(ctx, W, H);      break;
    }
  }

  /**
   * Full repaint: scene + playhead line. Use this for one-off redraws.
   */
  _render() {
    const ctx = this._ctx;
    const W   = this._canvas.clientWidth;
    const H   = this._canvas.clientHeight;
    if (!W || !H) return;

    this._renderScene(ctx, W, H);
    this._overlayPlayhead(ctx, W, H);
  }

  /**
   * Async render -- fetches missing data then re-renders.
   */
  async _renderWithFetch() {
    if (!this._analysis) return;

    // Start render immediately with cached data
    this._render();

    // Fetch missing data for the current mode
    let fetched = false;
    switch (this._filter) {
      case "pitch":
        await Promise.all([this._ensureUserPitch(), this._ensureRefPitch()]);
        fetched = true;
        break;
      case "expression":
        await this._ensureRefStars();
        this._populateTechniqueDropdown();
        fetched = true;
        break;
      case "volume":
        await Promise.all([this._ensureUserLoudness(), this._ensureRefLoudness()]);
        fetched = true;
        break;
    }

    if (fetched) this._render();
  }

  // ---- Private: playhead overlay ----

  /** Draw just the playhead line on top of the current canvas content. */
  _overlayPlayhead(ctx, W, H) {
    if (!this._analysis) return;
    const x = this._playheadFraction * W;
    ctx.save();
    ctx.strokeStyle = PLAYHEAD_COLOR;
    ctx.lineWidth   = 1.5;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
    ctx.restore();
  }

  /** Called from setPlayheadPosition for non-rAF updates. */
  _drawPlayhead() {
    const ctx = this._ctx;
    const W   = this._canvas.clientWidth;
    const H   = this._canvas.clientHeight;
    if (!W || !H || !this._analysis) return;
    this._render();
  }

  // ---- Private: time -> x coordinate ----

  _timeToX(songTime, W) {
    const dur = this._analysis?.duration_s ?? 1;
    return (songTime / dur) * W;
  }

  // ---- Private: renderer -- Mimic Score ----

  _drawMimicGraph(ctx, W, H) {
    const notes      = this._analysis?.notes ?? [];
    const techniques = this._analysis?.techniques ?? [];
    if (!notes.length) { this._drawEmpty(ctx, W, H, "No note data"); return; }

    const techMap  = new Map(techniques.map(t => [t.note_index, t]));
    const duration = this._analysis.duration_s || 1;

    // Per-note scores keyed to the note's midpoint in time
    const scored = [];
    for (const note of notes) {
      const score = computeNoteScore(note, techMap);
      if (score != null) scored.push({ t: (note.start_s + note.end_s) / 2, score });
    }
    if (!scored.length) { this._drawEmpty(ctx, W, H, "No score data"); return; }

    // --- Gaussian kernel smoothing over time ---
    // sigma scales with the song: ~3 s for a 3-minute track, minimum 2 s.
    const sigma      = Math.max(1.0, duration / 120);
    const inv2sig2   = 1 / (2 * sigma * sigma);
    const numSamples = Math.min(400, Math.ceil(W));

    const samples = [];
    for (let i = 0; i <= numSamples; i++) {
      const t = (i / numSamples) * duration;
      let wSum = 0, sSum = 0;
      for (const n of scored) {
        const d = t - n.t;
        const w = Math.exp(-(d * d) * inv2sig2);
        wSum += w;
        sSum += w * n.score;
      }
      samples.push({ t, score: wSum > 1e-6 ? sSum / wSum : 0 });
    }

    // Auto-scale Y to the smoothed range so the curve fills the canvas
    const allScores  = samples.map(s => s.score);
    const scoreMin   = Math.min(...allScores);
    const scoreMax   = Math.max(...allScores);
    const scoreRange = Math.max(0.05, scoreMax - scoreMin);
    const mean       = allScores.reduce((a, b) => a + b, 0) / allScores.length;

    const PAD_T = 10, PAD_B = 6;
    const scoreToY = s =>
      PAD_T + (1 - (s - scoreMin) / scoreRange) * (H - PAD_T - PAD_B);
    const meanY = scoreToY(mean);
    const baseY = meanY; // fills are bounded by the center line, not the bottom

    // Map samples to canvas points
    const pts = samples.map(s => ({
      x:     (s.t / duration) * W,
      y:     scoreToY(s.score),
      score: s.score,
    }));

    // --- Split into green / red runs at the mean threshold ---
    // Each run: array of {x, y, score}; crossover endpoints are interpolated.
    const greenRuns = [], redRuns = [];
    let curRun = [pts[0]];
    (pts[0].score >= mean ? greenRuns : redRuns).push(curRun);

    for (let i = 1; i < pts.length; i++) {
      const prev = pts[i - 1], curr = pts[i];
      const prevAbove = prev.score >= mean;
      const currAbove = curr.score >= mean;

      if (prevAbove !== currAbove) {
        // Interpolate the exact crossover point
        const frac   = (mean - prev.score) / (curr.score - prev.score);
        const crossPt = { x: prev.x + frac * (curr.x - prev.x), y: meanY, score: mean };
        curRun.push(crossPt);
        curRun = [crossPt, curr];
        (currAbove ? greenRuns : redRuns).push(curRun);
      } else {
        curRun.push(curr);
      }
    }

    // Helper: fill a run's area down to baseY
    const fillRun = (run, color) => {
      if (run.length < 2) return;
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.moveTo(run[0].x, baseY);
      for (const p of run) ctx.lineTo(p.x, p.y);
      ctx.lineTo(run[run.length - 1].x, baseY);
      ctx.closePath();
      ctx.fill();
    };

    // Helper: stroke a run's curve
    const strokeRun = (run, color) => {
      if (run.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2;
      ctx.lineJoin    = "round";
      ctx.lineCap     = "round";
      ctx.moveTo(run[0].x, run[0].y);
      for (let i = 1; i < run.length; i++) ctx.lineTo(run[i].x, run[i].y);
      ctx.stroke();
    };

    // Draw filled areas first (between curve and center line)
    for (const run of greenRuns) fillRun(run, "rgba(34, 197, 94, 0.30)");
    for (const run of redRuns)   fillRun(run, "rgba(239, 68, 68, 0.30)");

    // Center / average line
    ctx.save();
    ctx.strokeStyle = "rgba(120, 113, 108, 0.35)";
    ctx.lineWidth   = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, meanY);
    ctx.lineTo(W, meanY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Draw the colored curve on top
    for (const run of greenRuns) strokeRun(run, GOOD_COLOR);
    for (const run of redRuns)   strokeRun(run, BAD_COLOR);
  }

  // ---- Private: renderer -- Pitch ----

  _drawPitchGraph(ctx, W, H) {
    const refNotes  = this._reference?.notes ?? [];
    const userPitch = this._cache.userPitch;
    const analysis  = this._analysis;
    const offset    = analysis.global_offset_s ?? 0;
    const duration  = analysis.duration_s ?? Infinity;
    let   octaveShift = analysis.octave_shift_semitones ?? 0;

    if (!refNotes.length && !userPitch) {
      this._drawEmpty(ctx, W, H, "Loading pitch data…");
      return;
    }

    const refMidi = refNotes.map(n => n.midi_pitch).filter(Boolean);
    if (!refMidi.length) { this._drawEmpty(ctx, W, H, "No pitch data"); return; }

    // Auto-correct octave: if the user's median pitch (after existing shift) is
    // more than half an octave from the reference median, snap to the nearest
    // octave multiple.  This handles cases where octave_shift_semitones is
    // absent or wrong in the analysis JSON.
    if (userPitch?.frames?.length) {
      const voicedMidis = userPitch.frames
        .map(f => hzToMidi(f.f0_hz)).filter(m => m != null);
      if (voicedMidis.length) {
        voicedMidis.sort((a, b) => a - b);
        const userMedian = voicedMidis[Math.floor(voicedMidis.length / 2)] + octaveShift;
        const refSorted  = [...refMidi].sort((a, b) => a - b);
        const refMedian  = refSorted[Math.floor(refSorted.length / 2)];
        const residual   = refMedian - userMedian;
        const correction = Math.round(residual / 12) * 12;
        if (Math.abs(correction) >= 6) {
          console.debug(`[PitchGraph] auto-octave-correcting by ${correction} semitones (residual ${residual.toFixed(1)})`);
          octaveShift += correction;
        }
      }
    }

    // MIDI range anchored to the reference only — no user-pitch expansion needed.
    const rawMin = Math.min(...refMidi);
    const rawMax = Math.max(...refMidi);
    const midiMin = rawMin - 2;
    const midiMax = rawMax + 2;

    const PAD_TOP = 6, PAD_BOTTOM = 6;
    const midiToY = m => PAD_TOP + (1 - (m - midiMin) / (midiMax - midiMin)) * (H - PAD_TOP - PAD_BOTTOM);

    // --- Assign user pitch frames to reference notes (binary search, O(F log N)) ---
    // Build a sorted list of voiced frames with the octave-corrected MIDI value.
    const voicedFrames = [];
    if (userPitch?.frames?.length) {
      for (const f of userPitch.frames) {
        const midi = hzToMidi(f.f0_hz);
        if (midi == null) continue;
        const t = f.time - offset;
        if (t < 0 || t > duration) continue;
        voicedFrames.push({ t, midi: midi + octaveShift });
      }
      voicedFrames.sort((a, b) => a.t - b.t);
    }

    // Sort reference notes by start time for binary search.
    const sortedNotes = refNotes
      .map((n, i) => ({ ...n, _i: i }))
      .sort((a, b) => a.start_s - b.start_s);

    // noteMidis[i] = array of voiced user MIDI values that fall inside refNotes[i].
    const noteMidis = refNotes.map(() => []);
    for (const frame of voicedFrames) {
      let lo = 0, hi = sortedNotes.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const n   = sortedNotes[mid];
        if      (frame.t < n.start_s) hi = mid - 1;
        else if (frame.t > n.end_s)   lo = mid + 1;
        else { noteMidis[n._i].push(frame.midi); break; }
      }
    }

    // Color a block by median cents deviation from its reference MIDI pitch.
    // Opacity decreases with inaccuracy — secondary encoding for colorblind users.
    // In-tune notes are solid; off-tune notes fade, making them visually distinct
    // regardless of hue perception.
    const blockColor = (refPitch, userMidis) => {
      if (!userMidis.length) return REF_MIDI_BLOCK_COLOR; // no user data — grey
      const sorted    = [...userMidis].sort((a, b) => a - b);
      const median    = sorted[Math.floor(sorted.length / 2)];
      const absCents  = Math.abs((median - refPitch) * 100);
      if (absCents < 25) return "rgba(34, 197, 94, 0.85)";  // green — in tune,       solid
      if (absCents < 75) return "rgba(234, 179, 8, 0.55)";  // amber — slightly off,  faded
      return             "rgba(239, 68,  68, 0.35)";         // red   — far off,       very faded
    };

    // Draw color-coded MIDI blocks.
    for (let i = 0; i < refNotes.length; i++) {
      const note = refNotes[i];
      const x1   = this._timeToX(note.start_s, W);
      const x2   = this._timeToX(note.end_s,   W);
      const y1   = midiToY(note.midi_pitch + 0.45);
      const y2   = midiToY(note.midi_pitch - 0.45);
      ctx.save();
      ctx.fillStyle = blockColor(note.midi_pitch, noteMidis[i]);
      ctx.beginPath();
      ctx.roundRect(x1, y1, Math.max(x2 - x1, 2), Math.max(y2 - y1, 2), 2);
      ctx.fill();
      ctx.restore();
    }

    // Y-axis note-name labels (spaced to avoid overlap).
    const uniqueMidi  = [...new Set(refMidi)].sort((a, b) => a - b);
    const minLabelGap = 12;
    let lastLabelY    = -Infinity;
    ctx.save();
    ctx.font      = "9px 'DM Sans', sans-serif";
    ctx.fillStyle = "rgba(120, 113, 108, 0.7)";
    ctx.textAlign = "right";
    for (const midi of uniqueMidi) {
      const y = midiToY(midi);
      if (y - lastLabelY < minLabelGap) continue;
      ctx.fillText(midiToName(midi), -4, y + 3);
      lastLabelY = y;
    }
    ctx.restore();

    // Show a subtle loading hint when user pitch is still being fetched.
    if (!userPitch) this._drawEmpty(ctx, W, H, "Loading pitch data…");
  }

  /**
   * Draw a continuous pitch line from frame data.
   * @param {PitchFrame[]} frames
   * @param {number} offsetS - global_offset_s to apply to frame times
   * @param {Function} midiToY
   * @param {string} color
   * @param {number[]} dash - line dash pattern
   * @param {number} width
   * @param {number} octaveShift - semitones to add to detected MIDI
   */
  _drawPitchLine(ctx, W, H, frames, offsetS, midiToY, color, dash, width, octaveShift = 0) {
    const duration    = this._analysis?.duration_s ?? Infinity;
    // Gaps shorter than this (seconds) are bridged — hides consonant/breath blips.
    const GAP_BRIDGE  = 0.18;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth   = width;
    ctx.lineCap     = "round";
    ctx.lineJoin    = "round";
    if (dash.length) ctx.setLineDash(dash);

    // Downsample: aim for at most 1 point per 2 CSS pixels
    const step = Math.max(1, Math.floor(frames.length / (W / 2)));

    ctx.beginPath();
    let penDown       = false;
    let lastVoicedT   = -Infinity; // song-time of the last voiced frame
    let lastVoicedX   = 0;
    let lastVoicedY   = 0;

    for (let i = 0; i < frames.length; i += step) {
      const f        = frames[i];
      const songTime = f.time - offsetS;
      if (songTime < 0 || songTime > duration) { penDown = false; continue; }

      const midi = hzToMidi(f.f0_hz);
      if (midi == null) {
        // Unvoiced — don't lift pen yet; we'll decide when the next voiced
        // frame arrives (gap bridging).
        penDown = false;
        continue;
      }

      const x = this._timeToX(songTime, W);
      const y = midiToY(midi + octaveShift);

      if (!penDown) {
        const gap = songTime - lastVoicedT;
        if (gap <= GAP_BRIDGE && lastVoicedT > 0) {
          // Short gap — interpolate a connecting segment instead of lifting
          ctx.lineTo(x, y);
        } else {
          ctx.moveTo(x, y);
        }
        penDown = true;
      } else {
        ctx.lineTo(x, y);
      }

      lastVoicedT = songTime;
      lastVoicedX = x;
      lastVoicedY = y;
    }
    ctx.stroke();
    ctx.restore();
  }

  // ---- Private: renderer -- Timing ----

  _drawTimingGraph(ctx, W, H) {
    const notes    = this._analysis?.notes ?? [];
    const refNotes = this._reference?.notes ?? [];

    // Collect per-note arrival offsets, keyed to the reference note's midpoint.
    const refMap = new Map(refNotes.map(n => [n.index, n]));
    const scored = [];
    for (const note of notes) {
      if (note.arrival_offset_ms == null) continue;
      const ref = refMap.get(note.note_index);
      if (!ref) continue;
      scored.push({ t: (ref.start_s + ref.end_s) / 2, offsetMs: note.arrival_offset_ms });
    }

    if (!scored.length) { this._drawEmpty(ctx, W, H, "No timing data"); return; }

    const duration = this._analysis.duration_s || 1;

    // --- Gaussian kernel smoothing (same sigma logic as reference match) ---
    const sigma      = Math.max(1.0, duration / 120);
    const inv2sig2   = 1 / (2 * sigma * sigma);
    const numSamples = Math.min(400, Math.ceil(W));

    const samples = [];
    for (let i = 0; i <= numSamples; i++) {
      const t = (i / numSamples) * duration;
      let wSum = 0, sSum = 0;
      for (const n of scored) {
        const d = t - n.t;
        const w = Math.exp(-(d * d) * inv2sig2);
        wSum += w;
        sSum += w * n.offsetMs;
      }
      samples.push({ t, offsetMs: wSum > 1e-6 ? sSum / wSum : 0 });
    }

    // Symmetric Y range so 0ms sits exactly at the canvas midpoint.
    const PAD_T    = 10, PAD_B = 6;
    const maxAbs   = Math.max(30, ...samples.map(s => Math.abs(s.offsetMs)));
    const halfH    = H / 2 - Math.max(PAD_T, PAD_B);
    const centerY  = H / 2;
    // positive offsetMs = late = below center (higher Y); negative = early = above
    const offsetToY = ms => centerY + (ms / maxAbs) * halfH;

    const pts = samples.map(s => ({
      x: (s.t / duration) * W,
      y: offsetToY(s.offsetMs),
      offsetMs: s.offsetMs,
    }));

    // --- Split into early / late runs at the center line (offsetMs = 0) ---
    const earlyRuns = [], lateRuns = [];
    let curRun = [pts[0]];
    (pts[0].offsetMs <= 0 ? earlyRuns : lateRuns).push(curRun);

    for (let i = 1; i < pts.length; i++) {
      const prev = pts[i - 1], curr = pts[i];
      const prevEarly = prev.offsetMs <= 0;
      const currEarly = curr.offsetMs <= 0;

      if (prevEarly !== currEarly) {
        const frac    = (0 - prev.offsetMs) / (curr.offsetMs - prev.offsetMs);
        const crossPt = { x: prev.x + frac * (curr.x - prev.x), y: centerY, offsetMs: 0 };
        curRun.push(crossPt);
        curRun = [crossPt, curr];
        (currEarly ? earlyRuns : lateRuns).push(curRun);
      } else {
        curRun.push(curr);
      }
    }

    const fillRun = (run, color) => {
      if (run.length < 2) return;
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.moveTo(run[0].x, centerY);
      for (const p of run) ctx.lineTo(p.x, p.y);
      ctx.lineTo(run[run.length - 1].x, centerY);
      ctx.closePath();
      ctx.fill();
    };

    const strokeRun = (run, color) => {
      if (run.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2;
      ctx.lineJoin    = "round";
      ctx.lineCap     = "round";
      ctx.moveTo(run[0].x, run[0].y);
      for (let i = 1; i < run.length; i++) ctx.lineTo(run[i].x, run[i].y);
      ctx.stroke();
    };

    // Fill: blue above center (early / rushing), amber below (late / dragging)
    for (const run of earlyRuns) fillRun(run, "rgba(59, 111, 212, 0.25)");
    for (const run of lateRuns)  fillRun(run, "rgba(234, 179, 8, 0.28)");

    // Center / on-beat line
    ctx.save();
    ctx.strokeStyle = "rgba(120, 113, 108, 0.35)";
    ctx.lineWidth   = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(W, centerY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Colored curve on top of fills
    for (const run of earlyRuns) strokeRun(run, "#3b6fd4");
    for (const run of lateRuns)  strokeRun(run, "#d97706");

    // Axis labels drawn left of the canvas (same pattern as pitch labels)
    ctx.save();
    ctx.font      = "9px 'DM Sans', sans-serif";
    ctx.fillStyle = "rgba(120, 113, 108, 0.7)";
    ctx.textAlign = "right";
    ctx.fillText("early", -4, PAD_T + 9);
    ctx.fillText("on beat", -4, centerY + 3);
    ctx.fillText("late",  -4, H - PAD_B - 2);
    ctx.restore();
  }

  // ---- Private: renderer -- Technique ----

  _populateTechniqueDropdown() {
    const sel = this._techniqueSelect;
    // Only rebuild if empty
    if (sel.options.length > 0) return;

    // Determine which techniques appear in analysis data
    const techniques = this._analysis?.techniques ?? [];
    const refTechs   = new Set();
    const userTechs  = new Set();
    for (const t of techniques) {
      (t.reference_techniques ?? []).forEach(x => refTechs.add(x));
      (t.user_techniques      ?? []).forEach(x => userTechs.add(x));
    }

    // Also check reference STARS phonemes
    const refStars = this._cache.refStars;
    if (refStars?.phonemes) {
      for (const ph of refStars.phonemes) {
        for (const [tech, val] of Object.entries(ph.techniques ?? {})) {
          if (val) refTechs.add(tech);
        }
      }
    }

    const allTechs = STARS_TECH_NAMES.filter(
      t => refTechs.has(t) || userTechs.has(t)
    );
    // Fallback to full list if nothing found
    const techList = allTechs.length ? allTechs : STARS_TECH_NAMES;

    sel.innerHTML = "";
    for (const t of techList) {
      const opt = document.createElement("option");
      opt.value       = t;
      opt.textContent = TECH_DISPLAY[t] || t;
      sel.appendChild(opt);
    }

    // Default to vibrato if present, else first
    const defaultTech = techList.includes("vibrato") ? "vibrato" : techList[0];
    sel.value       = defaultTech;
    this._technique = defaultTech;
  }

  _drawTechniqueGraph(ctx, W, H) {
    const refNotes   = this._reference?.notes ?? [];
    const techniques = this._analysis?.techniques ?? [];
    if (!refNotes.length) { this._drawEmpty(ctx, W, H, "No note data"); return; }

    const tech = this._technique;

    // Build lookup: note_index -> { userHas, refHas }
    const techMap = new Map(techniques.map(t => [t.note_index, {
      userHas: t.user_techniques?.includes(tech) ?? false,
      refHas:  t.reference_techniques?.includes(tech) ?? false,
    }]));

    // Also check reference STARS phonemes for reference technique presence
    const refStars = this._cache.refStars;
    const refStarsTechNotes = new Set();
    if (refStars?.phonemes) {
      for (const ph of refStars.phonemes) {
        if (ph.techniques?.[tech]) {
          // Map phoneme time range to reference notes
          for (const refNote of refNotes) {
            if (ph.start_s < refNote.end_s && ph.end_s > refNote.start_s) {
              refStarsTechNotes.add(refNote.index);
            }
          }
        }
      }
    }

    const LANE_PAD   = 6;
    const LABEL_H    = 24;  // pixels reserved above each note row for the label
    const laneH      = (H - LANE_PAD * 3 - LABEL_H * 2) / 2;
    const NOTE_RADIUS = 3;
    const NOTE_GAP   = 2;

    // User lane sits below its label; reference lane sits below a second label.
    const userLabelY = LANE_PAD;
    const userLaneY  = userLabelY + LABEL_H;
    const refLabelY  = userLaneY + laneH + LANE_PAD;
    const refLaneY   = refLabelY + LABEL_H;

    // Row labels drawn above their respective note rows
    ctx.save();
    ctx.font      = "9px 'DM Sans', sans-serif";
    ctx.fillStyle = "rgba(120, 113, 108, 0.8)";
    ctx.textAlign = "left";
    ctx.fillText("Your Performance", 12, userLabelY + LABEL_H - 12);
    ctx.fillText("Original Song",    12, refLabelY  + LABEL_H - 12);
    ctx.restore();

    for (const refNote of refNotes) {
      const x1 = this._timeToX(refNote.start_s, W);
      const x2 = this._timeToX(refNote.end_s,   W);
      const nw  = Math.max(x2 - x1 - NOTE_GAP, 2);
      const nx  = x1 + NOTE_GAP / 2;

      const info = techMap.get(refNote.index);
      const userHas = info?.userHas ?? false;
      const refHas  = (info?.refHas ?? false) || refStarsTechNotes.has(refNote.index);

      // User row
      ctx.save();
      ctx.fillStyle = userHas ? TECH_NOTE_HIT : TECH_NOTE_DEFAULT;
      ctx.beginPath();
      ctx.roundRect(nx, userLaneY, nw, laneH, NOTE_RADIUS);
      ctx.fill();
      ctx.restore();

      // Reference row
      ctx.save();
      ctx.fillStyle = refHas ? TECH_NOTE_HIT : TECH_NOTE_DEFAULT;
      ctx.beginPath();
      ctx.roundRect(nx, refLaneY, nw, laneH, NOTE_RADIUS);
      ctx.fill();
      ctx.restore();
    }
  }

  // ---- Private: renderer -- Volume ----

  _drawVolumeGraph(ctx, W, H) {
    const userLoudness = this._cache.userLoudness;
    const refLoudness  = this._cache.refLoudness;

    if (!userLoudness && !refLoudness) {
      this._drawEmpty(ctx, W, H, "Loading volume data…");
      return;
    }

    const PAD_TOP = 8, PAD_BOTTOM = 8;

    // Gather all dB values to establish y-axis range
    const allDb = [];
    if (refLoudness?.frames) {
      refLoudness.frames.forEach(f => { if (f.rms_db > -100) allDb.push(f.rms_db); });
    }
    if (userLoudness?.frames) {
      userLoudness.frames.forEach(f => { if (f.rms_db > -100) allDb.push(f.rms_db); });
    }

    const dbMin = allDb.length ? Math.max(Math.min(...allDb) - 4, -80) : -60;
    const dbMax = allDb.length ? Math.min(Math.max(...allDb) + 4,   0) : 0;

    const dbToY = db => PAD_TOP + (1 - (db - dbMin) / (dbMax - dbMin)) * (H - PAD_TOP - PAD_BOTTOM);

    /**
     * Build canvas path points from a loudness track.
     * For the reference track offset is 0; for the user track, apply global_offset_s.
     */
    const buildPath = (track, offsetS) => {
      const frames = track?.frames ?? [];
      const dur    = this._analysis?.duration_s ?? 1;
      const step   = Math.max(1, Math.floor(frames.length / (W / 2)));
      const pts    = [];
      for (let i = 0; i < frames.length; i += step) {
        const f       = frames[i];
        const songT   = f.time - offsetS;
        if (songT < 0 || songT > dur) continue;
        const x = this._timeToX(songT, W);
        const y = dbToY(Math.max(f.rms_db, dbMin));
        pts.push([x, y]);
      }
      return pts;
    };

    const offset = this._analysis?.global_offset_s ?? 0;
    const refPts  = buildPath(refLoudness,  0);
    const userPts = buildPath(userLoudness, offset);

    // Draw reference: filled area first, then stroke
    if (refPts.length > 1) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(refPts[0][0], H - PAD_BOTTOM);
      for (const [x, y] of refPts) ctx.lineTo(x, y);
      ctx.lineTo(refPts[refPts.length - 1][0], H - PAD_BOTTOM);
      ctx.closePath();
      ctx.fillStyle = VOL_REF_FILL;
      ctx.fill();

      ctx.beginPath();
      let first = true;
      for (const [x, y] of refPts) {
        first ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        first = false;
      }
      ctx.strokeStyle = VOL_REF_STROKE;
      ctx.lineWidth   = 1.5;
      ctx.stroke();
      ctx.restore();
    }

    // Draw user: solid blue line
    if (userPts.length > 1) {
      ctx.save();
      ctx.beginPath();
      let first = true;
      for (const [x, y] of userPts) {
        first ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        first = false;
      }
      ctx.strokeStyle = VOL_USER_STROKE;
      ctx.lineWidth   = 2;
      ctx.lineJoin    = "round";
      ctx.stroke();
      ctx.restore();
    }

    // dB axis ticks (right side)
    ctx.save();
    ctx.font      = "9px 'DM Sans', sans-serif";
    ctx.fillStyle = "rgba(120, 113, 108, 0.6)";
    ctx.textAlign = "right";
    const tickStep = Math.ceil((dbMax - dbMin) / 4);
    for (let db = Math.ceil(dbMin / tickStep) * tickStep; db <= dbMax; db += tickStep) {
      const y = dbToY(db);
      ctx.fillText(`${db}dB`, W - 4, y + 3);
      ctx.save();
      ctx.strokeStyle = "rgba(120, 113, 108, 0.12)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W - 40, y);
      ctx.stroke();
      ctx.restore();
    }
    ctx.restore();
  }

  // ---- Private: empty state ----

  _drawEmpty(ctx, W, H, message) {
    ctx.save();
    ctx.font      = "11px 'DM Sans', sans-serif";
    ctx.fillStyle = "rgba(120, 113, 108, 0.5)";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(message, W / 2, H / 2);
    ctx.restore();
  }
}
