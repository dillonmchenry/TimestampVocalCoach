import WaveSurfer from "https://unpkg.com/wavesurfer.js@7.8.6/dist/wavesurfer.esm.js";

/**
 * Prefix a /api/... path with the configured API base URL.
 * window.API_BASE is set by config.js (empty string for same-origin local dev,
 * or the full RunPod URL for Vercel-hosted deployments).
 */
function apiUrl(path) {
  const base = (window.API_BASE || "").replace(/\/$/, "");
  return base + path;
}

const songSelect = document.getElementById("songSelect");
const songBadges = document.getElementById("songBadges");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const analyzeButton = document.getElementById("analyzeButton");
const status = document.getElementById("status");
const analysisProgress = document.getElementById("analysisProgress");
const analysisCancelBtn = document.getElementById("analysisCancelBtn");
const uploadSection = document.querySelector("section.upload");
const results = document.getElementById("results");
const waveformDiv = document.getElementById("waveform");
const sectionRibbon = document.getElementById("sectionRibbon");
const timelineDiv = document.getElementById("timeline");
const offsetSummary = document.getElementById("offsetSummary");
const playPause = document.getElementById("playPause");
const mixRefVocalBtn = document.getElementById("mixRefVocal");
const mixInstrumentalBtn = document.getElementById("mixInstrumental");
const highlightList = document.getElementById("highlightList");
const highlightFilters = document.getElementById("highlightFilters");
const basisFilters = document.getElementById("basisFilters");
const overviewTiles = document.getElementById("overviewTiles");
const segmentInfo = document.getElementById("segmentInfo");
const fastProfile = document.getElementById("fastProfile");

let pendingFile = null;
let analysisAbortController = null;
let wavesurfer = null;
let currentMedia = null;
let currentDuration = 0;
let currentAnalysis = null;
let currentPlaybackSongId = null;
let mixRefVocalOn = false;
let mixInstrumentalOn = false;
let refVocalMedia = null;
let instrumentalMedia = null;
let playbackSyncAbort = null;

const MIX_LAYER_VOLUME = 0.85;
const MIX_SYNC_THRESHOLD_S = 0.08;
let currentReference = null;
let currentReferenceSongId = null;

const TYPE_LABEL = {
  // Existing
  best_pitch_phrase: "Cleanest run",
  pitch_struggle: "Tricky passage",
  sharp_flat_note: "Note callout",
  expressive_match: "Matched expression",
  expressive_moment: "Your expression",
  missed_expression: "Missed expression",
  vocal_texture: "Vocal colour",
  late_entrance: "Watch entrance",
  timing_consistency: "Timing drift",
  fade_within_notes: "Voice fading",
  dynamic_drop: "Pulling back",
  dynamic_surge: "Great energy",
  section_strength: "Section strength",
  section_weakness: "Section weakness",
  section_delta: "Section comparison",
  section_dynamic_contrast: "Dynamic contrast",
  best_overall_section: "Best section",
  weakest_overall_section: "Focus area",
  // Sprint 2: continuous pitch
  scoop_habit: "Pitch approach",
  pitch_overshoot: "Overshooting",
  clean_attack: "Clean attack",
  falling_release: "Falling release",
  steady_sustain: "Steady sustain",
  pitch_instability: "Pitch wobble",
  vibrato_quality: "Vibrato quality",
  consistent_vibrato: "Consistent vibrato",
  wide_vibrato: "Wide vibrato",
  delayed_vibrato: "Vibrato timing",
  straight_tone_control: "Straight tone",
  // Sprint 2: continuous loudness
  breathy_onset: "Breathy onset",
  clean_onset: "Clean onset",
  sforzando_attack: "Strong accent",
  note_crescendo: "Note crescendo",
  note_swell: "Note swell",
  support_fade: "Support fade",
  dynamic_sustain: "Steady volume",
  release_cutoff: "Abrupt ending",
  // Sprint 2: cross-dimensional
  breath_support_issue: "Breath support",
  registration_strain: "Register push",
  controlled_crescendo: "Dynamic control",
  loud_pitch_instability: "Loud + off pitch",
  soft_passage_control: "Quiet accuracy",
  vibrato_with_support: "Supported vibrato",
  scoop_with_fade: "Compound onset",
  technique_accuracy_tradeoff: "Expression vs pitch",
  expressive_stability: "Expression + pitch",
  high_note_control: "High note control",
  // Sprint 2: phrase / section
  rushed_phrase: "Rushing",
  dragged_phrase: "Dragging",
  rhythmic_precision: "Locked timing",
  phrase_pitch_arc: "Phrase arc",
  section_improvement: "Getting better",
  section_regression: "Dropping off",
  section_vibrato_contrast: "Vibrato contrast",
};

/** Maps backend moment.type -> feedback_basis (mirrors MOMENT_FEEDBACK_BASIS in highlights.py).
 *  The actual basis for dual-basis types is stamped by the backend on moment.feedback_basis. */
const MOMENT_FEEDBACK_BASIS = {
  // Existing
  best_pitch_phrase: "absolute",
  pitch_struggle: "absolute",
  sharp_flat_note: "absolute",
  late_entrance: "absolute",
  timing_consistency: "absolute",
  section_delta: "comparative",
  fade_within_notes: "absolute",
  dynamic_drop: "absolute",
  dynamic_surge: "absolute",
  section_strength: "absolute",
  section_weakness: "absolute",
  best_overall_section: "absolute",
  weakest_overall_section: "absolute",
  section_dynamic_contrast: "absolute",
  expressive_match: "comparative",
  expressive_moment: "comparative",
  missed_expression: "comparative",
  vocal_texture: "comparative",
  // Sprint 2 (dual defaults to absolute; actual basis stamped by backend)
  scoop_habit: "absolute",
  pitch_overshoot: "absolute",
  clean_attack: "absolute",
  falling_release: "absolute",
  steady_sustain: "absolute",
  pitch_instability: "absolute",
  vibrato_quality: "absolute",
  consistent_vibrato: "absolute",
  wide_vibrato: "absolute",
  delayed_vibrato: "comparative",
  straight_tone_control: "absolute",
  breathy_onset: "absolute",
  clean_onset: "absolute",
  sforzando_attack: "absolute",
  note_crescendo: "absolute",
  note_swell: "comparative",
  support_fade: "absolute",
  dynamic_sustain: "absolute",
  release_cutoff: "absolute",
  breath_support_issue: "absolute",
  registration_strain: "absolute",
  controlled_crescendo: "comparative",
  loud_pitch_instability: "absolute",
  soft_passage_control: "absolute",
  vibrato_with_support: "absolute",
  scoop_with_fade: "absolute",
  technique_accuracy_tradeoff: "comparative",
  expressive_stability: "comparative",
  high_note_control: "absolute",
  rushed_phrase: "absolute",
  dragged_phrase: "absolute",
  rhythmic_precision: "absolute",
  phrase_pitch_arc: "absolute",
  section_improvement: "absolute",
  section_regression: "absolute",
  section_vibrato_contrast: "comparative",
};

const BASIS_DISPLAY = {
  absolute: "Technique",
  comparative: "Style",
};

/** Maps backend moment.type -> feedback category (matches highlight engine). */
const MOMENT_TYPE_CATEGORY = {
  // Existing
  best_pitch_phrase: "pitch",
  pitch_struggle: "pitch",
  sharp_flat_note: "pitch",
  section_strength: "pitch",
  section_weakness: "pitch",
  best_overall_section: "pitch",
  weakest_overall_section: "pitch",
  expressive_match: "expression",
  expressive_moment: "expression",
  missed_expression: "expression",
  vocal_texture: "expression",
  late_entrance: "timing",
  timing_consistency: "timing",
  section_delta: "timing",
  fade_within_notes: "volume",
  dynamic_drop: "volume",
  dynamic_surge: "volume",
  section_dynamic_contrast: "volume",
  // Sprint 2: continuous pitch
  scoop_habit: "pitch",
  pitch_overshoot: "pitch",
  clean_attack: "pitch",
  falling_release: "pitch",
  steady_sustain: "pitch",
  pitch_instability: "pitch",
  vibrato_quality: "expression",
  consistent_vibrato: "expression",
  wide_vibrato: "expression",
  delayed_vibrato: "expression",
  straight_tone_control: "expression",
  // Sprint 2: continuous loudness
  breathy_onset: "volume",
  clean_onset: "volume",
  sforzando_attack: "volume",
  note_crescendo: "volume",
  note_swell: "volume",
  support_fade: "volume",
  dynamic_sustain: "volume",
  release_cutoff: "volume",
  // Sprint 2: cross-dimensional
  breath_support_issue: "pitch",
  registration_strain: "pitch",
  controlled_crescendo: "volume",
  loud_pitch_instability: "pitch",
  soft_passage_control: "pitch",
  vibrato_with_support: "expression",
  scoop_with_fade: "pitch",
  technique_accuracy_tradeoff: "expression",
  expressive_stability: "expression",
  high_note_control: "pitch",
  // Sprint 2: phrase / section
  rushed_phrase: "timing",
  dragged_phrase: "timing",
  rhythmic_precision: "timing",
  phrase_pitch_arc: "pitch",
  section_improvement: "pitch",
  section_regression: "pitch",
  section_vibrato_contrast: "expression",
};

const CATEGORY_DISPLAY = {
  pitch: "Pitch",
  expression: "Expression",
  timing: "Timing",
  volume: "Volume",
};

const PITCH_GOOD_TYPES = new Set([
  "best_pitch_phrase",
  "section_strength",
  "best_overall_section",
  // Sprint 2 affirming types that should render with good styling
  "clean_attack",
  "steady_sustain",
  "consistent_vibrato",
  "straight_tone_control",
  "clean_onset",
  "dynamic_sustain",
  "controlled_crescendo",
  "soft_passage_control",
  "vibrato_with_support",
  "expressive_stability",
  "high_note_control",
  "rhythmic_precision",
  "section_improvement",
]);

function cardCategoryClass(moment) {
  const cat = MOMENT_TYPE_CATEGORY[moment.type] || "pitch";
  if (cat === "pitch") {
    return PITCH_GOOD_TYPES.has(moment.type) ? "cat-pitch-good" : "cat-pitch-bad";
  }
  return `cat-${cat}`;
}

function regionCategoryClass(moment) {
  const cat = MOMENT_TYPE_CATEGORY[moment.type] || "pitch";
  if (cat === "pitch") {
    return PITCH_GOOD_TYPES.has(moment.type) ? "region-pitch-good" : "region-pitch-bad";
  }
  return `region-${cat}`;
}

function cardHeadingText(moment) {
  const cat = momentCategoryKey(moment);
  return CATEGORY_DISPLAY[cat] || "Feedback";
}

function momentCategoryKey(moment) {
  return MOMENT_TYPE_CATEGORY[moment.type] || "pitch";
}

/** Active highlight filter: null = show all categories. */
let activeHighlightFilter = null;
/** Active basis filter: "all" | "absolute" | "comparative". */
let activeBasisFilter = "all";

function applyHighlightFilter() {
  const cards = highlightList.querySelectorAll("article.card");
  let visible = 0;
  for (const card of cards) {
    const catMatch =
      !activeHighlightFilter || card.dataset.category === activeHighlightFilter;
    const basisMatch =
      activeBasisFilter === "all" || card.dataset.basis === activeBasisFilter;
    card.classList.toggle("hidden", !(catMatch && basisMatch));
    if (catMatch && basisMatch) visible += 1;
  }

  // Mirror the category filter onto timeline regions so they stay in sync.
  for (const region of timelineDiv.querySelectorAll(".region[data-category]")) {
    region.classList.toggle(
      "hidden",
      !!(activeHighlightFilter && region.dataset.category !== activeHighlightFilter)
    );
  }

  let empty = highlightList.querySelector(".filter-empty");
  if ((activeHighlightFilter || activeBasisFilter !== "all") && visible === 0 && cards.length > 0) {
    if (!empty) {
      empty = document.createElement("div");
      empty.className = "empty-cards filter-empty";
      highlightList.appendChild(empty);
    }
    empty.textContent = "No highlights match the current filters.";
    empty.classList.remove("hidden");
  } else if (empty) {
    empty.remove();
  }
}

function setHighlightFilter(category) {
  activeHighlightFilter = activeHighlightFilter === category ? null : category;
  for (const btn of highlightFilters.querySelectorAll(".highlight-filter-badge")) {
    btn.classList.toggle("active", btn.dataset.filter === activeHighlightFilter);
  }
  applyHighlightFilter();
}

function setBasisFilter(basis) {
  activeBasisFilter = basis;
  for (const btn of basisFilters.querySelectorAll(".basis-filter-badge")) {
    btn.classList.toggle("active", btn.dataset.basis === activeBasisFilter);
  }
  applyHighlightFilter();
}

function initHighlightFilterBadges() {
  if (!highlightFilters || highlightFilters.dataset.bound) return;
  highlightFilters.dataset.bound = "1";
  for (const btn of highlightFilters.querySelectorAll(".highlight-filter-badge")) {
    btn.addEventListener("click", () => {
      setHighlightFilter(btn.dataset.filter);
    });
  }
}

function initBasisFilterBadges() {
  if (!basisFilters || basisFilters.dataset.bound) return;
  basisFilters.dataset.bound = "1";
  for (const btn of basisFilters.querySelectorAll(".basis-filter-badge")) {
    btn.addEventListener("click", () => {
      setBasisFilter(btn.dataset.basis);
    });
  }
}

// Section-kind -> CSS class for the ribbon coloring.
const SECTION_KIND_CLASS = {
  intro: "kind-intro",
  verse: "kind-verse",
  pre_chorus: "kind-prechorus",
  chorus: "kind-chorus",
  bridge: "kind-bridge",
  refrain: "kind-refrain",
  outro: "kind-outro",
};

const LAST_SONG_KEY = "vocalCoach.lastSongId";

let allSongs = [];

async function loadSongs() {
  status.textContent = "Loading songs…";
  try {
    const r = await fetch(apiUrl("/api/songs"));
    if (!r.ok) throw new Error(`status ${r.status}`);
    const data = await r.json();
    allSongs = data.songs || [];
    if (!allSongs.length) {
      songSelect.innerHTML = "";
      const opt = document.createElement("option");
      opt.textContent = "(no songs found — run scripts/import_ultrastar.py)";
      songSelect.appendChild(opt);
      songSelect.disabled = true;
      status.textContent = "No songs available.";
      return;
    }
    populateSongSelect(allSongs);
    // Restore last selection from localStorage if it still exists.
    const remembered = localStorage.getItem(LAST_SONG_KEY);
    if (remembered && allSongs.some((s) => s.song_id === remembered)) {
      songSelect.value = remembered;
    }
    songSelect.disabled = false;
    songSelect.addEventListener("change", () => {
      currentReference = null;
      currentReferenceSongId = null;
      const song = songSelect.selectedOptions[0]?._song;
      if (song) localStorage.setItem(LAST_SONG_KEY, song.song_id);
      updateSongBadges(song);
    });
    updateSongBadges(songSelect.selectedOptions[0]?._song);
    status.textContent = "";
  } catch (e) {
    status.textContent = `Failed to load songs: ${e.message}`;
    status.classList.add("error");
  }
}

function populateSongSelect(songs) {
  const remembered = songSelect.value;
  songSelect.innerHTML = "";
  for (const s of songs) {
    const opt = document.createElement("option");
    opt.value = s.song_id;
    opt.textContent = `${s.title} — ${s.artist || "Unknown artist"}`;
    opt._song = s;
    songSelect.appendChild(opt);
  }
  if (remembered && songs.some((s) => s.song_id === remembered)) {
    songSelect.value = remembered;
  }
}

function fmtDuration(secs) {
  if (!isFinite(secs)) return "?";
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function updateSongBadges(song) {
  songBadges.innerHTML = "";
  if (!song) return;
  const badges = [
    { label: song.language || "English", kind: "lang" },
    { label: song.genre || "Pop", kind: "genre" },
    { label: fmtDuration(song.duration_s), kind: "duration" },
  ];
  for (const b of badges) {
    const span = document.createElement("span");
    span.className = `badge badge-${b.kind}`;
    span.textContent = b.label;
    songBadges.appendChild(span);
  }
}

function setAnalyzeLoading(loading) {
  analyzeButton.disabled = loading || !pendingFile;
  analyzeButton.classList.toggle("is-loading", loading);
  const label = analyzeButton.querySelector(".analyze-label");
  if (label) label.textContent = loading ? "Analyzing…" : "Analyze";
}

/**
 * Show the analysis progress bar and advance it to the given step index (0–4).
 * Pass 5 to mark all steps as completed (used on job completion before hiding).
 * Also hides the rest of the upload section so the progress bar is the sole focus.
 */
function showAnalysisProgress(activeStep) {
  uploadSection.classList.add("upload--analyzing");
  analysisProgress.classList.remove("hidden");

  const stepEls = analysisProgress.querySelectorAll(".progress-step");
  const connectorFills = analysisProgress.querySelectorAll(".step-connector-fill");

  stepEls.forEach((el, i) => {
    const circle = el.querySelector(".step-circle");
    el.classList.remove("is-active", "is-completed");
    if (i < activeStep) {
      circle.dataset.state = "completed";
      el.classList.add("is-completed");
    } else if (i === activeStep) {
      circle.dataset.state = "active";
      el.classList.add("is-active");
    } else {
      circle.dataset.state = "pending";
    }
  });

  // Connector i (between step i and step i+1) fills when step i is completed.
  connectorFills.forEach((fill, i) => {
    fill.classList.toggle("filled", activeStep > i);
  });
}

/** Hide the progress bar, restore the upload section, and reset all step states. */
function hideAnalysisProgress() {
  uploadSection.classList.remove("upload--analyzing");
  analysisProgress.classList.add("hidden");
  analysisProgress.querySelectorAll(".step-circle").forEach((c) => {
    c.dataset.state = "pending";
  });
  analysisProgress.querySelectorAll(".progress-step").forEach((el) => {
    el.classList.remove("is-active", "is-completed");
  });
  analysisProgress.querySelectorAll(".step-connector-fill").forEach((f) => {
    f.classList.remove("filled");
  });
}

analysisCancelBtn.addEventListener("click", () => {
  analysisAbortController?.abort();
  // hideAnalysisProgress() is called by the AbortError catch in the polling loop
});

function setPendingFile(file) {
  pendingFile = file;
  if (file) {
    status.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    status.classList.remove("error");
  } else {
    status.textContent = "";
  }
  setAnalyzeLoading(false);
}

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer?.files?.[0];
  if (file) setPendingFile(file);
});
fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) setPendingFile(file);
});

analyzeButton.addEventListener("click", async () => {
  if (!pendingFile) return;
  const songId = songSelect.value;
  if (!songId) return;

  analysisAbortController = new AbortController();
  const { signal } = analysisAbortController;

  setAnalyzeLoading(true);
  showAnalysisProgress(0);
  status.classList.remove("error");
  status.textContent = "";

  const fd = new FormData();
  fd.append("file", pendingFile);
  fd.append("stars_profile", currentStarsProfile());

  try {
    const r = await fetch(apiUrl(`/api/songs/${encodeURIComponent(songId)}/analyze`), {
      method: "POST",
      body: fd,
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`HTTP ${r.status}: ${detail.slice(0, 200)}`);
    }
    const { job_id } = await r.json();
    const analysis = await pollJob(job_id, signal);
    // Flash all steps complete before transitioning to results
    showAnalysisProgress(5);
    await new Promise((res) => setTimeout(res, 600));
    hideAnalysisProgress();
    status.textContent = `Done. Performance ID: ${analysis.perf_id}`;
    await renderAnalysis(songId, analysis);
  } catch (e) {
    if (e.name === "AbortError") {
      hideAnalysisProgress();
    } else {
      console.error(e);
      hideAnalysisProgress();
      status.textContent = `Analysis failed: ${e.message}`;
      status.classList.add("error");
    }
  } finally {
    setAnalyzeLoading(false);
    analysisAbortController = null;
  }
});

/**
 * Sleep for `ms` milliseconds, but resolve immediately if `signal` is aborted.
 * Throws a DOMException("AbortError") when the signal fires.
 */
function sleepOrAbort(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new DOMException("cancelled", "AbortError")); return; }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("cancelled", "AbortError"));
    }, { once: true });
  });
}

/**
 * Poll GET /api/jobs/{job_id} every 2 s until the job reaches "done" or
 * "error", then return the analysis result or throw.
 * Updates the analysis progress bar on each poll via job.step (0–4).
 * Respects an optional AbortSignal for user-initiated cancellation.
 */
async function pollJob(jobId, signal) {
  while (true) {
    await sleepOrAbort(2000, signal);
    const r = await fetch(apiUrl(`/api/jobs/${encodeURIComponent(jobId)}`), { signal });
    if (!r.ok) throw new Error(`Job poll failed: HTTP ${r.status}`);
    const job = await r.json();
    if (job.step != null) showAnalysisProgress(job.step);
    if (job.status === "done") return job.result;
    if (job.status === "error") throw new Error(`Analysis error: ${job.error}`);
    // "pending" or "running" — keep polling
  }
}

async function getReferenceAnnotation(songId) {
  if (currentReference && currentReferenceSongId === songId) {
    return currentReference;
  }
  try {
    const r = await fetch(
      apiUrl(`/api/songs/${encodeURIComponent(songId)}/reference_annotation`),
    );
    if (!r.ok) throw new Error(`status ${r.status}`);
    currentReference = await r.json();
    currentReferenceSongId = songId;
    return currentReference;
  } catch (e) {
    console.warn(`Failed to load reference annotation: ${e.message}`);
    currentReference = null;
    currentReferenceSongId = null;
    return null;
  }
}

function fmtTime(t) {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

const SECTION_BEST_MIN_NOTES = 4;

function sectionBlendedScore(section, notes, techniques) {
  const pct = section.pct_in_tune;
  if (pct == null) return null;

  const secNoteIndices = new Set();
  for (const n of notes) {
    const mid = 0.5 * (n.start_s + n.end_s);
    if (mid >= section.start_s && mid < section.end_s) {
      secNoteIndices.add(n.note_index);
    }
  }

  let expressiveCount = 0;
  for (const t of techniques) {
    const hasExpr =
      (t.matched && t.matched.length) || (t.user_added && t.user_added.length);
    if (secNoteIndices.has(t.note_index) && hasExpr) {
      expressiveCount += 1;
    }
  }
  const exprDensity = expressiveCount / Math.max(1, secNoteIndices.size);

  const arrivalVals = notes
    .filter(
      (n) =>
        n.arrival_offset_ms != null && secNoteIndices.has(n.note_index),
    )
    .map((n) => n.arrival_offset_ms);
  let timing = 0.5;
  if (arrivalVals.length) {
    const meanAbs =
      arrivalVals.reduce((sum, v) => sum + Math.abs(v), 0) / arrivalVals.length;
    timing = Math.max(0, 1 - meanAbs / 200);
  }

  return 0.5 * pct + 0.3 * exprDensity + 0.2 * timing;
}

function computeStrongestSection(analysis) {
  const sections = analysis?.sections;
  if (!sections?.length) return null;
  const notes = analysis.notes || [];
  const techniques = analysis.techniques || [];
  let bestName = null;
  let bestScore = -1;
  for (const section of sections) {
    if ((section.note_count || 0) < SECTION_BEST_MIN_NOTES) continue;
    const score = sectionBlendedScore(section, notes, techniques);
    if (score != null && score > bestScore) {
      bestScore = score;
      bestName = section.name;
    }
  }
  return bestName;
}

function renderOverviewTiles(overview, analysis) {
  overviewTiles.innerHTML = "";
  if (!overview) {
    overviewTiles.hidden = true;
    return;
  }
  overviewTiles.hidden = false;

  const tiles = [];
  if (overview.mimic_score != null) {
    tiles.push({
      label: "Mimic score",
      value: `${overview.mimic_score.toFixed(0)}`,
      sub: "/ 100",
      kind: "headline",
    });
  }
  if (overview.pct_in_tune != null) {
    tiles.push({
      label: "In tune",
      value: `${(overview.pct_in_tune * 100).toFixed(0)}%`,
      sub: `${overview.note_count} notes scored`,
    });
  }
  if (overview.median_cents != null) {
    const med = overview.median_cents;
    const direction = med < -3 ? "flat" : med > 3 ? "sharp" : "centered";
    tiles.push({
      label: "Median pitch",
      value: `${med >= 0 ? "+" : ""}${med.toFixed(0)}c`,
      sub: direction,
    });
  }
  if (overview.octave_shift_semitones) {
    const semis = overview.octave_shift_semitones;
    tiles.push({
      label: "Octave shift",
      value: `${semis > 0 ? "+" : ""}${semis} semitones`,
      sub: semis > 0 ? "you sang higher" : "you sang lower",
    });
  }
  if (overview.technique_match_rate != null) {
    tiles.push({
      label: "Technique match",
      value: `${(overview.technique_match_rate * 100).toFixed(0)}%`,
      sub: "vs reference",
    });
  }
  const strongestSection =
    overview.strongest_section ?? computeStrongestSection(analysis);
  if (strongestSection) {
    tiles.push({
      label: "Strongest section",
      value: strongestSection,
      sub: "pitch · expression · timing",
    });
  }
  if (overview.arrival_offset_ms_mean != null) {
    const ms = overview.arrival_offset_ms_mean;
    tiles.push({
      label: "Avg timing",
      value: `${ms >= 0 ? "+" : ""}${ms.toFixed(0)} ms`,
      sub: ms >= 0 ? "behind beat" : "ahead of beat",
    });
  }
  for (const tile of tiles) {
    const card = document.createElement("div");
    card.className = `tile ${tile.kind || ""}`;
    card.innerHTML = `
      <span class="tile-label">${tile.label}</span>
      <span class="tile-value">${tile.value}</span>
      <span class="tile-sub">${tile.sub || ""}</span>
    `;
    overviewTiles.appendChild(card);
  }
}

function renderPerformanceSummary(summary) {
  const el = document.getElementById("performanceSummary");
  if (!el) return;
  if (!summary || !summary.narrative) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }

  const takeawaysHtml = summary.takeaways && summary.takeaways.length
    ? `<div class="summary-takeaways">
        <h4 class="summary-section-heading">Key takeaways</h4>
        <ol class="summary-list">
          ${summary.takeaways.map(t => `<li>${t}</li>`).join("")}
        </ol>
      </div>`
    : "";

  const trendsHtml = summary.trends && summary.trends.length
    ? `<div class="summary-trends">
        <h4 class="summary-section-heading">Patterns observed</h4>
        <ul class="summary-list summary-trend-list">
          ${summary.trends.map(t => `<li>${t}</li>`).join("")}
        </ul>
      </div>`
    : "";

  el.innerHTML = `
    <div class="summary-inner">
      <div class="summary-header">
        <span class="summary-label">Summary</span>
      </div>
      <p class="summary-narrative">${summary.narrative}</p>
      ${takeawaysHtml}
      ${trendsHtml}
    </div>
  `;
  el.hidden = false;
}

function renderSectionRibbon(reference, analysis) {
  sectionRibbon.innerHTML = "";
  if (!reference || !Array.isArray(reference.sections) || !reference.sections.length) {
    return;
  }
  const totalDur = currentDuration;
  const offset = analysis.global_offset_s || 0;
  // For partial recordings, only show sections within the recorded segment.
  const segmentEnd = analysis.segment_end_song_s ?? Infinity;
  const trendByName = {};
  for (const trend of analysis.sections || []) {
    trendByName[trend.name] = trend;
  }
  for (const section of reference.sections) {
    // Skip sections that start entirely after the recording ended.
    if (section.start_s >= segmentEnd) continue;
    // Clamp the section end to the recording boundary.
    const clampedEnd = Math.min(section.end_s, segmentEnd);
    const userStart = section.start_s + offset;
    const userEnd = clampedEnd + offset;
    const left = (userStart / totalDur) * 100;
    const width = ((userEnd - userStart) / totalDur) * 100;
    const trend = trendByName[section.name];
    const band = document.createElement("div");
    const kindClass = SECTION_KIND_CLASS[section.kind] || "kind-unknown";
    band.className = `section-band ${kindClass}`;
    band.style.left = `${Math.max(0, left)}%`;
    band.style.width = `${Math.max(0.5, width)}%`;
    const pctLabel =
      trend && trend.pct_in_tune != null
        ? ` · ${(trend.pct_in_tune * 100).toFixed(0)}% in tune`
        : "";
    band.title = `${section.name}${section.kind ? ` (${section.kind})` : ""}${pctLabel}`;
    band.innerHTML = `<span class="section-band-label">${section.name}</span>`;
    band.addEventListener("click", () => {
      seekAndPlay(userStart);
    });
    sectionRibbon.appendChild(band);
  }
}

function lyricSnippet(reference, noteIndices) {
  if (!reference || !Array.isArray(reference.notes) || !noteIndices?.length) return "";
  const MAX_WORDS = 8;
  const seenWordIndices = new Set();
  const tokens = [];
  for (const idx of noteIndices) {
    if (tokens.length >= MAX_WORDS) break;
    const note = reference.notes[idx];
    if (!note) continue;
    const raw = (note.lyric_word || "").trim();
    if (!raw) continue;
    const wi = note.word_index;
    if (wi != null && seenWordIndices.has(wi)) continue;
    if (wi != null) seenWordIndices.add(wi);
    for (const w of raw.split(/\s+/)) {
      tokens.push(w);
      if (tokens.length >= MAX_WORDS) break;
    }
  }
  if (!tokens.length) return "";
  let snippet = tokens.join(" ");
  if (tokens.length >= MAX_WORDS) snippet += "…";
  return snippet;
}

function keyStatFor(moment) {
  const d = moment.detail || {};
  if (moment.type === "best_pitch_phrase" || moment.type === "pitch_struggle") {
    if (d.mean_pct_in_tune != null) {
      return `${(d.mean_pct_in_tune * 100).toFixed(0)}% in tune`;
    }
  }
  if (moment.type === "section_strength" || moment.type === "section_weakness") {
    if (d.pct_in_tune != null) {
      return `${(d.pct_in_tune * 100).toFixed(0)}% in tune`;
    }
  }
  if (moment.type === "sharp_flat_note" && d.median_cents != null) {
    const c = d.median_cents;
    return `${c >= 0 ? "+" : ""}${c.toFixed(0)}c ${d.direction || ""}`;
  }
  if (moment.type === "late_entrance" && d.arrival_offset_ms != null) {
    const ms = d.arrival_offset_ms;
    return `${ms >= 0 ? "+" : ""}${ms.toFixed(0)} ms`;
  }
  if (moment.type === "timing_consistency" && d.mean_abs_offset_ms != null) {
    return `avg ${d.mean_abs_offset_ms.toFixed(0)}ms ${d.direction || "off"}`;
  }
  if (moment.type === "vocal_texture" && d.consecutive_notes != null) {
    return `${d.consecutive_notes} notes`;
  }
  if (moment.type === "expressive_match" || moment.type === "expressive_moment") {
    if (d.matched_note_count != null && d.window_size != null) {
      return `${d.matched_note_count}/${d.window_size} notes`;
    }
    if (d.user_note_count != null && d.window_size != null) {
      return `${d.user_note_count}/${d.window_size} notes`;
    }
  }
  if (moment.type === "missed_expression" && d.missed_note_count != null) {
    return `${d.missed_note_count} notes`;
  }
  if (moment.type === "fade_within_notes" && d.fading_note_count != null) {
    return `${d.fading_note_count} notes fading`;
  }
  if (moment.type === "dynamic_drop" && d.mean_rms_delta_db != null) {
    return `${Math.abs(d.mean_rms_delta_db).toFixed(1)} dB quieter`;
  }
  if (moment.type === "dynamic_surge" && d.mean_rms_delta_db != null) {
    return `+${d.mean_rms_delta_db.toFixed(1)} dB`;
  }
  if (moment.type === "best_overall_section" || moment.type === "weakest_overall_section") {
    if (d.pct_in_tune != null) {
      return `${(d.pct_in_tune * 100).toFixed(0)}% in tune`;
    }
  }
  if (moment.type === "section_dynamic_contrast" && d.delta_db != null) {
    return `${Math.abs(d.delta_db).toFixed(1)} dB gap`;
  }
  if (moment.type === "section_delta") {
    if (d.delta != null) return `${(d.delta * 100).toFixed(0)}% gap`;
    if (d.cents_delta != null) return `${Math.abs(d.cents_delta).toFixed(0)}c gap`;
    if (d.gap != null) return `${(d.gap * 100).toFixed(0)}% drop`;
  }
  return `score ${moment.score.toFixed(2)}`;
}

function momentDomId(moment) {
  return moment.id || `${moment.type}:${moment.start_s}`;
}

function scrollToHighlightCard(momentId) {
  if (!momentId || !highlightList) return;
  const safe =
    typeof CSS !== "undefined" && CSS.escape
      ? CSS.escape(momentId)
      : momentId.replace(/["\\]/g, "\\$&");
  const card = highlightList.querySelector(
    `article.card[data-moment-id="${safe}"]`,
  );
  if (!card || card.classList.contains("hidden")) return;
  // Center the card in the horizontal list. Prefer scrollLeft over
  // scrollIntoView so we don't also nudge the page vertically.
  const listRect = highlightList.getBoundingClientRect();
  const cardRect = card.getBoundingClientRect();
  const delta =
    cardRect.left + cardRect.width / 2 - (listRect.left + listRect.width / 2);
  highlightList.scrollBy({ left: delta, behavior: "smooth" });
}

function renderCoachingCards(reference, analysis) {
  highlightList.innerHTML = "";
  activeHighlightFilter = null;
  activeBasisFilter = "all";
  if (highlightFilters) {
    highlightFilters.hidden = true;
    for (const btn of highlightFilters.querySelectorAll(".highlight-filter-badge")) {
      btn.classList.remove("active");
    }
  }
  if (basisFilters) {
    basisFilters.hidden = true;
    for (const btn of basisFilters.querySelectorAll(".basis-filter-badge")) {
      btn.classList.toggle("active", btn.dataset.basis === "all");
    }
  }
  if (!analysis.highlights?.moments?.length) {
    const empty = document.createElement("div");
    empty.className = "empty-cards";
    empty.textContent = "No coaching moments — try another take.";
    highlightList.appendChild(empty);
    return;
  }
  initHighlightFilterBadges();
  initBasisFilterBadges();
  if (highlightFilters) highlightFilters.hidden = false;
  if (basisFilters) basisFilters.hidden = false;

  for (const moment of analysis.highlights.moments) {
    // Skip low-confidence cards defensively (should already be filtered server-side).
    if (moment.confidence === "low") continue;
    // Skip moments that span more than 25% of the song — too broad for a
    // point-in-time card.  These are song-wide trends better suited to the
    // performance summary.
    if (analysis.duration_s > 0 && (moment.end_s - moment.start_s) / analysis.duration_s > 0.25) continue;

    const card = document.createElement("article");
    const scopeClass = moment.scope === "section" ? "scope-section" : "scope-local";
    const category = momentCategoryKey(moment);
    const basis = moment.feedback_basis || MOMENT_FEEDBACK_BASIS[moment.type] || "absolute";
    const confidenceClass = moment.confidence === "medium" ? " confidence-medium" : "";
    card.className = `card ${moment.type} ${cardCategoryClass(moment)} ${scopeClass}${confidenceClass}`;
    card.dataset.category = category;
    card.dataset.basis = basis;
    card.dataset.momentId = momentDomId(moment);

    const userStart = moment.start_s + analysis.global_offset_s;
    const userEnd = moment.end_s + analysis.global_offset_s;
    const lyric = lyricSnippet(reference, moment.note_indices);
    const sectionTag = moment.section_names?.length
      ? `<p class="card-section">${moment.section_names.join(" · ")}</p>`
      : "";
    const lyricEl = lyric ? `<blockquote class="card-lyric">“${lyric}”</blockquote>` : "";
    const basisLabel = `<span class="card-basis-label basis-${basis}">${BASIS_DISPLAY[basis] || basis}</span>`;
    const evidenceBadge =
      moment.confidence === "medium"
        ? `<span class="evidence-badge">Limited evidence</span>`
        : "";

    card.innerHTML = `
      <header class="card-header">
        <h5>${cardHeadingText(moment)}${basisLabel}</h5>
      </header>
      <p class="card-title">${moment.title}</p>
      <p class="card-summary">${moment.summary}</p>
      ${sectionTag}
      ${lyricEl}
      ${evidenceBadge}
      <div class="card-footer">
        <div class="card-stat">
          <span class="card-stat-value">${keyStatFor(moment)}</span>
          <span class="card-stat-meta">${fmtTime(userStart)}–${fmtTime(userEnd)}</span>
        </div>
        <button class="card-play" type="button">Play from here</button>
      </div>
    `;
    card.querySelector(".card-play").addEventListener("click", (e) => {
      e.stopPropagation();
      seekAndPlay(userStart);
      scrollToHighlightCard(card.dataset.momentId);
    });
    card.addEventListener("click", () => {
      seekAndPlay(userStart);
      scrollToHighlightCard(card.dataset.momentId);
    });
    highlightList.appendChild(card);
  }
  applyHighlightFilter();
}

function currentStarsProfile() {
  return fastProfile && fastProfile.checked ? "fast" : "full";
}

// Build a normalized peak envelope from the per-frame RMS (dB) loudness track
// so the waveform renders without the browser decoding the full WAV.
async function fetchPeaks(songId, perfId) {
  try {
    const r = await fetch(
      apiUrl(`/api/songs/${encodeURIComponent(songId)}/performances/${encodeURIComponent(
        perfId,
      )}/loudness`),
    );
    if (!r.ok) return null;
    const track = await r.json();
    const frames = track.frames || [];
    if (!frames.length) return null;
    // dB -> linear amplitude, normalized to [0, 1].
    const lin = frames.map((f) => Math.pow(10, (f.rms_db ?? -120) / 20));
    const maxAmp = Math.max(...lin, 1e-6);
    const peaks = lin.map((v) => v / maxAmp);
    return peaks;
  } catch (e) {
    console.warn("Failed to build peaks from loudness:", e);
    return null;
  }
}

function songTimeFromUser(userTime, offsetS) {
  return userTime - (offsetS ?? 0);
}

function getPerformanceTime() {
  if (wavesurfer && typeof wavesurfer.getCurrentTime === "function") {
    return wavesurfer.getCurrentTime();
  }
  return currentMedia?.currentTime ?? 0;
}

function isPerformancePlaying() {
  if (wavesurfer && typeof wavesurfer.isPlaying === "function") {
    return wavesurfer.isPlaying();
  }
  return Boolean(currentMedia && !currentMedia.paused);
}

function ensureMixMedia() {
  if (!refVocalMedia) {
    refVocalMedia = new Audio();
    refVocalMedia.preload = "auto";
    refVocalMedia.volume = MIX_LAYER_VOLUME;
  }
  if (!instrumentalMedia) {
    instrumentalMedia = new Audio();
    instrumentalMedia.preload = "auto";
    instrumentalMedia.volume = MIX_LAYER_VOLUME;
  }
}

function mixLayerUrl(songId, kind) {
  const base = apiUrl(`/api/songs/${encodeURIComponent(songId)}/audio`);
  return kind === "instrumental" ? `${base}/instrumental` : `${base}/reference`;
}

function prepareMixLayers(songId) {
  ensureMixMedia();
  refVocalMedia.src = mixLayerUrl(songId, "reference");
  instrumentalMedia.src = mixLayerUrl(songId, "instrumental");
  refVocalMedia.pause();
  instrumentalMedia.pause();
}

function pauseMixLayers() {
  refVocalMedia?.pause();
  instrumentalMedia?.pause();
}

function syncMixLayer(media, enabled, songTime, playing) {
  if (!media || !enabled) {
    media?.pause();
    return;
  }
  const dur = media.duration;
  if (!isFinite(dur) || dur <= 0) return;
  if (songTime < 0 || songTime > dur) {
    media.pause();
    return;
  }
  if (Math.abs(media.currentTime - songTime) > MIX_SYNC_THRESHOLD_S) {
    media.currentTime = songTime;
  }
  if (playing) {
    if (media.paused) media.play().catch(() => {});
  } else {
    media.pause();
  }
}

function syncMixLayers(userTime, playing) {
  const offset = currentAnalysis?.global_offset_s ?? 0;
  const songT = songTimeFromUser(userTime, offset);
  syncMixLayer(refVocalMedia, mixRefVocalOn, songT, playing);
  syncMixLayer(instrumentalMedia, mixInstrumentalOn, songT, playing);
}

function updateMixToggleUi(song) {
  const hasInst = Boolean(song?.has_instrumental);
  if (mixInstrumentalBtn) {
    mixInstrumentalBtn.disabled = !hasInst;
    mixInstrumentalBtn.title = hasInst
      ? ""
      : "This song has no instrumental track";
  }
  if (mixRefVocalBtn) {
    mixRefVocalBtn.disabled = false;
  }
}

function resetMixToggles() {
  mixRefVocalOn = false;
  mixInstrumentalOn = false;
  pauseMixLayers();
  for (const btn of [mixRefVocalBtn, mixInstrumentalBtn]) {
    if (!btn) continue;
    btn.classList.remove("active");
    btn.setAttribute("aria-pressed", "false");
  }
}

function setMixToggle(kind, on) {
  if (kind === "ref") {
    mixRefVocalOn = on;
    if (mixRefVocalBtn) {
      mixRefVocalBtn.classList.toggle("active", on);
      mixRefVocalBtn.setAttribute("aria-pressed", String(on));
    }
  } else {
    if (mixInstrumentalBtn?.disabled) return;
    mixInstrumentalOn = on;
    if (mixInstrumentalBtn) {
      mixInstrumentalBtn.classList.toggle("active", on);
      mixInstrumentalBtn.setAttribute("aria-pressed", String(on));
    }
  }
  syncMixLayers(getPerformanceTime(), isPerformancePlaying());
}

function detachPlaybackMixSync() {
  playbackSyncAbort?.abort();
  playbackSyncAbort = null;
}

function attachPlaybackMixSync() {
  detachPlaybackMixSync();
  playbackSyncAbort = new AbortController();
  const { signal } = playbackSyncAbort;
  const sync = () => {
    syncMixLayers(getPerformanceTime(), isPerformancePlaying());
  };
  if (currentMedia) {
    for (const ev of ["play", "pause", "timeupdate", "seeked", "ended"]) {
      currentMedia.addEventListener(ev, sync, { signal });
    }
  }
  if (wavesurfer) {
    for (const ev of ["play", "pause", "timeupdate", "seeking", "interaction"]) {
      wavesurfer.on(ev, sync);
    }
  }
}

// Seek to a (user-time) second and play, preferring WaveSurfer but falling
// back to the raw media element so playback works even if the waveform
// renderer never reached a ready state.
function seekAndPlay(userStart) {
  const t = Math.max(0, userStart);
  if (wavesurfer) {
    try {
      wavesurfer.setTime(t);
      const p = wavesurfer.play();
      if (p && typeof p.catch === "function") p.catch(() => {});
      syncMixLayers(t, true);
      return;
    } catch (e) {
      console.warn("WaveSurfer play failed, using media element:", e);
    }
  }
  if (currentMedia) {
    try {
      currentMedia.currentTime = t;
      currentMedia.play().catch(() => {});
      syncMixLayers(t, true);
    } catch (e) {
      console.warn("Media element play failed:", e);
    }
  }
}

async function renderAnalysis(songId, analysis) {
  results.hidden = false;
  currentAnalysis = analysis;
  currentPlaybackSongId = songId;
  currentDuration = analysis.duration_s;

  // Show a segment indicator when the user recorded only part of the song.
  const song = allSongs.find((s) => s.song_id === songId);
  if (analysis.segment_end_song_s != null && song) {
    segmentInfo.textContent =
      `Analyzed ${fmtMmSs(0)}\u2013${fmtMmSs(analysis.segment_end_song_s)} of ${fmtMmSs(song.duration_s)}`;
    segmentInfo.hidden = false;
  } else {
    segmentInfo.hidden = true;
    segmentInfo.textContent = "";
  }

  resetMixToggles();
  prepareMixLayers(songId);
  updateMixToggleUi(allSongs.find((s) => s.song_id === songId));
  detachPlaybackMixSync();
  if (wavesurfer) {
    try {
      wavesurfer.destroy();
    } catch (e) {
      /* ignore */
    }
    wavesurfer = null;
  }

  const audioUrl = apiUrl(`/api/songs/${encodeURIComponent(songId)}/performances/${encodeURIComponent(
    analysis.perf_id,
  )}/audio`);

  // Dedicated streaming media element: plays via HTTP range requests and is
  // a reliable fallback for the play-from-here actions regardless of whether
  // the visual waveform decode succeeds.
  const mediaEl = new Audio();
  mediaEl.preload = "auto";
  mediaEl.src = audioUrl;
  currentMedia = mediaEl;

  // Precomputed peaks from the loudness track avoid a fragile in-browser
  // decode of the full (40 MB+) performance WAV.
  const peaks = await fetchPeaks(songId, analysis.perf_id);

  wavesurfer = WaveSurfer.create({
    container: waveformDiv,
    waveColor: "#cec5bb",
    progressColor: "#3b6fd4",
    height: 128,
    normalize: true,
    media: mediaEl,
    ...(peaks ? { peaks: [peaks], duration: analysis.duration_s } : {}),
  });
  wavesurfer.on("error", (err) => {
    console.error("WaveSurfer error:", err);
    status.textContent = `Waveform failed to render (${err}). Playback may still work.`;
    status.classList.add("error");
  });
  // If we couldn't build peaks, fall back to decoding the audio for the wave.
  if (!peaks) {
    wavesurfer.load(audioUrl);
  }
  attachPlaybackMixSync();

  // Overview tiles (above the waveform).
  renderOverviewTiles(analysis.overview, analysis);

  // Performance summary (LLM narrative + takeaways + trends, if available).
  renderPerformanceSummary(analysis.performance_summary);

  // Region markers in song time. NOTE: the Wavesurfer waveform is the
  // user vocal in *user* time; we shift song-time regions back to user
  // time using the analysis.global_offset_s.
  timelineDiv.innerHTML = "";
  const totalDur = currentDuration;
  for (const moment of analysis.highlights.moments) {
    // Skip song-wide trends (>25% of duration) — same filter as cards.
    if (analysis.duration_s > 0 && (moment.end_s - moment.start_s) / analysis.duration_s > 0.25) continue;
    const userStart = moment.start_s + analysis.global_offset_s;
    const userEnd = moment.end_s + analysis.global_offset_s;
    const left = (userStart / totalDur) * 100;
    const width = ((userEnd - userStart) / totalDur) * 100;
    const region = document.createElement("div");
    region.className = `region ${regionCategoryClass(moment)}`;
    region.style.left = `${Math.max(0, left)}%`;
    region.style.width = `${Math.max(1, width)}%`;
    region.title = `${moment.title}\n${moment.summary}`;
    region.dataset.momentId = momentDomId(moment);
    region.dataset.category = momentCategoryKey(moment);
    region.addEventListener("click", () => {
      seekAndPlay(userStart);
      scrollToHighlightCard(region.dataset.momentId);
    });
    timelineDiv.appendChild(region);
  }

  // Section ribbon (verses/choruses/...) under the waveform.
  const reference = await getReferenceAnnotation(songId);
  renderSectionRibbon(reference, analysis);

  // Rich coaching cards.
  renderCoachingCards(reference, analysis);
}

playPause.addEventListener("click", () => {
  if (wavesurfer) {
    try {
      wavesurfer.playPause();
      syncMixLayers(getPerformanceTime(), isPerformancePlaying());
      return;
    } catch (e) {
      /* fall through to media element */
    }
  }
  if (currentMedia) {
    if (currentMedia.paused) currentMedia.play().catch(() => {});
    else currentMedia.pause();
    syncMixLayers(getPerformanceTime(), isPerformancePlaying());
  }
});

mixRefVocalBtn?.addEventListener("click", () => {
  setMixToggle("ref", !mixRefVocalOn);
});

mixInstrumentalBtn?.addEventListener("click", () => {
  setMixToggle("instrumental", !mixInstrumentalOn);
});

// ---------------------------------------------------------------------------
// Sprint 3 stretch: karaoke sing-along mode
// ---------------------------------------------------------------------------

const modeUploadBtn = document.getElementById("modeUpload");
const modeKaraokeBtn = document.getElementById("modeKaraoke");
const uploadPanel = document.getElementById("uploadPanel");
const karaokePanel = document.getElementById("karaokePanel");
const karaokeRecordBtn = document.getElementById("karaokeRecord");
const karaokeStopBtn = document.getElementById("karaokeStop");
const karaokeCancelBtn = document.getElementById("karaokeCancel");
const karaokeTime = document.getElementById("karaokeTime");
const karaokeLyrics = document.getElementById("karaokeLyrics");
const karaokeAudio = document.getElementById("karaokeAudio");
const karaokeViz = document.getElementById("karaokeViz");
const karaokeRefVocalBtn = document.getElementById("karaokeRefVocal");

let karaokeScriptNode = null;   // ScriptProcessorNode used for WAV capture
let karaokePcmBuffers = [];     // Float32Array chunks collected during recording
let karaokeRecordSr = 44100;    // actual AudioContext sample rate
let karaokeStream = null;
let karaokeRafHandle = null;
let karaokeStartTs = 0;
let karaokeWordRows = [];
let karaokeLinesArr = [];       // grouped line-level entries for vertical display
let karaokeLastScrollFocusIdx = -1; // tracks which line we last scrolled to

// Reference vocal playback during karaoke.
let karaokeRefVocalMedia = null;
let karaokeRefVocalOn = false;

// Web Audio API handles for the live mic waveform.
let karaokeAudioCtx = null;
let karaokeAnalyser = null;
let karaokeVizData = null;  // Uint8Array reused each frame

/**
 * Encode collected PCM buffers into a 16-bit mono WAV Blob.
 * Runs entirely in the browser — the server receives a plain WAV file that
 * soundfile can open without any conversion or audioread fallback.
 */
function _wavEncode(buffers, sampleRate) {
  const totalSamples = buffers.reduce((s, b) => s + b.length, 0);
  const ab = new ArrayBuffer(44 + totalSamples * 2);
  const v = new DataView(ab);
  const ws = (off, str) => { for (let i = 0; i < str.length; i++) v.setUint8(off + i, str.charCodeAt(i)); };
  ws(0, "RIFF"); v.setUint32(4, 36 + totalSamples * 2, true);
  ws(8, "WAVE"); ws(12, "fmt ");
  v.setUint32(16, 16, true);       // fmt chunk size
  v.setUint16(20, 1, true);        // PCM
  v.setUint16(22, 1, true);        // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); // byte rate
  v.setUint16(32, 2, true);        // block align
  v.setUint16(34, 16, true);       // bits per sample
  ws(36, "data"); v.setUint32(40, totalSamples * 2, true);
  let off = 44;
  for (const buf of buffers) {
    for (let i = 0; i < buf.length; i++) {
      const s = Math.max(-1, Math.min(1, buf[i]));
      v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      off += 2;
    }
  }
  return new Blob([ab], { type: "audio/wav" });
}

// A timing gap larger than this triggers a line break — but only once the
// current line has reached LYRIC_MIN_WORDS, so short words like "Yesterday"
// don't end up alone on a line after a brief pause.
const LYRIC_LINE_GAP_S = 0.8;
// A gap this long always forces a new line regardless of word count
// (covers true section rests / instrumentals).
const LYRIC_HARD_GAP_S = 5.0;
// Minimum words a line must accumulate before a normal gap can break it.
const LYRIC_MIN_WORDS = 4;
// Absolute maximum words before forcing a line break regardless of timing.
const LYRIC_MAX_WORDS = 12;
// How far above the container top the active line is positioned (px).
const LYRIC_SCROLL_OFFSET_PX = 80;

function setMode(mode) {
  const isUpload = mode === "upload";
  modeUploadBtn.classList.toggle("active", isUpload);
  modeKaraokeBtn.classList.toggle("active", !isUpload);
  uploadPanel.hidden = !isUpload;
  karaokePanel.hidden = isUpload;
}

modeUploadBtn.addEventListener("click", () => setMode("upload"));
modeKaraokeBtn.addEventListener("click", async () => {
  setMode("karaoke");
  const songId = songSelect.value;
  if (!songId) return;
  const reference = await getReferenceAnnotation(songId);
  buildKaraokeLyrics(reference);
});

function buildKaraokeLyrics(reference) {
  karaokeLyrics.innerHTML = "";
  karaokeWordRows = [];
  karaokeLinesArr = [];
  karaokeLastScrollFocusIdx = -1;
  if (!reference || !Array.isArray(reference.notes)) {
    karaokeLyrics.innerHTML = `<p class="hint">Lyrics will appear here once a reference annotation is available.</p>`;
    return;
  }

  // Deduplicate notes into unique words by word_index.
  const wordsMap = new Map();
  for (const note of reference.notes) {
    const wordIdx = note.word_index;
    if (wordIdx == null) continue;
    if (!wordsMap.has(wordIdx)) {
      wordsMap.set(wordIdx, {
        word: (note.lyric_word || "").trim(),
        start_s: note.start_s,
        end_s: note.end_s,
      });
    } else {
      const entry = wordsMap.get(wordIdx);
      entry.end_s = Math.max(entry.end_s, note.end_s);
    }
  }
  const words = [...wordsMap.values()].filter((w) => w.word);
  words.sort((a, b) => a.start_s - b.start_s);

  // Group words into lines using a minimum-word guard so short words don't
  // end up alone. A soft gap break (>= LYRIC_LINE_GAP_S) only fires once the
  // current line has at least LYRIC_MIN_WORDS words. A hard gap
  // (>= LYRIC_HARD_GAP_S, e.g. an instrumental rest) always breaks immediately.
  const rawLines = [];
  let currentLine = [];
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const prev = i > 0 ? words[i - 1] : null;
    const gap = prev ? w.start_s - prev.end_s : 0;
    if (currentLine.length > 0) {
      const hardBreak = gap >= LYRIC_HARD_GAP_S;
      const softBreak = gap >= LYRIC_LINE_GAP_S && currentLine.length >= LYRIC_MIN_WORDS;
      const maxBreak  = currentLine.length >= LYRIC_MAX_WORDS;
      if (hardBreak || softBreak || maxBreak) {
        rawLines.push(currentLine);
        currentLine = [];
      }
    }
    currentLine.push(w);
  }
  if (currentLine.length > 0) rawLines.push(currentLine);

  // Top spacer lets the first line scroll to the visual center of the container.
  const topSpacer = document.createElement("div");
  topSpacer.className = "karaoke-lyric-spacer";
  topSpacer.style.height = `${LYRIC_SCROLL_OFFSET_PX}px`;
  karaokeLyrics.appendChild(topSpacer);

  // Build DOM: one .karaoke-line per group.
  for (const lineWords of rawLines) {
    const lineDiv = document.createElement("div");
    lineDiv.className = "karaoke-line upcoming";

    const wordItems = [];
    for (const w of lineWords) {
      const span = document.createElement("span");
      span.className = "karaoke-word";
      span.textContent = w.word + " ";
      lineDiv.appendChild(span);
      const item = { span, start_s: w.start_s, end_s: w.end_s };
      wordItems.push(item);
      karaokeWordRows.push(item);
    }

    karaokeLyrics.appendChild(lineDiv);
    karaokeLinesArr.push({
      div: lineDiv,
      words: wordItems,
      start_s: lineWords[0].start_s,
      end_s: lineWords[lineWords.length - 1].end_s,
    });
  }

  // Bottom spacer so the last line can also be scrolled to the visual center.
  const botSpacer = document.createElement("div");
  botSpacer.className = "karaoke-lyric-spacer";
  botSpacer.style.height = `${LYRIC_SCROLL_OFFSET_PX + 160}px`;
  karaokeLyrics.appendChild(botSpacer);
}

function fmtMmSs(t) {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function _drawKaraokeWaveform() {
  if (!karaokeAnalyser || !karaokeVizData) return;
  const canvas = karaokeViz;
  // Sync canvas pixel size to its CSS layout size once per draw.
  const W = canvas.clientWidth || 400;
  const H = canvas.clientHeight || 64;
  if (canvas.width !== W) canvas.width = W;
  if (canvas.height !== H) canvas.height = H;

  const ctx = canvas.getContext("2d");
  karaokeAnalyser.getByteTimeDomainData(karaokeVizData);

  ctx.clearRect(0, 0, W, H);

  // Subtle filled waveform so silence = flat line, singing = tall waves.
  const accent = "#3b6fd4";
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1.5;
  ctx.beginPath();

  // Also draw a filled area under the waveform for visibility.
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, "rgba(59,111,212,0.22)");
  grad.addColorStop(1, "rgba(59,111,212,0.02)");

  const sliceW = W / karaokeVizData.length;
  let x = 0;
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  for (let i = 0; i < karaokeVizData.length; i++) {
    const y = ((karaokeVizData[i] / 128.0) * H) / 2;
    ctx.lineTo(x, y);
    x += sliceW;
  }
  ctx.lineTo(W, H / 2);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Stroke the waveform on top.
  x = 0;
  ctx.beginPath();
  for (let i = 0; i < karaokeVizData.length; i++) {
    const y = ((karaokeVizData[i] / 128.0) * H) / 2;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    x += sliceW;
  }
  ctx.strokeStyle = "#3b6fd4";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function applyKaraokeLyricsAtTime(t) {
  if (karaokeLinesArr.length === 0) return;

  // Find which line is actively being sung.
  let activeLineIdx = -1;
  for (let i = 0; i < karaokeLinesArr.length; i++) {
    const ln = karaokeLinesArr[i];
    if (t >= ln.start_s && t < ln.end_s) { activeLineIdx = i; break; }
  }

  // During gaps between lines, pivot around the last completed line so the
  // next upcoming line stays visible rather than the display going blank.
  let focusIdx = activeLineIdx;
  if (focusIdx === -1) {
    for (let i = karaokeLinesArr.length - 1; i >= 0; i--) {
      if (karaokeLinesArr[i].end_s <= t) { focusIdx = i; break; }
    }
  }

  // Apply sung / active / upcoming classes to every line.
  // No display toggling — all lines stay in the DOM; the container scrolls.
  for (let i = 0; i < karaokeLinesArr.length; i++) {
    const ln = karaokeLinesArr[i];
    const isActive = i === activeLineIdx;
    const isSung   = activeLineIdx !== -1 ? i < activeLineIdx : i <= focusIdx;
    const isUpcoming = !isActive && !isSung;

    ln.div.classList.toggle("active",   isActive);
    ln.div.classList.toggle("sung",     isSung);
    ln.div.classList.toggle("upcoming", isUpcoming);
  }

  // Smooth-scroll the container so the active (or focus) line sits just below
  // the top padding. Only trigger scrollTo when the focus line changes to
  // avoid fighting the browser's inertia scroll.
  if (focusIdx !== karaokeLastScrollFocusIdx) {
    karaokeLastScrollFocusIdx = focusIdx;
    if (focusIdx >= 0) {
      const targetLine = karaokeLinesArr[focusIdx].div;
      const containerRect = karaokeLyrics.getBoundingClientRect();
      const lineRect = targetLine.getBoundingClientRect();
      const currentScroll = karaokeLyrics.scrollTop;
      const targetScroll = lineRect.top - containerRect.top + currentScroll - LYRIC_SCROLL_OFFSET_PX;
      karaokeLyrics.scrollTo({ top: Math.max(0, targetScroll), behavior: "smooth" });
    }
  }
}

function updateKaraokeHighlight() {
  const t = (performance.now() - karaokeStartTs) / 1000;
  karaokeTime.textContent = fmtMmSs(t);
  _drawKaraokeWaveform();
  applyKaraokeLyricsAtTime(t);

  // Keep reference vocal in sync with the instrumental.
  if (karaokeRefVocalOn && karaokeRefVocalMedia && !karaokeRefVocalMedia.paused) {
    const drift = Math.abs(karaokeRefVocalMedia.currentTime - karaokeAudio.currentTime);
    if (drift > MIX_SYNC_THRESHOLD_S) {
      karaokeRefVocalMedia.currentTime = karaokeAudio.currentTime;
    }
  }

  karaokeRafHandle = requestAnimationFrame(updateKaraokeHighlight);
}

function resetKaraokeLyricsView() {
  karaokeTime.textContent = "0:00";
  karaokeLastScrollFocusIdx = -1;
  karaokeLyrics.scrollTop = 0;
  applyKaraokeLyricsAtTime(0);
}

function _handleKaraokeEnded() {
  endKaraokeSession({ analyze: true });
}

async function endKaraokeSession({ analyze }) {
  karaokeAudio.removeEventListener("ended", _handleKaraokeEnded);
  if (!karaokeScriptNode) return;

  // Capture duration before tearing anything down.
  const recordingDuration = (performance.now() - karaokeStartTs) / 1000;

  // Stop capturing PCM — disconnect before closing AudioContext.
  karaokeScriptNode.disconnect();
  karaokeScriptNode = null;
  // Encode collected PCM to WAV before the AudioContext is torn down.
  const wavBlob = analyze ? _wavEncode(karaokePcmBuffers, karaokeRecordSr) : null;
  karaokePcmBuffers = [];

  karaokeAudio.pause();
  karaokeRecordBtn.disabled = false;
  karaokeStopBtn.disabled = true;
  karaokeCancelBtn.disabled = true;
  cancelAnimationFrame(karaokeRafHandle);

  // Stop reference vocal playback and reset toggle.
  if (karaokeRefVocalMedia) {
    karaokeRefVocalMedia.pause();
    karaokeRefVocalMedia = null;
  }
  karaokeRefVocalOn = false;
  karaokeRefVocalBtn.classList.remove("active");
  karaokeRefVocalBtn.setAttribute("aria-pressed", "false");

  if (karaokeStream) {
    karaokeStream.getTracks().forEach((t) => t.stop());
    karaokeStream = null;
  }
  if (karaokeAudioCtx) {
    karaokeAudioCtx.close().catch(() => {});
    karaokeAudioCtx = null;
    karaokeAnalyser = null;
    karaokeVizData = null;
  }
  karaokeViz.hidden = true;
  resetKaraokeLyricsView();

  if (!analyze || !wavBlob) {
    status.textContent = "";
    status.classList.remove("error");
    return;
  }

  const songId = songSelect.value;
  if (!songId) return;

  analysisAbortController = new AbortController();
  const { signal } = analysisAbortController;

  showAnalysisProgress(0);
  status.textContent = "";
  status.classList.remove("error");
  const fd = new FormData();
  fd.append("file", wavBlob, "recording.wav");
  fd.append("stars_profile", currentStarsProfile());
  fd.append("recording_duration_s", recordingDuration.toFixed(3));
  try {
    const r = await fetch(
      apiUrl(`/api/songs/${encodeURIComponent(songId)}/analyze`),
      { method: "POST", body: fd },
    );
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`HTTP ${r.status}: ${detail.slice(0, 200)}`);
    }
    const { job_id } = await r.json();
    const analysis = await pollJob(job_id, signal);
    // Flash all steps complete before transitioning to results
    showAnalysisProgress(5);
    await new Promise((res) => setTimeout(res, 600));
    hideAnalysisProgress();
    status.textContent = `Done. Performance ID: ${analysis.perf_id}`;
    await renderAnalysis(songId, analysis);
  } catch (e) {
    if (e.name === "AbortError") {
      hideAnalysisProgress();
    } else {
      console.error(e);
      hideAnalysisProgress();
      status.textContent = `Analysis failed: ${e.message}`;
      status.classList.add("error");
    }
  } finally {
    analysisAbortController = null;
  }
}

karaokeRecordBtn.addEventListener("click", async () => {
  const songId = songSelect.value;
  if (!songId) return;
  if (!navigator.mediaDevices) {
    status.textContent = "Browser does not support microphone access.";
    status.classList.add("error");
    return;
  }
  try {
    karaokeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    status.textContent = `Mic permission denied: ${e.message}`;
    status.classList.add("error");
    return;
  }
  karaokePcmBuffers = [];

  // Build Web Audio graph: mic source -> analyser (waveform viz) + scriptNode (PCM capture).
  karaokeAudioCtx = new AudioContext();
  karaokeRecordSr = karaokeAudioCtx.sampleRate;
  karaokeAnalyser = karaokeAudioCtx.createAnalyser();
  karaokeAnalyser.fftSize = 512;
  karaokeAnalyser.smoothingTimeConstant = 0.6;
  karaokeVizData = new Uint8Array(karaokeAnalyser.fftSize);

  const micSrc = karaokeAudioCtx.createMediaStreamSource(karaokeStream);
  micSrc.connect(karaokeAnalyser);

  // ScriptProcessorNode captures raw PCM float samples into karaokePcmBuffers.
  // Must be connected to destination to fire onaudioprocess reliably.
  karaokeScriptNode = karaokeAudioCtx.createScriptProcessor(4096, 1, 1);
  karaokeScriptNode.onaudioprocess = (e) => {
    karaokePcmBuffers.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  micSrc.connect(karaokeScriptNode);
  karaokeScriptNode.connect(karaokeAudioCtx.destination);
  karaokeViz.hidden = false;

  // Play instrumental
  karaokeAudio.src = apiUrl(`/api/songs/${encodeURIComponent(songId)}/audio/instrumental`);
  karaokeAudio.hidden = false;
  karaokeAudio.currentTime = 0;
  karaokeAudio.addEventListener("ended", _handleKaraokeEnded);
  await karaokeAudio.play().catch(() => {
    /* autoplay restrictions may block; the user can click the visible controls */
  });

  // Preload reference vocal so the toggle can start it instantly.
  karaokeRefVocalMedia = new Audio();
  karaokeRefVocalMedia.preload = "auto";
  karaokeRefVocalMedia.src = apiUrl(`/api/songs/${encodeURIComponent(songId)}/audio/reference`);
  karaokeRefVocalMedia.volume = MIX_LAYER_VOLUME;

  karaokeStartTs = performance.now();
  karaokeRecordBtn.disabled = true;
  karaokeStopBtn.disabled = false;
  karaokeCancelBtn.disabled = false;
  status.textContent = "Recording… sing along to the highlighted lyrics.";
  status.classList.remove("error");
  cancelAnimationFrame(karaokeRafHandle);
  updateKaraokeHighlight();
});

karaokeStopBtn.addEventListener("click", () => endKaraokeSession({ analyze: true }));

karaokeCancelBtn.addEventListener("click", () => endKaraokeSession({ analyze: false }));

karaokeRefVocalBtn.addEventListener("click", () => {
  // Only active during a recording session.
  if (!karaokeRefVocalMedia) return;
  karaokeRefVocalOn = !karaokeRefVocalOn;
  karaokeRefVocalBtn.classList.toggle("active", karaokeRefVocalOn);
  karaokeRefVocalBtn.setAttribute("aria-pressed", String(karaokeRefVocalOn));
  if (karaokeRefVocalOn) {
    karaokeRefVocalMedia.currentTime = karaokeAudio.currentTime;
    karaokeRefVocalMedia.play().catch(() => {});
  } else {
    karaokeRefVocalMedia.pause();
  }
});

loadSongs();
