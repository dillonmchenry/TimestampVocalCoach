import WaveSurfer from "https://unpkg.com/wavesurfer.js@7.8.6/dist/wavesurfer.esm.js";
import { AnalysisGraph } from "./analysis-graph.js";

/**
 * Prefix a /api/... path with the configured API base URL.
 * window.API_BASE is set by config.js (empty string for same-origin local dev,
 * or the full RunPod URL for Vercel-hosted deployments).
 */
function apiUrl(path) {
  const base = (window.API_BASE || "").replace(/\/$/, "");
  return base + path;
}

const carouselTrack = document.getElementById("carouselTrack");
const carouselLeft = document.getElementById("carouselLeft");
const carouselRight = document.getElementById("carouselRight");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const analyzeButton = document.getElementById("analyzeButton");
const status = document.getElementById("status");
const analysisProgress = document.getElementById("analysisProgress");
const analysisCancelBtn = document.getElementById("analysisCancelBtn");
const uploadSection = document.querySelector("section.upload");
const resultsContainer = document.getElementById("results-container");
// Per-active-block DOM references — updated by activateBlock() whenever the active block changes
let waveformDiv = null;
let sectionRibbon = null;
let timelineDiv = null;
const offsetSummary = document.getElementById("offsetSummary");
// Per-active-block playback bar buttons — updated by activateBlock() on each switch
let playPause = null;
let mixRefVocalBtn = null;
let mixInstrumentalBtn = null;
let highlightList = null;
let highlightFilters = null;
let sectionStoryBanner = null;
const basisFilters = document.getElementById("basisFilters");
let overviewTiles = null;
let overviewHeading = null;
let segmentInfo = null;
let backToFullSongBtn = null;
const addPerformanceBar = document.getElementById("addPerformanceBar");
const addPerformanceBtn = document.getElementById("addPerformanceBtn");
const pickerSection = document.querySelector("section.picker");

let pendingFile = null;
let analysisAbortController = null;
let wavesurfer = null;
let analysisGraph = null;
let currentMedia = null;
let currentDuration = 0;
let currentAnalysis = null;
let currentPlaybackSongId = null;
let mixRefVocalOn = false;
let mixInstrumentalOn = false;
let refVocalMedia = null;
let instrumentalMedia = null;
let playbackSyncAbort = null;

// Section focus state
let focusedSection = null;   // { name, kind, start_s, end_s } in song time, or null
let fullPeaks = null;        // cached full-recording peaks array for waveform restoration
let fullAudioUrl = null;     // cached audio URL for waveform restoration

// Per-active-block element reference for the LLM summary panel
let performanceSummaryEl = null;
// AbortController that cleans up playback bar click listeners when the active block changes
let playbackListenersAbort = null;

// All rendered analyses in this session; each entry is a block-state object
let analyses = [];
let activeAnalysisIdx = -1;

const MIX_LAYER_VOLUME = 0.85;
const MIX_SYNC_THRESHOLD_S = 0.08;
// Marker pushed into history.state when at least one analysis result is visible.
// The popstate listener uses this to intercept accidental Back navigations.
const RESULTS_STATE = { secondpass: "results" };
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

// Card types that do NOT show the "Show coaching & comparison" disclosure section.
// Mirrors _NO_TRY_THIS_TYPES in highlights.py. Affirming cards (positive reinforcement)
// and timing cards (practice strategies, not vocal exercises) are excluded because the
// expanded section adds nothing useful — there is no corrective drill and no meaningful
// audio comparison to make.
const NO_DISCLOSURE_TYPES = new Set([
  // Timing cards
  "late_entrance", "timing_consistency", "rushed_phrase",
  "dragged_phrase", "rhythmic_precision", "section_delta",
  // Affirming / positive-reinforcement cards
  "best_pitch_phrase", "section_strength", "best_overall_section",
  "clean_attack", "steady_sustain", "consistent_vibrato",
  "straight_tone_control", "clean_onset", "dynamic_sustain",
  "controlled_crescendo", "soft_passage_control",
  "vibrato_with_support", "expressive_stability",
  "high_note_control", "section_improvement", "dynamic_surge",
  "expressive_match", "expressive_moment",
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
  if (!highlightList || !timelineDiv) return;
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
  equalizeCardHeights();
}

/**
 * Measure the tallest visible collapsed card and apply that as min-height to
 * all collapsed cards so the disclosure links stay on the same horizontal
 * plane.  Uses align-items: flex-start on the parent, so expanded cards grow
 * freely beyond the min-height without stretching their siblings.
 */
function equalizeCardHeights() {
  if (!highlightList) return;
  // Target the inner .card-collapsed panel (not the outer .card) so the
  // disclosure link anchors to the bottom of a fixed-height area. Expanded
  // cards grow by pushing .card-expanded *below* this area, not by shrinking it.
  const panels = Array.from(highlightList.querySelectorAll("article.card:not(.hidden) .card-collapsed"));
  // Clear first so we measure natural heights in the next frame.
  panels.forEach(p => p.style.minHeight = "");
  requestAnimationFrame(() => {
    if (!panels.length) return;
    const maxH = Math.max(...panels.map(p => p.offsetHeight));
    if (maxH > 0) panels.forEach(p => p.style.minHeight = maxH + "px");
  });
}

function setHighlightFilter(category) {
  activeHighlightFilter = activeHighlightFilter === category ? null : category;
  if (highlightFilters) {
    for (const btn of highlightFilters.querySelectorAll(".highlight-filter-badge")) {
      btn.classList.toggle("active", btn.dataset.filter === activeHighlightFilter);
    }
  }
  applyHighlightFilter();
  // Sync the analysis graph to the new filter
  if (analysisGraph) analysisGraph.setFilter(activeHighlightFilter);
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
let selectedSongId = "";

async function loadSongs() {
  status.textContent = "Loading songs…";
  try {
    const r = await fetch(apiUrl("/api/songs"));
    if (!r.ok) throw new Error(`status ${r.status}`);
    const data = await r.json();
    allSongs = data.songs || [];
    if (!allSongs.length) {
      carouselTrack.textContent = "No songs found — run scripts/import_ultrastar.py";
      status.textContent = "No songs available.";
      return;
    }
    // Restore last selection from localStorage if it still exists.
    const remembered = localStorage.getItem(LAST_SONG_KEY);
    const initialId =
      remembered && allSongs.some((s) => s.song_id === remembered)
        ? remembered
        : allSongs[0].song_id;
    renderCarousel(allSongs, initialId);
    status.textContent = "";
  } catch (e) {
    status.textContent = `Failed to load songs: ${e.message}`;
    status.classList.add("error");
  }
}

function fmtDuration(secs) {
  if (!isFinite(secs)) return "?";
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

// Placeholder gradient pairs keyed by first letter of song_id, cycling through a palette.
const PLACEHOLDER_GRADIENTS = [
  ["#4a5568", "#2d3748"],
  ["#4c6a52", "#2d4a34"],
  ["#6a4c52", "#4a2d34"],
  ["#4c4a6a", "#2d2d4a"],
  ["#6a5c3a", "#4a3e22"],
  ["#3a5c6a", "#224a52"],
  ["#6a3a52", "#4a2238"],
  ["#3a6a5c", "#224a42"],
];

function selectCardById(songId, { animate = true } = {}) {
  selectedSongId = songId;
  const song = allSongs.find((s) => s.song_id === songId);
  if (song) localStorage.setItem(LAST_SONG_KEY, songId);

  let selectedCard = null;
  carouselTrack.querySelectorAll(".song-card").forEach((card) => {
    const isSelected = card.dataset.songId === songId;
    card.classList.toggle("selected", isSelected);
    if (isSelected) selectedCard = card;
  });

  // Scroll so the selected card is centered in the track viewport.
  if (selectedCard) {
    const trackRect = carouselTrack.getBoundingClientRect();
    const cardRect = selectedCard.getBoundingClientRect();
    const targetScroll =
      carouselTrack.scrollLeft +
      (cardRect.left - trackRect.left) -
      (trackRect.width - cardRect.width) / 2;
    carouselTrack.scrollTo({
      left: Math.max(0, targetScroll),
      behavior: animate ? "smooth" : "instant",
    });
  }

  // Reset reference cache when song changes.
  currentReference = null;
  currentReferenceSongId = null;
}

function renderCarousel(songs, initialId) {
  carouselTrack.innerHTML = "";

  songs.forEach((song, idx) => {
    const card = document.createElement("div");
    card.className = "song-card";
    card.dataset.songId = song.song_id;

    // Cover image or placeholder.
    if (song.cover_url) {
      const img = document.createElement("img");
      img.className = "song-card__cover";
      img.src = apiUrl(song.cover_url);
      img.alt = `${song.title} cover`;
      img.onerror = () => img.replaceWith(makeCoverPlaceholder(song, idx));
      card.appendChild(img);
    } else {
      card.appendChild(makeCoverPlaceholder(song, idx));
    }

    // Checkmark badge (shown when selected).
    const check = document.createElement("div");
    check.className = "song-card__check";
    check.innerHTML =
      '<svg viewBox="0 0 12 12"><polyline points="2,6 5,9 10,3"/></svg>';
    card.appendChild(check);

    // Title + artist.
    const info = document.createElement("div");
    info.className = "song-card__info";
    const title = document.createElement("div");
    title.className = "song-card__title";
    title.textContent = song.title;
    const artist = document.createElement("div");
    artist.className = "song-card__artist";
    artist.textContent = song.artist || "Unknown artist";
    info.appendChild(title);
    info.appendChild(artist);
    card.appendChild(info);

    // Tags: language, first genre tag, duration.
    const tags = document.createElement("div");
    tags.className = "song-card__tags";
    const tagDefs = [
      { label: song.language || "English", kind: "lang" },
      { label: (song.genre_tags && song.genre_tags[0]) || "Pop", kind: "genre" },
      { label: fmtDuration(song.duration_s), kind: "duration" },
    ];
    for (const t of tagDefs) {
      const span = document.createElement("span");
      span.className = `song-tag song-tag--${t.kind}`;
      span.textContent = t.label;
      tags.appendChild(span);
    }
    card.appendChild(tags);

    card.addEventListener("click", () => selectCardById(song.song_id));
    carouselTrack.appendChild(card);
  });

  // Set initial selection without animation so the card is already centered on load.
  selectCardById(initialId, { animate: false });
  updateCarouselArrows();
}

function makeCoverPlaceholder(song, idx) {
  const div = document.createElement("div");
  div.className = "song-card__cover-placeholder";
  const [a, b] = PLACEHOLDER_GRADIENTS[idx % PLACEHOLDER_GRADIENTS.length];
  div.style.setProperty("--placeholder-a", a);
  div.style.setProperty("--placeholder-b", b);
  // Two-letter initials from title.
  const words = song.title.trim().split(/\s+/);
  div.textContent = words.length > 1
    ? (words[0][0] + words[1][0]).toUpperCase()
    : song.title.slice(0, 2).toUpperCase();
  return div;
}

function updateCarouselArrows() {
  if (!carouselTrack) return;
  const atStart = carouselTrack.scrollLeft <= 4;
  const atEnd =
    carouselTrack.scrollLeft + carouselTrack.clientWidth >=
    carouselTrack.scrollWidth - 4;
  carouselLeft.disabled = atStart;
  carouselRight.disabled = atEnd;
}

carouselLeft.addEventListener("click", () => {
  carouselTrack.scrollBy({ left: -180, behavior: "smooth" });
});
carouselRight.addEventListener("click", () => {
  carouselTrack.scrollBy({ left: 180, behavior: "smooth" });
});
carouselTrack.addEventListener("scroll", updateCarouselArrows);

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
  uploadSection.classList.remove("upload--collapsed");
  uploadSection.classList.add("upload--analyzing");
  if (pickerSection) pickerSection.hidden = true;
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
  const songId = selectedSongId;
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
      if (analyses.length > 0) collapseUploadSection();
      else if (pickerSection) pickerSection.hidden = false;
    } else {
      console.error(e);
      hideAnalysisProgress();
      status.textContent = `Analysis failed: ${e.message}`;
      status.classList.add("error");
      if (analyses.length > 0) collapseUploadSection();
      else if (pickerSection) pickerSection.hidden = false;
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
  if (!overviewTiles) return;
  overviewTiles.innerHTML = "";
  if (!overview) {
    overviewTiles.hidden = true;
    if (overviewHeading) overviewHeading.hidden = true;
    return;
  }
  overviewTiles.hidden = false;
  if (overviewHeading) overviewHeading.hidden = false;

  // Mimic score circle
  if (overview.mimic_score != null) {
    overviewTiles.appendChild(_buildMimicTile(overview.mimic_score));
  }

  // Pitch card
  if (overview.pct_in_tune != null) {
    const med = overview.median_cents;
    const pills = [];
    if (med != null && Math.abs(med) >= 3) {
      const isSharp = med > 0;
      pills.push({
        text: `${isSharp ? "Slightly sharp" : "Slightly flat"} ${isSharp ? "+" : ""}${med.toFixed(0)} cents`,
        cls: isSharp ? "tile-pill--sharp" : "tile-pill--flat",
      });
    }
    if (overview.octave_shift_semitones) {
      const semis = overview.octave_shift_semitones;
      pills.push({ text: `Octave shift: ${semis > 0 ? "+" : ""}${semis} semitones` });
    }
    overviewTiles.appendChild(_buildCard({
      iconClass: "tile-card-icon--pitch",
      iconSvg: _iconPitch(),
      name: "Pitch",
      value: `${(overview.pct_in_tune * 100).toFixed(0)}% in tune`,
      sub: `${overview.note_count} notes scored`,
      pills,
    }));
  }

  // Technique card
  if (overview.technique_match_rate != null) {
    const pct = overview.technique_match_rate * 100;
    const sub = pct >= 70 ? "Close to the reference"
      : pct >= 40 ? "Some variation from reference"
      : "Far from reference";
    overviewTiles.appendChild(_buildCard({
      iconClass: "tile-card-icon--tech",
      iconSvg: _iconTech(),
      name: "Technique",
      value: `${pct.toFixed(0)}% match`,
      sub,
    }));
  }

  // Timing card
  if (overview.arrival_offset_ms_mean != null) {
    const ms = overview.arrival_offset_ms_mean;
    const absMs = Math.abs(ms);
    const direction = ms >= 0 ? "behind the beat" : "ahead of the beat";
    const sub = absMs < 30 ? "Very tight timing" : absMs < 80 ? "Just " + direction : direction;
    overviewTiles.appendChild(_buildCard({
      iconClass: "tile-card-icon--timing",
      iconSvg: _iconTiming(),
      name: "Timing",
      value: `${ms >= 0 ? "+" : ""}${ms.toFixed(0)} ms`,
      sub,
    }));
  }

  // Strongest section card
  const strongestSection = overview.strongest_section ?? computeStrongestSection(analysis);
  if (strongestSection) {
    const tags = _strongestSectionTags(strongestSection, analysis);
    overviewTiles.appendChild(_buildCard({
      iconClass: "tile-card-icon--section",
      iconSvg: _iconSection(),
      name: "Strongest Section",
      value: strongestSection,
      sub: "",
      pills: tags.map(t => ({ text: t })),
    }));
  }
}

function _buildMimicTile(score) {
  const tier = score <= 25 ? "Beginner"
    : score <= 50 ? "Developing"
    : score <= 75 ? "Intermediate"
    : score <= 90 ? "Advanced"
    : "Expert";
  const r = 44;
  const cx = 56;
  const cy = 56;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - score / 100);
  const tile = document.createElement("div");
  tile.className = "tile-mimic";
  tile.innerHTML = `
    <span class="tile-card-name">Mimic Score</span>
    <svg width="112" height="112" viewBox="0 0 112 112">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--border)" stroke-width="8"/>
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--accent)" stroke-width="8"
        stroke-dasharray="${circ.toFixed(2)}" stroke-dashoffset="${offset.toFixed(2)}"
        stroke-linecap="round" transform="rotate(-90 ${cx} ${cy})"/>
      <text x="${cx}" y="${cy - 6}" text-anchor="middle" dominant-baseline="middle"
        font-size="28" font-weight="700" fill="var(--accent)" font-family="inherit">${score.toFixed(0)}</text>
      <text x="${cx}" y="${cy + 18}" text-anchor="middle" dominant-baseline="middle"
        font-size="11" fill="var(--muted)" font-family="inherit">/ 100</text>
    </svg>
    <span class="tile-mimic-label">${tier}</span>
  `;
  return tile;
}

function _buildCard({ iconClass, iconSvg, name, value, sub, pills }) {
  const pillsHtml = pills && pills.length
    ? `<div class="tile-card-pills">${pills.map(p => `<span class="tile-pill ${p.cls || ""}">${p.text}</span>`).join("")}</div>`
    : "";
  const card = document.createElement("div");
  card.className = "tile-card";
  card.innerHTML = `
    <div class="tile-card-header">
      <div class="tile-card-icon ${iconClass}">${iconSvg}</div>
      <span class="tile-card-name">${name}</span>
    </div>
    <div class="tile-card-value">${value}</div>
    ${sub ? `<div class="tile-card-sub">${sub}</div>` : ""}
    ${pillsHtml}
  `;
  return card;
}

function _iconPitch() {
  return `<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="var(--bad)" stroke-width="2" stroke-linecap="round"><path d="M2 10 Q5 4 8 10 Q11 16 14 10"/></svg>`;
}

function _iconTech() {
  return `<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="var(--good)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="2,9 6,13 14,4"/></svg>`;
}

function _iconTiming() {
  return `<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="8" r="6"/><polyline points="8,4 8,8 11,10"/></svg>`;
}

function _iconSection() {
  return `<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="var(--accent)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="8,1.5 10.2,6 15,6.7 11.5,10.1 12.4,15 8,12.6 3.6,15 4.5,10.1 1,6.7 5.8,6"/></svg>`;
}

function _strongestSectionTags(sectionName, analysis) {
  if (!analysis) return ["Pitch", "Expression", "Timing"];
  const section = (analysis.sections || []).find(s => s.name === sectionName);
  if (!section) return ["Pitch", "Expression", "Timing"];

  const notes = analysis.notes || [];
  const techniques = analysis.techniques || [];
  const scores = [];

  if (section.pct_in_tune != null) {
    scores.push({ label: "Pitch", score: section.pct_in_tune });
  }

  const secNoteIndices = new Set();
  for (const n of notes) {
    const mid = 0.5 * (n.start_s + n.end_s);
    if (mid >= section.start_s && mid < section.end_s) secNoteIndices.add(n.note_index);
  }
  let exprCount = 0;
  for (const t of techniques) {
    const hasExpr = (t.matched && t.matched.length) || (t.user_added && t.user_added.length);
    if (secNoteIndices.has(t.note_index) && hasExpr) exprCount++;
  }
  scores.push({ label: "Expression", score: exprCount / Math.max(1, secNoteIndices.size) });

  if (section.arrival_offset_ms_mean != null) {
    scores.push({ label: "Timing", score: Math.max(0, 1 - Math.abs(section.arrival_offset_ms_mean) / 200) });
  }

  if (section.rms_delta_db != null) {
    scores.push({ label: "Volume", score: Math.max(0, 1 - Math.abs(section.rms_delta_db) / 4) });
  }

  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, 3).map(s => s.label);
}

function renderPerformanceSummary(summary) {
  const el = performanceSummaryEl;
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
  if (!sectionRibbon) return;
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
      setSectionFocus(section).catch(console.error);
    });
    sectionRibbon.appendChild(band);
  }
}

// ---- Section focus helpers -----------------------------------------------

/**
 * Render ALL highlight regions across the full recording into timelineDiv.
 * Called at initial render and when restoring from section view.
 */
function renderFullRegions() {
  if (!timelineDiv) return;
  timelineDiv.innerHTML = "";
  if (!currentAnalysis) return;
  const totalDur = currentDuration;
  const offset   = currentAnalysis.global_offset_s ?? 0;
  for (const moment of currentAnalysis.highlights?.moments ?? []) {
    if (
      currentAnalysis.duration_s > 0 &&
      (moment.end_s - moment.start_s) / currentAnalysis.duration_s > 0.25
    ) continue;
    const userStart = moment.start_s + offset;
    const userEnd   = moment.end_s   + offset;
    const left  = (userStart / totalDur) * 100;
    const width = ((userEnd - userStart) / totalDur) * 100;
    const region = document.createElement("div");
    region.className = `region ${regionCategoryClass(moment)}`;
    region.style.left  = `${Math.max(0, left)}%`;
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
}

/**
 * Render highlight regions filtered and positioned relative to a section's
 * user-time range.
 */
function renderSectionRegions(section) {
  if (!timelineDiv) return;
  timelineDiv.innerHTML = "";
  if (!currentAnalysis) return;
  const offset          = currentAnalysis.global_offset_s ?? 0;
  const userSecStart    = section.start_s + offset;
  const userSecEnd      = section.end_s   + offset;
  const sectionDuration = userSecEnd - userSecStart;
  if (sectionDuration <= 0) return;

  for (const moment of currentAnalysis.highlights?.moments ?? []) {
    if (
      currentAnalysis.duration_s > 0 &&
      (moment.end_s - moment.start_s) / currentAnalysis.duration_s > 0.25
    ) continue;
    const userStart = moment.start_s + offset;
    const userEnd   = moment.end_s   + offset;
    // Only show moments that overlap this section
    if (userEnd <= userSecStart || userStart >= userSecEnd) continue;
    const relStart = Math.max(0, userStart - userSecStart);
    const relEnd   = Math.min(sectionDuration, userEnd - userSecStart);
    const left  = (relStart / sectionDuration) * 100;
    const width = ((relEnd - relStart) / sectionDuration) * 100;
    const region = document.createElement("div");
    region.className = `region ${regionCategoryClass(moment)}`;
    region.style.left  = `${Math.max(0, left)}%`;
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
}

/**
 * Render the section ribbon with only the focused section, spanning full width.
 */
function renderSectionFocusedRibbon(section) {
  if (!sectionRibbon) return;
  sectionRibbon.innerHTML = "";
  const kindClass = SECTION_KIND_CLASS[section.kind] || "kind-unknown";
  const band = document.createElement("div");
  band.className = `section-band ${kindClass}`;
  band.style.left  = "0%";
  band.style.width = "100%";
  band.title = section.name;
  band.innerHTML = `<span class="section-band-label">${section.name}</span>`;
  band.title = "Back to full song";
  band.addEventListener("click", () => clearSectionFocus().catch(console.error));
  sectionRibbon.appendChild(band);
}

/**
 * Recreate WaveSurfer zoomed to the given section.
 * Uses the full peaks array + zoom so all existing playback wiring continues
 * to work correctly (the same currentMedia element is reused).
 */
/**
 * Find the scrollable element inside the WaveSurfer renderer.
 *
 * WaveSurfer v7 renders into a child <div> whose internals live inside a
 * shadow root.  We must traverse that shadow root to reach the .scroll
 * container.  Falls back gracefully for builds that don't use shadow DOM.
 */
function findWaveSurferScrollEl() {
  if (!waveformDiv) return null;
  // v7: container → rendererDiv (shadow host) → shadowRoot → .scroll
  const rendererDiv = waveformDiv.firstElementChild;
  if (rendererDiv?.shadowRoot) {
    return (
      rendererDiv.shadowRoot.querySelector(".scroll") ||
      rendererDiv.shadowRoot.firstElementChild
    );
  }
  // Non-shadow-DOM fallback
  return (
    waveformDiv.querySelector(".scroll") ||
    waveformDiv.firstElementChild
  );
}

function buildSectionWaveform(section) {
  detachPlaybackMixSync();
  if (wavesurfer) {
    try { wavesurfer.destroy(); } catch (e) { /* ignore */ }
    wavesurfer = null;
  }

  const offset          = currentAnalysis?.global_offset_s ?? 0;
  const userSecStart    = section.start_s + offset;
  const sectionDuration = Math.max(1, section.end_s - section.start_s);

  // Compute zoom before creation — WaveSurfer v7's zoom() method silently
  // no-ops when peaks are used (it guards on decodedData), so we must pass
  // minPxPerSec as a constructor option so the peaks render at the right width.
  const containerWidth = waveformDiv.clientWidth || 800;
  const minPxPerSec    = containerWidth / sectionDuration;

  wavesurfer = WaveSurfer.create({
    container:     waveformDiv,
    waveColor:     "#cec5bb",
    progressColor: "#3b6fd4",
    height:        128,
    normalize:     true,
    media:         currentMedia,
    autoCenter:    false,
    autoScroll:    false,
    minPxPerSec:   minPxPerSec,
    ...(fullPeaks ? { peaks: [fullPeaks], duration: currentDuration } : {}),
  });

  wavesurfer.on("error", (err) => {
    console.error("WaveSurfer error:", err);
  });

  // Once rendered, scroll the shadow-DOM scroll container to the section
  // start and lock it so the user can't scroll away.
  const applyScroll = () => {
    const scrollEl = findWaveSurferScrollEl();
    if (scrollEl) {
      scrollEl.scrollLeft = Math.round(userSecStart * minPxPerSec);
      scrollEl.style.overflow = "hidden";
    }
  };
  wavesurfer.on("ready", applyScroll);
  // Belt-and-suspenders: also apply after two animation frames in case the
  // ready event fired before our listener was registered.
  requestAnimationFrame(() => requestAnimationFrame(applyScroll));

  if (!fullPeaks && fullAudioUrl) wavesurfer.load(fullAudioUrl);

  attachPlaybackMixSync();
}

/**
 * Recreate WaveSurfer with the full audio at the normal (fit-to-container) zoom.
 * Used when restoring from section view.
 */
function buildFullWaveform() {
  detachPlaybackMixSync();
  if (wavesurfer) {
    try { wavesurfer.destroy(); } catch (e) { /* ignore */ }
    wavesurfer = null;
  }

  wavesurfer = WaveSurfer.create({
    container:     waveformDiv,
    waveColor:     "#cec5bb",
    progressColor: "#3b6fd4",
    height:        128,
    normalize:     true,
    media:         currentMedia,
    ...(fullPeaks ? { peaks: [fullPeaks], duration: currentDuration } : {}),
  });

  wavesurfer.on("error", (err) => {
    console.error("WaveSurfer error:", err);
    status.textContent = `Waveform failed to render (${err}). Playback may still work.`;
    status.classList.add("error");
  });

  if (!fullPeaks && fullAudioUrl) wavesurfer.load(fullAudioUrl);

  attachPlaybackMixSync();
}

/**
 * Show the section story banner for the given section name, or hide it.
 * Stories come from analysis.section_stories (list of SectionStory objects).
 */
function updateSectionStoryBanner(sectionName) {
  if (!sectionStoryBanner) return;
  // The label cell is the previous sibling element inside the grid.
  const labelCell = sectionStoryBanner.previousElementSibling;
  const stories = currentAnalysis?.section_stories ?? [];
  const story = stories.find(s => s.section_name === sectionName);
  if (!story) {
    sectionStoryBanner.hidden = true;
    if (labelCell) labelCell.hidden = true;
    return;
  }
  const narrative = story.narrative || story.deterministic_summary || "";
  if (!narrative) {
    sectionStoryBanner.hidden = true;
    if (labelCell) labelCell.hidden = true;
    return;
  }
  sectionStoryBanner.innerHTML = `<p class="story-narrative">${narrative}</p>`;
  sectionStoryBanner.hidden = false;
  if (labelCell) labelCell.hidden = false;
}

/**
 * Enter section view: zoom all timeline components to the selected section
 * and filter coaching cards to that section only.
 */
async function setSectionFocus(section) {
  // If already focused on this exact section, no-op
  if (focusedSection && focusedSection.name === section.name) return;

  focusedSection = section;

  // 1. Show the back button in the playback controls bar
  if (backToFullSongBtn) backToFullSongBtn.hidden = false;

  // 2. Update section ribbon to show only focused section spanning full width
  renderSectionFocusedRibbon(section);

  // 3. Scope analysis graph to section song-time range
  if (analysisGraph) analysisGraph.setTimeRange(section.start_s, section.end_s);

  // 4. Recreate waveform zoomed to section
  buildSectionWaveform(section);

  // 5. Scope highlight regions to section
  renderSectionRegions(section);

  // 6. Filter coaching cards to section
  const reference = await getReferenceAnnotation(currentPlaybackSongId).catch(() => null);
  renderCoachingCards(reference ?? currentReference, currentAnalysis);

  // 7. Show section story banner above cards
  updateSectionStoryBanner(section.name);
}

/**
 * Leave section view: restore all timeline components to the full recording.
 */
async function clearSectionFocus() {
  focusedSection = null;

  // Hide the back button
  if (backToFullSongBtn) backToFullSongBtn.hidden = true;

  // Restore full analysis graph range
  if (analysisGraph) analysisGraph.setTimeRange(null, null);

  // Restore section ribbon
  if (currentReference && currentAnalysis) {
    renderSectionRibbon(currentReference, currentAnalysis);
  }

  // Restore full waveform (zoom reset)
  buildFullWaveform();

  // Restore all highlight regions
  renderFullRegions();

  // Restore all coaching cards
  const reference = await getReferenceAnnotation(currentPlaybackSongId).catch(() => null);
  renderCoachingCards(reference ?? currentReference, currentAnalysis);

  // Hide section story banner and its label cell
  if (sectionStoryBanner) {
    sectionStoryBanner.hidden = true;
    const labelCell = sectionStoryBanner.previousElementSibling;
    if (labelCell) labelCell.hidden = true;
  }
}

// ---- End section focus helpers -------------------------------------------

// ---------------------------------------------------------------------------
// Multi-analysis session management
// ---------------------------------------------------------------------------

/**
 * Switch the "active" result block to analyses[idx].
 * Updates all module-level per-block variables (wavesurfer, currentAnalysis,
 * waveformDiv, etc.) to point at the chosen block's state and DOM elements.
 * Pauses the previously active block.
 */
function activateBlock(idx) {
  if (idx < 0 || idx >= analyses.length) return;

  // Pause and save state of the currently active block
  if (activeAnalysisIdx >= 0 && activeAnalysisIdx !== idx) {
    const prev = analyses[activeAnalysisIdx];
    try { prev.wavesurfer?.pause(); } catch (e) { /* ignore */ }
    prev.mediaEl?.pause();
    prev.refVocalMedia?.pause();
    prev.instrumentalMedia?.pause();
    // Save mutable interaction state back to the block
    prev.focusedSection  = focusedSection;
    prev.mixRefVocalOn   = mixRefVocalOn;
    prev.mixInstrumentalOn = mixInstrumentalOn;
  }

  // Detach playback sync from the old block before switching
  detachPlaybackMixSync();

  activeAnalysisIdx = idx;
  const block = analyses[idx];

  // Update module-level playback state to the new block
  wavesurfer           = block.wavesurfer;
  analysisGraph        = block.analysisGraph;
  currentMedia         = block.mediaEl;
  currentDuration      = block.analysis.duration_s;
  currentAnalysis      = block.analysis;
  currentPlaybackSongId = block.songId;
  focusedSection       = block.focusedSection;
  fullPeaks            = block.fullPeaks;
  fullAudioUrl         = block.fullAudioUrl;
  refVocalMedia        = block.refVocalMedia;
  instrumentalMedia    = block.instrumentalMedia;
  mixRefVocalOn        = block.mixRefVocalOn;
  mixInstrumentalOn    = block.mixInstrumentalOn;

  // Update per-block DOM element references
  const container = block.container;
  waveformDiv        = container.querySelector('[data-role="waveform"]');
  sectionRibbon      = container.querySelector('[data-role="sectionRibbon"]');
  timelineDiv        = container.querySelector('[data-role="timeline"]');
  overviewTiles      = container.querySelector('[data-role="overviewTiles"]');
  overviewHeading    = container.querySelector('[data-role="overviewHeading"]');
  highlightList      = container.querySelector('[data-role="highlightList"]');
  sectionStoryBanner = container.querySelector('[data-role="sectionStoryBanner"]');
  segmentInfo        = container.querySelector('[data-role="segmentInfo"]');
  performanceSummaryEl = container.querySelector('[data-role="performanceSummary"]');
  // Playback bar buttons are now per-block (inside the template)
  playPause          = container.querySelector('[data-role="playPause"]');
  mixRefVocalBtn     = container.querySelector('[data-role="mixRefVocal"]');
  mixInstrumentalBtn = container.querySelector('[data-role="mixInstrumental"]');
  backToFullSongBtn  = container.querySelector('[data-role="backToFullSong"]');
  highlightFilters   = container.querySelector('[data-role="highlightFilters"]');

  // Wire click handlers to this block's playback bar
  wirePlaybackBarListeners();

  // Sync playback bar UI to this block's state
  const song = allSongs.find(s => s.song_id === block.songId);
  updateMixToggleUi(song);
  if (mixRefVocalBtn) {
    mixRefVocalBtn.classList.toggle("active", mixRefVocalOn);
    mixRefVocalBtn.setAttribute("aria-pressed", String(mixRefVocalOn));
  }
  if (mixInstrumentalBtn) {
    mixInstrumentalBtn.classList.toggle("active", mixInstrumentalOn);
    mixInstrumentalBtn.setAttribute("aria-pressed", String(mixInstrumentalOn));
  }
  if (backToFullSongBtn) backToFullSongBtn.hidden = (block.focusedSection == null);

  // Re-attach playback sync for the newly active block
  if (block.wavesurfer || block.mediaEl) {
    attachPlaybackMixSync();
  }

  updatePlayPauseIcon();

  // Mark the active block visually
  document.querySelectorAll(".result-block").forEach((el, i) => {
    el.classList.toggle("result-block--active", i === idx);
  });
}

/**
 * Permanently remove the analysis block at `idx`, clean up its audio
 * resources, and reset UI state accordingly.
 */
function removeBlock(idx) {
  const block = analyses[idx];
  if (!block) return;

  // Stop and destroy audio for this block
  if (activeAnalysisIdx === idx) detachPlaybackMixSync();
  try { block.wavesurfer?.pause(); block.wavesurfer?.destroy(); } catch (e) { /* ignore */ }
  block.mediaEl?.pause();
  block.refVocalMedia?.pause();
  block.instrumentalMedia?.pause();

  block.container.remove();
  analyses.splice(idx, 1);

  if (analyses.length === 0) {
    // Back to the empty initial state — reset all module-level pointers
    activeAnalysisIdx    = -1;
    wavesurfer           = null;
    analysisGraph        = null;
    currentMedia         = null;
    currentAnalysis      = null;
    currentPlaybackSongId = null;
    focusedSection       = null;
    fullPeaks            = null;
    fullAudioUrl         = null;
    refVocalMedia        = null;
    instrumentalMedia    = null;
    playPause            = null;
    mixRefVocalBtn       = null;
    mixInstrumentalBtn   = null;
    backToFullSongBtn    = null;
    highlightFilters     = null;
    highlightList        = null;
    sectionStoryBanner   = null;
    overviewTiles        = null;
    overviewHeading      = null;
    segmentInfo          = null;
    performanceSummaryEl = null;
    resultsContainer.hidden = true;
    expandUploadSection();
  } else {
    // Pick the block to activate after deletion
    let newIdx = activeAnalysisIdx;
    if (activeAnalysisIdx === idx) {
      newIdx = Math.max(0, idx - 1);
    } else if (activeAnalysisIdx > idx) {
      newIdx = activeAnalysisIdx - 1;
    }
    activeAnalysisIdx = -1; // force activateBlock to re-run
    activateBlock(newIdx);
  }
}

/** Collapse the upload section to the compact "Add a performance" bar. */
function collapseUploadSection() {
  uploadSection.classList.add("upload--collapsed");
  if (pickerSection) pickerSection.hidden = true;
}

/** Expand the upload section back to full upload/karaoke interface. */
function expandUploadSection() {
  uploadSection.classList.remove("upload--collapsed");
  if (pickerSection) pickerSection.hidden = false;
  // Reset pending file so the user starts fresh
  pendingFile = null;
  if (fileInput) fileInput.value = "";
  status.textContent = "";
  setAnalyzeLoading(false);
}

// Wire the "Add a performance" compact bar to re-expand the upload section
addPerformanceBar?.addEventListener("click", () => {
  expandUploadSection();
});

// ---------------------------------------------------------------------------

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
  if (!highlightList) return;
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
  // Build the working moments list, filtering to section if in section view.
  let moments = analysis.highlights?.moments ?? [];
  if (focusedSection) {
    const secStart = focusedSection.start_s;
    const secEnd   = focusedSection.end_s;
    moments = moments.filter(m => m.end_s > secStart && m.start_s < secEnd);
  }

  if (!moments.length) {
    const empty = document.createElement("div");
    empty.className = "empty-cards";
    empty.textContent = focusedSection
      ? `No coaching moments in ${focusedSection.name}.`
      : "No coaching moments — try another take.";
    highlightList.appendChild(empty);
    return;
  }
  initHighlightFilterBadges();
  initBasisFilterBadges();
  if (highlightFilters) highlightFilters.hidden = false;
  if (basisFilters) basisFilters.hidden = false;

  for (const moment of moments) {
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
    const evidenceBadge =
      moment.confidence === "medium"
        ? `<span class="evidence-badge">Limited evidence</span>`
        : "";


    const tryThisEl = moment.practice_tip
      ? `<div class="card-try-this">
           <h6 class="card-try-this-label">Try this</h6>
           <p class="card-try-this-text">${moment.practice_tip}</p>
         </div>`
      : "";

    const hasDisclosure = !NO_DISCLOSURE_TYPES.has(moment.type);

    // V2: corrective cards are collapsed by default with a disclosure link.
    // Affirming and timing cards show all content immediately (no disclosure).
    card.className = `card ${hasDisclosure ? "collapsed" : ""} ${moment.type} ${cardCategoryClass(moment)} ${scopeClass}${confidenceClass}`.trim();

    const songStart = moment.start_s;
    const songEnd = moment.end_s;

    const disclosureFooter = hasDisclosure ? `
        <hr class="card-divider" />
        <a class="card-disclosure" role="button" tabindex="0">Show coaching &amp; comparison</a>` : "";

    const expandedSection = hasDisclosure ? `
      <div class="card-expanded">
        ${lyricEl}
        ${tryThisEl}
        <div class="card-actions">
          <button class="card-play-take" type="button">&#9654; Play your take</button>
          <button class="card-play-ref" type="button">&#9654; Play reference</button>
        </div>
      </div>` : "";

    card.innerHTML = `
      <div class="card-collapsed">
        <header class="card-header">
          <h5>${cardHeadingText(moment)}</h5>
          <span class="card-stat-meta">${fmtTime(userStart)}&ndash;${fmtTime(userEnd)}</span>
        </header>
        <p class="card-title">${moment.title}</p>
        <p class="card-summary">${moment.summary}</p>
        ${sectionTag}
        ${evidenceBadge}
        ${disclosureFooter}
      </div>
      ${expandedSection}
    `;

    if (hasDisclosure) {
      const disclosureLink = card.querySelector(".card-disclosure");
      const expandedEl = card.querySelector(".card-expanded");

      disclosureLink.addEventListener("click", (e) => {
        e.stopPropagation();
        const isExpanded = expandedEl.classList.contains("open");
        expandedEl.classList.toggle("open", !isExpanded);
        disclosureLink.textContent = isExpanded
          ? "Show coaching & comparison"
          : "Hide coaching & comparison";
        card.classList.toggle("collapsed", isExpanded);
        if (isExpanded) {
          // Re-equalize heights after the collapse animation (0.28s) finishes.
          setTimeout(equalizeCardHeights, 300);
        }
      });
      disclosureLink.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          disclosureLink.click();
        }
      });

      card.querySelector(".card-play-take").addEventListener("click", (e) => {
        e.stopPropagation();
        playRanged(userStart, userEnd);
      });

      card.querySelector(".card-play-ref").addEventListener("click", (e) => {
        e.stopPropagation();
        playReferenceRanged(songStart, songEnd);
      });
    }

    card.addEventListener("click", () => {
      playRanged(userStart, userEnd);
      scrollToHighlightCard(card.dataset.momentId);
    });
    highlightList.appendChild(card);
  }
  applyHighlightFilter();
}

/**
 * Seek the user's performance to userStart and stop playback at userEnd.
 * Uses WaveSurfer when available; falls back to the raw media element.
 */
function playRanged(userStart, userEnd) {
  const t = Math.max(0, userStart);

  function attachStop(media) {
    const stopListener = () => {
      if (media.currentTime >= userEnd) {
        media.pause();
        media.removeEventListener("timeupdate", stopListener);
      }
    };
    media.addEventListener("timeupdate", stopListener);
  }

  if (wavesurfer) {
    try {
      wavesurfer.setTime(t);
      const media = wavesurfer.getMediaElement
        ? wavesurfer.getMediaElement()
        : wavesurfer.media;
      if (media) attachStop(media);
      const p = wavesurfer.play();
      if (p && typeof p.catch === "function") p.catch(() => {});
      syncMixLayers(t, true);
      return;
    } catch (e) {
      console.warn("WaveSurfer ranged play failed, using media element:", e);
    }
  }
  if (currentMedia) {
    try {
      currentMedia.currentTime = t;
      attachStop(currentMedia);
      currentMedia.play().catch(() => {});
      syncMixLayers(t, true);
    } catch (e) {
      console.warn("Media ranged play failed:", e);
    }
  }
}

/**
 * Play the reference vocal audio from songStart to songEnd (song time, no offset)
 * then stop. Pauses the user performance while the snippet plays.
 */
function playReferenceRanged(songStart, songEnd) {
  ensureMixMedia();
  if (!refVocalMedia.src) {
    console.warn("Reference audio not loaded yet");
    return;
  }
  // Pause the user performance while reference snippet plays.
  if (wavesurfer && wavesurfer.isPlaying()) wavesurfer.pause();
  else currentMedia?.pause();
  pauseMixLayers();

  refVocalMedia.currentTime = Math.max(0, songStart);
  const stopRef = () => {
    if (refVocalMedia.currentTime >= songEnd) {
      refVocalMedia.pause();
      refVocalMedia.removeEventListener("timeupdate", stopRef);
    }
  };
  refVocalMedia.addEventListener("timeupdate", stopRef);
  refVocalMedia.play().catch((e) => {
    console.warn("Reference ranged play failed:", e);
    refVocalMedia.removeEventListener("timeupdate", stopRef);
  });
}

function currentStarsProfile() {
  return "fast";
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
    // Keep the play/pause icon in sync with actual media state
    currentMedia.addEventListener("play",  updatePlayPauseIcon, { signal });
    currentMedia.addEventListener("pause", updatePlayPauseIcon, { signal });
    currentMedia.addEventListener("ended", updatePlayPauseIcon, { signal });
  }
  if (wavesurfer) {
    for (const ev of ["play", "pause", "timeupdate", "seeking", "interaction"]) {
      wavesurfer.on(ev, sync);
    }
    // Drive the graph playhead loop from WaveSurfer events
    wavesurfer.on("play",  () => {
      if (analysisGraph) analysisGraph.startPlayheadLoop(getPerformanceTime);
    });
    wavesurfer.on("pause", () => {
      if (analysisGraph) {
        analysisGraph.stopPlayheadLoop();
        analysisGraph.setPlayheadUserTime(getPerformanceTime());
      }
    });
    wavesurfer.on("seeking", () => {
      if (analysisGraph) {
        analysisGraph.setPlayheadUserTime(getPerformanceTime());
      }
    });
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
  // --- 1. Create a new result block from the template ---
  const template = document.getElementById("resultBlockTemplate");
  const blockEl = template.content.cloneNode(true).firstElementChild;

  const blockIdx = analyses.length;

  // --- 2. Register the block state (shell; wavesurfer/graph set below) ---
  const blockState = {
    songId,
    analysis,
    container: blockEl,
    wavesurfer: null,
    analysisGraph: null,
    mediaEl: null,
    fullPeaks: null,
    fullAudioUrl: null,
    focusedSection: null,
    refVocalMedia: null,
    instrumentalMedia: null,
    mixRefVocalOn: false,
    mixInstrumentalOn: false,
  };
  analyses.push(blockState);

  // Push a history entry the first time results appear so that a browser Back
  // gesture is intercepted by the popstate listener instead of leaving the page.
  if (!history.state?.secondpass) {
    history.pushState(RESULTS_STATE, "");
  }

  // --- 3. Append block and reveal results container ---
  resultsContainer.hidden = false;
  resultsContainer.appendChild(blockEl);

  // Activate this block: updates all module-level per-block variables so the
  // render helpers below write into the correct DOM elements.
  activateBlock(blockIdx);

  // --- 4. Initialise per-block state ---
  focusedSection = null;
  fullPeaks = null;
  fullAudioUrl = null;
  if (backToFullSongBtn) backToFullSongBtn.hidden = true;
  currentDuration = analysis.duration_s;

  // Segment indicator when only part of the song was recorded (karaoke)
  const song = allSongs.find((s) => s.song_id === songId);
  if (analysis.segment_end_song_s != null && song) {
    segmentInfo.textContent =
      `Analyzed ${fmtMmSs(0)}\u2013${fmtMmSs(analysis.segment_end_song_s)} of ${fmtMmSs(song.duration_s)}`;
    segmentInfo.hidden = false;
  } else {
    segmentInfo.hidden = true;
    segmentInfo.textContent = "";
  }

  // --- 5. Set up mix layers for this block ---
  resetMixToggles();
  // ensureMixMedia() will create fresh Audio elements (refVocalMedia / instrumentalMedia are null
  // for new blocks, so we get brand-new instances per block).
  prepareMixLayers(songId);
  updateMixToggleUi(allSongs.find((s) => s.song_id === songId));
  // Save the new mix media back to the block state immediately so activateBlock
  // can pause them if another block is activated mid-analysis.
  blockState.refVocalMedia    = refVocalMedia;
  blockState.instrumentalMedia = instrumentalMedia;

  // --- 6. Build performance audio element ---
  const audioUrl = apiUrl(`/api/songs/${encodeURIComponent(songId)}/performances/${encodeURIComponent(
    analysis.perf_id,
  )}/audio`);
  fullAudioUrl = audioUrl;
  blockState.fullAudioUrl = audioUrl;

  // Dedicated streaming media element: plays via HTTP range requests and is
  // a reliable fallback for the play-from-here actions regardless of whether
  // the visual waveform decode succeeds.
  const mediaEl = new Audio();
  mediaEl.preload = "auto";
  mediaEl.src = audioUrl;
  currentMedia = mediaEl;
  blockState.mediaEl = mediaEl;

  // --- 7. Fetch precomputed peaks ---
  const peaks = await fetchPeaks(songId, analysis.perf_id);
  fullPeaks = peaks;
  blockState.fullPeaks = peaks;

  // Unhide the timeline grid BEFORE WaveSurfer.create so that the waveform
  // container has non-zero dimensions when the renderer measures it.
  const timelineGrid = blockEl.querySelector('[data-role="timelineGrid"]');
  if (timelineGrid) timelineGrid.hidden = false;

  // --- 8. Create WaveSurfer for this block ---
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
  blockState.wavesurfer = wavesurfer;
  attachPlaybackMixSync();

  // --- 9. Create AnalysisGraph for this block ---
  const graphCanvas     = blockEl.querySelector('[data-role="analysisGraphCanvas"]');
  const graphLabel      = blockEl.querySelector('[data-role="analysisGraphLabel"]');
  const graphTechSelect = blockEl.querySelector('[data-role="techniqueSelect"]');
  if (graphCanvas && graphLabel && graphTechSelect) {
    analysisGraph = new AnalysisGraph(
      graphCanvas,
      graphLabel,
      graphTechSelect,
      apiUrl,
      seekAndPlay,
    );
    const reference = await getReferenceAnnotation(songId);
    analysisGraph.setAnalysis(analysis, reference);
    analysisGraph.setFilter(activeHighlightFilter);
  }
  blockState.analysisGraph = analysisGraph;

  // --- 10. Render content into this block's DOM elements ---
  renderOverviewTiles(analysis.overview, analysis);
  renderPerformanceSummary(analysis.performance_summary);

  // Region markers in song time. NOTE: the Wavesurfer waveform is the
  // user vocal in *user* time; we shift song-time regions back to user
  // time using the analysis.global_offset_s.
  renderFullRegions();

  // Section ribbon (verses/choruses/...) under the waveform.
  // getReferenceAnnotation caches the result; this is effectively free if
  // we already fetched it above for the analysis graph.
  const sectionReference = await getReferenceAnnotation(songId);
  renderSectionRibbon(sectionReference, analysis);

  // Rich coaching cards.
  renderCoachingCards(sectionReference, analysis);

  // --- 11. Make the block interactive: clicking it activates its playback ---
  // Use a live index lookup so this still works after earlier blocks are deleted.
  blockEl.addEventListener("pointerdown", () => {
    const liveIdx = analyses.indexOf(blockState);
    if (liveIdx !== -1 && activeAnalysisIdx !== liveIdx) activateBlock(liveIdx);
  }, { capture: true });

  // Delete button — remove this block and clean up its resources
  const deleteBtn = blockEl.querySelector('[data-role="deleteBlock"]');
  deleteBtn?.addEventListener("click", (e) => {
    e.stopPropagation(); // don't trigger the pointerdown activation above
    const liveIdx = analyses.indexOf(blockState);
    if (liveIdx !== -1) removeBlock(liveIdx);
  });

  // --- 12. Collapse the upload section to the compact "Add a performance" bar ---
  collapseUploadSection();
}

function updatePlayPauseIcon() {
  if (!playPause) return;
  const playing = isPerformancePlaying();
  playPause.classList.toggle("is-playing", playing);
  playPause.setAttribute("aria-label", playing ? "Pause" : "Play");
  playPause.title = playing ? "Pause" : "Play";
}

/**
 * Wire click handlers onto the currently active block's playback bar buttons.
 * Old listeners are cleaned up via AbortController so switching blocks never
 * accumulates stale handlers.
 */
function wirePlaybackBarListeners() {
  playbackListenersAbort?.abort();
  playbackListenersAbort = new AbortController();
  const { signal } = playbackListenersAbort;

  playPause?.addEventListener("click", () => {
    if (wavesurfer) {
      try {
        wavesurfer.playPause();
        syncMixLayers(getPerformanceTime(), isPerformancePlaying());
        updatePlayPauseIcon();
        return;
      } catch (e) {
        /* fall through to media element */
      }
    }
    if (currentMedia) {
      if (currentMedia.paused) currentMedia.play().catch(() => {});
      else currentMedia.pause();
      syncMixLayers(getPerformanceTime(), isPerformancePlaying());
      updatePlayPauseIcon();
    }
  }, { signal });

  mixRefVocalBtn?.addEventListener("click", () => {
    setMixToggle("ref", !mixRefVocalOn);
  }, { signal });

  mixInstrumentalBtn?.addEventListener("click", () => {
    setMixToggle("instrumental", !mixInstrumentalOn);
  }, { signal });

  backToFullSongBtn?.addEventListener("click", () => {
    clearSectionFocus().catch(console.error);
  }, { signal });
}

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
let karaokeRefVocalOn = true;  // default ON per advisor feedback

// Web Audio API handles for the live mic waveform.
let karaokeAudioCtx = null;
let karaokeAnalyser = null;
let karaokeVizData = null;  // Uint8Array reused each frame

/**
 * Encode collected PCM buffers into a 16-bit mono WAV Blob.
 * Runs entirely in the browser — the server receives a plain WAV file that
 * soundfile can open without any conversion or audioread fallback.
 *
 * A peak-scan pass normalizes the signal if any sample exceeds 1.0 to
 * prevent hard digital clipping on hot mic inputs, while leaving the
 * dynamics untouched for normal recordings.
 */
function _wavEncode(buffers, sampleRate) {
  // Scan for the true peak across all collected chunks.
  let peak = 0;
  for (const buf of buffers) {
    for (let i = 0; i < buf.length; i++) {
      const abs = Math.abs(buf[i]);
      if (abs > peak) peak = abs;
    }
  }
  // Only normalize when the signal actually clips; leave clean recordings as-is.
  const gain = peak > 1.0 ? 1.0 / peak : 1.0;
  if (peak > 1.0) {
    console.warn(`[_wavEncode] peak=${peak.toFixed(3)}, auto-normalized to avoid clipping`);
  }

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
      const s = Math.max(-1, Math.min(1, buf[i] * gain));
      v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      off += 2;
    }
  }
  return new Blob([ab], { type: "audio/wav" });
}

// A timing gap larger than this triggers a line break — but only once the
// current line has reached LYRIC_MIN_WORDS real words, so short entries like
// "Yesterday" (a single lyric_word that is just one token) don't get stranded.
const LYRIC_LINE_GAP_S = 0.8;
// A gap this long always forces a new line regardless of word count
// (covers true section rests / instrumentals).
const LYRIC_HARD_GAP_S = 5.0;
// Minimum real space-separated word tokens before a normal gap can break.
const LYRIC_MIN_WORDS = 4;
// Maximum real word tokens before forcing a break regardless of timing.
// Now that phrase_break_before handles primary line splits, this is only
// a last resort for unusually long phrases (e.g. rap-heavy or ad-lib sections).
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
  const songId = selectedSongId;
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
  // phrase_break_before is True on the first note of each word that
  // immediately follows an UltraStar phrase-break marker (``-``).  We keep
  // that flag so the line-grouping step below can use it as a hard break.
  const wordsMap = new Map();
  for (const note of reference.notes) {
    const wordIdx = note.word_index;
    if (wordIdx == null) continue;
    if (!wordsMap.has(wordIdx)) {
      wordsMap.set(wordIdx, {
        word: (note.lyric_word || "").trim(),
        start_s: note.start_s,
        end_s: note.end_s,
        phrase_break_before: !!note.phrase_break_before,
      });
    } else {
      const entry = wordsMap.get(wordIdx);
      entry.end_s = Math.max(entry.end_s, note.end_s);
    }
  }
  const words = [...wordsMap.values()].filter((w) => w.word);
  words.sort((a, b) => a.start_s - b.start_s);

  // Count actual space-separated tokens in the accumulated line (used for
  // the LYRIC_MAX_WORDS hard cap so no single line becomes too long).
  const tokenCount = (line) =>
    line.reduce((n, w) => n + w.word.trim().split(/\s+/).length, 0);

  // Group words into display lines.
  // Primary rule: break whenever a word has phrase_break_before = true
  // (derived directly from the UltraStar "-" phrase-break markers in the
  // chart — the most reliable signal available).
  // Fallback rules for songs without that data or very long phrases:
  //   • hard gap >= LYRIC_HARD_GAP_S always breaks,
  //   • soft gap >= LYRIC_LINE_GAP_S breaks once LYRIC_MIN_WORDS are met,
  //   • LYRIC_MAX_WORDS forces a break regardless of timing.
  const rawLines = [];
  let currentLine = [];
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const prev = i > 0 ? words[i - 1] : null;
    const gap = prev ? w.start_s - prev.end_s : 0;
    if (currentLine.length > 0) {
      const tokens = tokenCount(currentLine);
      const phraseBreak = !!w.phrase_break_before;
      const hardBreak   = gap >= LYRIC_HARD_GAP_S;
      const softBreak   = gap >= LYRIC_LINE_GAP_S && tokens >= LYRIC_MIN_WORDS;
      const maxBreak    = tokens >= LYRIC_MAX_WORDS;
      if (phraseBreak || hardBreak || softBreak || maxBreak) {
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

  // Stop reference vocal playback and reset toggle to default ON for next take.
  if (karaokeRefVocalMedia) {
    karaokeRefVocalMedia.pause();
    karaokeRefVocalMedia = null;
  }
  karaokeRefVocalOn = true;
  karaokeRefVocalBtn.classList.add("active");
  karaokeRefVocalBtn.setAttribute("aria-pressed", "true");

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

  const songId = selectedSongId;
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
      if (analyses.length > 0) collapseUploadSection();
      else if (pickerSection) pickerSection.hidden = false;
    } else {
      console.error(e);
      hideAnalysisProgress();
      status.textContent = `Analysis failed: ${e.message}`;
      status.classList.add("error");
      if (analyses.length > 0) collapseUploadSection();
      else if (pickerSection) pickerSection.hidden = false;
    }
  } finally {
    analysisAbortController = null;
  }
}

karaokeRecordBtn.addEventListener("click", async () => {
  const songId = selectedSongId;
  if (!songId) return;
  if (!navigator.mediaDevices) {
    status.textContent = "Browser does not support microphone access.";
    status.classList.add("error");
    return;
  }
  try {
    karaokeStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
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
  // Route through a zero-gain node so onaudioprocess fires reliably (it
  // requires the node to be connected to the graph) but mic audio does not
  // reach the speakers, preventing acoustic feedback and AEC interference.
  const muteNode = karaokeAudioCtx.createGain();
  muteNode.gain.value = 0;
  karaokeScriptNode.connect(muteNode);
  muteNode.connect(karaokeAudioCtx.destination);
  karaokeViz.hidden = false;

  // Play instrumental
  karaokeAudio.src = apiUrl(`/api/songs/${encodeURIComponent(songId)}/audio/instrumental`);
  karaokeAudio.hidden = false;
  karaokeAudio.currentTime = 0;
  karaokeAudio.addEventListener("ended", _handleKaraokeEnded);
  await karaokeAudio.play().catch(() => {
    /* autoplay restrictions may block; the user can click the visible controls */
  });

  // Load reference vocal and start it immediately (default ON).
  karaokeRefVocalMedia = new Audio();
  karaokeRefVocalMedia.preload = "auto";
  karaokeRefVocalMedia.src = apiUrl(`/api/songs/${encodeURIComponent(songId)}/audio/reference`);
  karaokeRefVocalMedia.volume = MIX_LAYER_VOLUME;

  // Sync button UI to current state before potentially auto-playing.
  karaokeRefVocalBtn.classList.toggle("active", karaokeRefVocalOn);
  karaokeRefVocalBtn.setAttribute("aria-pressed", String(karaokeRefVocalOn));

  if (karaokeRefVocalOn) {
    // Start reference vocal in sync with the instrumental (same user gesture).
    karaokeRefVocalMedia.currentTime = karaokeAudio.currentTime;
    karaokeRefVocalMedia.play().catch(() => {});
  }

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

// Intercept browser Back while analysis results are held in memory.
// Without this, a horizontal overscroll that escapes the card list (or a
// trackpad/swipe gesture starting at the viewport edge) would navigate away
// and destroy the in-memory session.
window.addEventListener("popstate", () => {
  if (analyses.length > 0) {
    // Re-push so consecutive Back presses are also intercepted.
    history.pushState(RESULTS_STATE, "");
    // Restore the results view in case it was hidden.
    resultsContainer.hidden = false;
    collapseUploadSection();
    activateBlock(activeAnalysisIdx >= 0 ? activeAnalysisIdx : analyses.length - 1);
  }
  // If analyses is empty the event is not cancelled, letting the browser
  // navigate away normally (e.g. to a previous site).
});

loadSongs();
