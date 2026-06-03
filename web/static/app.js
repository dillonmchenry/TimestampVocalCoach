import WaveSurfer from "https://unpkg.com/wavesurfer.js@7.8.6/dist/wavesurfer.esm.js";

const songSelect = document.getElementById("songSelect");
const songBadges = document.getElementById("songBadges");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const analyzeButton = document.getElementById("analyzeButton");
const status = document.getElementById("status");
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
const overviewTiles = document.getElementById("overviewTiles");
const fastProfile = document.getElementById("fastProfile");

let pendingFile = null;
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
};

/** Maps backend moment.type -> feedback category (matches highlight engine). */
const MOMENT_TYPE_CATEGORY = {
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
]);

function cardCategoryClass(moment) {
  const cat = MOMENT_TYPE_CATEGORY[moment.type] || "pitch";
  if (cat === "pitch") {
    return PITCH_GOOD_TYPES.has(moment.type) ? "cat-pitch-good" : "cat-pitch-bad";
  }
  return `cat-${cat}`;
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

function applyHighlightFilter() {
  const cards = highlightList.querySelectorAll("article.card");
  let visible = 0;
  for (const card of cards) {
    const match =
      !activeHighlightFilter || card.dataset.category === activeHighlightFilter;
    card.classList.toggle("hidden", !match);
    if (match) visible += 1;
  }
  let empty = highlightList.querySelector(".filter-empty");
  if (activeHighlightFilter && visible === 0 && cards.length > 0) {
    if (!empty) {
      empty = document.createElement("div");
      empty.className = "empty-cards filter-empty";
      highlightList.appendChild(empty);
    }
    const label = CATEGORY_DISPLAY[activeHighlightFilter] || activeHighlightFilter;
    empty.textContent = `No ${label.toLowerCase()} highlights in this take.`;
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

function initHighlightFilterBadges() {
  if (!highlightFilters || highlightFilters.dataset.bound) return;
  highlightFilters.dataset.bound = "1";
  for (const btn of highlightFilters.querySelectorAll(".highlight-filter-badge")) {
    btn.addEventListener("click", () => {
      setHighlightFilter(btn.dataset.filter);
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
    const r = await fetch("/api/songs");
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

  setAnalyzeLoading(true);
  status.classList.remove("error");
  status.textContent = "";

  const fd = new FormData();
  fd.append("file", pendingFile);
  fd.append("stars_profile", currentStarsProfile());

  try {
    const r = await fetch(`/api/songs/${encodeURIComponent(songId)}/analyze`, {
      method: "POST",
      body: fd,
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`HTTP ${r.status}: ${detail.slice(0, 200)}`);
    }
    const analysis = await r.json();
    status.textContent = `Done. Performance ID: ${analysis.perf_id}`;
    await renderAnalysis(songId, analysis);
  } catch (e) {
    console.error(e);
    status.textContent = `Analysis failed: ${e.message}`;
    status.classList.add("error");
  } finally {
    setAnalyzeLoading(false);
  }
});

async function getReferenceAnnotation(songId) {
  if (currentReference && currentReferenceSongId === songId) {
    return currentReference;
  }
  try {
    const r = await fetch(
      `/api/songs/${encodeURIComponent(songId)}/reference_annotation`,
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

function renderSectionRibbon(reference, analysis) {
  sectionRibbon.innerHTML = "";
  if (!reference || !Array.isArray(reference.sections) || !reference.sections.length) {
    return;
  }
  const totalDur = currentDuration;
  const offset = analysis.global_offset_s || 0;
  const trendByName = {};
  for (const trend of analysis.sections || []) {
    trendByName[trend.name] = trend;
  }
  for (const section of reference.sections) {
    const userStart = section.start_s + offset;
    const userEnd = section.end_s + offset;
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
  const seenWords = new Set();
  const words = [];
  for (const idx of noteIndices) {
    const note = reference.notes[idx];
    if (!note) continue;
    const word = (note.lyric_word || "").trim();
    if (!word) continue;
    const key = `${note.word_index}:${word}`;
    if (seenWords.has(key)) continue;
    seenWords.add(key);
    words.push(word);
    if (words.length >= 12) break;
  }
  return words.join(" ");
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
  card.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
}

function renderCoachingCards(reference, analysis) {
  highlightList.innerHTML = "";
  activeHighlightFilter = null;
  if (highlightFilters) {
    highlightFilters.hidden = true;
    for (const btn of highlightFilters.querySelectorAll(".highlight-filter-badge")) {
      btn.classList.remove("active");
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
  if (highlightFilters) highlightFilters.hidden = false;

  for (const moment of analysis.highlights.moments) {
    const card = document.createElement("article");
    const scopeClass = moment.scope === "section" ? "scope-section" : "scope-local";
    const category = momentCategoryKey(moment);
    card.className = `card ${moment.type} ${cardCategoryClass(moment)} ${scopeClass}`;
    card.dataset.category = category;
    card.dataset.momentId = momentDomId(moment);

    const userStart = moment.start_s + analysis.global_offset_s;
    const userEnd = moment.end_s + analysis.global_offset_s;
    const lyric = lyricSnippet(reference, moment.note_indices);
    const sectionTag = moment.section_names?.length
      ? `<p class="card-section">${moment.section_names.join(" · ")}</p>`
      : "";
    const lyricEl = lyric ? `<blockquote class="card-lyric">“${lyric}”</blockquote>` : "";

    card.innerHTML = `
      <header class="card-header">
        <h5>${cardHeadingText(moment)}</h5>
      </header>
      <p class="card-title">${moment.title}</p>
      <p class="card-summary">${moment.summary}</p>
      ${sectionTag}
      ${lyricEl}
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
    });
    card.addEventListener("click", () => {
      seekAndPlay(userStart);
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
      `/api/songs/${encodeURIComponent(songId)}/performances/${encodeURIComponent(
        perfId,
      )}/loudness`,
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
  const base = `/api/songs/${encodeURIComponent(songId)}/audio`;
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

  const audioUrl = `/api/songs/${encodeURIComponent(songId)}/performances/${encodeURIComponent(
    analysis.perf_id,
  )}/audio`;

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

  // Region markers in song time. NOTE: the Wavesurfer waveform is the
  // user vocal in *user* time; we shift song-time regions back to user
  // time using the analysis.global_offset_s.
  timelineDiv.innerHTML = "";
  const totalDur = currentDuration;
  for (const moment of analysis.highlights.moments) {
    if (moment.scope === "section") continue; // shown via the section ribbon instead
    const userStart = moment.start_s + analysis.global_offset_s;
    const userEnd = moment.end_s + analysis.global_offset_s;
    const left = (userStart / totalDur) * 100;
    const width = ((userEnd - userStart) / totalDur) * 100;
    const region = document.createElement("div");
    region.className = `region ${moment.type}`;
    region.style.left = `${Math.max(0, left)}%`;
    region.style.width = `${Math.max(1, width)}%`;
    region.title = `${moment.title}\n${moment.summary}`;
    region.dataset.momentId = momentDomId(moment);
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

let karaokeMediaRecorder = null;
let karaokeChunks = [];
let karaokeStream = null;
let karaokeRafHandle = null;
let karaokeStartTs = 0;
let karaokeWordRows = [];

// Web Audio API handles for the live mic waveform.
let karaokeAudioCtx = null;
let karaokeAnalyser = null;
let karaokeVizData = null;  // Uint8Array reused each frame

// Rolling-window constants (seconds).
const LYRIC_PAST_S = 1.0;    // keep just-sung words visible for this long after they end
const LYRIC_FUTURE_S = 3.5;  // show this much of upcoming lyrics ahead of current time

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
  if (!reference || !Array.isArray(reference.notes)) {
    karaokeLyrics.innerHTML = `<p class="hint">Lyrics will appear here once a reference annotation is available.</p>`;
    return;
  }
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
  for (const w of words) {
    const span = document.createElement("span");
    span.className = "karaoke-word";
    span.textContent = w.word + " ";
    span.dataset.start = w.start_s;
    span.dataset.end = w.end_s;
    // Initially show only the first window so the preview doesn't dump all lyrics.
    span.style.display = w.start_s <= LYRIC_FUTURE_S ? "" : "none";
    karaokeLyrics.appendChild(span);
    karaokeWordRows.push({ span, start_s: w.start_s, end_s: w.end_s });
  }
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
  let activeIdx = -1;
  for (let i = 0; i < karaokeWordRows.length; i++) {
    const row = karaokeWordRows[i];
    if (t >= row.start_s && t < row.end_s) {
      activeIdx = i;
      break;
    }
  }

  const windowStart = t - LYRIC_PAST_S;
  const windowEnd = t + LYRIC_FUTURE_S;
  for (let i = 0; i < karaokeWordRows.length; i++) {
    const row = karaokeWordRows[i];
    const inWindow = row.end_s >= windowStart && row.start_s <= windowEnd;
    row.span.style.display = inWindow ? "" : "none";
    row.span.classList.toggle("active", i === activeIdx);
    row.span.classList.toggle("sung", i < activeIdx && inWindow);
  }
}

function updateKaraokeHighlight() {
  const t = (performance.now() - karaokeStartTs) / 1000;
  karaokeTime.textContent = fmtMmSs(t);
  _drawKaraokeWaveform();
  applyKaraokeLyricsAtTime(t);
  karaokeRafHandle = requestAnimationFrame(updateKaraokeHighlight);
}

function resetKaraokeLyricsView() {
  karaokeTime.textContent = "0:00";
  applyKaraokeLyricsAtTime(0);
}

async function endKaraokeSession({ analyze }) {
  if (!karaokeMediaRecorder) return;
  const recorder = karaokeMediaRecorder;
  const stopped = new Promise((resolve) => {
    recorder.onstop = resolve;
  });
  recorder.stop();
  karaokeAudio.pause();
  karaokeRecordBtn.disabled = false;
  karaokeStopBtn.disabled = true;
  karaokeCancelBtn.disabled = true;
  cancelAnimationFrame(karaokeRafHandle);
  await stopped;
  karaokeMediaRecorder = null;
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
  const recordedChunks = karaokeChunks;
  karaokeChunks = [];
  resetKaraokeLyricsView();

  if (!analyze) {
    status.textContent = "";
    status.classList.remove("error");
    return;
  }

  const songId = songSelect.value;
  if (!songId) return;
  const mime = recorder.mimeType || "audio/webm";
  const ext = mime.includes("ogg") ? "ogg" : mime.includes("mp4") ? "m4a" : "webm";
  const blob = new Blob(recordedChunks, { type: mime });
  status.textContent = "Analyzing karaoke take…";
  const fd = new FormData();
  fd.append("file", blob, `karaoke.${ext}`);
  fd.append("stars_profile", currentStarsProfile());
  try {
    const r = await fetch(
      `/api/songs/${encodeURIComponent(songId)}/analyze`,
      { method: "POST", body: fd },
    );
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`HTTP ${r.status}: ${detail.slice(0, 200)}`);
    }
    const analysis = await r.json();
    status.textContent = `Done. Performance ID: ${analysis.perf_id}`;
    await renderAnalysis(songId, analysis);
  } catch (e) {
    console.error(e);
    status.textContent = `Analysis failed: ${e.message}`;
    status.classList.add("error");
  }
}

karaokeRecordBtn.addEventListener("click", async () => {
  const songId = songSelect.value;
  if (!songId) return;
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    status.textContent = "Browser does not support MediaRecorder.";
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
  karaokeChunks = [];
  karaokeMediaRecorder = new MediaRecorder(karaokeStream);
  karaokeMediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) karaokeChunks.push(e.data);
  };
  karaokeMediaRecorder.start();

  // Wire mic stream into the Web Audio AnalyserNode for the live waveform.
  karaokeAudioCtx = new AudioContext();
  karaokeAnalyser = karaokeAudioCtx.createAnalyser();
  karaokeAnalyser.fftSize = 512;
  karaokeAnalyser.smoothingTimeConstant = 0.6;
  karaokeVizData = new Uint8Array(karaokeAnalyser.fftSize);
  karaokeAudioCtx.createMediaStreamSource(karaokeStream).connect(karaokeAnalyser);
  karaokeViz.hidden = false;

  // Play instrumental
  karaokeAudio.src = `/api/songs/${encodeURIComponent(songId)}/audio/instrumental`;
  karaokeAudio.hidden = false;
  karaokeAudio.currentTime = 0;
  await karaokeAudio.play().catch(() => {
    /* autoplay restrictions may block; the user can click the visible controls */
  });

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

loadSongs();
