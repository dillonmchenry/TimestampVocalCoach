import WaveSurfer from "https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js";

const songSelect = document.getElementById("songSelect");
const songMeta = document.getElementById("songMeta");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const analyzeButton = document.getElementById("analyzeButton");
const status = document.getElementById("status");
const results = document.getElementById("results");
const waveformDiv = document.getElementById("waveform");
const timelineDiv = document.getElementById("timeline");
const offsetSummary = document.getElementById("offsetSummary");
const playPause = document.getElementById("playPause");
const highlightList = document.getElementById("highlightList");

let pendingFile = null;
let wavesurfer = null;
let currentDuration = 0;

const TYPE_LABEL = {
  best_pitch_phrase: "Cleanest run",
  pitch_struggle: "Tricky passage",
  expressive_match: "Matched expression",
  expressive_moment: "Your expression",
  missed_expression: "Missed expression",
  late_entrance: "Watch entrance",
};

async function loadSongs() {
  status.textContent = "Loading songs…";
  try {
    const r = await fetch("/api/songs");
    if (!r.ok) throw new Error(`status ${r.status}`);
    const data = await r.json();
    songSelect.innerHTML = "";
    if (!data.songs.length) {
      const opt = document.createElement("option");
      opt.textContent = "(no songs found — run scripts/import_ultrastar.py)";
      songSelect.appendChild(opt);
      songSelect.disabled = true;
      status.textContent = "No songs available.";
      return;
    }
    for (const s of data.songs) {
      const opt = document.createElement("option");
      opt.value = s.song_id;
      opt.textContent = `${s.title}: ${s.artist || "Unknown artist"}`;
      opt._song = s;
      songSelect.appendChild(opt);
    }
    songSelect.disabled = false;
    songSelect.addEventListener("change", () => updateSongMeta(songSelect.selectedOptions[0]?._song));
    updateSongMeta(songSelect.selectedOptions[0]?._song);
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

function updateSongMeta(song) {
  if (!song) {
    songMeta.innerHTML = "";
    return;
  }
  const featureBits = [];
  featureBits.push(song.has_instrumental ? "instrumental" : "—");
  featureBits.push(song.has_reference_pitch ? "ref pitch" : "no ref pitch");
  featureBits.push(song.has_reference_stars ? "ref STARS" : "no ref STARS");
  songMeta.innerHTML = `
    <span><strong>Duration:</strong> ${fmtDuration(song.duration_s)}</span>
  `;
}

function setPendingFile(file) {
  pendingFile = file;
  if (file) {
    status.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    status.classList.remove("error");
    analyzeButton.disabled = false;
  } else {
    analyzeButton.disabled = true;
  }
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

  analyzeButton.disabled = true;
  status.classList.remove("error");
  status.textContent = "Analyzing… this can take a minute or two depending on STARS";

  const fd = new FormData();
  fd.append("file", pendingFile);

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
    renderAnalysis(songId, analysis);
  } catch (e) {
    console.error(e);
    status.textContent = `Analysis failed: ${e.message}`;
    status.classList.add("error");
  } finally {
    analyzeButton.disabled = false;
  }
});

function renderAnalysis(songId, analysis) {
  results.hidden = false;
  currentDuration = analysis.duration_s;
  const offsetMsLabel = `Estimated user offset: ${(analysis.global_offset_s * 1000).toFixed(0)} ms`;
  const shift = analysis.octave_shift_semitones || 0;
  let summary = offsetMsLabel;
  if (shift !== 0) {
    const octaves = Math.abs(shift) / 12;
    const dir = shift < 0 ? "below" : "above";
    const sign = shift > 0 ? "+" : "";
    const oneOrMany = octaves === 1 ? "octave" : "octaves";
    summary += ` · Auto-transposed ${sign}${shift} semitones (${octaves} ${oneOrMany} ${dir} the chart)`;
  }
  offsetSummary.textContent = summary;

  if (wavesurfer) {
    wavesurfer.destroy();
    wavesurfer = null;
  }
  wavesurfer = WaveSurfer.create({
    container: waveformDiv,
    waveColor: "#3b4254",
    progressColor: "#6cb4ff",
    height: 128,
    normalize: true,
  });
  const audioUrl = `/api/songs/${encodeURIComponent(songId)}/performances/${encodeURIComponent(
    analysis.perf_id
  )}/audio`;
  wavesurfer.load(audioUrl);

  // Region markers in song time. NOTE: the Wavesurfer waveform is the
  // user vocal in *user* time; we shift song-time regions back to user
  // time using the analysis.global_offset_s.
  const offsetMs = analysis.global_offset_s * 1000;
  timelineDiv.innerHTML = "";
  const totalDur = currentDuration;
  for (const moment of analysis.highlights.moments) {
    const userStart = moment.start_s + analysis.global_offset_s;
    const userEnd = moment.end_s + analysis.global_offset_s;
    const left = (userStart / totalDur) * 100;
    const width = ((userEnd - userStart) / totalDur) * 100;
    const region = document.createElement("div");
    region.className = `region ${moment.type}`;
    region.style.left = `${Math.max(0, left)}%`;
    region.style.width = `${Math.max(1, width)}%`;
    region.title = `${moment.title}\n${moment.summary}`;
    region.addEventListener("click", () => {
      if (!wavesurfer) return;
      wavesurfer.setTime(Math.max(0, userStart));
      wavesurfer.play();
    });
    timelineDiv.appendChild(region);
  }

  highlightList.innerHTML = "";
  for (const moment of analysis.highlights.moments) {
    const li = document.createElement("li");
    li.className = moment.type;
    const userStart = moment.start_s + analysis.global_offset_s;
    const userEnd = moment.end_s + analysis.global_offset_s;
    li.innerHTML = `
      <span class="title">${moment.title}</span>
      <span class="summary">${moment.summary}</span>
      <span class="meta">${fmtTime(userStart)}–${fmtTime(userEnd)} · score ${moment.score.toFixed(2)} · ${moment.type}</span>
    `;
    li.addEventListener("click", () => {
      if (!wavesurfer) return;
      wavesurfer.setTime(Math.max(0, userStart));
      wavesurfer.play();
    });
    highlightList.appendChild(li);
  }
}

function fmtTime(t) {
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

playPause.addEventListener("click", () => {
  if (!wavesurfer) return;
  wavesurfer.playPause();
});

loadSongs();
