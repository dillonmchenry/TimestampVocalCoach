// API base URL for SecondPass.
//
// Local dev (served by FastAPI on the same origin): leave empty.
// Vercel / external hosting: set to the full RunPod pod URL, e.g.
//   window.API_BASE = "https://abc123xyz-8000.proxy.runpod.net";
//
// This file is overwritten at Vercel build time from the API_BASE env var.
// To update the RunPod URL: change API_BASE in Vercel → Settings → Environment
// Variables, then trigger a redeploy (takes ~10 seconds).
window.API_BASE = "";
