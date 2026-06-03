"""Threshold sweep on held-out clips to find optimal per-technique decision boundary.

Usage:
    python scripts/threshold_sweep.py [student_dir]

Defaults to stars_student_v4. Supports both frame-level (v3) and
phoneme-level (v4+) technique head modes.
"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sklearn.metrics import f1_score
from vocal_coach.student_runner import load_student

TECH_NAMES = ['bubble','breathe','pharyngeal','vibrato','glissando','mixed','falsetto','weak','strong']
corpus = Path('data/student_corpus')
manifest = [json.loads(l) for l in (corpus / 'manifest.jsonl').read_text().splitlines()]
held = [r for r in manifest if r['split'] == 'held_out']

student_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('stars_student_v4')
model, phone_vocab, resolved_device = load_student(student_dir, device='cuda')
use_phoneme_level = getattr(model.config, 'phoneme_level_tech', False)
print(f"Loaded {student_dir} | phoneme_level_tech={use_phoneme_level}")

all_probs  = {t: [] for t in TECH_NAMES}
all_labels = {t: [] for t in TECH_NAMES}

for rec in held:
    feat_dir = corpus / rec['feature_dir']
    mel_path = feat_dir / 'mel.npy'
    f0_path  = feat_dir / 'f0.npy'
    lbl_path = feat_dir / 'labels.json'
    if not (mel_path.is_file() and f0_path.is_file() and lbl_path.is_file()):
        continue
    mel = torch.from_numpy(np.load(str(mel_path))).float().unsqueeze(0).to(resolved_device)
    f0  = torch.from_numpy(np.load(str(f0_path))).float().unsqueeze(0).to(resolved_device)
    labels   = json.loads(lbl_path.read_text())

    ph_starts = labels.get('ph_start_frames', [])
    techniques = labels.get('techniques', {})
    n_frames   = mel.shape[1]
    n_phones   = len(ph_starts)

    if n_phones < 2:
        continue

    with torch.no_grad():
        out = model(mel, f0)

    ph_ends = ph_starts[1:] + [n_frames]
    for pi in range(n_phones):
        s = int(ph_starts[pi])
        e = int(min(ph_ends[pi], n_frames))
        if e <= s:
            continue

        if use_phoneme_level:
            # Pool h over the phoneme span, run technique head.
            ph_h = out.h[0, s:e].mean(dim=0, keepdim=True)      # (1, d)
            ph_logit = model.technique_head(ph_h).squeeze(0)     # (K,)
            ph_prob = torch.sigmoid(ph_logit).detach().cpu().numpy()  # (K,)
        else:
            # Legacy: average frame-level sigmoid probs.
            frame_probs = torch.sigmoid(out.technique_logits.squeeze(0)).cpu().numpy()
            ph_prob = frame_probs[s:e].mean(axis=0)              # (K,)

        for ki, tname in enumerate(TECH_NAMES):
            label_val = techniques.get(tname, [0]*n_phones)
            lv = int(label_val[pi]) if pi < len(label_val) else 0
            all_probs[tname].append(float(ph_prob[ki]))
            all_labels[tname].append(lv)

print(f"\nEvaluated {len(held)} held-out clips (phoneme-level comparison)")
print()
print(f"{'technique':<14} {'pos_mean':>9} {'neg_mean':>9} {'support':>8} {'opt_thresh':>11} {'opt_F1':>7}")
print("-" * 65)
opt_f1s = []
opt_thresholds = {}
for name in TECH_NAMES:
    lbl = np.array(all_labels[name])
    prb = np.array(all_probs[name])
    pos_mask = lbl > 0
    neg_mask = lbl == 0
    pos_mean = float(prb[pos_mask].mean()) if pos_mask.any() else float('nan')
    neg_mean = float(prb[neg_mask].mean()) if neg_mask.any() else float('nan')
    support  = int(pos_mask.sum())
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (prb >= t).astype(int)
        f1 = f1_score(lbl, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    opt_f1s.append(best_f1)
    opt_thresholds[name] = round(best_t, 2)
    print(f"{name:<14} {pos_mean:>9.4f} {neg_mean:>9.4f} {support:>8d} {best_t:>11.2f} {best_f1:>7.3f}")

print()
macro_opt = np.mean(opt_f1s)
print(f"Macro F1 at optimal per-class thresholds: {macro_opt:.3f}")
macro_50 = np.mean([
    f1_score(np.array(all_labels[t]), (np.array(all_probs[t]) >= 0.5).astype(int), zero_division=0)
    for t in TECH_NAMES
])
print(f"Macro F1 at 0.50 threshold:               {macro_50:.3f}")
print()
print("Optimal TECH_THRESHOLDS dict:")
print("TECH_THRESHOLDS = {")
for name in TECH_NAMES:
    print(f'    "{name}": {opt_thresholds[name]},')
print("}")
