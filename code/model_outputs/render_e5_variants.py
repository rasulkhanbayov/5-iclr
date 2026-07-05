import sys
import json
sys.path.insert(0, "/ephemeral/sgf_work/perturbation")
from perturb_renderer import render_relabeled_scale_video

OUT_DIR = "/ephemeral/sgf_work/perturbed_clips/e5"
import os
os.makedirs(OUT_DIR, exist_ok=True)

SESSIONS = ["02182026", "05052026"]
ALPHAS = [0.25, 0.5, 2, 4]
ORIENTATIONS = [False, True]  # rotate_orientation

manifest = []
for session in SESSIONS:
    for alpha in ALPHAS:
        for rotated in ORIENTATIONS:
            tag = f"{session}_alpha{alpha}_{'rot' if rotated else 'orig'}"
            out_path = os.path.join(OUT_DIR, f"{tag}.mp4")
            meta = render_relabeled_scale_video(session, alpha, out_path, rotate_orientation=rotated)
            manifest.append(meta)
            print(f"{tag}: new_label={meta['new_label']} n_frames={meta['n_frames']}", flush=True)

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nwrote {len(manifest)} variants to {OUT_DIR}")
