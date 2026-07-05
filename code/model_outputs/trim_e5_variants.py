"""
Trim the E5 scale-clip variants the same way trim_scale_clips.py did for E2 --
02182026's source (scale v.mp4) is 144s/8657 frames; showing the model the
full untrimmed clip makes each inference call take 7+ minutes. The ruler is
static throughout, so 3s is enough (same justification as trim_scale_clips.py).
05052026's source is already short (919 frames, ~15s at its fps) but trimmed
too for consistency across the E5 condition grid.
"""
import sys
import os
import json
sys.path.insert(0, "/ephemeral/sgf_work/model_outputs")
from trim_scale_clips import trim_clip

SRC_DIR = "/ephemeral/sgf_work/perturbed_clips/e5"
OUT_DIR = "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/e5"
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(SRC_DIR, "manifest.json")) as f:
    manifest = json.load(f)

results = []
for entry in manifest:
    src = entry["out_path"]
    name = os.path.basename(src)
    out_path = os.path.join(OUT_DIR, name)
    meta = trim_clip(src, out_path)
    meta.update({k: entry[k] for k in ("session", "alpha", "rotated_orientation", "new_label")})
    results.append(meta)
    print(f"{name}: {meta['n_frames']} frames", flush=True)

# Also trim base (alpha=1, unperturbed) originals for both sessions/orientations,
# needed as the SFS denominator (pred_base) -- E2 already has the unrotated base
# trims (trimmed_scale_clips/{session}_..._base_trim.mp4); only the ROTATED base
# variant is new for E5.
import cv2
sys.path.insert(0, "/ephemeral/sgf_work/perturbation")
from perturb_renderer import SESSION_LABEL_CONFIG

for session in ["02182026", "05052026"]:
    cfg = SESSION_LABEL_CONFIG[session]
    cap = cv2.VideoCapture(cfg["scale_clip"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_keep = int(fps * 3.0)
    out_path = os.path.join(OUT_DIR, f"{session}_base_rot_trim.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (h, w))
    n_written = 0
    for _ in range(n_keep):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
        n_written += 1
    cap.release()
    writer.release()
    results.append({"session": session, "alpha": 1, "rotated_orientation": True,
                     "new_label": cfg["original_label"], "out": out_path, "n_frames": n_written})
    print(f"{session}_base_rot_trim.mp4: {n_written} frames", flush=True)

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {len(results)} trimmed variants to {OUT_DIR}")
