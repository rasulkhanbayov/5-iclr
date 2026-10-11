"""
Trim scale-reference clips to a short duration for VLM input. The ruler/comb
is static throughout the whole original clip (confirmed in Phase 1), so a few
seconds is enough -- showing the model a 144-second clip (02182026's scale v.mp4)
made a single inference call take 7+ minutes instead of ~1 minute, since decord
must still index/decode a much longer container even at low sample fps.

Rebased 2026-08-11 to this project's own layout (paths, session scope).
05052026 dropped: the main experiments use session 02182026 only.
"""
import cv2
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

TRIM_SECONDS = 3.0

DATA_ROOT = sgf_paths.RAW_VIDEO_DIR
PERTURBED_DIR = sgf_paths.PERTURBED_CLIP_DIR
RESAMPLED_DIR = sgf_paths.RESAMPLED_CLIP_DIR
OUT_DIR = sgf_paths.TRIMMED_SCALE_DIR


def trim_clip(src_path, out_path, trim_seconds=TRIM_SECONDS):
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_keep = int(fps * trim_seconds)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    n_written = 0
    for _ in range(n_keep):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n_written += 1
    cap.release()
    writer.release()
    return {"src": src_path, "out": out_path, "n_frames": n_written, "fps": fps}


if __name__ == "__main__":
    jobs = [
        # E2 base + relabeled (R_alpha) conditions
        (f"{DATA_ROOT}/02182026/scale v.mp4",
         os.path.join(OUT_DIR, "02182026_scalev_base_trim.mp4")),
        (os.path.join(PERTURBED_DIR, "02182026_relabel_alpha0.5.mp4"),
         os.path.join(OUT_DIR, "02182026_scalev_alpha0.5_trim.mp4")),
        (os.path.join(PERTURBED_DIR, "02182026_relabel_alpha2.mp4"),
         os.path.join(OUT_DIR, "02182026_scalev_alpha2_trim.mp4")),
        # X4 resampling (S_s) conditions -- scale clip only; the droplet
        # fall-clips are trimmed to their own motion window already by
        # prepare_clips.py and do not need this 3s cap.
        (os.path.join(RESAMPLED_DIR, "02182026_resample_s0.5.mp4"),
         os.path.join(OUT_DIR, "02182026_scalev_resample_s0.5_trim.mp4")),
        (os.path.join(RESAMPLED_DIR, "02182026_resample_s2.mp4"),
         os.path.join(OUT_DIR, "02182026_scalev_resample_s2_trim.mp4")),
    ]
    os.makedirs(OUT_DIR, exist_ok=True)
    for src, out in jobs:
        if not os.path.exists(src):
            print(f"SKIP (missing source): {src}")
            continue
        result = trim_clip(src, out)
        print(result)
