"""
Improvement plan A2: contrast-enhance the 5 E3 items that failed the control
question, all from 05052026, all traced to that session's low-contrast,
heavily-vignetted footage (see project_sgf_e3_results.md). A whole-frame
global contrast stretch or CLAHE was tried first and barely helped (see
conversation, a2_stretched_0.08cg.png / a2_clahe_0.08cg.png) -- the ENTIRE
frame has low dynamic range, not just the surface strip, so global
normalization doesn't concentrate the available contrast where it's needed.
What worked: crop to the impact region (a fixed ROI around the linked
droplet's final position, not the whole frame) and contrast-stretch ONLY that
local crop -- this makes the surface boundary and the droplet's impact shape
clearly legible (verified by hand, a2_impact_sequence_stretched.png shows a
droplet visibly flattened against a now-legible surface line at frame 490).

Re-encodes each of the 5 failing fall clips as a cropped+stretched version
and re-runs the E3 control question on those 5 only (not all 12 -- the other
7 already pass).
"""
import sys
import os
import json
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

sys.path.insert(0, os.path.join(sgf_paths.CODE_DIR, "oracle"))
from droplet_oracle import (find_motion_window, build_static_background,
                             detect_droplet_blobs, link_primary_droplet, TRUE_FPS)

FAILING_CLIPS = ["cainhcg 0.08 b.mp4", "cainhg0.02 2.mp4", "0.08cg.mp4", "0.08cg2.mp4", "0.08cg3.mp4"]
SRC_DIR = sgf_paths.raw_video("05052026")
FALL_CLIP_DIR = sgf_paths.FALL_CLIP_DIR
OUT_DIR = sgf_paths.ENHANCED_FALL_CLIP_DIR
os.makedirs(OUT_DIR, exist_ok=True)

CROP_HALF_W = 220
CROP_HALF_H = 260


def find_impact_roi(video_path):
    """Locate the linked droplet's final (lowest, near-impact) position to
    center the crop on -- same track-linking already used by the oracle, just
    reused here to find WHERE the impact happens rather than to measure
    velocity."""
    cap = cv2.VideoCapture(video_path)
    window = find_motion_window(cap)
    if window is None:
        cap.release()
        return None
    start, end, _ = window
    bg = build_static_background(cap)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fall_start = max(0, start - 15)
    fall_end = min(n_total - 1, end + 5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fall_start)
    track = []
    for idx in range(fall_start, fall_end + 1):
        ok, f = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        track.append({"frame": idx, "blobs": detect_droplet_blobs(gray, bg)})
    cap.release()
    linked = link_primary_droplet(track)
    if not linked:
        return None
    last = linked[-1]
    return float(last["cx"]), float(last["cy"]), window


def render_enhanced_clip(video_path, out_path, roi_cx, roi_cy, window,
                          pre_margin=15, post_margin=65, output_fps=15):
    """Crop to a fixed ROI around the impact point and contrast-stretch each
    cropped frame independently (local stats per frame, not one global
    lo/hi over the whole clip -- lighting drifts slightly frame to frame in
    this footage, a per-frame stretch keeps every frame maximally legible)."""
    start, end, _ = window
    cap = cv2.VideoCapture(video_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fall_start = max(0, start - pre_margin)
    fall_end = min(n_total - 1, end + post_margin)

    x0, x1 = int(roi_cx - CROP_HALF_W), int(roi_cx + CROP_HALF_W)
    y0, y1 = int(roi_cy - CROP_HALF_H), int(roi_cy + CROP_HALF_H)
    h_full, w_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    x0, x1 = max(0, x0), min(w_full, x1)
    y0, y1 = max(0, y0), min(h_full, y1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, fall_start)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, output_fps, (x1 - x0, y1 - y0))
    n_written = 0
    for _ in range(fall_end - fall_start + 1):
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lo, hi = gray.min(), gray.max()
        stretched = np.clip((gray.astype(float) - lo) * 255.0 / (hi - lo + 1e-6), 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
        writer.write(out_bgr)
        n_written += 1
    cap.release()
    writer.release()
    return {"n_frames": n_written, "crop": (x0, y0, x1, y1), "extracted": (fall_start, fall_end)}


def main():
    manifest = []
    for name in FAILING_CLIPS:
        src = os.path.join(SRC_DIR, name)
        roi = find_impact_roi(src)
        if roi is None:
            print(f"SKIP {name}: could not locate impact ROI", flush=True)
            continue
        cx, cy, window = roi
        out_path = os.path.join(OUT_DIR, name.replace(".mp4", "_enhanced.mp4"))
        meta = render_enhanced_clip(src, out_path, cx, cy, window)
        meta.update({"clip": name, "out_path": out_path, "roi_center": (cx, cy)})
        manifest.append(meta)
        print(f"{name}: crop={meta['crop']} n_frames={meta['n_frames']} -> {out_path}", flush=True)

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {len(manifest)} enhanced clips to {OUT_DIR}")


if __name__ == "__main__":
    main()
