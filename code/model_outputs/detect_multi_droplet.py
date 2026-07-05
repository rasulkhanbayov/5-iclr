"""
Multi-droplet clip detector, built to fix E3's 05052026 control-question failures
(see memory project_sgf_e3_results.md). Some clips (e.g. cainhcg 0.08 b.mp4) have
a SECOND droplet already resting on the surface throughout the clip, unrelated to
the falling droplet the oracle/prompt are asking about -- confuses both models and
(potentially) the oracle's own motion-window detection. Flag these before they're
used in an E2/E3/etc. clip selection, rather than discovering the confound after
the fact.

Distinct from _is_static_artifact (which filters out tiny fixed-position noise
like a dust speck): this checks for a second SUBSTANTIAL blob (droplet-sized) that
persists across the extraction window at a position clearly separate from the
main linked-droplet trajectory.
"""
import sys
sys.path.insert(0, "/ephemeral/sgf_work/oracle")
from droplet_oracle import (detect_droplet_blobs, build_static_background,
                             find_motion_window, link_primary_droplet)
import numpy as np
import cv2


def _raw_dark_blobs(gray_frame, dark_thresh_frac=1.0, min_area=200, max_area=15000,
                     edge_margin_frac=0.15):
    """
    Detect dark objects directly on the RAW frame (no background subtraction),
    so an object present for the WHOLE clip (like a resting droplet already
    merged into the background-subtraction model) is still found.

    Guards against false positives from vignette/shadow: max_area caps blob
    size to droplet-scale (vignette corners are tens of thousands of px vs. a
    real droplet's hundreds to low thousands), edge_margin_frac excludes a
    border strip near the frame edges where vignetting is darkest. A
    single-frame edge-contrast check was tried and found unreliable (this
    footage's real droplets are themselves blurry-edged -- gradient ~20 --
    while some vignette noise spikes higher; see
    project_sgf_e3_results.md/multi-droplet detector notes). Real
    discrimination comes from TEMPORAL PERSISTENCE at a stable position
    (_position_persistence_frac below), not single-frame shape.
    """
    h, w = gray_frame.shape
    my, mx = int(h * edge_margin_frac), int(w * edge_margin_frac)
    thresh = gray_frame.mean() - gray_frame.std() * dark_thresh_frac
    mask = (gray_frame < thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask[:my, :] = 0
    mask[-my:, :] = 0
    mask[:, :mx] = 0
    mask[:, -mx:] = 0
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    blobs = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            blobs.append({"cx": float(cx), "cy": float(cy), "area_px": int(area)})
    return blobs


def detect_multi_droplet(video_path, pre_margin=15, post_margin=15,
                          min_area=200, separation_px=60,
                          position_stability_px=15, min_persistence_frac=0.8):
    """
    Returns {"is_multi_droplet": bool, "reason": str, "details": {...}}.

    Method: link the primary (falling) droplet via background-subtraction as
    usual. Separately, collect all raw-frame dark blobs (see _raw_dark_blobs)
    across the window that sit farther than separation_px from the primary
    droplet's position in their frame -- these are "secondary blob"
    candidates. A resting droplet shows up as a TEMPORALLY STABLE cluster of
    such candidates (same position, most/all frames); vignette/shadow noise
    is comparatively unstable frame-to-frame (different spurious positions,
    or the same position but not persistent). Cluster secondary candidates by
    position (within position_stability_px) and flag the clip only if the
    largest cluster persists across >= min_persistence_frac of checked
    frames -- this is the discriminating signal that single-frame area/edge
    filters could not provide (see _raw_dark_blobs docstring).
    """
    cap = cv2.VideoCapture(video_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window = find_motion_window(cap)
    if window is None:
        cap.release()
        return {"is_multi_droplet": None, "reason": "no motion window found", "details": {}}
    start, end, _ = window
    fall_start = max(0, start - pre_margin)
    fall_end = min(n_total - 1, end + post_margin)

    bg = build_static_background(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fall_start)
    track = []
    raw_gray_by_frame = {}
    for idx in range(fall_start, fall_end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_gray_by_frame[idx] = gray
        blobs = detect_droplet_blobs(gray, bg)
        track.append({"frame": idx, "blobs": blobs})
    cap.release()

    linked = link_primary_droplet(track)
    if not linked:
        return {"is_multi_droplet": None, "reason": "no linkable primary droplet", "details": {}}
    primary_by_frame = {p["frame"]: (p["cx"], p["cy"]) for p in linked}

    candidates = []  # (frame_idx, cx, cy)
    n_checked = 0
    for frame_idx, gray in raw_gray_by_frame.items():
        if frame_idx not in primary_by_frame:
            continue
        px, py = primary_by_frame[frame_idx]
        n_checked += 1
        for b in _raw_dark_blobs(gray, min_area=min_area):
            if np.hypot(b["cx"] - px, b["cy"] - py) > separation_px:
                candidates.append((frame_idx, b["cx"], b["cy"]))

    if not candidates or n_checked == 0:
        return {"is_multi_droplet": False, "reason": "no secondary blob candidates found",
                "details": {"n_checked": n_checked}}

    # cluster candidates by position (simple greedy clustering -- few points, fine)
    positions = np.array([(c[1], c[2]) for c in candidates])
    used = np.zeros(len(candidates), dtype=bool)
    best_cluster_size = 0
    best_cluster_center = None
    for i in range(len(candidates)):
        if used[i]:
            continue
        dists = np.hypot(positions[:, 0] - positions[i, 0], positions[:, 1] - positions[i, 1])
        in_cluster = dists <= position_stability_px
        cluster_frames = {candidates[j][0] for j in range(len(candidates)) if in_cluster[j]}
        if len(cluster_frames) > best_cluster_size:
            best_cluster_size = len(cluster_frames)
            best_cluster_center = positions[i].tolist()
        used |= in_cluster

    persistence_frac = best_cluster_size / n_checked
    is_multi = persistence_frac >= min_persistence_frac
    return {
        "is_multi_droplet": is_multi,
        "reason": (f"stable secondary object at ~{best_cluster_center} present in "
                   f"{best_cluster_size}/{n_checked} frames ({persistence_frac:.0%})"),
        "details": {"best_cluster_size": best_cluster_size, "n_checked": n_checked,
                    "persistence_frac": persistence_frac, "cluster_center": best_cluster_center},
    }


if __name__ == "__main__":
    import json
    with open("/ephemeral/sgf_work/model_outputs/e2_selected_clips.json") as f:
        clips = json.load(f)
    results = []
    for c in clips:
        r = detect_multi_droplet(c["path"])
        r["clip"] = c["path"].split("/")[-1]
        r["session"] = c["session"]
        results.append(r)
        print(f"{r['session']:12s} {r['clip']:25s} multi={r['is_multi_droplet']} -- {r['reason']}", flush=True)
    with open("/ephemeral/sgf_work/results/e3/multi_droplet_check.json", "w") as f:
        json.dump(results, f, indent=2)
