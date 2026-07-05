"""
Per-clip extraction window tuning, replacing prepare_clips.py's fixed
pre_margin=15/post_margin=15.

Root cause being fixed (see memory project_sgf_e3_results.md): a fixed
post_margin ends the extracted clip at an arbitrary point relative to each
clip's OWN rebound dynamics. Some droplets settle fast; others (e.g.
05052026/0.08cg.mp4) are still mid-bounce -- flattened one moment, airborne
and elongated the next -- at frame end+15. The fixed cutoff doesn't know or
care which. Confirmed by hand-labeling: 11/12 clips show entirely normal
single-droplet splash/rebound physics with no CV confound, so the fix that
actually matters is picking a better end frame, not detecting bad clips.

Method: instead of a fixed post_margin, extend forward from the motion
window's end frame and measure local frame-to-frame pixel motion in a ROI
around the linked droplet's last known position. Keep extending while
motion stays "active" (above a floor relative to the clip's own peak motion
during the fall); stop once motion has stayed low for min_settle_frames in a
row, which is the actual visual signature of "the droplet has settled" (or
at least stopped changing shape quickly) rather than an arbitrary count.
Cap how far this can extend (max_post_margin) so a droplet that never fully
settles (e.g. keeps micro-jittering) doesn't blow up the clip length.
"""
import sys
sys.path.insert(0, "/ephemeral/sgf_work/oracle")
from droplet_oracle import find_motion_window, TRUE_FPS
import cv2
import numpy as np


def _roi_motion_series(cap, start_frame, end_frame, cx, cy, roi_half=80):
    """Frame-to-frame absdiff sum within a ROI centered on (cx, cy), for
    frames [start_frame, end_frame] inclusive. ROI keeps the measurement
    local to the droplet/impact area so far-field noise (vignette, unrelated
    motion elsewhere in frame) doesn't drown out the signal."""
    h_lo, h_hi = int(cy - roi_half), int(cy + roi_half)
    w_lo, w_hi = int(cx - roi_half), int(cx + roi_half)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    scores = []
    prev = None
    for idx in range(start_frame, end_frame + 1):
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        roi = g[max(0, h_lo):min(h, h_hi), max(0, w_lo):min(w, w_hi)]
        if prev is not None and prev.shape == roi.shape:
            scores.append((idx, float(cv2.absdiff(roi, prev).mean())))
        prev = roi
    return scores


def _roi_noise_floor(cap, cx, cy, n_baseline=15, roi_half=80):
    """
    Measure this clip's OWN pre-fall sensor-noise floor in the same ROI used
    for post-window motion scoring, from the early baseline frames also used
    by build_static_background. Needed because the floor varies a lot
    clip-to-clip (e.g. cainhsds1.mp4 ~2.6-4.0 vs cainhsds2.mp4 ~0.5-1.0,
    presumably lighting/backlight-brightness dependent) -- a threshold that's
    a fixed fraction of THIS clip's peak motion doesn't track that, and can
    sit permanently above a noisy clip's true settled floor (confirmed on
    cainhsds1.mp4: post-impact scores plateau at ~2.6-4, comfortably settled
    by eye, but never dip below a peak-relative threshold of 1.16).
    """
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, n // 4), n_baseline).astype(int)
    h_lo, h_hi = int(cy - roi_half), int(cy + roi_half)
    w_lo, w_hi = int(cx - roi_half), int(cx + roi_half)
    prev = None
    scores = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        roi = g[max(0, h_lo):min(h, h_hi), max(0, w_lo):min(w, w_hi)]
        if prev is not None and prev.shape == roi.shape:
            scores.append(float(cv2.absdiff(roi, prev).mean()))
        prev = roi
    return float(np.median(scores)) if scores else 0.5


def find_settled_end(video_path, linked_track, motion_window, min_settle_frames=8,
                      settle_margin=1.5, plateau_frac=0.15, max_post_margin=80):
    """
    Returns the frame index at which the droplet's local motion has settled,
    starting the search from the motion window's end. Primary criterion:
    min_settle_frames consecutive frames score below settle_margin times
    this clip's OWN pre-fall noise floor (see _roi_noise_floor). Some clips
    (e.g. cainhcg1.mp4) never fully return to that floor -- visual inspection
    confirmed the droplet IS a stable resting dome by ~frame 460, but a
    persistent low-amplitude oscillation (residual meniscus wobble or a
    compression artifact, ~0.5 units in a ~1.1 unit floor) keeps scores just
    above the strict threshold out past +250 frames. Secondary, looser
    criterion: min_settle_frames consecutive frames all below plateau_frac of
    the fall's OWN peak motion (a real, sustained drop from the fall's
    dynamics, even if not all the way down to the static-baseline floor).
    Whichever criterion is met first wins; max_post_margin is the hard cap
    for clips that satisfy neither (e.g. a splash that's still actively
    reconfiguring).
    """
    start, end, _ = motion_window
    if not linked_track:
        return end + 15, {"reason": "no linked track; fell back to fixed +15"}

    last = linked_track[-1]
    cx, cy = last["cx"], last["cy"]
    cap = cv2.VideoCapture(video_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    noise_floor = _roi_noise_floor(cap, cx, cy)
    fall_scores = _roi_motion_series(cap, max(0, start - 5), end, cx, cy)
    peak = max((s for _, s in fall_scores), default=0.0)
    probe_end = min(n_total - 1, end + max_post_margin)
    post_scores = _roi_motion_series(cap, end + 1, probe_end, cx, cy)
    cap.release()

    strict_thresh = noise_floor * settle_margin
    plateau_thresh = peak * plateau_frac if peak > 0 else strict_thresh

    strict_run = plateau_run = 0
    chosen = probe_end  # default: neither criterion met within budget, use the cap
    met = None
    for frame_idx, score in post_scores:
        strict_run = strict_run + 1 if score < strict_thresh else 0
        plateau_run = plateau_run + 1 if score < plateau_thresh else 0
        if strict_run >= min_settle_frames:
            chosen, met = frame_idx, "strict"
            break
        if plateau_run >= min_settle_frames:
            chosen, met = frame_idx, "plateau"
            break

    details = {
        "noise_floor": noise_floor, "peak_fall_motion": peak,
        "strict_threshold": strict_thresh, "plateau_threshold": plateau_thresh,
        "post_scores": post_scores, "criterion_met": met,
        "hit_cap": met is None,
    }
    return chosen, details


if __name__ == "__main__":
    import json
    with open("/ephemeral/sgf_work/model_outputs/e2_selected_clips.json") as f:
        clips = json.load(f)

    results = []
    for c in clips:
        path = c["path"]
        cap = cv2.VideoCapture(path)
        window = find_motion_window(cap)
        cap.release()
        if window is None:
            print(f"SKIP {path}: no motion window")
            continue
        start, end, _ = window

        # Reuse the already-computed linked-track last position if present in
        # e2_selected_clips.json; otherwise recompute it fresh.
        if "impact_frame" in c.get("velocity", {}) and c["velocity"]["impact_frame"]:
            # approximate cx,cy unavailable from this JSON; recompute via oracle
            pass
        from droplet_oracle import build_static_background, detect_droplet_blobs, link_primary_droplet
        cap = cv2.VideoCapture(path)
        bg = build_static_background(cap)
        fall_start = max(0, start - 15)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fall_start)
        track = []
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fall_end_probe = min(n_total - 1, end + 5)
        for idx in range(fall_start, fall_end_probe + 1):
            ok, f = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            track.append({"frame": idx, "blobs": detect_droplet_blobs(gray, bg)})
        cap.release()
        linked = link_primary_droplet(track)

        chosen_end, details = find_settled_end(path, linked, window)
        old_end = min(n_total - 1, end + 15)
        result = {
            "clip": path.split("/")[-1], "session": c["session"],
            "motion_window": [start, end],
            "old_fixed_end": old_end,
            "new_settled_end": chosen_end,
            "extra_frames_vs_fixed": chosen_end - old_end,
            "hit_cap": details.get("hit_cap", False),
        }
        results.append(result)
        print(f"{c['session']:12s} {result['clip']:22s} window_end={end} "
              f"old_end={old_end} new_end={chosen_end} "
              f"(+{result['extra_frames_vs_fixed']}) hit_cap={result['hit_cap']}", flush=True)

    with open("/ephemeral/sgf_work/results/e3/settled_window_tuning.json", "w") as f:
        json.dump(results, f, indent=2)
