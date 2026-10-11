"""
SGF CV oracle: droplet centroid/area tracking + velocity, built from scratch.

Calibration and true frame rate are session-level facts confirmed by the human
authors, not derived from container metadata (which is wrong: reports 60fps,
true capture rate is 2996fps) or guessed from pixels alone. See memory
project_sgf_dataset_scope.md / project_sgf_frame_rate.md for how these numbers
were obtained.
"""
from __future__ import annotations
import cv2
import numpy as np

TRUE_FPS = 2996.0

SESSION_CALIBRATION_PX_PER_MM = {
    # CORRECTED 2026-08-22: 02182026's ruler is graduated in 1/16" (1.5875mm)
    # per tick, not 1mm -- confirmed by the human authors (session predates
    # 05052026 and the two "later" sessions, all of which use a genuine 1mm
    # ruler; 02182026 is the one "earlier" inch-graduated exception). The
    # original 104.52 was measured as px PER TICK assuming 1 tick = 1mm; the
    # corrected value divides by the true 1.5875mm tick pitch, exactly the
    # same correction already applied below for new_videos/05122026's pin
    # target. Free-fall cross-check (Appendix B.2/"Oracle error and scope"):
    # under the old 104.52, oracle velocities disagreed with v=sqrt(2gh) by a
    # mean 40% across 22 high-confidence 02182026 clips; under the corrected
    # value the mean disagreement drops to 6%.
    "02182026": 104.52 / 1.5875,
    # 05052026 is confirmed absent from drops_videos/ as of 2026-08-11 (the old
    # the old GPU-server that held it no longer exists). Kept here as a
    # historical record, not live data -- run_batch.py's SESSIONS dict does not
    # include it, so this key is never looked up. This session used a genuine
    # 1mm ruler (one of the "later" sessions), so 66.19 is NOT corrected.
    "05052026": 66.19,
    "new_videos/05122026": 68.19 / 1.5875,  # 68.19 px per 1/16" pin pitch
}


def detect_droplet_blobs(gray_frame, bg_gray, min_area=15, max_area=20000):
    """
    Background-subtraction blob detector. Droplet is darker than the bright
    backlit background. Returns list of (cx, cy, area_px, bbox).
    """
    diff = cv2.absdiff(gray_frame, bg_gray)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    blobs = []
    for i in range(1, n_labels):  # skip label 0 = background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            blobs.append({"cx": float(cx), "cy": float(cy), "area_px": int(area),
                          "bbox": (int(x), int(y), int(w), int(h))})
    return blobs


def build_static_background(cap, n_samples=15):
    """Median background from frames BEFORE any motion (pre-fall baseline)."""
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, n // 4), n_samples).astype(int)  # early frames only
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _motion_scores_over_indices(cap, indices):
    """
    Seeks via cap.set(POS_FRAMES) for a handful of far-apart indices (the fine
    pass; small window, few seeks, fine). NOT used for the coarse pass anymore
    -- see _motion_scores_sequential.
    """
    scores = {}
    prev = None
    prev_idx = None
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            gap = idx - prev_idx
            scores[idx] = cv2.absdiff(g, prev).sum() / max(gap, 1)
        prev, prev_idx = g, idx
    return scores


def _motion_scores_sequential(cap, n_frames, step):
    """
    Coarse motion scan via ONE sequential decode pass (read every frame, keep
    every `step`-th), not repeated cap.set() seeks. On long clips (some
    sessions have 5,000-12,000+ frames) hundreds of POS_FRAMES seeks each pay
    a keyframe-search cost and the coarse scan alone took minutes; a single
    sequential pass reading every frame is dramatically faster since decoders
    are optimized for forward sequential access, not random seeks.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    scores = {}
    prev = None
    prev_idx = None
    idx = 0
    while idx < n_frames:
        ok, f = cap.read()
        if not ok:
            break
        if idx % step == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                gap = idx - prev_idx
                scores[idx] = cv2.absdiff(g, prev).sum() / max(gap, 1)
            prev, prev_idx = g, idx
        idx += 1
    return scores


def find_motion_window(cap, coarse_step=5, motion_threshold_frac=0.15):
    """
    Two-pass motion localization: a coarse scan (sequential decode, one frame
    read per loop iteration, scored every coarse_step frames) finds the rough
    impact region cheaply, then a dense seek-based scan only within a small
    window around the coarse peak pins down exact start/end. The coarse pass
    reads every frame sequentially rather than seeking (cap.set) to each
    sampled index -- on long clips (some sessions have 5,000-12,000+ frames)
    hundreds of POS_FRAMES seeks each pay a keyframe-search cost, which made
    the coarse scan itself the bottleneck (minutes per clip, worse than the
    original whole-clip-seek version this replaced).
    """
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    coarse_scores = _motion_scores_sequential(cap, n, coarse_step)
    if not coarse_scores:
        return None
    peak_idx = max(coarse_scores, key=coarse_scores.get)

    # dense scan in a window around the coarse peak
    lo = max(0, peak_idx - coarse_step * 4)
    hi = min(n - 1, peak_idx + coarse_step * 4)
    fine_idxs = list(range(lo, hi + 1))
    fine_scores = _motion_scores_over_indices(cap, fine_idxs)
    if not fine_scores:
        return None

    thresh = max(fine_scores.values()) * motion_threshold_frac
    active = sorted(i for i, s in fine_scores.items() if s > thresh)
    if not active:
        return None
    return active[0], active[-1], fine_scores


def track_droplet(video_path, session_key, pre_impact_margin=20, post_impact_margin=5):
    """
    Full pipeline: locate motion window, build pre-fall background, detect the
    falling droplet blob per frame, and compute pre-impact velocity from the
    linear fit of centroid position over the fall frames.
    Returns dict with per-frame track and computed pre-impact velocity (mm/s).
    """
    px_per_mm = SESSION_CALIBRATION_PX_PER_MM[session_key]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 0:
        cap.release()
        return {"error": "unreadable or corrupted video file (0 frames / failed to open)"}

    window = find_motion_window(cap)
    if window is None:
        cap.release()
        return {"error": "no motion detected"}
    start, end, scores = window

    bg = build_static_background(cap)

    track = []
    fall_start = max(0, start - pre_impact_margin)
    fall_end = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, end + post_impact_margin)
    for idx in range(fall_start, fall_end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        blobs = detect_droplet_blobs(gray, bg)
        track.append({"frame": idx, "blobs": blobs})
    cap.release()

    return {
        "session": session_key,
        "px_per_mm": px_per_mm,
        "fps": TRUE_FPS,
        "motion_window": (start, end),
        "track": track,
    }


def _is_static_artifact(track, frame_idx_in_track, cx, cy, radius_px=5, min_span=15):
    """
    Some clips have a persistent non-droplet blob (dust speck / lens blemish)
    that survives background subtraction and sits at a fixed position for the
    whole clip. Naively seeding link_primary_droplet from "first frame with
    exactly one blob" locks onto this artifact instead of the real droplet
    when the artifact appears earlier in the padded track (found by hand on
    new_videos/05122026/0.028tx1.mp4: a blob at a fixed (954, 303.7) with
    near-constant ~165px area persisted for 30+ frames while the real falling
    droplet, at a different position, only appeared starting several frames
    later). Check whether a same-position blob (within radius_px) recurs
    across at least min_span frames before/after this one in the track.
    """
    count = 0
    for t in track[max(0, frame_idx_in_track - min_span):frame_idx_in_track + min_span]:
        for b in t["blobs"]:
            if np.hypot(b["cx"] - cx, b["cy"] - cy) < radius_px:
                count += 1
                break
    return count >= min_span  # present in most of a +/-min_span window => static, not falling


def link_primary_droplet(track, max_jump_px=60, static_check_radius=5, static_check_span=15):
    """
    Some clips contain multiple simultaneous droplets/satellites (e.g. multi-drop
    dispensing events), so per-frame blob lists can have >1 entry and picking
    "the" droplet isn't automatic. Greedily link frame-to-frame by nearest
    centroid distance, seeded from the first non-static blob found (checking
    ALL blobs in a frame, not just frames with exactly one blob -- a static
    artifact like a dust speck can coexist with the real falling droplet in
    the same frame the whole time it's in view, so restricting the seed
    search to single-blob frames can skip every frame the real droplet is
    actually visible in; found by hand on new_videos/05122026/0.028tx1.mp4,
    where the real droplet only ever appears in 2-3-blob frames alongside a
    persistent static blob). Returns a list of (frame, cx, cy, area_px) for
    the linked track; frames where linking fails (no blob within max_jump_px)
    end the track early rather than jumping to an unrelated blob.
    """
    seed_i, seed_blob = None, None
    for i, t in enumerate(track):
        for b in t["blobs"]:
            if not _is_static_artifact(track, i, b["cx"], b["cy"], static_check_radius, static_check_span):
                seed_i, seed_blob = i, b
                break
        if seed_i is not None:
            break
    if seed_i is None:
        return []

    linked = [{"frame": track[seed_i]["frame"], **seed_blob}]
    for t in track[seed_i + 1:]:
        last = linked[-1]
        if not t["blobs"]:
            continue
        dists = [np.hypot(b["cx"] - last["cx"], b["cy"] - last["cy"]) for b in t["blobs"]]
        best = int(np.argmin(dists))
        if dists[best] > max_jump_px:
            break
        linked.append({"frame": t["frame"], **t["blobs"][best]})
    return linked


def compute_pre_impact_velocity(linked_track, px_per_mm, fps=TRUE_FPS, n_window=5,
                                 decel_frac=0.85):
    """
    Velocity from the free-fall plateau just before contact, not averaged over
    the whole fall and not from the final points at the peak. The droplet
    accelerates under gravity so a whole-window linear fit underestimates the
    impact velocity, but the last few points before cy peaks are already
    inside the contact/deceleration cushion (the droplet is touching the
    surface and slowing down) and underestimate it just as badly the other
    way. Verified by hand on 0.028tx.mp4: per-frame velocity (finite diff of
    cy) is a flat ~24 px/frame plateau from well before impact, then visibly
    drops starting 2-3 frames before the cy peak — that drop is deceleration
    on contact, not signal. See memory project_sgf_dataset_scope.md.

    Method: find the cy peak (impact_i), walk backward from it while
    frame-to-frame velocity stays within decel_frac of the plateau's own
    median (i.e. skip the trailing points where velocity has visibly dropped),
    then fit a line over an n_window-point segment ending at that cutoff.
    """
    if len(linked_track) < 3:
        return None
    cy = np.array([p["cy"] for p in linked_track])
    frames = np.array([p["frame"] for p in linked_track])
    diffs = np.diff(cy)

    impact_i = len(cy)
    for i, d in enumerate(diffs):
        if d < 0:
            impact_i = i + 1
            break

    if impact_i < n_window + 2:
        # Track is too short to see the deceleration/impact signature at all --
        # this happens when detection loses the droplet mid-fall (occlusion by
        # a static artifact, frame edge, etc.) rather than the droplet actually
        # reaching the surface. Found on new_videos/05122026: droplet enters
        # frame and is trackable for only 3-4 frames before link breaks, well
        # before any deceleration would show. Can't apply the plateau method
        # (nothing to walk back from), but a plain average finite-difference
        # velocity over the visible span is still real signal, just noisier
        # and unverified against a deceleration cutoff. Flag it as such rather
        # than silently returning None (which looks identical to "no data").
        if impact_i < 3:
            return None
        seg_frames, seg_cy = frames[:impact_i], cy[:impact_i]
        slope, _ = np.polyfit(seg_frames, seg_cy, 1)
        v_mps = (slope * fps / px_per_mm) / 1000
        return {"v_pre_impact_mps": float(v_mps), "impact_frame": None,
                "plateau_end_frame": None, "n_points_used": len(seg_frames),
                "low_confidence": True,
                "reason": "track too short to detect impact/deceleration; plain average velocity over visible fall"}

    # per-frame velocities up to (not including) the peak
    pre_impact_diffs = diffs[:impact_i - 1]
    if len(pre_impact_diffs) < n_window:
        return None
    plateau_v = np.median(pre_impact_diffs)  # robust to the trailing deceleration tail

    cutoff = impact_i - 1
    for i in range(len(pre_impact_diffs) - 1, -1, -1):
        if pre_impact_diffs[i] < decel_frac * plateau_v:
            cutoff = i  # exclude this decelerated step and everything after it
        else:
            break

    seg_hi = cutoff + 1  # index into cy/frames, exclusive-safe
    seg_lo = max(0, seg_hi - n_window)
    seg_frames = frames[seg_lo:seg_hi]
    seg_cy = cy[seg_lo:seg_hi]
    if len(seg_frames) < 2:
        return None
    slope, _ = np.polyfit(seg_frames, seg_cy, 1)
    v_mps = (slope * fps / px_per_mm) / 1000
    return {"v_pre_impact_mps": float(v_mps), "impact_frame": int(frames[impact_i - 1]),
            "plateau_end_frame": int(frames[seg_hi - 1]), "n_points_used": len(seg_frames)}
