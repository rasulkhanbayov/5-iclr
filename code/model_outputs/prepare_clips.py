"""
Extract the fall+impact window from a droplet clip (reusing the oracle's motion
detection) and re-encode it as a slow, watchable video for VLM consumption.

Qwen2.5-VL samples video at ~2fps by default (qwen_vl_utils.vision_process.FPS).
At the true 2996fps capture rate, the fall+impact event is only ~20-40 frames
(~7-13ms) -- naive 2fps sampling of the ORIGINAL clip would show the model
almost nothing. Fix: extract just the motion-window frames and re-encode them
as a new, short video played back at a normal rate (e.g. 15fps), so the model
sees the full motion sequence stretched into human-watchable time. The
TRUE_FPS-to-output-fps ratio must be recorded so velocity units can be
converted back correctly when interpreting model output (the model reports
its own velocity estimate directly in m/s, using the visible calibration, so
no unit back-conversion is actually needed on our end -- this note is a
safeguard for future readers, not a required step in this script).

post_margin end point: originally a fixed +15 frames past the motion window,
which cut some clips (esp. 05052026) off mid-bounce, before the droplet
visibly settles or its impact becomes unambiguous (root cause of E3's
05052026 control-question failures, see memory project_sgf_e3_results.md).
Now uses tune_extraction_window.find_settled_end's per-clip motion-based
settle detection instead of a fixed pad -- see that module's docstring for
the settle criteria and why a single fixed margin doesn't work across clips.
"""
import os
import sys
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "oracle"))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
from droplet_oracle import (find_motion_window, TRUE_FPS, build_static_background,
                             detect_droplet_blobs, link_primary_droplet)
from tune_extraction_window import find_settled_end
import cv2


def extract_fall_clip(droplet_video_path, out_path, output_fps=15, pre_margin=15,
                       post_margin=None, max_post_margin=80):
    """
    Locate the fall+impact motion window and re-encode just that window as a
    slow clip. Returns metadata about the extracted window.

    post_margin=None (default) uses per-clip settle detection
    (tune_extraction_window.find_settled_end) to pick the end frame. Pass an
    int to force the old fixed-margin behavior (kept for callers/tests that
    want it).
    """
    cap = cv2.VideoCapture(droplet_video_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window = find_motion_window(cap)
    if window is None:
        cap.release()
        return {"error": "no motion window found"}
    start, end, _ = window
    fall_start = max(0, start - pre_margin)

    settle_details = None
    if post_margin is not None:
        fall_end = min(n_total - 1, end + post_margin)
    else:
        bg = build_static_background(cap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fall_start)
        probe_track = []
        probe_end = min(n_total - 1, end + 5)
        for idx in range(fall_start, probe_end + 1):
            ok, f = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            probe_track.append({"frame": idx, "blobs": detect_droplet_blobs(gray, bg)})
        linked = link_primary_droplet(probe_track)
        fall_end, settle_details = find_settled_end(
            droplet_video_path, linked, window, max_post_margin=max_post_margin)

    cap.set(cv2.CAP_PROP_POS_FRAMES, fall_start)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, output_fps, (w, h))
    n_written = 0
    for _ in range(fall_end - fall_start + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n_written += 1
    cap.release()
    writer.release()

    real_duration_s = n_written / TRUE_FPS
    playback_duration_s = n_written / output_fps
    result = {
        "source": droplet_video_path,
        "out_path": out_path,
        "motion_window_frames": (start, end),
        "extracted_frames": (fall_start, fall_end),
        "n_frames_written": n_written,
        "output_fps": output_fps,
        "real_duration_s": real_duration_s,
        "playback_duration_s": playback_duration_s,
        "slowdown_factor": playback_duration_s / real_duration_s if real_duration_s else None,
    }
    if settle_details is not None:
        result["settle_criterion_met"] = settle_details.get("criterion_met")
        result["settle_hit_cap"] = settle_details.get("hit_cap")
    return result
