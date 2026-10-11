"""
SGF resampling ("S_s") perturbation renderer.

Per main.tex Definition 1 (Conditions) and sgf_runbook_v2.txt block T1: unlike
relabeling R_alpha, which edits ONLY the scale-reference clip's printed/asserted
label and leaves every pixel of the droplet clip untouched, resampling S_s
rescales the WHOLE FRAME -- droplet pixels AND ruler/comb pixels together -- by
a factor s, while what the scale asserts about its own physical length is held
FIXED. Concretely: (m, c) -> (s*m, c/s). The object's apparent pixel-space
motion scales by s (a real measurer reading the same, unchanged asserted
calibration off a s-times-larger ruler image must therefore report s times the
original quantity), while the asserted calibration itself does not change.

This is a genuinely different operation from perturb_renderer.py's relabeling:
- Relabeling touches ONLY the scale-reference clip (repaints a printed number
  or adds a text banner); the droplet clip is never opened.
- Resampling touches BOTH clips (droplet fall-clip AND scale-reference clip),
  resizing every frame by s, and must NOT touch either clip's printed/asserted
  label -- the whole point is that the label stays what it already said.

Implementation: uniform image resize (cv2.resize) of every frame by factor s,
centered so the frame's original center point stays put (needed because s>1
crops toward the center by construction of a same-size output canvas, and s<1
would otherwise let the frame drift off-center on one side only). Output frame
size is held FIXED at the input's (w, h) so the video's own metadata (encoded
resolution) carries no signal about which condition it is -- only the resize
factor of the CONTENT changes. This matches relabeling's convention of keeping
every other pixel/metadata property constant across conditions so the model
cannot key off anything but the perturbed quantity.

For s>1 (zoom in): the resized frame is LARGER than the output canvas, so it is
center-cropped to fit. This is geometrically necessary (a same-size canvas
cannot hold a strictly larger image) and mirrors what a real camera would show
if physically zoomed in on the same scene -- some field of view is lost at the
edges, but the ruler and droplet (both near-center in this dataset's framing,
confirmed during dataset scoping) remain fully visible in every clip
checked during dataset scoping. For s<1 (zoom out): the resized frame is
SMALLER than the canvas and is centered on a black background, matching how a
real zoomed-out shot would show empty space beyond the original frame's edges
rather than fabricating scene content that was never captured.

Per runbook v2 block T1, use s in {0.5, 2}, mirroring relabeling's own
alpha in {0.5, 2} so the two perturbations are as directly comparable as
possible (same magnitudes, opposite semantics).
"""
from __future__ import annotations
import cv2
import numpy as np


def _resize_frame_centered(frame_bgr, s):
    """
    Resize frame_bgr by factor s, re-centered on a canvas of the ORIGINAL
    (w, h) so output video dimensions never change across conditions.
    s > 1: resized frame is larger than canvas -> center-crop to canvas size.
    s < 1: resized frame is smaller than canvas -> center on black background.
    s == 1 is not a valid resampling condition (see Definition 1: s != 1) and
    is rejected rather than silently treated as a no-op, since a caller
    passing s=1 almost certainly meant to pass an actual condition.
    """
    if s == 1:
        raise ValueError("s must differ from 1 to render a resampling condition")
    h, w = frame_bgr.shape[:2]
    new_w, new_h = max(1, round(w * s)), max(1, round(h * s))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if s > 1:
        # resized is larger than canvas: center-crop
        x0 = (new_w - w) // 2
        y0 = (new_h - h) // 2
        canvas[:, :] = resized[y0:y0 + h, x0:x0 + w]
    else:
        # resized is smaller than canvas: center-paste on black
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def render_resampled_video(in_path, out_path, s):
    """
    Apply the S_s resampling perturbation to every frame of in_path (either
    the droplet fall-clip OR the scale-reference clip -- this function does
    not distinguish between them, since the same uniform resize applies to
    both, per Definition 1's "rescales the whole frame ... object and ruler
    together"). Neither a printed label nor a text banner is touched here;
    callers must ensure the input clip's existing label/banner (from the
    BASE, unperturbed condition) is what gets resized, not a relabeled one --
    mixing S_s with R_alpha on the same clip is a different, unspecified
    condition and is not implemented.

    Returns metadata mirroring perturb_renderer.py's render_* return shape.
    """
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0 or w == 0 or h == 0:
        cap.release()
        raise RuntimeError(f"could not read video properties from {in_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    n_written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(_resize_frame_centered(frame, s))
        n_written += 1
    cap.release()
    writer.release()
    return {"in_path": in_path, "out_path": out_path, "s": s,
            "frame_size": (w, h), "n_frames": n_written}


def render_resampled_pair(droplet_clip_path, scale_clip_path,
                           out_droplet_path, out_scale_path, s):
    """
    Convenience wrapper: render BOTH clips of a resampling condition (droplet
    fall-clip and scale-reference clip) at the same s, since every S_s item
    needs both to build the (droplet, scale) pair the model is queried on --
    same calling convention as query_quantity_and_calibration in
    run_qwen_inference.py, which takes a droplet clip and a scale clip
    separately.
    """
    droplet_meta = render_resampled_video(droplet_clip_path, out_droplet_path, s)
    scale_meta = render_resampled_video(scale_clip_path, out_scale_path, s)
    return {"droplet": droplet_meta, "scale": scale_meta, "s": s}
