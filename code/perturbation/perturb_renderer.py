"""
SGF calibration-relabeling perturbation renderer.

Per sgf_runbook.txt and main.tex Section 3.3: the perturbation relabels what the
in-frame scale ASSERTS about its own physical length, by overlaying new numeral
labels on the scale-reference clip. Tick/graduation pixel positions are left
completely untouched -- only the printed numbers change. The droplet clip itself
is NEVER touched; only the scale-reference clip is relabeled. This is NOT a frame
resample (that would be inert -- see runbook: uniformly rescaling object+ruler
pixels together with the ruler's asserted length fixed lets a faithful measurer
read a proportionally finer calibration and recover the same quantity, so the
ideal response would be 1 and the test would probe nothing).

For alpha != 1: relabel N -> N/alpha at the same tick, so a viewer computing
(graduation count) x (labeled unit) recovers a length that is alpha times the
original per graduation -- i.e. the asserted mm-per-tick becomes alpha*c.

Confirmed with the user (2026-07-04): deliver the perturbation as a modified
scale-reference video with the printed numbers digitally replaced (inpainted out,
then a new number drawn in) while tick marks stay pixel-identical.

new_videos/05122026 uses a pin-comb target (evenly-spaced vertical pins, no
printed numerals to inpaint/relabel). Confirmed with the user (2026-07-04):
for this session, assert the new pitch via an added text overlay (e.g. "pin
spacing: 1/8 in" for alpha=2 on a native 1/16in comb) rather than editing pins
that were never labeled with text in the first place. Pin pixel positions are
never touched, same principle as the ruler case.
"""
from __future__ import annotations
import cv2
import numpy as np


# Per-session config: where the printed number lives, what it originally reads,
# and how to draw a replacement. Found by hand via frame inspection (see
# CLAUDE.md's Dataset Scope section and the frame_previews/ captures from
# Phase 1). ROI coordinates are (x0, y0, x1, y1) in native 1280x512 pixels.
SESSION_LABEL_CONFIG = {
    "05052026": {
        "scale_clip": "/ephemeral/videos/05052026/scale2.mp4",
        "roi": (150, 0, 450, 300),
        "original_label": "30",
        "text_position": (170, 220),
        "font_scale": 2.6,
        "thickness": 6,
        "rotate90": False,
        # E5 rotate_orientation path: position chosen directly against the
        # rotated (512w x 1280h) frame, see e5_check_05052026_rot_v3*.png --
        # not a transform of text_position, see render_relabeled_scale_video.
        "rotated_text_position": (60, 550),
    },
    "02182026": {
        "scale_clip": "/ephemeral/videos/02182026/scale v.mp4",
        # ruler is physically rotated 90 degrees in-frame; numeral "12" reads
        # top-to-bottom, spanning roughly 330px tall (measured by hand from the
        # reference frame) -- much bigger than 05052026's "30", needs a larger
        # font scale or the rotated glyph is illegibly small.
        "roi": (400, 130, 750, 480),
        "original_label": "12",
        "text_position": (570, 300),
        "font_scale": 11.0,
        "thickness": 20,
        "rotate90": True,
        # E5 rotate_orientation path: rotated frame is 512w x 1280h; smaller
        # font than the unrotated case's 11.0 (that scale was tuned for a
        # rotate90-composited canvas at a different effective size) -- tuned
        # directly against the rotated output, see e5_check_02182026_rot_v3*.png.
        "rotated_text_position": (60, 700),
    },
}
# new_videos/05122026 uses a pin-comb target (no printed number to relabel this
# way) -- see relabel_comb_pitch() below for that session's mechanism instead.


def _dark_mask(gray_frame, roi, std_frac=0.3):
    x0, y0, x1, y1 = roi
    region = gray_frame[y0:y1, x0:x1].astype(float)
    thresh = region.mean() - region.std() * std_frac
    mask = (region < thresh).astype(np.uint8) * 255
    full_mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = mask
    return cv2.dilate(full_mask, np.ones((7, 7), np.uint8), iterations=3)


def relabel_ruler_frame(frame_bgr, roi, new_text, text_position, font_scale,
                         thickness, rotate90=False):
    """
    Erase the original printed number (inpaint) and draw a replacement at
    roughly the same position, matching the scene's low-contrast blurred look
    (this is underexposed high-speed footage, not crisp studio photography --
    a crisp overlay reads as obviously synthetic against the real texture).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = _dark_mask(gray, roi)
    inpainted = cv2.inpaint(frame_bgr, mask, 15, cv2.INPAINT_NS)

    bg_sample = inpainted[text_position[1] - 40:text_position[1] + 10,
                           text_position[0] - 20:text_position[0] + 60]
    bg_mean = bg_sample.mean() if bg_sample.size else 60.0
    text_gray = int(np.clip(bg_mean * 0.55, 10, 200))

    if not rotate90:
        out = inpainted.copy()
        cv2.putText(out, new_text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (text_gray,) * 3, thickness, cv2.LINE_AA)
    else:
        (tw, th), _ = cv2.getTextSize(new_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = 10
        canvas = np.full((th + pad * 2, tw + pad * 2, 3), int(bg_mean), dtype=np.uint8)
        cv2.putText(canvas, new_text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (text_gray,) * 3, thickness, cv2.LINE_AA)
        rotated = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
        rh, rw = rotated.shape[:2]
        x0, y0 = text_position[0] - rw // 2, text_position[1] - rh // 2
        x1, y1 = x0 + rw, y0 + rh
        fh, fw = inpainted.shape[:2]
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(fw, x1), min(fh, y1)
        out = inpainted.copy()
        if cx1 > cx0 and cy1 > cy0:
            rx0, ry0 = cx0 - x0, cy0 - y0
            patch = rotated[ry0:ry0 + (cy1 - cy0), rx0:rx0 + (cx1 - cx0)]
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            keep = gray_patch < (bg_mean - 5)  # keep only the drawn strokes, not the fill background
            region_out = out[cy0:cy1, cx0:cx1]
            for c in range(3):
                ch = region_out[:, :, c]
                ch[keep] = patch[:, :, c][keep]
            out[cy0:cy1, cx0:cx1] = region_out

    return cv2.GaussianBlur(out, (3, 3), 0)


def _build_relabel_patch(sample_frame_bgr, roi, new_text, text_position, font_scale,
                          thickness, rotate90):
    """
    Precompute the inpainted-background + new-label patch ONCE from a sample
    frame (the ruler is static across the whole clip -- verified by hand in
    Phase 1: identical tick positions across frames sampled through the full
    duration of every scale clip). cv2.inpaint costs ~1.3s/call at this ROI
    size; calling it per-frame made a 919-frame render take 20+ minutes for
    inpainting alone. Since the ROI content never changes, inpaint once and
    reuse the same patch for every frame -- reduces cost from O(n_frames) to
    O(1) expensive calls plus a cheap per-frame copy.
    Returns (roi, patch_bgr) where patch_bgr is the same size as the ROI.
    """
    relabeled_full = relabel_ruler_frame(
        sample_frame_bgr, roi, new_text, text_position, font_scale, thickness, rotate90)
    x0, y0, x1, y1 = roi
    return relabeled_full[y0:y1, x0:x1].copy()


def _build_erase_patch(sample_frame_bgr, roi):
    """
    Like _build_relabel_patch but WITHOUT drawing any label -- just the
    inpainted (digit-erased) background. Needed for the rotate_orientation
    path in render_relabeled_scale_video: erasing must happen in the
    ORIGINAL frame's coordinates (that's where the real printed digit
    actually is), but drawing the new label must happen AFTER the frame is
    rotated (so the label's position/orientation is chosen directly in the
    final frame's own coordinate system, not fought through a rotation
    transform). Trying to pre-rotate the label to compensate before frame
    rotation got the digit's rotation right but not its position -- see
    conversation, e5_check_*_rot_v2.png showed the label present but
    clipped/mispositioned after rotation. Draw-after-rotate avoids the
    problem entirely.
    """
    gray = cv2.cvtColor(sample_frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = _dark_mask(gray, roi)
    inpainted = cv2.inpaint(sample_frame_bgr, mask, 15, cv2.INPAINT_NS)
    x0, y0, x1, y1 = roi
    return inpainted[y0:y1, x0:x1].copy()


def render_relabeled_scale_video(session_key, alpha, out_path, rotate_orientation=False):
    """
    Render a full relabeled scale-reference video for the given session and
    alpha. Original calibration c = original_label / (pixels to that
    graduation). Pixel positions are untouched, so to make the new asserted
    calibration equal to alpha*c, the new label must satisfy
    new_label / pixels = alpha * (original_label / pixels), i.e.
    new_label = alpha * original_label (main.tex Proposition 1: c -> alpha*c
    with pixel measurement unchanged means the recovered/asserted length
    scales by exactly alpha). E.g. alpha=0.5 on a ruler labeled "30" ->
    relabel to "15" (a faithful reader now thinks each graduation is HALF as
    long, so it converts the same pixel distance to half the physical length).

    rotate_orientation=True (E5 robustness axis): additionally rotate the
    WHOLE frame 90 degrees counter-clockwise before writing, so the ruler's
    graduations run along the opposite frame axis from the original footage
    (both real sessions physically photograph a vertical ruler rotated
    in-frame -- there is no naturally-occurring "other orientation" footage
    to compare against, so this synthesizes one from the same real ruler).
    ROTATE_90_COUNTERCLOCKWISE (not CLOCKWISE) verified by hand to keep
    02182026's "12" digit upright after rotation (rotated_ccw_check_
    02182026.png).

    Erase-then-rotate-then-draw, NOT rotate-a-pre-labeled-patch: an earlier
    attempt drew the new label (with a pre-flipped rotate90) into the
    ORIGINAL frame's coordinates and rotated the whole composited frame
    afterward. That got the digit's own rotation right but not its position
    -- text_position/roi are original-frame coordinates, and where they land
    after a whole-frame rotation isn't a place you'd choose to put a label
    (confirmed broken visually: e5_check_*_rot_v2.png showed "7.5" clipped at
    the bottom edge and 02182026's "48" missing its second digit entirely).
    Fix: erase the original digit (inpaint only, no draw) in original
    coordinates -- that's genuinely where the real digit is and must be
    erased there -- then rotate the erased frame, THEN draw the new label
    fresh in the rotated frame's own coordinate system using
    ROTATED_TEXT_POSITION below, sidestepping the position-transform problem
    entirely.
    """
    cfg = SESSION_LABEL_CONFIG[session_key]
    orig_val = float(cfg["original_label"])
    new_val = orig_val * alpha
    new_text = str(int(new_val)) if new_val == int(new_val) else f"{new_val:.1f}"

    cap = cv2.VideoCapture(cfg["scale_clip"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, sample_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"could not read first frame of {cfg['scale_clip']}")
    roi = cfg["roi"]
    x0, y0, x1, y1 = roi

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w, out_h = (h, w) if rotate_orientation else (w, h)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    if not rotate_orientation:
        patch = _build_relabel_patch(sample_frame, roi, new_text, cfg["text_position"],
                                      cfg["font_scale"], cfg["thickness"], cfg["rotate90"])
        for _ in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            frame[y0:y1, x0:x1] = patch
            writer.write(frame)
    else:
        erase_patch = _build_erase_patch(sample_frame, roi)
        rot_pos = cfg["rotated_text_position"]
        # After ROTATE_90_COUNTERCLOCKWISE, a point (x, y) in a frame of size
        # (w, h) maps to (y, w - 1 - x) in the (h, w)-sized rotated frame --
        # used only to sanity-check rot_pos by hand, not computed at runtime;
        # rot_pos is tuned directly against the rotated output instead, since
        # the label needs to sit on legible background, not just anywhere
        # geometrically valid.
        for _ in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            frame[y0:y1, x0:x1] = erase_patch
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bg_sample = frame[max(0, rot_pos[1] - 30):rot_pos[1] + 10,
                               max(0, rot_pos[0] - 10):rot_pos[0] + 60]
            bg_mean = bg_sample.mean() if bg_sample.size else 60.0
            text_gray = int(np.clip(bg_mean * 0.55, 10, 200))
            cv2.putText(frame, new_text, rot_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        cfg["font_scale"], (text_gray,) * 3, cfg["thickness"], cv2.LINE_AA)
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            writer.write(frame)
    cap.release()
    writer.release()
    return {"session": session_key, "alpha": alpha, "original_label": cfg["original_label"],
            "new_label": new_text, "out_path": out_path, "n_frames": n,
            "rotated_orientation": rotate_orientation}


# Comb-style target config: no printed numerals exist to inpaint/relabel, so
# the perturbation is delivered as an added text banner (confirmed with the
# user 2026-07-04). native_pitch_inches is the TRUE physical pin spacing,
# confirmed by the user during Phase 1 oracle calibration (1/16in, matching
# main.tex's "vertical 1/16-inch graduation" claim). banner_roi is an empty
# region of the frame (above the pin tops) that never overlaps any pin.
COMB_LABEL_CONFIG = {
    "new_videos/05122026": {
        "scale_clip": "/ephemeral/videos/new_videos/05122026/scale2.mp4",
        "native_pitch_inches": 1.0 / 16,
        "banner_roi": (0, 0, 1280, 45),
        "text_position": (20, 32),
        "font_scale": 1.0,
        "thickness": 2,
    },
}


def _format_inches(value):
    """Render a fraction-of-an-inch value as '1/16 in' style text, falling back to decimal."""
    from fractions import Fraction
    frac = Fraction(value).limit_denominator(64)
    if frac.numerator == 1:
        return f"1/{frac.denominator} in"
    return f"{frac.numerator}/{frac.denominator} in" if frac.denominator != 1 else f'{frac.numerator} in'


def render_relabeled_comb_video(session_key, alpha, out_path):
    """
    Render a comb-target scale video with an added text banner asserting a
    pin-to-pin pitch of alpha * native_pitch (pin pixel positions untouched --
    same principle as the ruler relabeling, just delivered as new text rather
    than edited existing text, since there's nothing printed on the pins to
    inpaint).
    """
    cfg = COMB_LABEL_CONFIG[session_key]
    new_pitch = cfg["native_pitch_inches"] * alpha
    pitch_text = f"pin spacing: {_format_inches(new_pitch)}"

    cap = cv2.VideoCapture(cfg["scale_clip"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    x0, y0, x1, y1 = cfg["banner_roi"]
    banner = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    cv2.putText(banner, pitch_text, (cfg["text_position"][0] - x0, cfg["text_position"][1] - y0),
                cv2.FONT_HERSHEY_SIMPLEX, cfg["font_scale"], (255, 255, 255), cfg["thickness"], cv2.LINE_AA)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        region = frame[y0:y1, x0:x1]
        mask = banner.sum(axis=2) > 0
        region[mask] = banner[mask]
        frame[y0:y1, x0:x1] = region
        writer.write(frame)
    cap.release()
    writer.release()
    return {"session": session_key, "alpha": alpha,
            "native_pitch_inches": cfg["native_pitch_inches"], "new_pitch_text": pitch_text,
            "out_path": out_path, "n_frames": n}
