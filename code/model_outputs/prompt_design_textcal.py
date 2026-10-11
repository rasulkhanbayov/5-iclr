"""
T1: text-calibration positive control, per the main
paper's "The spatial axes lack a positive control" flag.

Rationale: every model evaluated so far falls in the blind class on the
spatial axes, so the design has been shown to detect the ABSENCE of spatial
grounding and never its PRESENCE. This removes the one step every evaluated
model failed at -- reading the ruler -- by supplying the true calibration
directly as text, while leaving the conversion step (pixel displacement,
frame rate -> physical velocity) intact. If a model can convert at all, this
policy should behave as grounded under R_alpha and invariant under S_s;
Theorem 1 of the main paper predicts (r_R, r_S) = (alpha, 1) here, against
the (1,1) measured for the uninstructed policy.

Mirrors E2's design as closely as possible so results are directly
comparable: same 27 clips, same 3 conditions per relation, same 3 prompt
seeds. Only the input differs -- no scale-reference video is shown; the
calibration is asserted in the system prompt instead, and relabeled there for
R_alpha. For S_s, the frame is resampled (droplet only) with the same
asserted calibration text held fixed, mirroring the visual S_s condition's
"ruler's asserted physical length held fixed" invariant.
"""

TEXTCAL_SYSTEM_PROMPT_TEMPLATE = (
    "You are a careful physics measurement assistant. You will be shown one "
    "short video clip: a high-speed video of a liquid droplet falling and "
    "impacting a surface. You are given the exact calibration for this video: "
    "each pixel of displacement corresponds to {calibration_mm_per_tick:.4f} "
    "millimeters. Use this calibration directly; do not estimate it from the "
    "video."
)

TEXTCAL_USER_PROMPT_TEMPLATE = """Video 1 shows the droplet falling and impacting the surface.

You are told the exact calibration for this video: {calibration_mm_per_tick:.4f} \
millimeters per pixel of displacement.

Using ONLY what is visible in this video and the stated calibration, answer the \
question below. Do not use any outside knowledge about typical droplet velocities \
-- measure directly from the pixel displacement across frames and convert using \
the stated calibration.

VELOCITY: What is the droplet's velocity (in meters per second) just before it \
impacts the surface? Estimate this from the droplet's pixel displacement across \
frames and the stated calibration of {calibration_mm_per_tick:.4f} mm/pixel.

Respond in EXACTLY this format, with no other text:
VELOCITY_MPS: <number>
"""

TEXTCAL_USER_PROMPT_SEEDS = [
    TEXTCAL_USER_PROMPT_TEMPLATE,
    TEXTCAL_USER_PROMPT_TEMPLATE.replace(
        "Using ONLY what is visible in this video and the stated calibration, "
        "answer the question below.",
        "Base your answer strictly on the video provided and the stated "
        "calibration; do not rely on prior assumptions."
    ),
    TEXTCAL_USER_PROMPT_TEMPLATE.replace(
        "measure directly from the pixel displacement across frames and "
        "convert using the stated calibration.",
        "measure carefully using only the pixel displacement and the given "
        "calibration."
    ),
]

# R_alpha conditions: base (true calibration) and two relabeled magnitudes,
# matching E2's alpha in {0.5, 2} exactly.
TEXTCAL_RALPHA_CONDITIONS = {
    "base": 1.0,
    "0.5": 0.5,
    "2": 2.0,
}

# S_s conditions: two resample magnitudes, matching X4's s in {0.5, 2}.
TEXTCAL_SS_MAGNITUDES = [0.5, 2.0]
