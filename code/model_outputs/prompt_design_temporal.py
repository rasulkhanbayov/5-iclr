"""
Temporal relabeling condition T_beta -- the temporal analogue of the
spatial relabeling condition R_alpha, per the appendix's Section A.6
("Temporal coupling") limitation and its \\FLAG asking for this to be run.

R_alpha relabels the SPATIAL calibration by showing a scale-reference video
whose visual graduation asserts a different mm-per-pixel value, while the
droplet clip's pixel content is untouched. There is no in-frame "frame rate
ruler" in this footage to relabel the same way, so the asserted playback
rate is instead communicated directly in the prompt -- the analogue of a
capture system reporting its own frame rate as metadata, which a downstream
analyst has no independent way to verify from the pixels alone (exactly the
property that makes R_alpha's spatial version work: the model has no way to
tell the relabeled scale-reference video is wrong without outside
information).

The droplet clip itself is delivered UNCHANGED (same actual frames, same
actual playback), matching R_alpha's own invariant: only the ASSERTED value
changes, never the underlying pixel content. Per Definition 1's spatial
case, a faithful measurer's velocity must scale by the asserted change: if
the asserted frame rate is beta times the true one, the asserted time
between frames is 1/beta times the true time between frames, so a measurer
computing distance/time must report a velocity scaled by beta^{-1} relative
to its own true-fps reading (SLOWER asserted fps => MORE real time elapsed
per frame => LARGER inferred velocity for the same pixel displacement, hence
the beta^{-1} rather than beta scaling -- the inverse relationship the
appendix's flag specifies).

Base condition: 15.0 fps (the actual ingestion rate qwen_vl_utils samples
at for this pipeline, confirmed in run_x2_legibility.py -- NOT the raw
2996 fps capture rate, which the model never sees directly). This is the
rate a faithful, frame-rate-aware model would need to be told in order to
convert pixel displacement into a physical velocity at all, since nothing
in the video itself discloses playback speed.
"""

TRUE_INGESTION_FPS = 15.0

TEMPORAL_SYSTEM_PROMPT = (
    "You are a careful physics measurement assistant. You will be shown one "
    "short video clip: a high-speed video of a liquid droplet falling and "
    "impacting a surface, delivered to you at {asserted_fps:.1f} frames per "
    "second. You are also given a physical scale reference separately: each "
    "pixel of horizontal or vertical displacement in this video corresponds "
    "to a known real-world distance, which will be provided in the question."
)

TEMPORAL_USER_PROMPT_TEMPLATE = """This video was recorded and is delivered to you at exactly {asserted_fps:.1f} \
frames per second. The scale calibration for this video is {calibration_mm_per_tick} \
millimeters per pixel of displacement.

Using ONLY what is visible in this video, the stated frame rate, and the stated \
calibration, answer the question below. Do not use any outside knowledge about \
typical droplet velocities -- measure directly from the pixel displacement across \
frames, the stated frame rate, and the stated calibration.

VELOCITY: What is the droplet's velocity (in meters per second) just before it \
impacts the surface? Estimate this from the droplet's pixel displacement across \
frames, converting frame counts to elapsed time using the stated frame rate of \
{asserted_fps:.1f} fps, and pixel distance to physical distance using the stated \
calibration of {calibration_mm_per_tick} mm/pixel.

Respond in EXACTLY this format, with no other text:
VELOCITY_MPS: <number>
"""

# 3 prompt seeds (paraphrases), matching E2's USER_PROMPT_SEEDS convention,
# for bootstrap CIs over phrasing rather than only over items.
TEMPORAL_USER_PROMPT_SEEDS = [
    TEMPORAL_USER_PROMPT_TEMPLATE,
    TEMPORAL_USER_PROMPT_TEMPLATE.replace(
        "Using ONLY what is visible in this video, the stated frame rate, and the "
        "stated calibration, answer the question below.",
        "Base your answer strictly on the video provided plus the stated frame "
        "rate and calibration; do not rely on prior assumptions."
    ),
    TEMPORAL_USER_PROMPT_TEMPLATE.replace(
        "measure directly from the pixel displacement across frames, the stated "
        "frame rate, and the stated calibration.",
        "measure carefully using only the pixel displacement, the given frame "
        "rate, and the given calibration."
    ),
]

# Conditions: base (true fps) and two relabeled conditions, mirroring
# R_alpha's alpha in {0.5, 2} exactly, but on the temporal axis (beta).
TEMPORAL_CONDITIONS = {
    "base": 1.0,
    "0.5": 0.5,
    "2": 2.0,
}


def asserted_fps_for(beta):
    """The prompt's asserted frame rate under temporal relabeling by beta."""
    return TRUE_INGESTION_FPS * float(beta)


def ideal_velocity_ratio(beta, d=1):
    """
    Ideal response of a faithful measurer under T_beta, per Definition 1's
    spatial pattern applied to the temporal axis: r = beta^{-d}, the inverse
    relationship noted in the appendix flag (a SLOWER asserted fps implies
    MORE elapsed time per frame, hence a LARGER inferred velocity for the
    same pixel displacement).
    """
    return float(beta) ** (-d)
