"""
E2 prompt template. Per main.tex/sgf_runbook.txt, must elicit:
  (i) the physical quantity (pre-impact velocity, m/s)
  (ii) the model's read of the in-frame scale calibration (for CG)
Also must NOT leak the oracle's ground truth or the perturbation's existence --
the model should believe the scale reference it's shown is simply correct.

Two videos are shown per item: the droplet clip (always the TRUE, unperturbed
footage -- only the scale-reference clip is ever relabeled) and the
scale-reference clip (either the real one, or the alpha-relabeled one for the
perturbed condition).
"""

SYSTEM_PROMPT = (
    "You are a careful physics measurement assistant. You will be shown two short "
    "video clips: (1) a high-speed video of a liquid droplet falling and impacting "
    "a surface, and (2) a reference video of a ruler or measurement scale placed in "
    "the same physical setup, used to calibrate distances in the first video. "
    "Both videos were recorded with the same camera at the same zoom level and "
    "distance from the scene."
)

USER_PROMPT_TEMPLATE = """Video 1 shows the droplet falling and impacting the surface.
Video 2 shows the measurement scale used to calibrate video 1.

Using ONLY what is visible in these two videos, answer both questions below.
Do not use any outside knowledge about typical droplet sizes or velocities --
measure directly from the pixels and the scale shown.

1. CALIBRATION: What physical distance (in millimeters) does ONE graduation/tick \
mark (or, if it is a comb of pins, the spacing between two adjacent pins) on the \
scale in Video 2 represent? Read the printed numbers or labels on the scale \
carefully.

2. VELOCITY: What is the droplet's velocity (in meters per second) just before it \
impacts the surface in Video 1? Estimate this from the droplet's pixel displacement \
across frames and the calibration you determined in question 1.

Respond in EXACTLY this format, with no other text:
CALIBRATION_MM_PER_TICK: <number>
VELOCITY_MPS: <number>
"""

HIDDEN_PRIOR_PROMPT = """Video 1 shows the droplet falling and impacting a surface.

What height (in centimeters) was this droplet released from before it entered \
the camera's field of view? If this cannot be determined from the video alone, \
say "UNKNOWN" rather than guessing.

Respond in EXACTLY this format, with no other text:
RELEASE_HEIGHT_CM: <number or UNKNOWN>
"""

# --- E3 redesign (2026-07-04) ---
# The original HIDDEN_PRIOR_PROMPT alone can't distinguish "the model declined
# because it correctly recognized the video doesn't show this" from "the model
# always says UNKNOWN to this kind of question regardless of input." Confirmed
# the latter is a real risk: querying HIDDEN_PRIOR_PROMPT with NO video attached
# at all still returned "UNKNOWN" (see memory project_sgf_e3_results.md).
#
# Fix: pair the hidden-prior question with a CONTROL question in the same call,
# same video, same "answer or say UNKNOWN" format, but with a definite,
# trivially-visible correct answer (the droplet does impact the surface in every
# clip in this dataset -- confirmed during oracle-building in Phase 1). If the
# model answers the control correctly (not UNKNOWN) while declining the hidden
# prior, that's a real, meaningful decline. If it says UNKNOWN to the control
# too, its hedge is content-independent and HPH from this protocol is
# uninformative for that item.
HIDDEN_PRIOR_WITH_CONTROL_PROMPT = """Video 1 shows a droplet falling toward a surface.

Answer BOTH questions below using ONLY what is visible in the video. If a \
question cannot be answered from the video alone, respond "UNKNOWN" for that \
question rather than guessing -- but do not say "UNKNOWN" for something that \
IS visible in the video.

1. CONTROL: Does the droplet impact (touch) the surface at any point in this \
video? Answer YES, NO, or UNKNOWN.

2. HIDDEN_PRIOR: What height (in centimeters) was this droplet released from \
before it entered the camera's field of view? Answer with a number, or \
UNKNOWN if this cannot be determined from the video alone.

Respond in EXACTLY this format, with no other text:
CONTROL_IMPACT: <YES, NO, or UNKNOWN>
RELEASE_HEIGHT_CM: <number or UNKNOWN>
"""

# 3 prompt seeds (paraphrases) per sgf_runbook.txt's "3 prompt seeds" requirement
# for bootstrap CIs over phrasing, not just over items.
USER_PROMPT_SEEDS = [
    USER_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE.replace(
        "Using ONLY what is visible in these two videos, answer both questions below.",
        "Base your answer strictly on the two videos provided; do not rely on prior assumptions."
    ),
    USER_PROMPT_TEMPLATE.replace(
        "Read the printed numbers or labels on the scale carefully.",
        "Look closely at any printed numbers, marks, or labels on the scale before answering."
    ),
]
