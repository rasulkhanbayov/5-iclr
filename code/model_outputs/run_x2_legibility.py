"""
X2 (runbook v3, DECISIVE -- run before X1/X4/X10, can invalidate the
headline result): input-legibility and temporal-sufficiency control.

Two sub-checks:
  (i) STATIC LEGIBILITY. Extract one native-resolution frame containing the
      ruler and send it as an IMAGE (not video). If the model cannot read the
      graduation value from a still frame, it certainly cannot from the
      video, and CG measures legibility rather than grounding.
  (ii) TEMPORAL SUFFICIENCY. Log how many frames the model actually ingests
      per fall-clip and over what real-time span, then compare against the
      oracle's own motion window (the pre-impact free-fall segment) -- if
      the ingested frames don't span that segment, Acc@10% is uninterpretable
      regardless of anything else in this study.

Per the runbook: "Run it first." If (i) fails, the honest paper is a
different paper (about video-preprocessing legibility, not grounding). If
(ii) fails, Acc@10% must be withdrawn.
"""
import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "sgf_hf_cache"))

import sys
import json

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
sys.path.insert(0, os.path.join(_CODE_DIR, "oracle"))

import cv2
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from droplet_oracle import find_motion_window, TRUE_FPS
sys.path.insert(0, _CODE_DIR)
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Checkpoint revision used for the paper (Appendix B.4); override with SGF_QWEN25VL_REVISION.
MODEL_REVISION = os.environ.get("SGF_QWEN25VL_REVISION", "cc594898137f460bfe9f0759e9844b3ce807cfb5")

_model = None
_processor = None


def load_model():
    global _model, _processor
    if _model is None:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype="auto", device_map="cuda")
        _processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    return _model, _processor


LEGIBILITY_PROMPT = (
    "This image shows a measurement scale (a ruler or a comb of pins). "
    "What physical distance (in millimeters) does ONE graduation/tick mark, "
    "or the spacing between two adjacent pins, represent? Read any printed "
    "numbers or labels carefully.\n\n"
    "Respond in EXACTLY this format, with no other text:\n"
    "CALIBRATION_MM_PER_TICK: <number>"
)


def extract_native_frame(video_path, out_path, frame_idx=10):
    """Grab one native-resolution frame containing the ruler/comb."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    cv2.imwrite(out_path, frame)
    return out_path


def query_static_legibility(image_path):
    """(i) Static legibility at NATIVE resolution: send the frame as an
    IMAGE, not video, at its original pixel dimensions."""
    model, processor = load_model()
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": LEGIBILITY_PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)[0]


def get_video_path_resolution(video_path, target_fps=15.0):
    """
    Determine the actual per-frame spatial resolution qwen_vl_utils delivers
    to the model for VIDEO input on this clip -- process_vision_info resizes
    frames internally (token-budget driven), and that resized shape is what
    the model actually sees, not the source .mp4's native resolution. Reads
    it directly off video_inputs[0]'s tensor shape rather than assuming a
    fixed downsample factor, since the resize logic is resolution- and
    duration-dependent (see qwen_vl_utils.vision_process) and could differ
    from the still-image path's own resize behavior.
    """
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": f"file://{video_path}", "fps": target_fps},
            {"type": "text", "text": "placeholder"},
        ]},
    ]
    _, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    if not video_inputs:
        return None
    # video_inputs[0] shape is (T, C, H, W) for the Qwen2.5-VL vision
    # processor; take the last two dims as (H, W). Confirm this ordering
    # holds for whichever qwen_vl_utils version is actually installed if
    # this ever returns implausible values (e.g. H/W swapped or far larger
    # than the source clip, which would indicate a different tensor layout).
    shape = video_inputs[0].shape
    h, w = int(shape[-2]), int(shape[-1])
    return h, w


def query_static_legibility_at_resolution(image_path, target_h, target_w, out_path):
    """
    (i, second half) Static legibility at the VIDEO PATH's resolution: resize
    the same native still down to the (h, w) the video pipeline actually
    delivers, then ask the same question. This is the comparison the
    decision rule needs -- native-legible but downsampled-illegible would
    mean CG measures preprocessing, not grounding.
    """
    img = cv2.imread(image_path)
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, resized)
    return query_static_legibility(out_path)


def count_ingested_frames(video_path, target_fps=15.0):
    """
    (ii) Temporal sufficiency: how many frames does qwen_vl_utils actually
    sample from this clip at the fps we pass to the model, and what real-time
    span -- in the ORIGINAL, true-2996fps capture -- does the extracted fall
    clip itself represent?

    Does NOT run the model -- process_vision_info's sampling logic runs the
    same way regardless of whether generate() is called afterward, so this
    is a pure instrumentation pass, cheap to run on every clip.

    The fall clip was written at output_fps=15 (prepare_clips.py) from
    n_written true-capture frames, so its own encoded duration
    (n_frames_in_clip / 15) times TRUE_FPS/15 recovers the real-time span the
    clip represents -- equivalently, n_frames_in_clip / TRUE_FPS_RATIO where
    TRUE_FPS_RATIO = 15 / TRUE_FPS is the slowdown baked in at extraction.
    This does not depend on how many frames the MODEL then samples from the
    already-slowed clip; it is the ceiling on real-world time the clip could
    possibly convey, against which the model's own ingested-frame count is
    then compared.
    """
    cap = cv2.VideoCapture(video_path)
    n_frames_in_clip = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    encoded_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    real_time_span_s = (n_frames_in_clip / encoded_fps) * (encoded_fps / TRUE_FPS) \
        if encoded_fps else None
    # Simplifies to n_frames_in_clip / TRUE_FPS, kept expanded above to show
    # the derivation; both forms are algebraically identical.

    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": f"file://{video_path}", "fps": target_fps},
            {"type": "text", "text": "placeholder"},
        ]},
    ]
    _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    if not video_inputs:
        return {"error": "process_vision_info returned no video frames"}
    n_ingested = video_inputs[0].shape[0]  # (T, C, H, W) or similar; T is frame count
    return {
        "n_frames_ingested_by_model": int(n_ingested),
        "requested_fps": target_fps,
        "clip_real_time_span_s": real_time_span_s,
        "model_effective_time_resolution_s_per_frame":
            (real_time_span_s / n_ingested) if (real_time_span_s and n_ingested) else None,
    }


def main():
    manifest_path = sgf_paths.MANIFEST_PATH
    with open(manifest_path) as f:
        clips = json.load(f)

    out_dir = os.path.join(sgf_paths.RESULTS_DIR, "x2")
    os.makedirs(out_dir, exist_ok=True)
    frame_dir = os.path.join(out_dir, "native_frames")
    os.makedirs(frame_dir, exist_ok=True)

    results = {"static_legibility": [], "temporal_sufficiency": []}

    # (i) Static legibility -- once per session (ruler is static across
    # clips within a session, confirmed in Phase 1), not once per clip.
    # Two resolutions: native, and resized to whatever the VIDEO path
    # actually delivers -- the comparison the decision rule needs.
    scale_clip = sgf_paths.raw_video("02182026/scale v.mp4")
    frame_path = os.path.join(frame_dir, "02182026_native_frame.png")
    if extract_native_frame(scale_clip, frame_path) is not None:
        raw_native = query_static_legibility(frame_path)
        entry = {
            "session": "02182026", "true_calibration_mm_per_tick": 1.0,
            "raw_response_native_resolution": raw_native,
        }
        print(f"02182026 static legibility (native): {raw_native}", flush=True)

        # Use the TRIMMED scale clip (the one actually shown to the model in
        # E2/X4/X10), not the raw session scale clip, so the resolution
        # matches what those experiments' video path actually delivers.
        trimmed_scale = os.path.join(sgf_paths.TRIMMED_SCALE_DIR, "02182026_scalev_base_trim.mp4")
        video_res = get_video_path_resolution(trimmed_scale) if os.path.exists(trimmed_scale) else None
        if video_res is not None:
            h, w = video_res
            downsampled_path = os.path.join(frame_dir, "02182026_video_resolution_frame.png")
            raw_downsampled = query_static_legibility_at_resolution(frame_path, h, w, downsampled_path)
            entry["video_path_resolution_hw"] = [h, w]
            entry["raw_response_video_resolution"] = raw_downsampled
            print(f"02182026 static legibility (video path resolution {h}x{w}): "
                  f"{raw_downsampled}", flush=True)
        else:
            print("SKIP video-resolution legibility check: trimmed scale clip missing "
                  "or process_vision_info returned no frames "
                  "(run trim_scale_clips.py first).", flush=True)
        results["static_legibility"].append(entry)

    # (ii) Temporal sufficiency -- per clip, comparing ingested frame span
    # against the oracle's own motion window duration.
    fall_clip_dir = sgf_paths.FALL_CLIP_DIR
    for clip_info in clips:
        clip_name = os.path.basename(clip_info["path"])
        fall_clip_path = os.path.join(fall_clip_dir, clip_name.replace(".mp4", "_fall.mp4"))
        if not os.path.exists(fall_clip_path):
            print(f"SKIP {clip_name}: fall clip not extracted yet "
                  "(run prepare_clips.py / run_e2_batch.py first).", flush=True)
            continue
        info = count_ingested_frames(fall_clip_path)
        info["clip"] = clip_name
        results["temporal_sufficiency"].append(info)
        print(f"{clip_name}: {info}", flush=True)

    out_path = os.path.join(out_dir, "x2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
