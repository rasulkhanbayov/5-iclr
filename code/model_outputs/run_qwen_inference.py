"""
E2 inference pipeline: Qwen2.5-VL-7B-Instruct on droplet + scale-reference clip
pairs. Elicits (i) the droplet's pre-impact velocity and (ii) the model's read
of the scale calibration, per main.tex/sgf_runbook.txt's E2 spec.
"""
import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "sgf_hf_cache"))

import sys
import json
import time

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
sys.path.insert(0, os.path.join(_CODE_DIR, "oracle"))

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from prompt_design import (SYSTEM_PROMPT, USER_PROMPT_SEEDS, HIDDEN_PRIOR_PROMPT,
                            HIDDEN_PRIOR_WITH_CONTROL_PROMPT,
                            INSTRUCTED_CONSTANT_SYSTEM_PROMPT,
                            INSTRUCTED_CONSTANT_USER_PROMPT)
from prompt_design_temporal import (TEMPORAL_SYSTEM_PROMPT, TEMPORAL_USER_PROMPT_SEEDS,
                                     asserted_fps_for)
from prompt_design_textcal import (TEXTCAL_SYSTEM_PROMPT_TEMPLATE, TEXTCAL_USER_PROMPT_SEEDS)
from response_parsing import (parse_quantity_calibration, parse_hidden_prior,
                               parse_hidden_prior_with_control, parse_velocity_only)

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


def _run_chat(messages, max_new_tokens=256):
    model, processor = load_model()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    # qwen_vl_utils 0.0.14 returns video_kwargs['fps'] as a per-video LIST (by
    # design, for multi-video inputs), but transformers 5.13.0's processor
    # validates 'fps' as a strict scalar (int/float/None) -- a real
    # version-compatibility gap between the two libraries, not our bug.
    # Work around it by dropping the list and letting each video keep its own
    # actual sampled fps (already baked into video_inputs' frame timestamps);
    # the processor does not strictly need 'fps' when metadata is available.
    video_kwargs.pop("fps", None)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt", **video_kwargs)
    inputs = inputs.to("cuda")
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
    return output_text[0]


def query_quantity_and_calibration(droplet_clip_path, scale_clip_path, prompt_seed_idx=0):
    """Elicit velocity + calibration read for one (droplet, scale) clip pair."""
    prompt = USER_PROMPT_SEEDS[prompt_seed_idx]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": f"file://{droplet_clip_path}", "fps": 15.0},
            {"type": "video", "video": f"file://{scale_clip_path}", "fps": 15.0},
            {"type": "text", "text": prompt},
        ]},
    ]
    raw = _run_chat(messages)
    return raw, parse_quantity_calibration(raw)


def query_temporal_relabeling(droplet_clip_path, calibration_mm_per_tick, beta,
                               prompt_seed_idx=0):
    """
    T_beta (temporal relabeling, appendix Section A.6's flagged analogue of
    R_alpha): the droplet clip's actual frames/playback are unchanged --
    only the ASSERTED frame rate in the prompt changes, from the true
    ingestion rate (15.0 fps) to beta times that. Single video (no
    scale-reference clip): the physical calibration is asserted directly in
    text here too, since there is no in-frame frame-rate ruler to show a
    relabeled version of the way R_alpha shows a relabeled scale video.
    """
    prompt_template = TEMPORAL_USER_PROMPT_SEEDS[prompt_seed_idx]
    asserted_fps = asserted_fps_for(beta)
    system_prompt = TEMPORAL_SYSTEM_PROMPT.format(asserted_fps=asserted_fps)
    user_prompt = prompt_template.format(
        asserted_fps=asserted_fps, calibration_mm_per_tick=calibration_mm_per_tick)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "video", "video": f"file://{droplet_clip_path}", "fps": 15.0},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    raw = _run_chat(messages)
    return raw, parse_velocity_only(raw)


def query_textcal_relabeling(droplet_clip_path, true_calibration_mm_per_tick, alpha,
                              prompt_seed_idx=0):
    """
    T1 (text-calibration positive control): R_alpha applied to a
    policy given the true calibration as TEXT rather than asked to read a
    scale-reference video, removing the perceptual step every evaluated
    model has failed at while leaving the pixel-to-physical conversion step
    intact. The droplet clip is the same UNCHANGED footage as every other
    condition; only the asserted calibration in the prompt is relabeled by
    alpha. No scale-reference video is shown.
    """
    prompt_template = TEXTCAL_USER_PROMPT_SEEDS[prompt_seed_idx]
    asserted_cal = true_calibration_mm_per_tick * alpha
    system_prompt = TEXTCAL_SYSTEM_PROMPT_TEMPLATE.format(calibration_mm_per_tick=asserted_cal)
    user_prompt = prompt_template.format(calibration_mm_per_tick=asserted_cal)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "video", "video": f"file://{droplet_clip_path}", "fps": 15.0},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    raw = _run_chat(messages)
    return raw, parse_velocity_only(raw)


def query_textcal_resampling(resampled_droplet_clip_path, true_calibration_mm_per_tick,
                              prompt_seed_idx=0):
    """
    T1's S_s half: the frame is resampled (droplet clip resized by s, per
    resample_renderer.py) while the asserted calibration in the TEXT is held
    fixed at the true value -- mirroring the visual S_s condition's "ruler's
    asserted physical length held fixed" invariant. No scale-reference video
    is shown, consistent with query_textcal_relabeling.
    """
    prompt_template = TEXTCAL_USER_PROMPT_SEEDS[prompt_seed_idx]
    system_prompt = TEXTCAL_SYSTEM_PROMPT_TEMPLATE.format(
        calibration_mm_per_tick=true_calibration_mm_per_tick)
    user_prompt = prompt_template.format(calibration_mm_per_tick=true_calibration_mm_per_tick)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "video", "video": f"file://{resampled_droplet_clip_path}", "fps": 15.0},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    raw = _run_chat(messages)
    return raw, parse_velocity_only(raw)


def query_instructed_constant(droplet_clip_path, scale_clip_path):
    """
    X10 (runbook v3): identical call shape to query_quantity_and_calibration
    (same two videos, same response format/parser) but with
    INSTRUCTED_CONSTANT_SYSTEM_PROMPT in place of SYSTEM_PROMPT, instructing
    the model to report a fixed answer regardless of video content. Single
    condition, no prompt-seed variation (see prompt_design.py's
    INSTRUCTED_CONSTANT_PROMPT_SEEDS docstring for why).
    """
    messages = [
        {"role": "system", "content": INSTRUCTED_CONSTANT_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": f"file://{droplet_clip_path}", "fps": 15.0},
            {"type": "video", "video": f"file://{scale_clip_path}", "fps": 15.0},
            {"type": "text", "text": INSTRUCTED_CONSTANT_USER_PROMPT},
        ]},
    ]
    raw = _run_chat(messages)
    return raw, parse_quantity_calibration(raw)


def query_hidden_prior(droplet_clip_path):
    """
    DEPRECATED for E3 scoring: this prompt alone cannot distinguish a genuine
    video-grounded decline from a content-independent default "UNKNOWN" (see
    memory project_sgf_e3_results.md -- the same "UNKNOWN" was returned with NO
    video attached at all). Kept only for reference; use
    query_hidden_prior_with_control instead.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": f"file://{droplet_clip_path}", "fps": 15.0},
            {"type": "text", "text": HIDDEN_PRIOR_PROMPT},
        ]},
    ]
    raw = _run_chat(messages, max_new_tokens=64)
    return raw, parse_hidden_prior(raw)


def query_hidden_prior_with_control(droplet_clip_path):
    """
    E3 redesign: pairs the hidden-prior question with a control question (does
    the droplet impact the surface? -- true for every clip in this dataset) in
    the SAME call, so a genuine content-independent "always say UNKNOWN" hedge
    is visible in the control answer, not just inferred from context.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": f"file://{droplet_clip_path}", "fps": 15.0},
            {"type": "text", "text": HIDDEN_PRIOR_WITH_CONTROL_PROMPT},
        ]},
    ]
    raw = _run_chat(messages, max_new_tokens=64)
    return raw, parse_hidden_prior_with_control(raw)


# parse_quantity_calibration, parse_hidden_prior, and
# parse_hidden_prior_with_control now live in response_parsing.py (imported
# above) so callers that only need the parser -- run_openrouter_inference.py,
# run_qwen3vl_inference.py -- don't have to import torch to get it.
