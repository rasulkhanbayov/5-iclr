"""
E2 inference pipeline: Qwen2.5-VL-7B-Instruct on droplet + scale-reference clip
pairs. Elicits (i) the droplet's pre-impact velocity and (ii) the model's read
of the scale calibration, per main.tex/sgf_runbook.txt's E2 spec.
"""
import os
os.environ.setdefault("HF_HOME", "/ephemeral/sgf_work/hf_cache")

import sys
import re
import json
import time

sys.path.insert(0, "/ephemeral/sgf_work/model_outputs")
sys.path.insert(0, "/ephemeral/sgf_work/oracle")

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from prompt_design import (SYSTEM_PROMPT, USER_PROMPT_SEEDS, HIDDEN_PRIOR_PROMPT,
                            HIDDEN_PRIOR_WITH_CONTROL_PROMPT)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

_model = None
_processor = None


def load_model():
    global _model, _processor
    if _model is None:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="cuda")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
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


def parse_quantity_calibration(raw_text):
    cal_match = re.search(r"CALIBRATION_MM_PER_TICK:\s*([\d.]+)", raw_text)
    vel_match = re.search(r"VELOCITY_MPS:\s*([\d.]+)", raw_text)
    return {
        "calibration_mm_per_tick": float(cal_match.group(1)) if cal_match else None,
        "velocity_mps": float(vel_match.group(1)) if vel_match else None,
        "parse_failed": cal_match is None or vel_match is None,
    }


def parse_hidden_prior(raw_text):
    # Primary: exact requested format.
    m = re.search(r"RELEASE_HEIGHT_CM:\s*(UNKNOWN|[\d.]+)", raw_text, re.IGNORECASE)
    if m is not None:
        val = m.group(1)
        if val.upper() == "UNKNOWN":
            return {"declined": True, "asserted_value_cm": None, "parse_failed": False}
        return {"declined": False, "asserted_value_cm": float(val), "parse_failed": False}

    # Fallback: model dropped the "RELEASE_HEIGHT_CM:" prefix but still answered
    # in substance (e.g. bare "UNKNOWN", or a bare number). A prompt-adherence
    # slip should not be scored as a parse failure when the substance (decline
    # vs assert) is unambiguous -- HPH cares about that, not exact formatting.
    stripped = raw_text.strip()
    if re.fullmatch(r"UNKNOWN\.?", stripped, re.IGNORECASE):
        return {"declined": True, "asserted_value_cm": None, "parse_failed": False}
    bare_num = re.fullmatch(r"([\d.]+)\s*(cm)?\.?", stripped, re.IGNORECASE)
    if bare_num:
        return {"declined": False, "asserted_value_cm": float(bare_num.group(1)), "parse_failed": False}

    return {"declined": None, "asserted_value_cm": None, "parse_failed": True}


def parse_hidden_prior_with_control(raw_text):
    control_match = re.search(r"CONTROL_IMPACT:\s*(YES|NO|UNKNOWN)", raw_text, re.IGNORECASE)
    height_match = re.search(r"RELEASE_HEIGHT_CM:\s*(UNKNOWN|[\d.]+)", raw_text, re.IGNORECASE)

    control_answer = control_match.group(1).upper() if control_match else None
    # Ground truth: every clip in this dataset shows the droplet impacting the
    # surface (confirmed during Phase 1 oracle-building -- the fall+impact
    # event is exactly what find_motion_window locates). So the correct control
    # answer is always YES.
    control_correct = control_answer == "YES"

    if height_match is None:
        return {
            "control_answer": control_answer, "control_correct": control_correct,
            "declined": None, "asserted_value_cm": None, "parse_failed": True,
        }
    val = height_match.group(1)
    declined = val.upper() == "UNKNOWN"
    return {
        "control_answer": control_answer, "control_correct": control_correct,
        "declined": declined,
        "asserted_value_cm": None if declined else float(val),
        "parse_failed": False,
    }
