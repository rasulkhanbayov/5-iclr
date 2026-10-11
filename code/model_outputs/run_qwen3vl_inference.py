"""
X1 (runbook v3): current-generation open-weight model half of the required
"one frontier proprietary AND one current open-weight model" addition.

MODEL CLASS VERIFIED 2026-08-13. Qwen3-VL is on a different transformers code
path than Qwen2.5-VL; AutoModelForImageTextToText is confirmed correct for
Qwen3-VL-30B-A3B-Instruct -- a smoke test on a ~16GB card, too
small for this model) got past model instantiation and failed only on CUDA
OOM during weight loading, not on a class/import error, which is real
evidence this loads correctly given enough GPU memory. Not yet run to
completion on an H200 (80GB-class memory this model actually needs).

Model size: Qwen3-VL-30B-A3B (MoE, ~30B total/~3B active) needs 80GB-class
memory -- H200, not A100 (see the sbatch script in slurm_jobs/).

Mirrors run_qwen_inference.py's call shape exactly (query_quantity_and_calibration
signature, parse_quantity_calibration reuse) so no downstream metric code
needs to change -- only which local model answers the query differs.
"""
import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "sgf_hf_cache"))

import sys

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
sys.path.insert(0, os.path.join(_CODE_DIR, "oracle"))

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from prompt_design import (SYSTEM_PROMPT, USER_PROMPT_SEEDS,
                            INSTRUCTED_CONSTANT_SYSTEM_PROMPT,
                            INSTRUCTED_CONSTANT_USER_PROMPT)
from prompt_design_textcal import (TEXTCAL_SYSTEM_PROMPT_TEMPLATE, TEXTCAL_USER_PROMPT_SEEDS)
from response_parsing import parse_quantity_calibration, parse_velocity_only

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# Checkpoint revision used for the paper (Appendix B.4); override with SGF_QWEN3VL_REVISION.
MODEL_REVISION = os.environ.get("SGF_QWEN3VL_REVISION", "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c")

# CHECK_BEFORE_RUNNING: as of the transformers version this project's
# 00_setup_env.sh installs (now pinned, see requirements.txt), confirm which
# class actually loads Qwen3-VL-30B-A3B-Instruct -- run:
#   python3 -c "from transformers import AutoModelForImageTextToText as M; \
#               m = M.from_pretrained('Qwen/Qwen3-VL-30B-A3B-Instruct', \
#               torch_dtype='auto', device_map='cuda'); print(type(m))"
# and fix the import/class below to match before running a real batch. Using
# AutoModelForImageTextToText here as the most likely current path (Qwen3-VL
# is documented as using this generic head in recent transformers releases),
# NOT verified against real hardware in this session.
from transformers import AutoModelForImageTextToText
sys.path.insert(0, _CODE_DIR)
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

_model = None
_processor = None


def load_model():
    global _model, _processor
    if _model is None:
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype="auto", device_map="cuda")
        _processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    return _model, _processor


def _run_chat(messages, max_new_tokens=256):
    model, processor = load_model()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    # Same qwen_vl_utils/transformers 'fps' list-vs-scalar workaround as
    # run_qwen_inference.py -- see that file's _run_chat for the full
    # explanation. Re-check this is still needed once Qwen3-VL's actual
    # processor version is confirmed; it may not apply if the Qwen3-VL
    # processor validates 'fps' differently.
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
    """Same call shape as run_qwen_inference.py's function of the same name."""
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


def query_textcal_relabeling(droplet_clip_path, true_calibration_mm_per_tick, alpha,
                              prompt_seed_idx=0):
    """T1 positive control, R_alpha half. Mirrors
    run_qwen_inference.py's function of the same name exactly."""
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
    """T1 positive control, S_s half. Mirrors run_qwen_inference.py's
    function of the same name exactly."""
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
    condition, no prompt-seed variation -- mirrors run_qwen_inference.py's
    function of the same name exactly.
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


if __name__ == "__main__":
    # Single-call smoke test -- run this BEFORE any batch job, per
    # CHECK_BEFORE_RUNNING above.
    droplet = os.path.join(sgf_paths.FALL_CLIP_DIR, "water_fall.mp4")
    scale = os.path.join(sgf_paths.TRIMMED_SCALE_DIR, "02182026_scalev_base_trim.mp4")
    raw, parsed = query_quantity_and_calibration(droplet, scale, prompt_seed_idx=0)
    print("RAW RESPONSE:\n", raw)
    print("\nPARSED:", parsed)
