"""
E2/E4 multi-model extension: query a video-capable model via OpenRouter
(Gemini 2.5 Flash / Flash-Lite) instead of the local Qwen2.5-VL pipeline.
Reuses the exact same prompts (prompt_design.py) and parsers
(run_qwen_inference.py's parse_quantity_calibration) so E2/E4 metric code
needs zero changes -- only the inference call differs.

OpenRouter's chat/completions endpoint accepts video as a data: URI in an
"image_url"-shaped content block for Gemini models (the video/inline_data
content type Google's own API uses isn't exposed the same way through
OpenRouter's unified schema; base64 data-URI is the documented approach for
video files under 20MB via OpenRouter -- see conversation for confirmation
this actually works, tested with a live call before committing to a batch).
"""
import os
import sys
import base64
import json
import requests

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
from prompt_design import (SYSTEM_PROMPT, USER_PROMPT_SEEDS,
                            INSTRUCTED_CONSTANT_SYSTEM_PROMPT,
                            INSTRUCTED_CONSTANT_USER_PROMPT)
from prompt_design_textcal import (TEXTCAL_SYSTEM_PROMPT_TEMPLATE, TEXTCAL_USER_PROMPT_SEEDS)
from response_parsing import parse_quantity_calibration, parse_velocity_only
sys.path.insert(0, _CODE_DIR)
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _video_data_url(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"


def query_quantity_and_calibration_openrouter(droplet_clip_path, scale_clip_path,
                                               prompt_seed_idx=0, model="google/gemini-2.5-flash-lite",
                                               max_tokens=256):
    """
    max_tokens default (256) was fine for gemini-2.5-flash-lite (a
    non-reasoning model). Reasoning models (e.g. gemini-3.1-pro-preview)
    spend a large chunk of the completion budget on internal reasoning
    tokens BEFORE writing the answer -- confirmed 2026-08-11: a smoke test
    against gemini-3.1-pro-preview returned reasoning_tokens=243 out of
    completion_tokens=252, truncating the actual answer mid-word. Pass a
    much larger max_tokens (e.g. 2048) for reasoning models.
    """
    prompt = USER_PROMPT_SEEDS[prompt_seed_idx]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Video 1 (droplet clip):"},
            {"type": "image_url", "image_url": {"url": _video_data_url(droplet_clip_path)}},
            {"type": "text", "text": "Video 2 (scale-reference clip):"},
            {"type": "image_url", "image_url": {"url": _video_data_url(scale_clip_path)}},
            {"type": "text", "text": prompt},
        ]},
    ]
    api_key = os.environ["OPENROUTER_API_KEY"]  # evaluated per-call, not at import time
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return raw, parse_quantity_calibration(raw), usage


def query_textcal_relabeling_openrouter(droplet_clip_path, true_calibration_mm_per_tick,
                                         alpha, prompt_seed_idx=0,
                                         model="google/gemini-3.1-pro-preview",
                                         max_tokens=2048):
    """T1 positive control, R_alpha half, via OpenRouter. No scale-reference
    video -- the asserted (relabeled) calibration is given entirely in text."""
    prompt_template = TEXTCAL_USER_PROMPT_SEEDS[prompt_seed_idx]
    asserted_cal = true_calibration_mm_per_tick * alpha
    system_prompt = TEXTCAL_SYSTEM_PROMPT_TEMPLATE.format(calibration_mm_per_tick=asserted_cal)
    user_prompt = prompt_template.format(calibration_mm_per_tick=asserted_cal)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Video 1 (droplet clip):"},
            {"type": "image_url", "image_url": {"url": _video_data_url(droplet_clip_path)}},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    api_key = os.environ["OPENROUTER_API_KEY"]
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return raw, parse_velocity_only(raw), usage


def query_textcal_resampling_openrouter(resampled_droplet_clip_path, true_calibration_mm_per_tick,
                                         prompt_seed_idx=0,
                                         model="google/gemini-3.1-pro-preview",
                                         max_tokens=2048):
    """T1 positive control, S_s half, via OpenRouter."""
    prompt_template = TEXTCAL_USER_PROMPT_SEEDS[prompt_seed_idx]
    system_prompt = TEXTCAL_SYSTEM_PROMPT_TEMPLATE.format(
        calibration_mm_per_tick=true_calibration_mm_per_tick)
    user_prompt = prompt_template.format(calibration_mm_per_tick=true_calibration_mm_per_tick)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Video 1 (droplet clip):"},
            {"type": "image_url", "image_url": {"url": _video_data_url(resampled_droplet_clip_path)}},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    api_key = os.environ["OPENROUTER_API_KEY"]
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return raw, parse_velocity_only(raw), usage


def query_instructed_constant_openrouter(droplet_clip_path, scale_clip_path,
                                          model="google/gemini-2.5-flash-lite",
                                          max_tokens=256):
    """
    X10 via OpenRouter: same INSTRUCTED_CONSTANT_SYSTEM_PROMPT/USER_PROMPT as
    run_qwen_inference.py's query_instructed_constant, just routed through
    OpenRouter's video-as-data-URI path instead of the local model.
    """
    messages = [
        {"role": "system", "content": INSTRUCTED_CONSTANT_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Video 1 (droplet clip):"},
            {"type": "image_url", "image_url": {"url": _video_data_url(droplet_clip_path)}},
            {"type": "text", "text": "Video 2 (scale-reference clip):"},
            {"type": "image_url", "image_url": {"url": _video_data_url(scale_clip_path)}},
            {"type": "text", "text": INSTRUCTED_CONSTANT_USER_PROMPT},
        ]},
    ]
    api_key = os.environ["OPENROUTER_API_KEY"]
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return raw, parse_quantity_calibration(raw), usage


if __name__ == "__main__":
    droplet = os.path.join(sgf_paths.FALL_CLIP_DIR, "water_fall.mp4")
    scale = os.path.join(sgf_paths.TRIMMED_SCALE_DIR, "02182026_scalev_base_trim.mp4")
    raw, parsed, usage = query_quantity_and_calibration_openrouter(droplet, scale, prompt_seed_idx=0)
    print("RAW RESPONSE:\n", raw)
    print("\nPARSED:", parsed)
    print("\nUSAGE:", json.dumps(usage, indent=2))
