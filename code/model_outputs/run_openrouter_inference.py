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

sys.path.insert(0, "/ephemeral/sgf_work/model_outputs")
from prompt_design import SYSTEM_PROMPT, USER_PROMPT_SEEDS
from run_qwen_inference import parse_quantity_calibration

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ["OPENROUTER_API_KEY"]


def _video_data_url(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"


def query_quantity_and_calibration_openrouter(droplet_clip_path, scale_clip_path,
                                               prompt_seed_idx=0, model="google/gemini-2.5-flash-lite"):
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
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": 256, "temperature": 0.0},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return raw, parse_quantity_calibration(raw), usage


if __name__ == "__main__":
    droplet = "/ephemeral/sgf_work/model_outputs/fall_clips/water_fall.mp4"
    scale = "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/02182026_scalev_base_trim.mp4"
    raw, parsed, usage = query_quantity_and_calibration_openrouter(droplet, scale, prompt_seed_idx=0)
    print("RAW RESPONSE:\n", raw)
    print("\nPARSED:", parsed)
    print("\nUSAGE:", json.dumps(usage, indent=2))
