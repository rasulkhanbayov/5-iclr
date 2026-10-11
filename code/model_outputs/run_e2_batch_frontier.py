"""
X1 (runbook v3): E2 main table, current-generation FRONTIER PROPRIETARY model
via OpenRouter -- the other required half of "one frontier proprietary AND
one current open-weight model" (run_e2_batch_qwen3vl.py is the open-weight
half). API-based: needs no GPU, no Slurm job, just network access and
OPENROUTER_API_KEY -- run directly on any node including the shared servers.

MODEL is set to google/gemini-3-pro-preview (chosen 2026-08-11; see the
constant's own comment below for the rationale). Mirrors
run_e2_batch_openrouter.py (the original run's structure) exactly, rescoped
to the 27-clip 02182026-only manifest.
"""
import os
os.environ.setdefault("OPENROUTER_API_KEY", "")  # must be set in environment before running

import sys
import json
import time

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
from run_openrouter_inference import query_quantity_and_calibration_openrouter
sys.path.insert(0, _CODE_DIR)
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

# Chosen 2026-08-11: Gemini 3.1 Pro Preview, Google's current flagship
# frontier model. "google/gemini-3-pro-preview" (no version number) 404s on
# OpenRouter as of this date -- confirmed via GET /api/v1/models that the
# real current slug is "google/gemini-3.1-pro-preview", which explicitly
# lists "video" in its input_modalities and prices at $2/M input + $12/M
# output tokens -- at 243 calls with ~1-2MB of video per call this is a
# small, bounded cost, not a budget risk. Same OpenRouter image_url-shaped
# data-URI workaround as the original Gemini-2.5-Flash-Lite run, confirmed
# working for the Gemini family specifically.
MODEL = "google/gemini-3.1-pro-preview"
# gemini-3.1-pro-preview is a reasoning model: a smoke test showed it can
# spend nearly its entire completion budget on internal reasoning tokens
# before writing the answer (243 of 252 tokens), truncating the response
# mid-word at the old default of 256. 2048 leaves ample room for reasoning
# plus the short CALIBRATION_MM_PER_TICK/VELOCITY_MPS answer.
MAX_TOKENS = 2048

FALL_CLIP_DIR = sgf_paths.FALL_CLIP_DIR
OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "e2", "e2_frontier_results.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

TRIMMED_SCALE_DIR = sgf_paths.TRIMMED_SCALE_DIR
SCALE_CLIPS = {
    "02182026": {
        "base": os.path.join(TRIMMED_SCALE_DIR, "02182026_scalev_base_trim.mp4"),
        0.5: os.path.join(TRIMMED_SCALE_DIR, "02182026_scalev_alpha0.5_trim.mp4"),
        2: os.path.join(TRIMMED_SCALE_DIR, "02182026_scalev_alpha2_trim.mp4"),
    },
}
TRUE_CALIBRATION_MM_PER_TICK = {"02182026": 1.5875}  # CORRECTED: 1/16in (1.5875mm/tick), not 1mm -- see droplet_oracle.py


def main():
    assert os.environ.get("OPENROUTER_API_KEY"), "set OPENROUTER_API_KEY before running"

    manifest_path = sgf_paths.MANIFEST_PATH
    with open(manifest_path) as f:
        clips = json.load(f)

    results = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
    done_keys = {(r["clip"], r["condition"], r["seed_idx"]) for r in results}

    total = len(clips) * 3 * 3
    done = len(results)
    total_cost = sum(r.get("usage", {}).get("cost", 0) for r in results)
    print(f"{total} total items, {done} already done, {total-done} remaining, "
          f"${total_cost:.4f} spent so far", flush=True)

    for clip_info in clips:
        session = clip_info["session"]
        droplet_path = sgf_paths.raw_video(clip_info["path"])
        clip_name = os.path.basename(droplet_path)
        true_v = clip_info["velocity"]["v_pre_impact_mps"]
        true_cal = TRUE_CALIBRATION_MM_PER_TICK[session]

        fall_clip_path = os.path.join(FALL_CLIP_DIR, clip_name.replace(".mp4", "_fall.mp4"))
        if not os.path.exists(fall_clip_path):
            print(f"SKIP {clip_name}: fall clip missing ({fall_clip_path}). "
                  "Run run_e2_batch.py (or prepare_clips.py directly) first.", flush=True)
            continue

        for condition in ["base", 0.5, 2]:
            scale_clip = SCALE_CLIPS[session][condition]
            for seed_idx in range(3):
                key = (clip_name, str(condition), seed_idx)
                if key in done_keys:
                    continue
                t0 = time.time()
                alpha = condition if condition != "base" else 1.0
                asserted_cal = true_cal * alpha
                try:
                    raw, parsed, usage = query_quantity_and_calibration_openrouter(
                        fall_clip_path, scale_clip, prompt_seed_idx=seed_idx, model=MODEL,
                        max_tokens=MAX_TOKENS)
                    entry = {
                        "clip": clip_name, "session": session, "condition": str(condition),
                        "seed_idx": seed_idx, "true_velocity_mps": true_v,
                        "asserted_calibration_mm_per_tick": asserted_cal,
                        "raw_response": raw, **parsed, "usage": usage,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    total_cost += usage.get("cost", 0)
                except Exception as e:
                    entry = {"clip": clip_name, "session": session, "condition": str(condition),
                              "seed_idx": seed_idx, "error": f"{type(e).__name__}: {e}"}
                results.append(entry)
                done += 1
                v = entry.get("velocity_mps")
                c = entry.get("calibration_mm_per_tick")
                print(f"[{done}/{total}] {clip_name} cond={condition} seed={seed_idx} "
                      f"-> v={v} cal={c} (${total_cost:.4f} total)", flush=True)
                with open(OUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH}, {len(results)} total entries, ${total_cost:.4f} total spend")


if __name__ == "__main__":
    main()
