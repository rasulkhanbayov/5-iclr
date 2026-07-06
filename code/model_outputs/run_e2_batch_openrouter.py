"""
E2 main table, second model: google/gemini-2.5-flash-lite via OpenRouter.
Mirrors run_e2_batch.py exactly (same clips, same conditions, same prompt
seeds, same fall-clip/scale-clip files) so E2/E4 metric code and comparisons
are apples-to-apples with the Qwen2.5-VL-7B run. Only the inference call
differs (run_openrouter_inference.query_quantity_and_calibration_openrouter
instead of the local Qwen model).

Model choice justified by a pre-batch cost/quality spot check (see memory
project_sgf_openrouter_e2_results.md): flash-lite cost ~$0.0003/call (full
108-call batch estimated well under $1 against a $31 budget) and, on a small
sample, tracked true velocity more closely than the pricier google/gemini-2.5-flash.
"""
import os
os.environ.setdefault("OPENROUTER_API_KEY", "")  # must be set in environment before running

import sys
import json
import time

sys.path.insert(0, "/ephemeral/sgf_work/model_outputs")
from run_openrouter_inference import query_quantity_and_calibration_openrouter

FALL_CLIP_DIR = "/ephemeral/sgf_work/model_outputs/fall_clips"
OUT_PATH = "/ephemeral/sgf_work/results/e2/e2_gemini_flash_lite_results.json"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

MODEL = "google/gemini-2.5-flash-lite"

SCALE_CLIPS = {
    "02182026": {
        "base": "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/02182026_scalev_base_trim.mp4",
        0.5: "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/02182026_scalev_alpha0.5_trim.mp4",
        2: "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/02182026_scalev_alpha2_trim.mp4",
    },
    "05052026": {
        "base": "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/05052026_scale2_base_trim.mp4",
        0.5: "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/05052026_scale2_alpha0.5_trim.mp4",
        2: "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/05052026_scale2_alpha2_trim.mp4",
    },
}

TRUE_CALIBRATION_MM_PER_TICK = {"02182026": 1.0, "05052026": 1.0}


def main():
    assert os.environ.get("OPENROUTER_API_KEY"), "set OPENROUTER_API_KEY before running"

    with open("/ephemeral/sgf_work/model_outputs/e2_selected_clips.json") as f:
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
        droplet_path = clip_info["path"]
        clip_name = os.path.basename(droplet_path)
        true_v = clip_info["velocity"]["v_pre_impact_mps"]
        true_cal = TRUE_CALIBRATION_MM_PER_TICK[session]

        fall_clip_path = os.path.join(FALL_CLIP_DIR, clip_name.replace(".mp4", "_fall.mp4"))
        if not os.path.exists(fall_clip_path):
            print(f"SKIP {clip_name}: no fall clip found", flush=True)
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
                        fall_clip_path, scale_clip, prompt_seed_idx=seed_idx, model=MODEL)
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
                print(f"[{done}/{total}] {clip_name} cond={condition} seed={seed_idx} -> "
                      f"v={v} cal={c} (total cost so far: ${total_cost:.4f})", flush=True)
                with open(OUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH}, {len(results)} total entries, ${total_cost:.4f} total cost")


if __name__ == "__main__":
    main()
