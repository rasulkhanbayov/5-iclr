"""
X1 (runbook v3): E2 main table, current-generation open-weight model.
Mirrors run_e2_batch.py exactly (same 27-clip manifest, same 3 conditions,
same 3 prompt seeds, same fall-clip/scale-clip files) so results are directly
comparable to the Qwen2.5-VL-7B and frontier-API runs. Only the inference
call differs (run_qwen3vl_inference instead of run_qwen_inference).

Do not run this as a full batch until run_qwen3vl_inference.py's
CHECK_BEFORE_RUNNING smoke test has passed -- see that file's module
docstring.
"""
import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "sgf_hf_cache"))

import sys
import json
import time

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
sys.path.insert(0, os.path.join(_CODE_DIR, "oracle"))

from prepare_clips import extract_fall_clip
from run_qwen3vl_inference import query_quantity_and_calibration
sys.path.insert(0, _CODE_DIR)
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

FALL_CLIP_DIR = sgf_paths.FALL_CLIP_DIR
os.makedirs(FALL_CLIP_DIR, exist_ok=True)

OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "e2", "e2_qwen3vl_results.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

TRIMMED_SCALE_DIR = sgf_paths.TRIMMED_SCALE_DIR

SCALE_CLIPS = {
    "02182026": {
        "base": os.path.join(TRIMMED_SCALE_DIR, "02182026_scalev_base_trim.mp4"),
        0.5: os.path.join(TRIMMED_SCALE_DIR, "02182026_scalev_alpha0.5_trim.mp4"),
        2: os.path.join(TRIMMED_SCALE_DIR, "02182026_scalev_alpha2_trim.mp4"),
    },
}

TRUE_CALIBRATION_MM_PER_TICK = {
    "02182026": 1.5875,  # CORRECTED: 1/16in (1.5875mm/tick), not 1mm -- see droplet_oracle.py
}


def main():
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
    print(f"{total} total items, {done} already done, {total-done} remaining", flush=True)

    for clip_info in clips:
        session = clip_info["session"]
        droplet_path = sgf_paths.raw_video(clip_info["path"])
        clip_name = os.path.basename(droplet_path)
        true_v = clip_info["velocity"]["v_pre_impact_mps"]
        true_cal = TRUE_CALIBRATION_MM_PER_TICK[session]

        fall_clip_path = os.path.join(FALL_CLIP_DIR, clip_name.replace(".mp4", "_fall.mp4"))
        if not os.path.exists(fall_clip_path):
            fall_info = extract_fall_clip(droplet_path, fall_clip_path, output_fps=15)
            if "error" in fall_info:
                print(f"SKIP {clip_name}: {fall_info['error']}", flush=True)
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
                    raw, parsed = query_quantity_and_calibration(fall_clip_path, scale_clip, prompt_seed_idx=seed_idx)
                    entry = {
                        "clip": clip_name, "session": session, "condition": str(condition),
                        "seed_idx": seed_idx, "true_velocity_mps": true_v,
                        "asserted_calibration_mm_per_tick": asserted_cal,
                        "raw_response": raw, **parsed,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                except Exception as e:
                    entry = {"clip": clip_name, "session": session, "condition": str(condition),
                              "seed_idx": seed_idx, "error": f"{type(e).__name__}: {e}"}
                results.append(entry)
                done += 1
                v = entry.get("velocity_mps")
                c = entry.get("calibration_mm_per_tick")
                print(f"[{done}/{total}] {clip_name} cond={condition} seed={seed_idx} -> v={v} cal={c}", flush=True)
                with open(OUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH}, {len(results)} total entries")


if __name__ == "__main__":
    main()
