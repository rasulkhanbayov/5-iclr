"""
Improvement plan B1: broaden E5's robustness sweep from 2 representative
clips to the full 12-clip E2 set. Zero new API cost (local Qwen2.5-VL only).
Written as a SEPARATE script (not editing run_e5_batch.py in place) so the
original 2-clip run and its results (e5_qwen_results.json, already reported
in main.tex) are preserved unchanged for comparison -- this writes to its own
output file.

Condition grid per clip: 4 alphas x 2 orientations x 3 seeds = 24 scaled
calls, + 2 orientations x 3 seeds = 6 base calls = 30 calls/clip x 12 clips
= 360 total. All scale-clip variants (alpha x orientation) already exist per
SESSION (not per clip) from the original E5 render, so no new rendering is
needed -- only new droplet-clip x scale-clip pairings.
"""
import os
os.environ.setdefault("HF_HOME", "/ephemeral/sgf_work/hf_cache")

import sys
import json
import time

sys.path.insert(0, "/ephemeral/sgf_work/model_outputs")
sys.path.insert(0, "/ephemeral/sgf_work/oracle")

from run_qwen_inference import query_quantity_and_calibration

FALL_CLIP_DIR = "/ephemeral/sgf_work/model_outputs/fall_clips"
SCALE_DIR = "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips"
OUT_PATH = "/ephemeral/sgf_work/results/e5/e5_qwen_results_full12.json"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

ALPHAS = [0.25, 0.5, 2, 4]
ORIENTATIONS = ["orig", "rot"]

TRUE_CALIBRATION_MM_PER_TICK = {"02182026": 1.0, "05052026": 1.0}


def load_all_12_clips():
    with open("/ephemeral/sgf_work/model_outputs/e2_selected_clips.json") as f:
        clips = json.load(f)
    out = []
    for c in clips:
        out.append({"path": c["path"], "session": c["session"],
                     "true_velocity_mps": c["velocity"]["v_pre_impact_mps"]})
    return out


def scale_clip_path(session, alpha, orientation):
    if alpha == 1:
        return os.path.join(SCALE_DIR, "e5", f"{session}_base_rot_trim.mp4") if orientation == "rot" \
            else os.path.join(SCALE_DIR, f"{'02182026_scalev' if session=='02182026' else '05052026_scale2'}_base_trim.mp4")
    suffix = "rot" if orientation == "rot" else "orig"
    return os.path.join(SCALE_DIR, "e5", f"{session}_alpha{alpha}_{suffix}.mp4")


def main():
    selected_clips = load_all_12_clips()

    results = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
    done_keys = {(r["clip"], r["condition"], r["orientation"], r["seed_idx"]) for r in results}

    total = len(selected_clips) * (len(ALPHAS) + 1) * len(ORIENTATIONS) * 3
    done = len(results)
    print(f"{total} total items, {done} already done, {total-done} remaining", flush=True)

    for clip_info in selected_clips:
        session = clip_info["session"]
        droplet_path = clip_info["path"]
        clip_name = os.path.basename(droplet_path)
        true_v = clip_info["true_velocity_mps"]
        true_cal = TRUE_CALIBRATION_MM_PER_TICK[session]

        fall_clip_path = os.path.join(FALL_CLIP_DIR, clip_name.replace(".mp4", "_fall.mp4"))
        if not os.path.exists(fall_clip_path):
            print(f"SKIP {clip_name}: no fall clip found at {fall_clip_path}", flush=True)
            continue

        conditions = ["base"] + ALPHAS
        for condition in conditions:
            for orientation in ORIENTATIONS:
                alpha = 1.0 if condition == "base" else condition
                scale_clip = scale_clip_path(session, alpha, orientation)
                if not os.path.exists(scale_clip):
                    print(f"SKIP {clip_name} cond={condition} orient={orientation}: "
                          f"missing scale clip {scale_clip}", flush=True)
                    continue
                for seed_idx in range(3):
                    key = (clip_name, str(condition), orientation, seed_idx)
                    if key in done_keys:
                        continue
                    t0 = time.time()
                    asserted_cal = true_cal * alpha
                    try:
                        raw, parsed = query_quantity_and_calibration(
                            fall_clip_path, scale_clip, prompt_seed_idx=seed_idx)
                        entry = {
                            "clip": clip_name, "session": session, "condition": str(condition),
                            "orientation": orientation, "seed_idx": seed_idx,
                            "true_velocity_mps": true_v, "asserted_calibration_mm_per_tick": asserted_cal,
                            "raw_response": raw, **parsed,
                            "elapsed_s": round(time.time() - t0, 1),
                        }
                    except Exception as e:
                        entry = {"clip": clip_name, "session": session, "condition": str(condition),
                                  "orientation": orientation, "seed_idx": seed_idx,
                                  "error": f"{type(e).__name__}: {e}"}
                    results.append(entry)
                    done += 1
                    v, c = entry.get("velocity_mps"), entry.get("calibration_mm_per_tick")
                    print(f"[{done}/{total}] {clip_name} cond={condition} orient={orientation} "
                          f"seed={seed_idx} -> v={v} cal={c}", flush=True)
                    with open(OUT_PATH, "w") as f:
                        json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH}, {len(results)} total entries")


if __name__ == "__main__":
    main()
