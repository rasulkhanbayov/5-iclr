import os
os.environ.setdefault("HF_HOME", "/ephemeral/sgf_work/hf_cache")

import sys
import json
import time

sys.path.insert(0, "/ephemeral/sgf_work/model_outputs")
sys.path.insert(0, "/ephemeral/sgf_work/oracle")

from prepare_clips import extract_fall_clip
from run_qwen_inference import query_quantity_and_calibration
from droplet_oracle import SESSION_CALIBRATION_PX_PER_MM

FALL_CLIP_DIR = "/ephemeral/sgf_work/model_outputs/fall_clips"
os.makedirs(FALL_CLIP_DIR, exist_ok=True)

OUT_PATH = "/ephemeral/sgf_work/results/e2/e2_qwen_results.json"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Trimmed to 3s (see trim_scale_clips.py): the ruler/comb is static all clip, so
# showing the model the full original (02182026's scale v.mp4 is 144s!) made a
# single inference call take 7+ minutes instead of ~1 minute -- decord still has
# to index/decode the whole long container even at low sample fps.
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

TRUE_CALIBRATION_MM_PER_TICK = {
    "02182026": 1.0,
    "05052026": 1.0,
}


def main():
    with open("/ephemeral/sgf_work/model_outputs/e2_selected_clips.json") as f:
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
        droplet_path = clip_info["path"]
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
                # The "correct" calibration reading for CG is whatever the CURRENTLY
                # DISPLAYED scale actually asserts: true_cal for the base (unperturbed)
                # clip, or alpha*true_cal for a relabeled clip (relabel formula is
                # new_label = alpha*original_label, see project_sgf_perturbation_renderer.md).
                # CG scores whether the model correctly reads what's in front of it,
                # not whether it somehow knows the pre-perturbation ground truth.
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
