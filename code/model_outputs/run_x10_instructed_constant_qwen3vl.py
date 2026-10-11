"""
X10: instructed-constant control for Qwen3-VL-30B-A3B, current-generation
open-weight half of X1. Mirrors run_x10_instructed_constant.py exactly (same
27-clip manifest, same base/alpha=0.5/alpha=2 scale-clip conditions, no
prompt-seed sweep) so results are directly comparable to the Qwen2.5-VL-7B
and frontier-API X10 runs. Only the inference call differs
(run_qwen3vl_inference instead of run_qwen_inference).

27 clips x 3 conditions x 1 seed = 81 calls.
"""
import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "sgf_hf_cache"))

import sys
import json
import time

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
sys.path.insert(0, os.path.join(_CODE_DIR, "oracle"))

from run_qwen3vl_inference import query_instructed_constant
sys.path.insert(0, _CODE_DIR)
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

TRIMMED_SCALE_DIR = sgf_paths.TRIMMED_SCALE_DIR
FALL_CLIP_DIR = sgf_paths.FALL_CLIP_DIR

OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "x10", "x10_qwen3vl_results.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

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
    if not os.path.exists(manifest_path):
        print(f"ERROR: {manifest_path} does not exist. Run build_e2_manifest.py first.", file=sys.stderr)
        sys.exit(1)
    with open(manifest_path) as f:
        clips = json.load(f)

    for src in SCALE_CLIPS["02182026"].values():
        if not os.path.exists(src):
            print(f"ERROR: {src} does not exist. Run trim_scale_clips.py first.", file=sys.stderr)
            sys.exit(1)

    valid_clip_names = {os.path.basename(c["path"]) for c in clips}
    results = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            loaded = json.load(f)
        results = [r for r in loaded if r["clip"] in valid_clip_names]
        n_dropped = len(loaded) - len(results)
        if n_dropped:
            print(f"Dropped {n_dropped} stale entries for clips no longer in "
                  f"the manifest: {sorted(set(r['clip'] for r in loaded if r['clip'] not in valid_clip_names))}",
                  flush=True)
    done_keys = {(r["clip"], r["condition"]) for r in results}

    total = len(clips) * 3  # 3 conditions, 1 seed
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
            print(f"SKIP {clip_name}: fall clip missing ({fall_clip_path}). "
                  "Run run_e2_batch_qwen3vl.py (or prepare_clips.py directly) first "
                  "so fall clips are extracted once and shared across conditions.",
                  flush=True)
            continue

        for condition in ["base", 0.5, 2]:
            key = (clip_name, str(condition))
            if key in done_keys:
                continue
            scale_clip = SCALE_CLIPS[session][condition]
            alpha = condition if condition != "base" else 1.0
            asserted_cal = true_cal * alpha
            t0 = time.time()
            try:
                raw, parsed = query_instructed_constant(fall_clip_path, scale_clip)
                entry = {
                    "clip": clip_name, "session": session, "condition": str(condition),
                    "seed_idx": 0, "true_velocity_mps": true_v,
                    "asserted_calibration_mm_per_tick": asserted_cal,
                    "raw_response": raw, **parsed,
                    "elapsed_s": round(time.time() - t0, 1),
                }
            except Exception as e:
                entry = {"clip": clip_name, "session": session, "condition": str(condition),
                          "seed_idx": 0, "error": f"{type(e).__name__}: {e}"}
            results.append(entry)
            done += 1
            v = entry.get("velocity_mps")
            c = entry.get("calibration_mm_per_tick")
            print(f"[{done}/{total}] {clip_name} cond={condition} -> v={v} cal={c}", flush=True)
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH}, {len(results)} total entries")


if __name__ == "__main__":
    main()
