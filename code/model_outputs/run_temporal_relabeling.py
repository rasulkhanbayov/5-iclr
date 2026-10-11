"""
T_beta: temporal relabeling, the temporal analogue of E2's spatial
relabeling R_alpha, run per the appendix's Section A.6 flag ("if the
temporal analogue is run, report it here"). See prompt_design_temporal.py
for the full design rationale.

Same 27-clip manifest, same 3 conditions (base, 0.5, 2 -- here read as beta,
the ASSERTED frame-rate multiplier, not a spatial alpha), same 3 prompt
seeds as E2, so results are directly comparable in structure. Only Qwen2.5-VL
is run here initially (the model with the most complete existing evidence);
extending to Gemini/Qwen3-VL is a direct rerun with the OpenRouter/Qwen3-VL
query functions once this establishes the analysis path works.
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
from run_qwen_inference import query_temporal_relabeling
from prompt_design_temporal import TEMPORAL_CONDITIONS
sys.path.insert(0, _CODE_DIR)
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

FALL_CLIP_DIR = sgf_paths.FALL_CLIP_DIR
os.makedirs(FALL_CLIP_DIR, exist_ok=True)

OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "e2", "temporal_qwen_results.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

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

    valid_clip_names = {os.path.basename(c["path"]) for c in clips}
    results = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            loaded = json.load(f)
        # Same stale-entry guard as run_e2_batch.py/run_x4_resampling.py.
        results = [r for r in loaded if r["clip"] in valid_clip_names]
        n_dropped = len(loaded) - len(results)
        if n_dropped:
            print(f"Dropped {n_dropped} stale entries for clips no longer in "
                  f"the manifest: {sorted(set(r['clip'] for r in loaded if r['clip'] not in valid_clip_names))}",
                  flush=True)
    done_keys = {(r["clip"], r["condition"], r["seed_idx"]) for r in results}

    total = len(clips) * len(TEMPORAL_CONDITIONS) * 3
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

        for condition, beta in TEMPORAL_CONDITIONS.items():
            for seed_idx in range(3):
                key = (clip_name, condition, seed_idx)
                if key in done_keys:
                    continue
                t0 = time.time()
                try:
                    raw, parsed = query_temporal_relabeling(
                        fall_clip_path, true_cal, beta, prompt_seed_idx=seed_idx)
                    entry = {
                        "clip": clip_name, "session": session, "condition": condition,
                        "beta": beta, "seed_idx": seed_idx, "true_velocity_mps": true_v,
                        "asserted_calibration_mm_per_tick": true_cal,
                        "raw_response": raw, **parsed,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                except Exception as e:
                    entry = {"clip": clip_name, "session": session, "condition": condition,
                              "beta": beta, "seed_idx": seed_idx,
                              "error": f"{type(e).__name__}: {e}"}
                results.append(entry)
                done += 1
                v = entry.get("velocity_mps")
                print(f"[{done}/{total}] {clip_name} cond={condition} (beta={beta}) "
                      f"seed={seed_idx} -> v={v}", flush=True)
                with open(OUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH}, {len(results)} total entries")


if __name__ == "__main__":
    main()
