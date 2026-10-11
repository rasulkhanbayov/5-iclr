"""
T1 positive control, R_alpha half: text-calibration relabeling. Per
the main paper's "The spatial axes lack a positive
control" flag. Removes the perceptual step (reading the ruler) by asserting
the calibration directly in text, leaving the pixel-to-physical conversion
step intact -- if a model can convert at all, this policy should behave as
grounded under R_alpha: r_R ~ alpha.

Same 27-clip manifest, same 3 conditions (base, 0.5, 2), same 3 prompt seeds
as E2, so results are directly comparable in structure. Run for all three
models: Qwen2.5-VL-7B (local), Qwen3-VL-30B-A3B (local), Gemini 3.1 Pro
(OpenRouter). Model selection via MODEL env var: "qwen25vl" (default),
"qwen3vl", or "gemini".
"""
import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "sgf_hf_cache"))

import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
sys.path.insert(0, os.path.join(_CODE_DIR, "oracle"))

MODEL = os.environ.get("MODEL", "qwen25vl")

if MODEL == "qwen25vl":
    from prepare_clips import extract_fall_clip
    from run_qwen_inference import query_textcal_relabeling as _query
    OUT_NAME = "t1_textcal_relabel_qwen_results.json"
elif MODEL == "qwen3vl":
    from prepare_clips import extract_fall_clip
    from run_qwen3vl_inference import query_textcal_relabeling as _query
    OUT_NAME = "t1_textcal_relabel_qwen3vl_results.json"
elif MODEL == "gemini":
    from prepare_clips import extract_fall_clip
    from run_openrouter_inference import query_textcal_relabeling_openrouter as _openrouter_query
    OUT_NAME = "t1_textcal_relabel_frontier_results.json"

    def _query(droplet, alpha_cal_base, alpha, prompt_seed_idx=0):
        raw, parsed, usage = _openrouter_query(droplet, alpha_cal_base, alpha, prompt_seed_idx=prompt_seed_idx)
        parsed["usage"] = usage
        return raw, parsed
else:
    print(f"ERROR: unknown MODEL={MODEL!r}. Use qwen25vl, qwen3vl, or gemini.", file=sys.stderr)
    sys.exit(1)

FALL_CLIP_DIR = sgf_paths.FALL_CLIP_DIR
os.makedirs(FALL_CLIP_DIR, exist_ok=True)

OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "t1", OUT_NAME)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

TRUE_CALIBRATION_MM_PER_TICK = {
    "02182026": 1.5875,  # CORRECTED: 1/16in (1.5875mm/tick), not 1mm -- see droplet_oracle.py
}

CONDITIONS = {"base": 1.0, "0.5": 0.5, "2": 2.0}


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
        results = [r for r in loaded if r["clip"] in valid_clip_names]
        n_dropped = len(loaded) - len(results)
        if n_dropped:
            print(f"Dropped {n_dropped} stale entries for clips no longer in "
                  f"the manifest: {sorted(set(r['clip'] for r in loaded if r['clip'] not in valid_clip_names))}",
                  flush=True)
    done_keys = {(r["clip"], r["condition"], r["seed_idx"]) for r in results}

    total = len(clips) * len(CONDITIONS) * 3
    done = len(results)
    print(f"[{MODEL}] {total} total items, {done} already done, {total-done} remaining", flush=True)

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

        for condition, alpha in CONDITIONS.items():
            for seed_idx in range(3):
                key = (clip_name, condition, seed_idx)
                if key in done_keys:
                    continue
                t0 = time.time()
                try:
                    raw, parsed = _query(fall_clip_path, true_cal, alpha, prompt_seed_idx=seed_idx)
                    entry = {
                        "clip": clip_name, "session": session, "condition": condition,
                        "alpha": alpha, "seed_idx": seed_idx, "true_velocity_mps": true_v,
                        "asserted_calibration_mm_per_tick": true_cal * alpha,
                        "raw_response": raw, **parsed,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                except Exception as e:
                    entry = {"clip": clip_name, "session": session, "condition": condition,
                              "alpha": alpha, "seed_idx": seed_idx,
                              "error": f"{type(e).__name__}: {e}"}
                results.append(entry)
                done += 1
                v = entry.get("velocity_mps")
                print(f"[{MODEL}][{done}/{total}] {clip_name} cond={condition} "
                      f"(alpha={alpha}) seed={seed_idx} -> v={v}", flush=True)
                with open(OUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH}, {len(results)} total entries")


if __name__ == "__main__":
    main()
