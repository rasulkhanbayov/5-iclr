"""
X10: instructed-constant control, added for the AAAI-27 Alignment Track
(runbook v3). Makes Theorem 1 (a change-expecting relation cannot separate a
calibration-blind measurer from a policy that is insensitive to its input)
observable on a real system: instruct the model to report a fixed answer
regardless of video content, then score it through the IDENTICAL SGF path
(SFS at alpha in {0.5, 2}, CG, Acc@10%) as every other E2 condition.

Same 27-clip manifest, same base/alpha=0.5/alpha=2 scale-clip conditions as
E2 (the instructed-constant manipulation is the SYSTEM PROMPT, not the video
shown -- the model still sees the same base/relabeled scale clips as the main
E2 conditions), but no prompt-seed sweep (see prompt_design.py's
INSTRUCTED_CONSTANT_PROMPT_SEEDS docstring: seed variation is a different
question from whether the model complies with an explicit constant-output
instruction at all).

Runbook v3 estimated 108 calls (12 clips x 3 conditions x 3 seeds) for this
condition. At the rescoped 27-clip, no-seed-sweep design used here, cost is
27 x 3 = 81 calls -- cheaper than the original estimate.
"""
import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "sgf_hf_cache"))

import sys
import json
import time

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
sys.path.insert(0, os.path.join(_CODE_DIR, "oracle"))

from run_qwen_inference import query_instructed_constant
sys.path.insert(0, _CODE_DIR)
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

TRIMMED_SCALE_DIR = sgf_paths.TRIMMED_SCALE_DIR
FALL_CLIP_DIR = sgf_paths.FALL_CLIP_DIR

OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "x10", "x10_qwen_results.json")
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
        # See run_e2_batch.py's identical fix: drop entries for clips no
        # longer in the manifest rather than silently carrying them forward.
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
                  "Run run_e2_batch.py (or prepare_clips.py directly) first "
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
