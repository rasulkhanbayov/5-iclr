"""
X10 for Gemini 3.1 Pro Preview (frontier model), mirroring
run_x10_instructed_constant.py exactly but via OpenRouter. See that file's
module docstring for the rationale (Corollary "Design rule for metamorphic
grounding probes" / Theorem 1 made observable).

API-based: needs no GPU/Slurm, just OPENROUTER_API_KEY. 81 calls
(27 clips x 3 conditions, no seed sweep), ~$2.1 estimated at the E2 run's
observed $0.026/call.
"""
import os
os.environ.setdefault("OPENROUTER_API_KEY", "")

import sys
import json
import time

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
from run_openrouter_inference import query_instructed_constant_openrouter
sys.path.insert(0, _CODE_DIR)
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

MODEL = "google/gemini-3.1-pro-preview"
MAX_TOKENS = 2048

TRIMMED_SCALE_DIR = sgf_paths.TRIMMED_SCALE_DIR
FALL_CLIP_DIR = sgf_paths.FALL_CLIP_DIR

OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "x10", "x10_frontier_results.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

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

    total = len(clips) * 3
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
            print(f"SKIP {clip_name}: fall clip missing ({fall_clip_path}).", flush=True)
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
                raw, parsed, usage = query_instructed_constant_openrouter(
                    fall_clip_path, scale_clip, model=MODEL, max_tokens=MAX_TOKENS)
                entry = {
                    "clip": clip_name, "session": session, "condition": str(condition),
                    "seed_idx": 0, "true_velocity_mps": true_v,
                    "asserted_calibration_mm_per_tick": asserted_cal,
                    "raw_response": raw, **parsed, "usage": usage,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                total_cost += usage.get("cost", 0)
            except Exception as e:
                entry = {"clip": clip_name, "session": session, "condition": str(condition),
                          "seed_idx": 0, "error": f"{type(e).__name__}: {e}"}
            results.append(entry)
            done += 1
            v = entry.get("velocity_mps")
            c = entry.get("calibration_mm_per_tick")
            print(f"[{done}/{total}] {clip_name} cond={condition} -> v={v} cal={c} "
                  f"(${total_cost:.4f} total)", flush=True)
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH}, {len(results)} total entries, ${total_cost:.4f} total spend")


if __name__ == "__main__":
    main()
