"""
X4 for Gemini 3.1 Pro Preview (frontier model), mirroring run_x4_resampling.py
exactly but via OpenRouter instead of the local Qwen pipeline. See that
file's module docstring for the theory (Theorem 2/Corollary 2) and method
(Definition 1 resampling) this implements.

API-based: needs no GPU/Slurm, just OPENROUTER_API_KEY. 162 calls
(27 clips x 2 resample levels x 3 seeds), same cost profile as the E2
frontier run (~$0.026/call observed there -> ~$4.2 total).
"""
import os
os.environ.setdefault("OPENROUTER_API_KEY", "")

import sys
import json
import time

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
from run_openrouter_inference import query_quantity_and_calibration_openrouter
sys.path.insert(0, _CODE_DIR)
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

MODEL = "google/gemini-3.1-pro-preview"
MAX_TOKENS = 2048  # see run_e2_batch_frontier.py: this model spends most of
                    # its completion budget on reasoning tokens before answering

RESAMPLED_DIR = sgf_paths.RESAMPLED_CLIP_DIR
TRIMMED_SCALE_DIR = sgf_paths.TRIMMED_SCALE_DIR

OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "e4_resampling", "x4_frontier_results.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

TRUE_CALIBRATION_MM_PER_TICK = {"02182026": 1.5875}  # CORRECTED: 1/16in (1.5875mm/tick), not 1mm -- see droplet_oracle.py

SCALE_CLIPS_RESAMPLED = {
    "02182026": {
        0.5: os.path.join(TRIMMED_SCALE_DIR, "02182026_scalev_resample_s0.5_trim.mp4"),
        2: os.path.join(TRIMMED_SCALE_DIR, "02182026_scalev_resample_s2_trim.mp4"),
    },
}


def main():
    assert os.environ.get("OPENROUTER_API_KEY"), "set OPENROUTER_API_KEY before running"

    manifest_path = sgf_paths.MANIFEST_PATH
    if not os.path.exists(manifest_path):
        print(f"ERROR: {manifest_path} does not exist. Run build_e2_manifest.py first.", file=sys.stderr)
        sys.exit(1)
    with open(manifest_path) as f:
        clips = json.load(f)

    for src in SCALE_CLIPS_RESAMPLED["02182026"].values():
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
    done_keys = {(r["clip"], r["s"], r["seed_idx"]) for r in results}

    total = len(clips) * 2 * 3
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

        for s in [0.5, 2]:
            resampled_droplet = os.path.join(
                RESAMPLED_DIR, clip_name.replace(".mp4", f"_fall_resample_s{s}.mp4"))
            if not os.path.exists(resampled_droplet):
                print(f"SKIP {clip_name} s={s}: resampled droplet clip missing "
                      f"({resampled_droplet}).", flush=True)
                continue
            scale_clip = SCALE_CLIPS_RESAMPLED[session][s]

            for seed_idx in range(3):
                key = (clip_name, str(s), seed_idx)
                if key in done_keys:
                    continue
                t0 = time.time()
                try:
                    raw, parsed, usage = query_quantity_and_calibration_openrouter(
                        resampled_droplet, scale_clip, prompt_seed_idx=seed_idx,
                        model=MODEL, max_tokens=MAX_TOKENS)
                    entry = {
                        "clip": clip_name, "session": session, "s": str(s),
                        "seed_idx": seed_idx, "true_velocity_mps": true_v,
                        "asserted_calibration_mm_per_tick": true_cal,
                        "raw_response": raw, **parsed, "usage": usage,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    total_cost += usage.get("cost", 0)
                except Exception as e:
                    entry = {"clip": clip_name, "session": session, "s": str(s),
                              "seed_idx": seed_idx, "error": f"{type(e).__name__}: {e}"}
                results.append(entry)
                done += 1
                v = entry.get("velocity_mps")
                print(f"[{done}/{total}] {clip_name} s={s} seed={seed_idx} "
                      f"-> v={v} (${total_cost:.4f} total)", flush=True)
                with open(OUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH}, {len(results)} total entries, ${total_cost:.4f} total spend")


if __name__ == "__main__":
    main()
