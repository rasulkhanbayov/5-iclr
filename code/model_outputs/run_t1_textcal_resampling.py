"""
T1 positive control, S_s half: text-calibration resampling. The asserted
calibration is held fixed in the text (the true value) while the frame is
resampled by s -- mirroring the visual S_s condition's invariant. If the
policy can convert at all, it should be resample-invariant: r_S ~ 1.

Same 27-clip manifest, s in {0.5, 2}, 3 seeds, matching X4's design. Model
selection via MODEL env var: "qwen25vl" (default), "qwen3vl", or "gemini".
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
    from run_qwen_inference import query_textcal_resampling as _query
    OUT_NAME = "t1_textcal_resample_qwen_results.json"
elif MODEL == "qwen3vl":
    from run_qwen3vl_inference import query_textcal_resampling as _query
    OUT_NAME = "t1_textcal_resample_qwen3vl_results.json"
elif MODEL == "gemini":
    from run_openrouter_inference import query_textcal_resampling_openrouter as _openrouter_query
    OUT_NAME = "t1_textcal_resample_frontier_results.json"

    def _query(resampled_droplet, true_cal, prompt_seed_idx=0):
        raw, parsed, usage = _openrouter_query(resampled_droplet, true_cal, prompt_seed_idx=prompt_seed_idx)
        parsed["usage"] = usage
        return raw, parsed
else:
    print(f"ERROR: unknown MODEL={MODEL!r}. Use qwen25vl, qwen3vl, or gemini.", file=sys.stderr)
    sys.exit(1)

RESAMPLED_DIR = sgf_paths.RESAMPLED_CLIP_DIR

OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "t1", OUT_NAME)
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
        results = [r for r in loaded if r["clip"] in valid_clip_names]
        n_dropped = len(loaded) - len(results)
        if n_dropped:
            print(f"Dropped {n_dropped} stale entries for clips no longer in "
                  f"the manifest: {sorted(set(r['clip'] for r in loaded if r['clip'] not in valid_clip_names))}",
                  flush=True)
    done_keys = {(r["clip"], r["s"], r["seed_idx"]) for r in results}

    total = len(clips) * 2 * 3  # 2 resample levels x 3 seeds
    done = len(results)
    print(f"[{MODEL}] {total} total items, {done} already done, {total-done} remaining", flush=True)

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

            for seed_idx in range(3):
                key = (clip_name, str(s), seed_idx)
                if key in done_keys:
                    continue
                t0 = time.time()
                try:
                    raw, parsed = _query(resampled_droplet, true_cal, prompt_seed_idx=seed_idx)
                    entry = {
                        "clip": clip_name, "session": session, "s": str(s),
                        "seed_idx": seed_idx, "true_velocity_mps": true_v,
                        "asserted_calibration_mm_per_tick": true_cal,
                        "raw_response": raw, **parsed,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                except Exception as e:
                    entry = {"clip": clip_name, "session": session, "s": str(s),
                              "seed_idx": seed_idx, "error": f"{type(e).__name__}: {e}"}
                results.append(entry)
                done += 1
                v = entry.get("velocity_mps")
                print(f"[{MODEL}][{done}/{total}] {clip_name} s={s} seed={seed_idx} -> v={v}", flush=True)
                with open(OUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH}, {len(results)} total entries")


if __name__ == "__main__":
    main()
