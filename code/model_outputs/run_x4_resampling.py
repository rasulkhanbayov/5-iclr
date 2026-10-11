"""
X4: resampling negative control (S_s), runbook v2 block T1 / runbook v3 X4.
Required by Theorem 2/Corollary 2 -- without it, every claim in the paper that
localizes the failure to scale-reading rather than input insensitivity is
capped at membership in the calibration-blind class B_ld (see sgf_main.tex
Section 4, Corollary "Design rule for metamorphic grounding probes").

Method (Definition 1, Conditions): resample the WHOLE frame -- droplet clip
AND scale clip -- by s in {0.5, 2}, holding what the scale ASSERTS fixed.
Uses resample_renderer.py's pre-rendered clips (../results/resampled_clips/),
not perturb_renderer.py's relabeling (that is a different condition, R_alpha,
already covered by run_e2_batch.py).

Same 27-clip manifest, same 3 prompt seeds, same query/parse path as E2 --
only the video pair differs (resampled instead of base/relabeled), so results
feed directly into classify_policy() in sgf_metrics.py alongside E2's r_R.
"""
import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "sgf_hf_cache"))

import sys
import json
import time

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE_DIR, "model_outputs"))
sys.path.insert(0, os.path.join(_CODE_DIR, "oracle"))

from run_qwen_inference import query_quantity_and_calibration
sys.path.insert(0, _CODE_DIR)
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

RESAMPLED_DIR = sgf_paths.RESAMPLED_CLIP_DIR
TRIMMED_SCALE_DIR = sgf_paths.TRIMMED_SCALE_DIR

OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "e4_resampling", "x4_qwen_results.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Base condition scale calibration is unchanged by resampling (Definition 1:
# resampling holds what the scale ASSERTS fixed) -- the true asserted
# calibration for every resampling item is therefore the session's own base
# value, same as E2's "base" condition, regardless of s.
TRUE_CALIBRATION_MM_PER_TICK = {
    "02182026": 1.5875,  # CORRECTED: 1/16in (1.5875mm/tick), not 1mm -- see droplet_oracle.py
}

SCALE_CLIPS_RESAMPLED = {
    "02182026": {
        0.5: os.path.join(TRIMMED_SCALE_DIR, "02182026_scalev_resample_s0.5_trim.mp4"),
        2: os.path.join(TRIMMED_SCALE_DIR, "02182026_scalev_resample_s2_trim.mp4"),
    },
}


def main():
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
        # See run_e2_batch.py's identical fix: drop entries for clips no
        # longer in the manifest rather than silently carrying them forward.
        results = [r for r in loaded if r["clip"] in valid_clip_names]
        n_dropped = len(loaded) - len(results)
        if n_dropped:
            print(f"Dropped {n_dropped} stale entries for clips no longer in "
                  f"the manifest: {sorted(set(r['clip'] for r in loaded if r['clip'] not in valid_clip_names))}",
                  flush=True)
    done_keys = {(r["clip"], r["s"], r["seed_idx"]) for r in results}

    total = len(clips) * 2 * 3  # 2 resample levels x 3 seeds, per runbook T1
    done = len(results)
    print(f"{total} total items, {done} already done, {total-done} remaining", flush=True)

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
                      f"({resampled_droplet}). Run resample_renderer.py over the "
                      "fall clips first.", flush=True)
                continue
            scale_clip = SCALE_CLIPS_RESAMPLED[session][s]

            for seed_idx in range(3):
                key = (clip_name, str(s), seed_idx)
                if key in done_keys:
                    continue
                t0 = time.time()
                try:
                    raw, parsed = query_quantity_and_calibration(
                        resampled_droplet, scale_clip, prompt_seed_idx=seed_idx)
                    entry = {
                        "clip": clip_name, "session": session, "s": str(s),
                        "seed_idx": seed_idx, "true_velocity_mps": true_v,
                        "asserted_calibration_mm_per_tick": true_cal,  # unchanged by S_s
                        "raw_response": raw, **parsed,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                except Exception as e:
                    entry = {"clip": clip_name, "session": session, "s": str(s),
                              "seed_idx": seed_idx, "error": f"{type(e).__name__}: {e}"}
                results.append(entry)
                done += 1
                v = entry.get("velocity_mps")
                print(f"[{done}/{total}] {clip_name} s={s} seed={seed_idx} -> v={v}", flush=True)
                with open(OUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH}, {len(results)} total entries")


if __name__ == "__main__":
    main()
