import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "sgf_hf_cache"))

import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

sys.path.insert(0, os.path.join(sgf_paths.CODE_DIR, "model_outputs"))

from run_qwen_inference import query_hidden_prior_with_control

FALL_CLIP_DIR = sgf_paths.FALL_CLIP_DIR
OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "e3", "e3v3_qwen_results.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def main():
    with open(sgf_paths.MANIFEST_PATH) as f:
        clips = json.load(f)

    results = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
    done_clips = {r["clip"] for r in results}

    total = len(clips)
    done = len(results)
    print(f"{total} total items, {done} already done", flush=True)

    for clip_info in clips:
        clip_name = os.path.basename(clip_info["path"])
        if clip_name in done_clips:
            continue
        fall_clip_path = os.path.join(FALL_CLIP_DIR, clip_name.replace(".mp4", "_fall.mp4"))
        if not os.path.exists(fall_clip_path):
            print(f"SKIP {clip_name}: no fall clip extracted", flush=True)
            continue

        t0 = time.time()
        try:
            raw, parsed = query_hidden_prior_with_control(fall_clip_path)
            entry = {"clip": clip_name, "session": clip_info["session"],
                     "raw_response": raw, **parsed, "elapsed_s": round(time.time() - t0, 1)}
        except Exception as e:
            entry = {"clip": clip_name, "session": clip_info["session"], "error": f"{type(e).__name__}: {e}"}
        results.append(entry)
        done += 1
        print(f"[{done}/{total}] {clip_name} -> control={entry.get('control_answer')} "
              f"(correct={entry.get('control_correct')}) declined={entry.get('declined')} "
              f"asserted={entry.get('asserted_value_cm')}", flush=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH}, {len(results)} total entries")


if __name__ == "__main__":
    main()
