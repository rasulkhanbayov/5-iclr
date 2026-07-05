import os
os.environ.setdefault("HF_HOME", "/ephemeral/sgf_work/hf_cache")

import sys
import json
import time

sys.path.insert(0, "/ephemeral/sgf_work/model_outputs")

from run_qwen_inference import query_hidden_prior_with_control

FALL_CLIP_DIR = "/ephemeral/sgf_work/model_outputs/fall_clips"
OUT_PATH = "/ephemeral/sgf_work/results/e3/e3v3_qwen_results.json"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def main():
    with open("/ephemeral/sgf_work/model_outputs/e2_selected_clips.json") as f:
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
