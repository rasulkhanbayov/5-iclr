"""
Improvement plan A2: rerun E3's control question on the 5 contrast-enhanced,
impact-cropped clips (enhance_e3_clips.py) that failed in e3v3. Only these 5
-- the other 7 already pass and don't need re-testing.
"""
import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "sgf_hf_cache"))

import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

sys.path.insert(0, os.path.join(sgf_paths.CODE_DIR, "model_outputs"))
from run_qwen_inference import query_hidden_prior_with_control

ENHANCED_DIR = sgf_paths.ENHANCED_FALL_CLIP_DIR
OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "e3", "e3_enhanced_retest_results.json")

FAILING_CLIPS = ["cainhcg 0.08 b.mp4", "cainhg0.02 2.mp4", "0.08cg.mp4", "0.08cg2.mp4", "0.08cg3.mp4"]


def main():
    results = []
    for name in FAILING_CLIPS:
        path = os.path.join(ENHANCED_DIR, name.replace(".mp4", "_enhanced.mp4"))
        if not os.path.exists(path):
            print(f"SKIP {name}: no enhanced clip found at {path}", flush=True)
            continue
        t0 = time.time()
        raw, parsed = query_hidden_prior_with_control(path)
        entry = {"clip": name, "session": "05052026", "raw_response": raw, **parsed,
                  "elapsed_s": round(time.time() - t0, 1)}
        results.append(entry)
        print(f"{name} -> control={entry.get('control_answer')} "
              f"(correct={entry.get('control_correct')}) declined={entry.get('declined')}", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    n_correct = sum(1 for r in results if r.get("control_correct"))
    print(f"\nwrote {OUT_PATH}: {n_correct}/{len(results)} now pass control (were 0/5 before)")


if __name__ == "__main__":
    main()
