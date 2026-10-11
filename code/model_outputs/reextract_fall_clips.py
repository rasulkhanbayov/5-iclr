import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)
sys.path.insert(0, os.path.join(sgf_paths.CODE_DIR, "model_outputs"))
sys.path.insert(0, os.path.join(sgf_paths.CODE_DIR, "oracle"))
from prepare_clips import extract_fall_clip

OUT_DIR = sgf_paths.FALL_CLIP_DIR
os.makedirs(OUT_DIR, exist_ok=True)

with open(sgf_paths.MANIFEST_PATH) as f:
    clips = json.load(f)

results = []
for c in clips:
    path = sgf_paths.raw_video(c["path"])
    clip_name = os.path.basename(path)
    out_path = os.path.join(OUT_DIR, clip_name.replace(".mp4", "_fall.mp4"))
    meta = extract_fall_clip(path, out_path)
    meta["clip"] = clip_name
    meta["session"] = c["session"]
    results.append(meta)
    print(f"{c['session']:12s} {clip_name:22s} window={meta.get('motion_window_frames')} "
          f"extracted={meta.get('extracted_frames')} n_frames={meta.get('n_frames_written')} "
          f"met={meta.get('settle_criterion_met')} hit_cap={meta.get('settle_hit_cap')}", flush=True)

with open(os.path.join(sgf_paths.RESULTS_DIR, "fall_clips_v2_extraction_meta.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {len(results)} clips to {OUT_DIR}")
