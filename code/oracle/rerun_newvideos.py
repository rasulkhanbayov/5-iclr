import sys
import json
import glob
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from droplet_oracle import track_droplet, link_primary_droplet, compute_pre_impact_velocity
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

folder = sgf_paths.raw_video("13.5cm_speed/05122026")
session = "new_videos/05122026"

results = []
for path in sorted(glob.glob(os.path.join(folder, "*.mp4"))):
    name = os.path.basename(path)
    if "scale" in name.lower():
        continue
    out = {"path": os.path.relpath(path, sgf_paths.RAW_VIDEO_DIR), "session": session}
    result = track_droplet(path, session)
    if "error" in result:
        out["error"] = result["error"]
        results.append(out)
        print(f"{name} -> ERROR: {result['error']}", flush=True)
        continue
    linked = link_primary_droplet(result["track"])
    out["linked_track_len"] = len(linked)
    if not linked:
        out["error"] = "no linkable primary droplet found"
        results.append(out)
        print(f"{name} -> ERROR: no link", flush=True)
        continue
    vel = compute_pre_impact_velocity(linked, result["px_per_mm"])
    out["velocity"] = vel
    results.append(out)
    print(f"{name} -> linked={len(linked)} v={vel}", flush=True)

with open(os.path.join(sgf_paths.RESULTS_DIR, "oracle", "newvideos_rerun.json"), "w") as f:
    json.dump(results, f, indent=2)
