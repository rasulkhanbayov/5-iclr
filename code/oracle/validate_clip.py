# Usage: python3 code/oracle/validate_clip.py <video.mp4> <session, e.g. 02182026>
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from droplet_oracle import track_droplet, TRUE_FPS
import numpy as np
import time

path, session = sys.argv[1], sys.argv[2]
print(f"=== {path} ===", flush=True)
t0 = time.time()
result = track_droplet(path, session)
print(f"track_droplet took {time.time()-t0:.1f}s", flush=True)

if "error" in result:
    print("ERROR:", result["error"])
    sys.exit(0)

px_per_mm = result["px_per_mm"]
start, end = result["motion_window"]
print(f"motion window: {start}-{end}, {len(result['track'])} tracked frames")

single = [(t["frame"], t["blobs"][0]["cy"]) for t in result["track"] if len(t["blobs"]) == 1]
if len(single) < 5:
    print("too few single-blob frames to fit velocity:", len(single))
    sys.exit(0)

frames = np.array([p[0] for p in single])
cy = np.array([p[1] for p in single])
diffs = np.diff(cy)
end_idx = len(cy)
for i, d in enumerate(diffs):
    if d < 0:
        end_idx = i + 1
        break

if end_idx < 4:
    print("fall segment too short:", end_idx, "points")
    sys.exit(0)

seg_frames = frames[:end_idx]
seg_cy = cy[:end_idx]
slope, intercept = np.polyfit(seg_frames, seg_cy, 1)
v_mps = (slope * TRUE_FPS / px_per_mm) / 1000
print(f"fall segment frames {seg_frames[0]}-{seg_frames[-1]} ({end_idx} pts), v_pre_impact = {v_mps:.3f} m/s")
