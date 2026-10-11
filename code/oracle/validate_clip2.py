# Usage: python3 code/oracle/validate_clip2.py <video.mp4> <session, e.g. 02182026>
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from droplet_oracle import track_droplet, link_primary_droplet, compute_pre_impact_velocity
import time

path, session = sys.argv[1], sys.argv[2]
print(f"=== {path} ===", flush=True)
t0 = time.time()
result = track_droplet(path, session)
print(f"track_droplet took {time.time()-t0:.1f}s", flush=True)

if "error" in result:
    print("ERROR:", result["error"])
    sys.exit(0)

linked = link_primary_droplet(result["track"])
print(f"linked track length: {len(linked)}")
if linked:
    print("first point:", linked[0], "last point:", linked[-1])

vel = compute_pre_impact_velocity(linked, result["px_per_mm"])
print("velocity result:", vel)
