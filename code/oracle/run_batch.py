import sys
import json
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def _init_worker():
    # Each worker process is one unit of parallelism; without this, OpenCV's own
    # internal thread pool (2+ threads/process) multiplies against ProcessPoolExecutor's
    # workers and oversubscribes the machine (saw load average 54 on 28 cores, workers
    # stalling on scheduling contention rather than doing useful work).
    import cv2
    cv2.setNumThreads(1)


sys.path.insert(0, '.')
from droplet_oracle import track_droplet, link_primary_droplet, compute_pre_impact_velocity

SESSIONS = {
    "02182026": "/ephemeral/videos/02182026",
    "05052026": "/ephemeral/videos/05052026",
    "new_videos/05122026": "/ephemeral/videos/new_videos/05122026",
}

OUT_DIR = "/ephemeral/sgf_work/oracle/results"
os.makedirs(OUT_DIR, exist_ok=True)


def list_clips():
    clips = []
    for session_key, folder in SESSIONS.items():
        for path in sorted(glob.glob(os.path.join(folder, "*.mp4"))):
            name = os.path.basename(path)
            if "scale" in name.lower():
                continue
            clips.append((path, session_key))
    return clips


def process_one(args):
    path, session_key = args
    t0 = time.time()
    out = {"path": path, "session": session_key}
    try:
        result = track_droplet(path, session_key)
        if "error" in result:
            out["error"] = result["error"]
            return out
        linked = link_primary_droplet(result["track"])
        out["linked_track_len"] = len(linked)
        out["motion_window"] = result["motion_window"]
        if not linked:
            out["error"] = "no linkable primary droplet found"
            return out
        vel = compute_pre_impact_velocity(linked, result["px_per_mm"])
        out["velocity"] = vel
        out["max_area_px"] = max((p["area_px"] for p in linked), default=None)
        out["n_blobs_multi_frame_count"] = sum(1 for t in result["track"] if len(t["blobs"]) > 1)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    clips = list_clips()
    print(f"{len(clips)} clips to process", flush=True)

    out_path = os.path.join(OUT_DIR, "oracle_batch_results.json")

    # resume: skip clips already processed in a prior (killed) run, recovered from its log
    already_done = set()
    if len(sys.argv) > 1 and sys.argv[1] == "--skip-file":
        with open(sys.argv[2]) as f:
            already_done = {line.strip() for line in f if line.strip()}
        print(f"resuming: skipping {len(already_done)} already-completed clips", flush=True)

    results = []
    if already_done and os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)  # seed with previously-recovered results so they aren't lost
    todo = [c for c in clips if os.path.basename(c[0]) not in already_done]
    print(f"{len(todo)} clips remaining after skip", flush=True)

    n_workers = 8  # cv2.setNumThreads(1) in workers removes the need for heavy headroom;
                   # 8 procs x 1 thread stays under the 28-core budget with room for the OS.
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        futures = {ex.submit(process_one, c): c for c in todo}
        done = 0
        for fut in as_completed(futures):
            path, session_key = futures[fut]
            done += 1
            try:
                res = fut.result()
            except Exception as e:
                res = {"path": path, "session": session_key, "error": f"worker crash: {e}"}
            results.append(res)
            status = "OK" if "error" not in res else f"ERROR: {res['error']}"
            v = res.get("velocity", {}).get("v_pre_impact_mps") if res.get("velocity") else None
            print(f"[{done}/{len(todo)}] {os.path.basename(path)} ({session_key}) -> {status} v={v}", flush=True)
            # incremental write so a kill/crash doesn't lose already-completed work
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nwrote {out_path}")
    n_ok = sum(1 for r in results if "error" not in r)
    print(f"summary: {n_ok}/{len(results)} succeeded")
