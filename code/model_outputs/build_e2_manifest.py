"""
Build the E2 clip-selection manifest for the Slurm-cluster re-run.

Rescoped 2026-08-11 after 05052026 was unavailable (the old GPU server that
held it no longer exists; see the paper's dataset-scope discussion). Per user decision, E2 proceeds on
02182026 alone, using EVERY high-confidence clip the rebased oracle finds
(not just a 6-clip subset as the original run did) -- this also directly
improves the paper's own flagged statistical-power problem (Theorem 4's power
analysis needed a 0.45-0.64 true grounding rate to detect at n=12 clips;
more clips tighten this).

Selection criteria, applied in this order:
  1. session == "02182026" (only valid ruler session with usable data)
  2. oracle successfully tracked the clip and returned a velocity
  3. NOT low_confidence (matches original E2's quality bar -- see
     droplet_oracle.py's confidence flag)
  4. filename does not contain "scale" (calibration clips are not droplet
     clips, already excluded by run_batch.py itself, kept here as a second
     guard in case this script is ever pointed at a raw oracle dump)

Output: e2_selected_clips.json, same shape as the original manifest
(list of {path, session, velocity: {v_pre_impact_mps}}), so every downstream
script (run_e2_batch.py, perturbation renderers) that reads this file needs no
format changes, only the new content.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

ORACLE_RESULTS = os.path.join(sgf_paths.RESULTS_DIR, "oracle", "oracle_batch_results.json")
OUT_PATH = sgf_paths.MANIFEST_PATH

TRUE_CALIBRATION_MM_PER_TICK = {
    "02182026": 1.5875,  # CORRECTED: 1/16in (1.5875mm/tick), not 1mm -- see droplet_oracle.py
}


def main():
    with open(ORACLE_RESULTS) as f:
        oracle = json.load(f)

    selected = []
    for r in oracle:
        if r.get("session") != "02182026":
            continue
        if "scale" in os.path.basename(r["path"]).lower():
            continue
        if "error" in r:
            continue
        vel = r.get("velocity")
        if not isinstance(vel, dict) or vel.get("v_pre_impact_mps") is None:
            continue
        if vel.get("low_confidence"):
            continue
        selected.append({
            "path": r["path"],
            "session": r["session"],
            "velocity": {"v_pre_impact_mps": vel["v_pre_impact_mps"]},
        })

    selected.sort(key=lambda c: os.path.basename(c["path"]))

    with open(OUT_PATH, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"selected {len(selected)} clips from 02182026")
    print(f"wrote {OUT_PATH}")
    n_calls_per_model = len(selected) * 3 * 3  # 3 conditions x 3 seeds, per E2's design
    print(f"E2 calls per model at this clip count: {n_calls_per_model}")


if __name__ == "__main__":
    main()
