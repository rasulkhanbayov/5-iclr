"""
Improvement plan B1 analysis: same stability-table computation as
run_e5_analysis.py, applied to the full 12-clip E5 sweep
(e5_qwen_results_full12.json, 360 items) instead of the original 2-clip
sample (60 items). Adds a NEW axis not possible with only 2 clips: stability
BY CLIP, to check whether the low overall SFS/CG is uniform across all 12
droplet clips or concentrated in a subset.
"""
import sys
import json
import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

sys.path.insert(0, sgf_paths.REPO_ROOT)
from sgf_metrics import scale_faithfulness_score, bootstrap_ci

RESULTS_PATH = os.path.join(sgf_paths.RESULTS_DIR, "e5", "e5_qwen_results_full12.json")
OUT_PATH = os.path.join(sgf_paths.RESULTS_DIR, "e5", "e5_stability_table_full12.json")

ALPHAS = [0.25, 0.5, 2, 4]
TRUE_CAL = 1.0


def build_index(results):
    idx = {}
    for r in results:
        key = (r["clip"], r["orientation"], r["seed_idx"], r["condition"])
        idx[key] = r
    return idx


def per_item_sfs(idx, clip, orientation, seed_idx, alpha):
    base = idx.get((clip, orientation, seed_idx, "base"))
    scaled = idx.get((clip, orientation, seed_idx, str(alpha)))
    if base is None or scaled is None:
        return None
    _, per = scale_faithfulness_score(
        np.array([base["velocity_mps"]]), np.array([scaled["velocity_mps"]]), alpha=alpha, d=1)
    return float(per[0])


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    idx = build_index(results)
    clips = sorted({r["clip"] for r in results})
    orientations = sorted({r["orientation"] for r in results})
    seeds = sorted({r["seed_idx"] for r in results})

    rows = []
    for clip in clips:
        for orientation in orientations:
            for seed_idx in seeds:
                for alpha in ALPHAS:
                    sfs = per_item_sfs(idx, clip, orientation, seed_idx, alpha)
                    if sfs is None:
                        continue
                    scaled = idx[(clip, orientation, seed_idx, str(alpha))]
                    cg_hit = float(abs(scaled["calibration_mm_per_tick"] - TRUE_CAL * alpha)
                                   / (TRUE_CAL * alpha) <= 0.10)
                    session = scaled["session"]
                    rows.append({"clip": clip, "session": session, "orientation": orientation,
                                 "seed_idx": seed_idx, "alpha": alpha, "sfs": sfs, "cg_hit": cg_hit,
                                 "base_velocity_mps": idx[(clip, orientation, seed_idx, "base")]["velocity_mps"],
                                 "scaled_velocity_mps": scaled["velocity_mps"]})

    sfs_all = np.array([r["sfs"] for r in rows])
    cg_all = np.array([r["cg_hit"] for r in rows])
    overall_sfs, sfs_lo, sfs_hi = bootstrap_ci(sfs_all)
    overall_cg, cg_lo, cg_hi = bootstrap_ci(cg_all)

    def stability_by(key):
        table = {}
        for val in sorted({r[key] for r in rows}, key=str):
            sub = [r["sfs"] for r in rows if r[key] == val]
            m, lo, hi = bootstrap_ci(np.array(sub))
            table[str(val)] = {"n": len(sub), "sfs_mean": m, "sfs_ci95": [lo, hi]}
        return table

    stability = {
        "by_alpha": stability_by("alpha"),
        "by_orientation": stability_by("orientation"),
        "by_seed": stability_by("seed_idx"),
        "by_clip": stability_by("clip"),
        "by_session": stability_by("session"),
    }

    def spread(table):
        means = [v["sfs_mean"] for v in table.values()]
        return {"min": min(means), "max": max(means), "range": max(means) - min(means)}

    stability_spread = {k: spread(v) for k, v in stability.items()}

    summary = {
        "n_items": len(rows), "n_clips": len(clips), "clips": clips,
        "note_scope": "Improvement plan B1: full 12-clip E5 sweep (all E2 clips), local Qwen only, no new API cost.",
        "overall_sfs": overall_sfs, "overall_sfs_ci95": [sfs_lo, sfs_hi],
        "overall_cg": overall_cg, "overall_cg_ci95": [cg_lo, cg_hi],
        "stability": stability,
        "stability_spread": stability_spread,
        "rows": rows,
    }

    import os
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"n_items = {len(rows)} across {len(clips)} clips")
    print(f"overall SFS = {overall_sfs:.3f} CI[{sfs_lo:.3f},{sfs_hi:.3f}]")
    print(f"overall CG = {overall_cg:.3f} CI[{cg_lo:.3f},{cg_hi:.3f}]")
    print()
    for axis, table in stability.items():
        print(f"-- stability by {axis} --")
        for val, v in table.items():
            print(f"  {val:>22s}: n={v['n']:3d}  SFS={v['sfs_mean']:.3f}  CI=[{v['sfs_ci95'][0]:.3f},{v['sfs_ci95'][1]:.3f}]")
        sp = stability_spread[axis]
        print(f"  range across levels: {sp['range']:.3f} (min {sp['min']:.3f}, max {sp['max']:.3f})")
        print()
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
