"""
E5 robustness stability table (sgf_runbook.txt): SFS stability across alpha
values, ruler orientation, and prompt phrasings, from e5_qwen_results.json
(60 items: 2 clips x 5 conditions [base + alpha in {0.25,0.5,2,4}] x 2
orientations x 3 seeds).

Per clip x orientation x seed, we have one base velocity and 4 scaled
velocities (one per alpha) -- compute SFS per (clip, orientation, seed,
alpha) quadruple against that same triple's base, then group/average along
whichever axis is being tested for stability, holding the others fixed by
averaging over them (not cherry-picking one slice).
"""
import sys
import json
import numpy as np

sys.path.insert(0, "/home/ubuntu/5-ICLR (work with Dr. Mariam)")
from sgf_metrics import scale_faithfulness_score, calibration_grounding, bootstrap_ci

RESULTS_PATH = "/ephemeral/sgf_work/results/e5/e5_qwen_results.json"
OUT_PATH = "/ephemeral/sgf_work/results/e5/e5_stability_table.json"

ALPHAS = [0.25, 0.5, 2, 4]
TRUE_CAL = 1.0  # mm per tick, both sessions


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

    # Full per-item table: one row per (clip, orientation, seed, alpha).
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
                    rows.append({"clip": clip, "orientation": orientation, "seed_idx": seed_idx,
                                 "alpha": alpha, "sfs": sfs, "cg_hit": cg_hit,
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
    }

    # Range of per-condition SFS means, the runbook's actual "stability" ask:
    # how much does the mean SFS move across levels of each axis?
    def spread(table):
        means = [v["sfs_mean"] for v in table.values()]
        return {"min": min(means), "max": max(means), "range": max(means) - min(means)}

    stability_spread = {k: spread(v) for k, v in stability.items()}

    summary = {
        "n_items": len(rows),
        "clips": clips, "note_scope": ("2 representative clips (water.mp4/02182026, "
                                        "0.08cg2.mp4/05052026), per user-agreed scope -- "
                                        "E5 tests metric/perturbation stability across "
                                        "conditions, not per-clip item coverage (already "
                                        "covered by E2's 12-clip run)."),
        "overall_sfs": overall_sfs, "overall_sfs_ci95": [sfs_lo, sfs_hi],
        "overall_cg": overall_cg, "overall_cg_ci95": [cg_lo, cg_hi],
        "stability": stability,
        "stability_spread": stability_spread,
        "IMPORTANT_alpha4_artifact": (
            "The by_alpha table's alpha=4 SFS (0.396, the highest of any alpha) is NOT "
            "evidence of better faithfulness at that perturbation strength -- it is a "
            "coincidental artifact of the model's near-constant-response pathology "
            "(same phenomenon as project_sgf_e2_finding.md / E4's confound). The model's "
            "reported velocities are drawn from a tiny fixed palette (0.3, 0.5, 0.6, 0.84, "
            "1.5, 1.875, 3.5 m/s) that barely varies with the actual displayed calibration. "
            "At alpha=4 specifically, EVERY item (all clips/orientations/seeds) happens to "
            "land on the same (base=0.5, scaled=1.5) pair -- a content-independent 3x ratio "
            "that coincidentally sits close to the ideal alpha=4 log-ratio target, scoring "
            "artificially high. At other alphas the same small palette produces ratios far "
            "from ideal, scoring low. This is luck-of-the-draw from a small discrete output "
            "set landing near a target by chance, not a real alpha-dependent faithfulness "
            "effect -- do not report the by_alpha spread (0.331) as showing the model is "
            "'more faithful under strong perturbation.' The orientation and seed axes did NOT "
            "show this coincidental-alignment problem (their spreads, 0.046 and 0.087, are "
            "genuine measures of low variability, not itself a positive faithfulness signal "
            "either, just consistently low)."
        ),
        "rows": rows,
    }

    import os
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"n_items = {len(rows)}")
    print(f"overall SFS = {overall_sfs:.3f} CI[{sfs_lo:.3f},{sfs_hi:.3f}]")
    print(f"overall CG = {overall_cg:.3f} CI[{cg_lo:.3f},{cg_hi:.3f}]")
    print()
    for axis, table in stability.items():
        print(f"-- stability by {axis} --")
        for val, v in table.items():
            print(f"  {val:>8s}: n={v['n']:3d}  SFS={v['sfs_mean']:.3f}  CI=[{v['sfs_ci95'][0]:.3f},{v['sfs_ci95'][1]:.3f}]")
        sp = stability_spread[axis]
        print(f"  range across levels: {sp['range']:.3f} (min {sp['min']:.3f}, max {sp['max']:.3f})")
        print()
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
