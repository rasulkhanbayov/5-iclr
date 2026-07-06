"""
E4 improvement A1 (per user-approved improvement plan): rerun the decoupling
analysis across a range of accuracy tolerances tau, not just the paper's
originally-stated 10%. Motivation: Acc@10% is exactly 0 for both Qwen2.5-VL-7B
and gemini-2.5-flash-lite (see e4_multimodel_decoupling.json), which makes the
literal per-item correlation undefined at that tolerance. But E2's secondary
Acc@30% row showed real divergence between the two models (Qwen 0.028, Gemini
0.333) -- this sweep checks every tolerance in between (and a bit beyond) to
find out (a) the lowest tau at which Acc@tau has enough variance to correlate
against SFS at all, and (b) whether that correlation, once computable, ever
resembles the ORIGINAL hypothesized "decoupled" pattern (high accuracy, low
SFS) rather than the confounded negative correlation seen in the continuous
relative-error substitute.

Uses the exact same data loaded by run_e4_multimodel.py (no new inference
calls -- this is purely a recompute over already-collected E2 responses).
"""
import sys
import json
import numpy as np

sys.path.insert(0, "/home/ubuntu/5-ICLR (work with Dr. Mariam)")
from sgf_metrics import scale_faithfulness_score, acc_at_tau, bootstrap_ci

OUT_PATH = "/ephemeral/sgf_work/results/e4/e4_tau_sweep.json"

TAUS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]


def load_triples(path, model_name):
    with open(path) as f:
        r = json.load(f)
    from collections import defaultdict
    groups = defaultdict(dict)
    for x in r:
        groups[(x["clip"], x["seed_idx"])][x["condition"]] = x
    complete = {k: v for k, v in groups.items() if set(v.keys()) == {"base", "0.5", "2"}}
    items = []
    for (clip, seed_idx), conds in complete.items():
        items.append({
            "model": model_name, "clip": clip, "seed_idx": seed_idx,
            "true_velocity_mps": conds["base"]["true_velocity_mps"],
            "base_velocity_mps": conds["base"]["velocity_mps"],
            "half_velocity_mps": conds["0.5"]["velocity_mps"],
            "double_velocity_mps": conds["2"]["velocity_mps"],
        })
    return items


def sfs_avg_for(items):
    base_v = np.array([it["base_velocity_mps"] for it in items])
    half_v = np.array([it["half_velocity_mps"] for it in items])
    dbl_v = np.array([it["double_velocity_mps"] for it in items])
    _, sfs_half = scale_faithfulness_score(base_v, half_v, alpha=0.5, d=1)
    _, sfs_dbl = scale_faithfulness_score(base_v, dbl_v, alpha=2, d=1)
    return (sfs_half + sfs_dbl) / 2.0


def bootstrap_corr(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan"), float("nan")
    r = float(np.corrcoef(a, b)[0, 1])
    n = len(a)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        aa, bb = a[idx], b[idx]
        if np.std(aa) == 0 or np.std(bb) == 0:
            continue
        boots.append(np.corrcoef(aa, bb)[0, 1])
    if not boots:
        return r, float("nan"), float("nan")
    return r, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def sweep_for(items, label):
    base_v = np.array([it["base_velocity_mps"] for it in items])
    true_v = np.array([it["true_velocity_mps"] for it in items])
    sfs_avg = sfs_avg_for(items)

    rows = []
    for tau in TAUS:
        acc_hits = (np.abs(base_v - true_v) / np.abs(true_v) <= tau).astype(float)
        acc_mean, acc_lo, acc_hi = bootstrap_ci(acc_hits)
        if np.std(acc_hits) == 0:
            r, ci_lo, ci_hi, status = float("nan"), float("nan"), float("nan"), "DEGENERATE"
        else:
            r, ci_lo, ci_hi = bootstrap_corr(acc_hits, sfs_avg)
            status = "ok" if not np.isnan(r) else "DEGENERATE"
        rows.append({
            "tau": tau, "acc_at_tau": acc_mean, "acc_ci95": [acc_lo, acc_hi],
            "n_hits": int(acc_hits.sum()), "n": len(acc_hits),
            "pearson_r_acc_vs_sfs": r, "r_ci95": [ci_lo, ci_hi], "status": status,
        })
    return {"label": label, "n": len(items), "sfs_avg_mean": float(sfs_avg.mean()), "rows": rows}


def main():
    qwen_items = load_triples("/ephemeral/sgf_work/results/e2/e2_qwen_results.json", "Qwen2.5-VL-7B-Instruct")
    gem_items = load_triples("/ephemeral/sgf_work/results/e2/e2_gemini_flash_lite_results.json", "gemini-2.5-flash-lite")
    pooled_items = qwen_items + gem_items

    results = {
        "Qwen2.5-VL-7B-Instruct": sweep_for(qwen_items, "Qwen2.5-VL-7B-Instruct"),
        "gemini-2.5-flash-lite": sweep_for(gem_items, "gemini-2.5-flash-lite"),
        "pooled": sweep_for(pooled_items, "pooled (n=72)"),
    }

    # Find, for each, the lowest tau where the correlation first becomes computable.
    for name, res in results.items():
        first_ok = next((r["tau"] for r in res["rows"] if r["status"] == "ok"), None)
        res["first_computable_tau"] = first_ok

    # Check whether gemini's Acc@tau "hits" concentrate in one session (same
    # near-constant-output palette issue as E2/E4/E5) -- if so the loosened-
    # tolerance correlation is still confounded, not a clean coupling signal.
    gem_raw = json.load(open("/ephemeral/sgf_work/results/e2/e2_gemini_flash_lite_results.json"))
    gem_base = [x for x in gem_raw if x["condition"] == "base"]
    from collections import Counter
    hit_sessions = Counter(x["session"] for x in gem_base
                            if abs(x["velocity_mps"] - x["true_velocity_mps"]) / x["true_velocity_mps"] <= 0.30)
    results["gemini-2.5-flash-lite"]["acc30_hit_session_breakdown"] = dict(hit_sessions)
    results["gemini-2.5-flash-lite"]["acc30_confound_note"] = (
        "All Acc@30% 'hits' for gemini come from a single session (05052026), whose true "
        "velocities (~1.1-1.2 m/s) happen to cluster near gemini's favorite near-constant output "
        "(~1.4-1.5 m/s). This is the same near-constant-response palette issue documented in "
        "project_sgf_e2_finding.md / e4_multimodel_decoupling.json, not evidence of broader "
        "measurement ability. The negative tau-sweep correlation should NOT be reported as a "
        "clean decoupling/coupling signal for this reason."
    )

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    for name, res in results.items():
        print(f"\n=== {name} (n={res['n']}, mean SFS={res['sfs_avg_mean']:.3f}) ===")
        print(f"{'tau':>6s} {'Acc@tau':>9s} {'n_hits':>7s} {'r(acc,sfs)':>12s} {'status':>12s}")
        for row in res["rows"]:
            r_str = f"{row['pearson_r_acc_vs_sfs']:.3f}" if not np.isnan(row["pearson_r_acc_vs_sfs"]) else "nan"
            print(f"{row['tau']:6.2f} {row['acc_at_tau']:9.3f} {row['n_hits']:7d} {r_str:>12s} {row['status']:>12s}")
        print(f"first computable at tau={res['first_computable_tau']}")

    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
