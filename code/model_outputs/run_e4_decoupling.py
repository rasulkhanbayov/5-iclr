"""
E4 decoupling analysis (sgf_runbook.txt): correlate per-item Acc@10% with
per-item SFS across E2's Qwen2.5-VL-7B results. Objective per the runbook is
literally "correlate Acc@10% with SFS across all items/models" -- with only
one model run (Qwen2.5-VL-7B; no API keys available in this environment, see
CLAUDE.md), "across models" degenerates to "across items" here. Flagging
that explicitly in the output rather than silently treating it as multi-model.

Per-item unit: one (clip, seed_idx) triple (base, alpha=0.5, alpha=2) from
e2_qwen_results.json -- confirmed all 36 such groups are complete with no
parse failures (see conversation). For each item:
  - Acc@10% hit: 1 if the base-condition velocity is within 10% of the
    oracle's true velocity, else 0 (this is what acc_at_tau averages over
    normally; E4 needs the per-item indicator, not the aggregate).
  - SFS: sgf_metrics.scale_faithfulness_score's per_item output, computed
    once for alpha=0.5 and once for alpha=2 (each needs its own call since
    the metric's target log(alpha) differs), then averaged per item -- one
    SFS number per item, matching one Acc@10% number per item, for a valid
    paired correlation.

IMPORTANT HONESTY RULE (runbook, verbatim): the paper's headline hypothesis
is that accuracy and faithfulness are DECOUPLED. Report whatever the data
show -- do not keep a "decoupled" framing if the correlation says otherwise.
"""
import sys
import json
import numpy as np

sys.path.insert(0, "/home/ubuntu/5-ICLR (work with Dr. Mariam)")
from sgf_metrics import scale_faithfulness_score, acc_at_tau, relative_error, bootstrap_ci

RESULTS_PATH = "/ephemeral/sgf_work/results/e2/e2_qwen_results.json"
OUT_PATH = "/ephemeral/sgf_work/results/e4/e4_decoupling.json"


def build_items(results):
    groups = {}
    for r in results:
        key = (r["clip"], r["seed_idx"])
        groups.setdefault(key, {})[r["condition"]] = r

    items = []
    for (clip, seed_idx), conds in groups.items():
        if set(conds.keys()) != {"base", "0.5", "2"}:
            continue
        if any(conds[c]["parse_failed"] for c in ("base", "0.5", "2")):
            continue
        base, half, dbl = conds["base"], conds["0.5"], conds["2"]
        items.append({
            "clip": clip, "seed_idx": seed_idx, "session": base["session"],
            "true_velocity_mps": base["true_velocity_mps"],
            "base_velocity_mps": base["velocity_mps"],
            "scaled_0.5_velocity_mps": half["velocity_mps"],
            "scaled_2_velocity_mps": dbl["velocity_mps"],
        })
    return items


def compute_per_item_metrics(items):
    true_v = np.array([it["true_velocity_mps"] for it in items])
    base_v = np.array([it["base_velocity_mps"] for it in items])
    half_v = np.array([it["scaled_0.5_velocity_mps"] for it in items])
    dbl_v = np.array([it["scaled_2_velocity_mps"] for it in items])

    rel_err = relative_error(base_v, true_v)
    acc_hits = (rel_err <= 0.10).astype(float)

    _, sfs_half_per_item = scale_faithfulness_score(base_v, half_v, alpha=0.5, d=1)
    _, sfs_dbl_per_item = scale_faithfulness_score(base_v, dbl_v, alpha=2, d=1)
    sfs_avg_per_item = (sfs_half_per_item + sfs_dbl_per_item) / 2.0

    for i, it in enumerate(items):
        it["relative_error"] = float(rel_err[i])
        it["acc10_hit"] = float(acc_hits[i])
        it["sfs_alpha0.5"] = float(sfs_half_per_item[i])
        it["sfs_alpha2"] = float(sfs_dbl_per_item[i])
        it["sfs_avg"] = float(sfs_avg_per_item[i])

    return acc_hits, sfs_avg_per_item, rel_err


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    items = build_items(results)
    n_dropped = len({(r["clip"], r["seed_idx"]) for r in results}) - len(items)

    acc_hits, sfs_avg, rel_err = compute_per_item_metrics(items)
    n = len(items)

    def bootstrap_corr(a, b, n_boot=10000, seed=0):
        rng = np.random.default_rng(seed)
        if np.std(a) == 0 or np.std(b) == 0:
            return float("nan"), float("nan"), float("nan")
        r = float(np.corrcoef(a, b)[0, 1])
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

    # Primary Acc@10% correlation: DEGENERATE. Every single item misses the
    # 10% tolerance (matches E2's headline Acc@10%=0.0) -- acc_hits is a
    # constant zero vector, so Pearson r is mathematically undefined, not
    # merely "small" or "zero." Reporting this honestly rather than
    # substituting a different threshold silently.
    r_acc10, ci_acc10_lo, ci_acc10_hi = bootstrap_corr(acc_hits, sfs_avg)

    # Since Acc@10% has no variance to correlate against, the runbook's
    # actual question ("does accuracy correlate with faithfulness") is
    # answered using the CONTINUOUS relative error instead (negated, so
    # higher = more accurate, for interpretability alongside SFS where
    # higher = more faithful). This is the closest computable analogue of
    # the intended analysis given this model's saturated Acc@10%.
    neg_rel_err = -rel_err
    r_continuous, ci_cont_lo, ci_cont_hi = bootstrap_corr(neg_rel_err, sfs_avg)

    overall_acc10 = float(np.mean(acc_hits))
    overall_sfs = float(np.mean(sfs_avg))

    summary = {
        "n_items": n,
        "n_items_dropped_incomplete_or_parse_failed": n_dropped,
        "unit_of_analysis": "(clip, seed_idx) triple, base/alpha=0.5/alpha=2 conditions, Qwen2.5-VL-7B-Instruct only",
        "caveat_models": ("Runbook's E4 objective is 'across all items/models'; only one model "
                          "(Qwen2.5-VL-7B-Instruct) was run in this environment (no API keys "
                          "available for closed models, see CLAUDE.md), so this is a "
                          "within-model, across-item correlation, not across-model. Should be "
                          "re-run once additional models are added."),
        "overall_acc_at_10pct": overall_acc10,
        "overall_sfs_avg": overall_sfs,
        "acc10_vs_sfs": {
            "pearson_r": r_acc10, "ci_95": [ci_acc10_lo, ci_acc10_hi],
            "status": "DEGENERATE" if np.std(acc_hits) == 0 else "ok",
            "caveat": ("acc10_hit is a constant (all 0) across all 36 items -- every single "
                       "item misses the 10% tolerance, matching E2's Acc@10%=0.0 finding. "
                       "Correlation with a zero-variance variable is mathematically "
                       "undefined (NaN), not 'zero' or 'no correlation' -- there is no "
                       "accuracy signal to decouple FROM at this threshold."),
        },
        "neg_relative_error_vs_sfs": {
            "pearson_r": r_continuous, "ci_95": [ci_cont_lo, ci_cont_hi],
            "caveat": ("Since Acc@10% is saturated at 0, this uses continuous relative "
                       "error (negated: higher = more accurate) as the closest computable "
                       "stand-in for 'accuracy' with actual variance across items."),
            "IMPORTANT_confound": ("This correlation (r=-0.48, CI excludes 0) is NOT clean "
                       "evidence of a genuine accuracy-faithfulness relationship. Per "
                       "project_sgf_e2_finding.md, base_velocity_mps only takes ~3 discrete "
                       "values (0.45-0.5, 1.5, 3.5) that are almost entirely determined by "
                       "seed_idx, not by clip content -- e.g. seed_idx=1 outputs 3.5 m/s for "
                       "EVERY clip. 'relative_error' here is therefore mostly measuring how "
                       "close each clip's true velocity happens to sit to whichever "
                       "near-constant value that prompt seed elicits, not the model actually "
                       "measuring more or less accurately. The negative correlation likely "
                       "reflects that the same near-constant-response pathology also flattens "
                       "SFS (a model that ignores clip content also tends to under-respond to "
                       "the scale perturbation) rather than a real tradeoff between being "
                       "accurate and being faithful. Any 'accuracy vs faithfulness' claim from "
                       "this analysis is confounded by E2's near-constant-response finding and "
                       "should not be reported as clean support for either 'coupled' or "
                       "'decoupled' framing without this caveat."),
        },
        "items": items,
    }

    import os
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"n_items = {n} (dropped {n_dropped})")
    print(f"overall Acc@10%% = {overall_acc10:.3f}")
    print(f"overall SFS (avg alpha=0.5/2) = {overall_sfs:.3f}")
    print(f"Pearson r(Acc@10% hit, SFS) = {r_acc10} (DEGENERATE: acc10 is a constant)")
    print(f"Pearson r(-relative_error, SFS) = {r_continuous:.3f}  95% CI [{ci_cont_lo:.3f}, {ci_cont_hi:.3f}]")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
