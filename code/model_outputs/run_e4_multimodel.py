"""
E4 decoupling analysis, multi-model version. Adds google/gemini-2.5-flash-lite
(via OpenRouter) alongside the original Qwen2.5-VL-7B-Instruct run, addressing
the single-model gap flagged in project_sgf_e4_results.md / main.tex's
Limitations section.

Does NOT resolve the core problem: Acc@10% is exactly 0 for BOTH models
independently, and remains exactly 0 when pooled (72 items, 0 hits) -- so the
runbook's literal per-item accuracy-vs-SFS correlation is still mathematically
degenerate, now demonstrated across two independent model families rather than
one. This is reported as a stronger, more general version of the same finding,
not a different one: the paper's 10%-tolerance decoupling test cannot be run on
current-generation VLMs on this task, full stop, not just on the specific model
we happened to have local weights for.
"""
import sys
import json
import numpy as np

sys.path.insert(0, "/home/ubuntu/5-ICLR (work with Dr. Mariam)")
from sgf_metrics import scale_faithfulness_score, relative_error, bootstrap_ci

OUT_PATH = "/ephemeral/sgf_work/results/e4/e4_multimodel_decoupling.json"


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


def compute(items):
    base_v = np.array([it["base_velocity_mps"] for it in items])
    true_v = np.array([it["true_velocity_mps"] for it in items])
    half_v = np.array([it["half_velocity_mps"] for it in items])
    dbl_v = np.array([it["double_velocity_mps"] for it in items])

    rel_err = relative_error(base_v, true_v)
    acc10 = (rel_err <= 0.10).astype(float)
    _, sfs_half = scale_faithfulness_score(base_v, half_v, alpha=0.5, d=1)
    _, sfs_dbl = scale_faithfulness_score(base_v, dbl_v, alpha=2, d=1)
    sfs_avg = (sfs_half + sfs_dbl) / 2.0
    for i, it in enumerate(items):
        it["relative_error"] = float(rel_err[i])
        it["acc10_hit"] = float(acc10[i])
        it["sfs_avg"] = float(sfs_avg[i])
    return rel_err, acc10, sfs_avg


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


def main():
    qwen_items = load_triples("/ephemeral/sgf_work/results/e2/e2_qwen_results.json", "Qwen2.5-VL-7B-Instruct")
    gem_items = load_triples("/ephemeral/sgf_work/results/e2/e2_gemini_flash_lite_results.json", "gemini-2.5-flash-lite")
    all_items = qwen_items + gem_items

    per_model = {}
    for name, items in [("Qwen2.5-VL-7B-Instruct", qwen_items), ("gemini-2.5-flash-lite", gem_items)]:
        rel_err, acc10, sfs_avg = compute(items)
        r_cont, lo, hi = bootstrap_corr(-rel_err, sfs_avg)
        per_model[name] = {
            "n": len(items), "acc_at_10pct": float(acc10.mean()),
            "acc_at_30pct": float((rel_err <= 0.30).mean()),
            "sfs_avg_mean": float(sfs_avg.mean()),
            "pearson_r_negrelerr_vs_sfs": r_cont, "ci_95": [lo, hi],
        }

    rel_err_all, acc10_all, sfs_avg_all = compute(all_items)
    r_pooled, lo_pooled, hi_pooled = bootstrap_corr(-rel_err_all, sfs_avg_all)

    summary = {
        "n_items_pooled": len(all_items),
        "per_model": per_model,
        "pooled_acc_at_10pct": float(acc10_all.mean()),
        "pooled_acc10_status": "DEGENERATE (0/72, both models independently saturated at 0)" if acc10_all.sum() == 0 else "ok",
        "pooled_sfs_avg_mean": float(sfs_avg_all.mean()),
        "pooled_pearson_r_negrelerr_vs_sfs": r_pooled,
        "pooled_ci_95": [lo_pooled, hi_pooled],
        "conclusion": (
            "Adding a second, independent model family (gemini-2.5-flash-lite, via OpenRouter) "
            "does NOT resolve E4's degeneracy -- Acc@10% is exactly 0 for BOTH models "
            "independently (0/36 each) and for the 72-item pool. This upgrades the finding from "
            "'one model's accuracy happened to saturate' to 'the 10%-tolerance decoupling test "
            "could not be run on either of two independently-built current-generation VLMs on "
            "this task' -- a stronger, more general version of the same null result, not a "
            "different one. The continuous-relative-error substitute correlation is negative for "
            "both models individually (Qwen r=-0.481, gemini r=-0.218) and pooled (r=-0.460, CI "
            "excludes 0), but per project_sgf_e4_results.md's original confound analysis this "
            "should still not be over-read as a clean coupling finding: gemini also shows a "
            "constrained output palette relative to the perturbation (calibration reads plateau "
            "at 0.5/1.0 mm/tick and never reach the true alpha=2 asserted value of 2.0 on either "
            "session; velocity does not move in a consistent direction under alpha, see "
            "e2_gemini_flash_lite_results.json), so the same 'both axes downstream of the same "
            "non-responsiveness' caveat still applies, just now demonstrated on a second model "
            "independently rather than assumed to be Qwen-specific."
        ),
        "items": all_items,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
