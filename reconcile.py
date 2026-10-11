"""
Reconciliation script for the Code and Data Supplement: recomputes every
table cell and headline number in the paper from results/repro_raw_outputs.csv
(built from the raw per-item result JSONs) and prints a MATCH/MISMATCH verdict
against the value written in the paper. This is the concrete implementation
of what sgf_metrics.py's --repro flag describes but does not itself execute
(as of this writing that flag only prints the required CSV schema).

Usage: python3 reconcile.py
"""
import csv
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from sgf_metrics import wilson, binom_test_two_sided, clopper_pearson

CSV_PATH = os.path.join(_HERE, "results", "repro_raw_outputs.csv")

# (description, expected value from the paper, tolerance)
CHECKS = []


def load_rows():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def check(desc, actual, expected, tol=0.001, fmt="{:.4f}"):
    ok = abs(actual - expected) <= tol
    verdict = "MATCH" if ok else "MISMATCH"
    print(f"[{verdict}] {desc}: computed={fmt.format(actual)} paper={fmt.format(expected)}")
    return ok


def main():
    rows = load_rows()
    all_ok = True

    for model in ["Qwen2.5-VL-7B-Instruct", "Gemini-3.1-Pro-Preview", "Qwen3-VL-30B-A3B-Instruct"]:
        model_rows = [r for r in rows if r["model"] == model and r["relation"] in ("base", "R")
                      and r["family"] == "spatial"]
        e2_rows = [r for r in model_rows if r["parsed"] == "1" and r["c_hat"]]
        hits = sum(1 for r in e2_rows
                   if abs(float(r["c_hat"]) - float(r["c_true"])) / float(r["c_true"]) <= 0.10)
        n = len(e2_rows)
        cg = hits / n if n else float("nan")
        print(f"\n=== {model}: CG over all E2 items (base + relabeled) ===")
        print(f"  n={n}, hits={hits}, CG={cg:.4f}")

    expected_cg = {
        "Qwen2.5-VL-7B-Instruct": (0, 243, 0.000),
        "Gemini-3.1-Pro-Preview": (1, 240, 0.004),
        "Qwen3-VL-30B-A3B-Instruct": (0, 243, 0.000),
    }
    print("\n=== CG reconciliation against Table 1 ===")
    for model, (exp_hits, exp_n, exp_cg) in expected_cg.items():
        model_rows = [r for r in rows if r["model"] == model and r["relation"] in ("base", "R")
                      and r["family"] == "spatial" and r["parsed"] == "1" and r["c_hat"]]
        hits = sum(1 for r in model_rows
                   if abs(float(r["c_hat"]) - float(r["c_true"])) / float(r["c_true"]) <= 0.10)
        n = len(model_rows)
        ok_n = check(f"{model}: n", n, exp_n, tol=0, fmt="{:.0f}")
        ok_hits = check(f"{model}: hits", hits, exp_hits, tol=0, fmt="{:.0f}")
        ok_cg = check(f"{model}: CG", hits / n if n else float("nan"), exp_cg, tol=0.001)
        all_ok = all_ok and ok_n and ok_hits and ok_cg

    print("\n=== Acc@10% reconciliation against Table 1 (base condition only) ===")
    expected_acc10 = {
        "Qwen2.5-VL-7B-Instruct": (0, 81, 0.000),
        "Gemini-3.1-Pro-Preview": (8, 80, 0.100),
        "Qwen3-VL-30B-A3B-Instruct": (0, 81, 0.000),
    }
    for model, (exp_hits, exp_n, exp_acc) in expected_acc10.items():
        base_rows = [r for r in rows if r["model"] == model and r["relation"] == "base"
                     and r["family"] == "spatial" and r["parsed"] == "1" and r["q_hat"]]
        hits = sum(1 for r in base_rows
                   if abs(float(r["q_hat"]) - float(r["q_oracle"])) / float(r["q_oracle"]) <= 0.10)
        n = len(base_rows)
        ok_n = check(f"{model}: n", n, exp_n, tol=0, fmt="{:.0f}")
        ok_hits = check(f"{model}: hits", hits, exp_hits, tol=0, fmt="{:.0f}")
        ok_acc = check(f"{model}: Acc@10%", hits / n if n else float("nan"), exp_acc, tol=0.001)
        all_ok = all_ok and ok_n and ok_hits and ok_acc

    print("\n=== Acc@30% reconciliation against Table 1 (base condition only) ===")
    expected_acc30 = {
        "Qwen2.5-VL-7B-Instruct": 0.074,
        "Gemini-3.1-Pro-Preview": 0.313,
        "Qwen3-VL-30B-A3B-Instruct": 0.037,
    }
    for model, exp_acc in expected_acc30.items():
        base_rows = [r for r in rows if r["model"] == model and r["relation"] == "base"
                     and r["family"] == "spatial" and r["parsed"] == "1" and r["q_hat"]]
        hits = sum(1 for r in base_rows
                   if abs(float(r["q_hat"]) - float(r["q_oracle"])) / float(r["q_oracle"]) <= 0.30)
        n = len(base_rows)
        ok_acc = check(f"{model}: Acc@30%", hits / n if n else float("nan"), exp_acc, tol=0.001)
        all_ok = all_ok and ok_acc

    print("\n=== Mean response ratios (r_R, r_S) against Table 1 ===")
    expected_ratios = {
        ("Qwen2.5-VL-7B-Instruct", "R", 0.5): 0.794,
        ("Qwen2.5-VL-7B-Instruct", "R", 2.0): 0.808,
        ("Qwen2.5-VL-7B-Instruct", "S", 0.5): 0.911,
        ("Qwen2.5-VL-7B-Instruct", "S", 2.0): 0.808,
        ("Gemini-3.1-Pro-Preview", "R", 0.5): 1.610,
        ("Gemini-3.1-Pro-Preview", "R", 2.0): 1.653,
        ("Gemini-3.1-Pro-Preview", "S", 0.5): 1.474,
        ("Gemini-3.1-Pro-Preview", "S", 2.0): 1.021,
        ("Qwen3-VL-30B-A3B-Instruct", "R", 0.5): 1.059,
        ("Qwen3-VL-30B-A3B-Instruct", "R", 2.0): 1.048,
        ("Qwen3-VL-30B-A3B-Instruct", "S", 0.5): 0.939,
        ("Qwen3-VL-30B-A3B-Instruct", "S", 2.0): 1.588,
        ("Qwen2.5-VL-7B-Instruct", "T", 0.5): 0.683,
        ("Qwen2.5-VL-7B-Instruct", "T", 2.0): 4.569,
    }
    # Keyed by (model, family, clip_id, seed): E2/X4's "base" and temporal's
    # "base" are DIFFERENT elicitations for the same clip/seed (different
    # prompt, different or absent scale video) and must never be conflated
    # when pairing a relabeled/resampled row with its own base row.
    base_by_key = {}
    for r in rows:
        if r["relation"] == "base" and r["parsed"] == "1" and r["q_hat"]:
            base_by_key[(r["model"], r["family"], r["clip_id"], r["seed"])] = float(r["q_hat"])

    # Table 1's r_R/r_S/r_T are all from the SPATIAL family (E2/X4) except
    # the T rows, which are their own "temporal" family; T1's R/S rows share
    # the same relation labels and magnitudes but belong to family "t1" and
    # must be excluded here, or reconciled separately (see T1 section below).
    ratio_family = {"R": "spatial", "S": "spatial", "T": "temporal"}
    for (model, relation, magnitude), expected in expected_ratios.items():
        magnitude_rows = [r for r in rows if r["model"] == model and r["relation"] == relation
                          and r["family"] == ratio_family[relation]
                          and abs(float(r["magnitude"]) - magnitude) < 1e-9
                          and r["parsed"] == "1" and r["q_hat"]]
        ratios = []
        for r in magnitude_rows:
            key = (r["model"], r["family"], r["clip_id"], r["seed"])
            if key in base_by_key and base_by_key[key]:
                ratios.append(float(r["q_hat"]) / base_by_key[key])
        mean_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        ok = check(f"{model} {relation}_{magnitude}: mean ratio", mean_ratio, expected, tol=0.001)
        all_ok = all_ok and ok

    print("\n=== T1 (text-calibration positive control) against Appendix B.3's per-model detail ===")
    # (model, relation, magnitude): (expected_mean, expected_median, tol)
    # Base rows for T1 belong to family "t1" (a third, distinct elicitation
    # from both the E2/X4 "spatial" base and the temporal "temporal" base).
    # All three models are checked here (Qwen3-VL included): the paper's
    # current framing is "no model produced the predicted (alpha,1)
    # signature," reported via both mean and median for every model,
    # including Qwen3-VL's near-miss (median R_alpha=2 lands exactly on the
    # grounded target 2.0, but the paired S_s median is too far from 1.0 to
    # classify as grounded).
    expected_t1 = {
        ("Qwen2.5-VL-7B-Instruct", "R", 0.5): (5.2625, 4.2540),
        ("Qwen2.5-VL-7B-Instruct", "R", 2.0): (5.7492, 4.0317),
        ("Qwen2.5-VL-7B-Instruct", "S", 0.5): (1.2517, 1.0000),
        ("Qwen2.5-VL-7B-Instruct", "S", 2.0): (2.1467, 1.0000),
        ("Gemini-3.1-Pro-Preview", "R", 0.5): (2.5250, 1.2680),
        ("Gemini-3.1-Pro-Preview", "R", 2.0): (1.6201, 1.2500),
        ("Gemini-3.1-Pro-Preview", "S", 0.5): (1.6026, 1.0000),
        ("Gemini-3.1-Pro-Preview", "S", 2.0): (2.1657, 1.2700),
        ("Qwen3-VL-30B-A3B-Instruct", "R", 0.5): (0.9384, 0.8000),
        ("Qwen3-VL-30B-A3B-Instruct", "R", 2.0): (4.4337, 2.0000),
        ("Qwen3-VL-30B-A3B-Instruct", "S", 0.5): (2.3674, 1.0000),
        ("Qwen3-VL-30B-A3B-Instruct", "S", 2.0): (1.5072, 1.5118),
    }
    for (model, relation, magnitude), (exp_mean, exp_median) in expected_t1.items():
        magnitude_rows = [r for r in rows if r["model"] == model and r["relation"] == relation
                          and r["family"] == "t1"
                          and abs(float(r["magnitude"]) - magnitude) < 1e-9
                          and r["parsed"] == "1" and r["q_hat"]]
        ratios = []
        for r in magnitude_rows:
            key = (r["model"], "t1", r["clip_id"], r["seed"])
            if key in base_by_key and base_by_key[key]:
                ratios.append(float(r["q_hat"]) / base_by_key[key])
        if ratios:
            mean_ratio = sum(ratios) / len(ratios)
            sorted_r = sorted(ratios)
            n = len(sorted_r)
            median_ratio = sorted_r[n // 2] if n % 2 else (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2
        else:
            mean_ratio = median_ratio = float("nan")
        ok_mean = check(f"T1 {model} {relation}_{magnitude}: mean", mean_ratio, exp_mean, tol=0.001)
        ok_median = check(f"T1 {model} {relation}_{magnitude}: median", median_ratio, exp_median, tol=0.001)
        all_ok = all_ok and ok_mean and ok_median

    print("\n=== T1 zero-velocity rates against Appendix B.3 ===")
    # (model, relation): expected rate. R_alpha's rate is over the full
    # relabel set (base + R, family "t1"), matching how the main evaluation's
    # own zero-rate is computed for comparison; S_s has no base row of its
    # own in this family, so its rate is over the 162 perturbed items alone.
    expected_zero_rate = {
        ("Qwen2.5-VL-7B-Instruct", "R"): 0.000,
        ("Qwen2.5-VL-7B-Instruct", "S"): 0.019,
        ("Gemini-3.1-Pro-Preview", "R"): 0.000,
        ("Gemini-3.1-Pro-Preview", "S"): 0.012,
        ("Qwen3-VL-30B-A3B-Instruct", "R"): 0.012,
        ("Qwen3-VL-30B-A3B-Instruct", "S"): 0.080,
    }
    for (model, relation), exp_rate in expected_zero_rate.items():
        include = (relation, "base") if relation == "R" else (relation,)
        t1_rows = [r for r in rows if r["model"] == model and r["family"] == "t1"
                   and r["relation"] in include and r["parsed"] == "1" and r["q_hat"]]
        n_zero = sum(1 for r in t1_rows if float(r["q_hat"]) == 0.0)
        rate = n_zero / len(t1_rows) if t1_rows else float("nan")
        ok_rate = check(f"{model} T1 {relation} zero-velocity rate", rate, exp_rate, tol=0.005, fmt="{:.3f}")
        all_ok = all_ok and ok_rate

    print(f"\n{'='*72}")
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- see MISMATCH lines above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
