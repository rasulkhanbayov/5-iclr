"""
Build the raw-per-item CSV that sgf_metrics.py --repro expects, from this
project's actual result JSON files. Resolves R4 concretely: assembles the
single file needed to verify every table cell against source data.

Reads the result JSONs under results/ (next to this file) and writes
results/repro_raw_outputs.csv, the input reconcile.py reads. The built CSV is
committed, so you only need to rerun this after regenerating results.

CORRECTED 2026-08-22: TRUE_CAL was 1.0 (assumed 1mm/tick); the 02182026
session's ruler is graduated in 1/16 inch (1.5875mm/tick), confirmed by
direct pixel measurement -- see the paper's Appendix B.2 and
code/oracle/droplet_oracle.py. E2 and X4 never told the model
the calibration value, so their raw responses are unchanged, but CG and
Acc@tau must be scored against the corrected constant -- hence MODELS_E2 and
MODELS_X4 below point at the *_rescored.json siblings, not the base files.

Schema (per sgf_metrics.py's RECONCILE_SPEC):
  model, clip_id, seed, relation, magnitude, family, q_hat, c_hat, c_true,
  q_oracle, oracle_conf, parsed
"""
import json
import csv
import os

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_CSV = os.path.join(REPO, "repro_raw_outputs.csv")

TRUE_CAL = 1.5875  # 02182026 session, mm per tick (1/16 inch)

MODELS_E2 = {
    "Qwen2.5-VL-7B-Instruct": f"{REPO}/e2/e2_qwen_results_rescored.json",
    "Gemini-3.1-Pro-Preview": f"{REPO}/e2/e2_frontier_results_rescored.json",
    "Qwen3-VL-30B-A3B-Instruct": f"{REPO}/e2/e2_qwen3vl_results_rescored.json",
}

MODELS_X4 = {
    "Qwen2.5-VL-7B-Instruct": f"{REPO}/e4_resampling/x4_qwen_results_rescored.json",
    "Gemini-3.1-Pro-Preview": f"{REPO}/e4_resampling/x4_frontier_results_rescored.json",
    "Qwen3-VL-30B-A3B-Instruct": f"{REPO}/e4_resampling/x4_qwen3vl_results_rescored.json",
}

TEMPORAL_PATH = f"{REPO}/e2/temporal_qwen_results.json"

MODELS_T1_RELABEL = {
    "Qwen2.5-VL-7B-Instruct": f"{REPO}/t1/t1_textcal_relabel_qwen_results.json",
    "Gemini-3.1-Pro-Preview": f"{REPO}/t1/t1_textcal_relabel_frontier_results.json",
    "Qwen3-VL-30B-A3B-Instruct": f"{REPO}/t1/t1_textcal_relabel_qwen3vl_results.json",
}

MODELS_T1_RESAMPLE = {
    "Qwen2.5-VL-7B-Instruct": f"{REPO}/t1/t1_textcal_resample_qwen_results.json",
    "Gemini-3.1-Pro-Preview": f"{REPO}/t1/t1_textcal_resample_frontier_results.json",
    "Qwen3-VL-30B-A3B-Instruct": f"{REPO}/t1/t1_textcal_resample_qwen3vl_results.json",
}

# oracle confidence lookup, built from the manifest used across all E2 runs
MANIFEST_PATH = os.path.join(REPO, "e2_selected_clips.json")


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    # All 27 clips in this manifest are high-confidence by construction (the
    # manifest is built from ONLY the high-confidence oracle tracks; see
    # build_e2_manifest.py and the main paper's Dataset section).
    conf_by_clip = {os.path.basename(c["path"]): "high" for c in manifest}

    rows = []

    # E2: relabeling R_alpha, all three models, all 3 conditions (base + 2 relabeled)
    for model, path in MODELS_E2.items():
        data = load(path)
        for d in data:
            clip = d["clip"]
            cond = d["condition"]
            alpha = 1.0 if cond == "base" else float(cond)
            relation = "base" if cond == "base" else "R"
            magnitude = 1.0 if cond == "base" else alpha
            parsed = 0 if (d.get("parse_failed") or "error" in d) else 1
            rows.append({
                "model": model, "clip_id": clip, "seed": d.get("seed_idx", ""),
                "relation": relation, "magnitude": magnitude, "family": "spatial",
                "q_hat": d.get("velocity_mps", ""),
                "c_hat": d.get("calibration_mm_per_tick", ""),
                "c_true": TRUE_CAL * alpha,
                "q_oracle": d.get("true_velocity_mps", ""),
                "oracle_conf": conf_by_clip.get(clip, "high"),
                "parsed": parsed,
            })

    # X4: resampling S_s, all three models
    for model, path in MODELS_X4.items():
        data = load(path)
        for d in data:
            clip = d["clip"]
            s = float(d["s"])
            parsed = 0 if (d.get("parse_failed") or "error" in d) else 1
            rows.append({
                "model": model, "clip_id": clip, "seed": d.get("seed_idx", ""),
                "relation": "S", "magnitude": s, "family": "spatial",
                "q_hat": d.get("velocity_mps", ""),
                "c_hat": d.get("calibration_mm_per_tick", ""),
                "c_true": TRUE_CAL,  # S_s holds the asserted calibration fixed
                "q_oracle": d.get("true_velocity_mps", ""),
                "oracle_conf": conf_by_clip.get(clip, "high"),
                "parsed": parsed,
            })

    # Temporal T_beta, Qwen2.5-VL only
    data = load(TEMPORAL_PATH)
    for d in data:
        clip = d["clip"]
        beta = d.get("beta", 1.0)
        cond = d["condition"]
        relation = "base" if cond == "base" else "T"
        parsed = 0 if (d.get("parse_failed") or "error" in d) else 1
        rows.append({
            "model": "Qwen2.5-VL-7B-Instruct", "clip_id": clip, "seed": d.get("seed_idx", ""),
            "relation": relation, "magnitude": beta, "family": "temporal",
            "q_hat": d.get("velocity_mps", ""),
            "c_hat": d.get("asserted_calibration_mm_per_tick", TRUE_CAL),
            "c_true": TRUE_CAL,
            "q_oracle": d.get("true_velocity_mps", ""),
            "oracle_conf": conf_by_clip.get(clip, "high"),
            "parsed": parsed,
        })

    # T1 positive control: R_alpha half (text-calibration relabeling)
    for model, path in MODELS_T1_RELABEL.items():
        data = load(path)
        for d in data:
            clip = d["clip"]
            cond = d["condition"]
            alpha = d.get("alpha", 1.0 if cond == "base" else float(cond))
            relation = "base" if cond == "base" else "R"
            parsed = 0 if (d.get("parse_failed") or "error" in d) else 1
            rows.append({
                "model": model, "clip_id": clip, "seed": d.get("seed_idx", ""),
                "relation": relation, "magnitude": alpha, "family": "t1",
                "q_hat": d.get("velocity_mps", ""),
                "c_hat": d.get("asserted_calibration_mm_per_tick", TRUE_CAL * alpha),
                "c_true": TRUE_CAL * alpha,
                "q_oracle": d.get("true_velocity_mps", ""),
                "oracle_conf": conf_by_clip.get(clip, "high"),
                "parsed": parsed,
            })

    # T1 positive control: S_s half (text-calibration resampling)
    for model, path in MODELS_T1_RESAMPLE.items():
        data = load(path)
        for d in data:
            clip = d["clip"]
            s = float(d["s"])
            parsed = 0 if (d.get("parse_failed") or "error" in d) else 1
            rows.append({
                "model": model, "clip_id": clip, "seed": d.get("seed_idx", ""),
                "relation": "S", "magnitude": s, "family": "t1",
                "q_hat": d.get("velocity_mps", ""),
                "c_hat": d.get("asserted_calibration_mm_per_tick", TRUE_CAL),
                "c_true": TRUE_CAL,
                "q_oracle": d.get("true_velocity_mps", ""),
                "oracle_conf": conf_by_clip.get(clip, "high"),
                "parsed": parsed,
            })

    fieldnames = ["model", "clip_id", "seed", "relation", "magnitude", "family",
                  "q_hat", "c_hat", "c_true", "q_oracle", "oracle_conf", "parsed"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_CSV}, {len(rows)} rows")
    from collections import Counter
    print("rows per model:", Counter(r["model"] for r in rows))
    print("rows per relation:", Counter(r["relation"] for r in rows))


if __name__ == "__main__":
    main()
