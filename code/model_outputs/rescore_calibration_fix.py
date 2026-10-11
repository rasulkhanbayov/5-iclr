"""
Re-score E2 (main evaluation) and X4 (resampling) for all three models
against the CORRECTED calibration constant (1.5875 mm/tick, 1/16", not the
original wrong 1.0 mm/tick assumption -- see droplet_oracle.py) and the
corrected oracle ground-truth velocities in the regenerated
e2_selected_clips.json manifest.

Model responses themselves are UNCHANGED: E2 and X4 never told the model the
calibration value (it reads the scale-reference video itself), so only the
scoring changes, not the raw_response/velocity_mps/calibration_mm_per_tick
fields already recorded. This script recomputes CG (against the corrected
true asserted calibration per condition) and Acc@tau (against the corrected
oracle velocity per clip) and writes a *_rescored.json sitting alongside each
original file, leaving the original untouched as a record of what was
actually asked/answered.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)

RESULTS_DIR = sgf_paths.RESULTS_DIR

TRUE_CAL = 1.5875  # 1/16", CORRECTED 2026-08-22

MANIFEST_PATH = sgf_paths.MANIFEST_PATH


def load_manifest_velocities():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    return {os.path.basename(c["path"]): c["velocity"]["v_pre_impact_mps"] for c in manifest}


def rescore_e2_file(path, true_vs):
    with open(path) as f:
        data = json.load(f)
    n_updated = 0
    for d in data:
        if d.get("parse_failed") or "error" in d:
            continue
        cond = d["condition"]
        alpha = 1.0 if cond == "base" else float(cond)
        d["true_velocity_mps_ORIGINAL"] = d.get("true_velocity_mps")
        d["asserted_calibration_mm_per_tick_ORIGINAL"] = d.get("asserted_calibration_mm_per_tick")
        clip = d["clip"]
        if clip in true_vs:
            d["true_velocity_mps"] = true_vs[clip]
        d["asserted_calibration_mm_per_tick"] = TRUE_CAL * alpha
        n_updated += 1
    out_path = path.replace(".json", "_rescored.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {out_path}, {n_updated} entries rescored")
    return out_path


def rescore_x4_file(path, true_vs):
    with open(path) as f:
        data = json.load(f)
    n_updated = 0
    for d in data:
        if d.get("parse_failed") or "error" in d:
            continue
        d["true_velocity_mps_ORIGINAL"] = d.get("true_velocity_mps")
        d["asserted_calibration_mm_per_tick_ORIGINAL"] = d.get("asserted_calibration_mm_per_tick")
        clip = d["clip"]
        if clip in true_vs:
            d["true_velocity_mps"] = true_vs[clip]
        d["asserted_calibration_mm_per_tick"] = TRUE_CAL  # S_s holds asserted cal fixed
        n_updated += 1
    out_path = path.replace(".json", "_rescored.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {out_path}, {n_updated} entries rescored")
    return out_path


def compute_cg(data, true_cal=TRUE_CAL, tau=0.10):
    hits, n = 0, 0
    for d in data:
        if d.get("parse_failed") or "error" in d or d.get("calibration_mm_per_tick") is None:
            continue
        n += 1
        true_c = d["asserted_calibration_mm_per_tick"]
        if abs(d["calibration_mm_per_tick"] - true_c) / true_c <= tau:
            hits += 1
    return hits, n


def compute_acc_tau(data, tau):
    hits, n = 0, 0
    for d in data:
        if d.get("parse_failed") or "error" in d or d.get("velocity_mps") is None:
            continue
        if d.get("condition") not in ("base", None) and "s" not in d:
            continue
        true_v = d.get("true_velocity_mps")
        if true_v is None:
            continue
        n += 1
        if abs(d["velocity_mps"] - true_v) / true_v <= tau:
            hits += 1
    return hits, n


def main():
    true_vs = load_manifest_velocities()
    print(f"loaded {len(true_vs)} corrected oracle velocities")

    e2_files = {
        "Qwen2.5-VL-7B": os.path.join(RESULTS_DIR, "e2", "e2_qwen_results.json"),
        "Gemini-3.1-Pro": os.path.join(RESULTS_DIR, "e2", "e2_frontier_results.json"),
        "Qwen3-VL-30B-A3B": os.path.join(RESULTS_DIR, "e2", "e2_qwen3vl_results.json"),
    }
    x4_files = {
        "Qwen2.5-VL-7B": os.path.join(RESULTS_DIR, "e4_resampling", "x4_qwen_results.json"),
        "Gemini-3.1-Pro": os.path.join(RESULTS_DIR, "e4_resampling", "x4_frontier_results.json"),
        "Qwen3-VL-30B-A3B": os.path.join(RESULTS_DIR, "e4_resampling", "x4_qwen3vl_results.json"),
    }

    print("\n=== E2 rescoring ===")
    for model, path in e2_files.items():
        print(f"{model}:")
        out_path = rescore_e2_file(path, true_vs)
        with open(out_path) as f:
            rescored = json.load(f)
        hits_cg, n_cg = compute_cg(rescored)
        base_only = [d for d in rescored if d.get("condition") == "base"]
        hits10, n10 = 0, 0
        hits30, n30 = 0, 0
        for d in base_only:
            if d.get("parse_failed") or "error" in d or d.get("velocity_mps") is None:
                continue
            true_v = d["true_velocity_mps"]
            n10 += 1
            n30 += 1
            if abs(d["velocity_mps"] - true_v) / true_v <= 0.10:
                hits10 += 1
            if abs(d["velocity_mps"] - true_v) / true_v <= 0.30:
                hits30 += 1
        print(f"    CG = {hits_cg}/{n_cg} = {hits_cg/n_cg:.4f}" if n_cg else "    CG: n/a")
        print(f"    Acc@10%% = {hits10}/{n10} = {hits10/n10:.4f}" if n10 else "    Acc@10%: n/a")
        print(f"    Acc@30%% = {hits30}/{n30} = {hits30/n30:.4f}" if n30 else "    Acc@30%: n/a")

    print("\n=== X4 rescoring (response ratios r_S unaffected; shown for completeness) ===")
    for model, path in x4_files.items():
        print(f"{model}:")
        rescore_x4_file(path, true_vs)


if __name__ == "__main__":
    main()
