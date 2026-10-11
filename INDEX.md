# Repository index

A map of what is where, what is important, and which file produces which
paper result. **Core** = needed for the paper's claims. **Legacy** = earlier
12-clip / pre-rescoping experiments reported in Appendix B.3–B.4 as carrying
no claim; kept for completeness.

## Start here

| File | What it is |
|---|---|
| [README.md](README.md) | Overview, setup, how to reproduce every number |
| [reconcile.py](reconcile.py) | **Core.** Recomputes every Table 1 / App. B.3 number from `results/repro_raw_outputs.csv` and checks it against the paper (65 checks) |
| [sgf_metrics.py](sgf_metrics.py) | **Core.** Metrics library behind the theory: design matrix `U`, rank/kernel, blind-policy bound (`blind_bound_cg`), Wilson / Clopper–Pearson / cluster-bootstrap / Fisher intervals, `classify_policy`, `min_detectable_rate`. Pure numpy. `python3 sgf_metrics.py` runs the smoke test |
| [build_repro_csv.py](build_repro_csv.py) | **Core.** Builds `results/repro_raw_outputs.csv` from the per-item result JSONs |
| [code/sgf_paths.py](code/sgf_paths.py) | **Core.** Single source of every filesystem path and env-var override |
| [HF_UPLOAD.md](HF_UPLOAD.md) | Authors' checklist for publishing the video dataset on Hugging Face |
| [CLAUDE.md](CLAUDE.md) | Conventions and pitfalls for AI coding assistants (and humans) |

## Directory tree

```
.
├── README.md  INDEX.md  CLAUDE.md  HF_UPLOAD.md  LICENSE  CITATION.cff
├── requirements.txt  .env.example  .gitignore
├── reconcile.py  sgf_metrics.py  build_repro_csv.py
├── code/
│   ├── sgf_paths.py                 all paths (results/, data/raw_videos/, data/derived/)
│   ├── oracle/                      classical-CV ground truth (no learned model)
│   ├── perturbation/                R_alpha relabeling and S_s resampling renderers
│   └── model_outputs/               prompts, parsers, inference clients, batch runners
├── slurm_jobs/                      sbatch scripts for the two local models
├── scripts/                         dataset download / upload (Hugging Face)
├── results/                         every per-item result JSON (committed)
├── examples/                        a few real frames and short clips (5 MB)
└── data/                            git-ignored; filled by scripts/download_data.py
    ├── raw_videos/02182026/...      raw recordings
    └── derived/                     fall_clips, trimmed_scale_clips, perturbed_clips, resampled_clips
```

## Paper result → code → output

| Paper element | Script(s) | Output in `results/` | Status |
|---|---|---|---|
| Oracle ground truth (pre-impact velocity per clip; §4, App. B.2) | `code/oracle/run_batch.py` → `droplet_oracle.py` | `oracle/oracle_batch_results.json` | Core |
| 27-clip evaluation manifest | `code/model_outputs/build_e2_manifest.py` | `e2_selected_clips.json` | Core |
| Perturbed inputs ($R_\alpha$, $S_s$) | `code/perturbation/render_conditions.py`, then `code/model_outputs/trim_scale_clips.py` | `data/derived/*` | Core |
| Fall-clip extraction (motion window) | `code/model_outputs/prepare_clips.py` (+ `tune_extraction_window.find_settled_end`) | `data/derived/fall_clips/`; tuning record `e3/settled_window_tuning.json` | Core |
| Sanity check (oracle×α vs fixed 1 m/s emitter; §4) | computed interactively with `sgf_metrics.py`, **no standalone script** | `e1/e1_positive_control.json` | Core |
| Main evaluation $R_\alpha$: Table 1 CG, $r_R$, Acc@10/30% | `run_e2_batch.py` (Qwen2.5-VL), `run_e2_batch_qwen3vl.py`, `run_e2_batch_frontier.py` (Gemini) | `e2/e2_{qwen,qwen3vl,frontier}_results.json` + `_rescored` | Core |
| Resampling control $S_s$: Table 1 $r_S$ | `run_x4_resampling.py`, `run_x4_resampling_qwen3vl.py`, `run_x4_resampling_frontier.py` | `e4_resampling/x4_*_results.json` + `_rescored` | Core |
| Temporal relabeling $T_\beta$: Table 1 $r_T$ | `run_temporal_relabeling.py` (+ `prompt_design_temporal.py`) | `e2/temporal_qwen_results.json` | Core |
| Instructed-constant control (Table 1) | `run_x10_instructed_constant{,_qwen3vl,_frontier}.py` | `x10/x10_*_results.json` | Core |
| Text-calibration positive control (main text, App. B.3) | `run_t1_textcal_relabeling.py`, `run_t1_textcal_resampling.py` with `MODEL=qwen25vl\|qwen3vl\|gemini` (+ `prompt_design_textcal.py`) | `t1/t1_textcal_{relabel,resample}_*_results.json` | Core |
| Legibility / temporal-sufficiency check | `run_x2_legibility.py` | `x2/x2_results.json`, `x2/native_frames/*.png` | Core |
| Calibration-correction rescoring (App. B.2) | `rescore_calibration_fix.py` | `*_rescored.json`; pre-fix reruns in `pre_calibration_fix_backup/` | Core |
| Reconciliation input | `build_repro_csv.py` | `repro_raw_outputs.csv` | Core |
| Decoupling analysis, 27 clips × 3 models (App. B.3) | computed interactively from the `_rescored` files, **no standalone script** | `e4/e4_decoupling_27clip.json` | Supporting |
| Per-experiment metric summaries | computed interactively, **no standalone script** | `e2/*metrics_summary.json`, `e4_resampling/x4_qwen3vl_metrics_summary.json`, `e3/*metrics_summary.json` | Supporting |
| Hidden-prior probe E3 (App. B.3) | `run_e3_batch.py`, `run_e3v2_batch.py`, `run_e3v3_batch.py`, `enhance_e3_clips.py`, `run_e3_enhanced_retest.py`, `detect_multi_droplet.py` | `e3/*.json` | Legacy |
| Robustness sweep E5 (App. B.3) | `render_e5_variants.py`, `trim_e5_variants.py`, `run_e5_batch.py`, `run_e5_batch_b1_full12.py`, `run_e5_analysis.py`, `run_e5_analysis_full12.py` | `e5/*.json` | Legacy |
| Old 12-clip decoupling / τ-sweep | `run_e4_decoupling.py`, `run_e4_multimodel.py`, `run_e4_tau_sweep.py` | `e4/e4_decoupling.json`, `e4_multimodel_decoupling.json`, `e4_tau_sweep.json` | Legacy (superseded by `e4_decoupling_27clip.json`) |
| Gemini-2.5-Flash-Lite run (App. B.4) | `run_e2_batch_openrouter.py`, `analyze_b2_cg_asymmetry.py` | `e2/e2_gemini_flash_lite_results.json` | Legacy |
| Low-confidence session re-run | `code/oracle/rerun_newvideos.py` | `oracle/newvideos_rerun.json` | Legacy |
| Earlier fall-clip re-extraction | `reextract_fall_clips.py` | `fall_clips_v2_extraction_meta.json` | Legacy |

The legacy E3/E5 scripts reference session `05052026`, which is not part of
the default dataset download (see HF_UPLOAD.md).

## `code/` in detail

**`code/oracle/`**
- `droplet_oracle.py`: background-subtraction blob tracking plus a
  free-fall-plateau velocity fit. Holds the per-session calibration
  `SESSION_CALIBRATION_PX_PER_MM` (corrected to 1/16 inch for 02182026).
- `run_batch.py`: runs the oracle over every clip of the evaluation session
  (parallel).
- `validate_clip.py`, `validate_clip2.py`: single-clip debugging helpers.

**`code/perturbation/`**
- `perturb_renderer.py`: $R_\alpha$. Repaints the ruler's printed number so
  the asserted calibration becomes α·c; pixels are untouched.
  `SESSION_LABEL_CONFIG` holds the hand-measured label ROI.
- `resample_renderer.py`: $S_s$. Rescales whole frames (droplet and ruler)
  by s, keeping the asserted label fixed.
- `render_conditions.py`: driver that renders every clip the main
  experiments consume.

**`code/model_outputs/`** (inference and prompts)
- `run_qwen_inference.py` (Qwen2.5-VL-7B), `run_qwen3vl_inference.py`
  (Qwen3-VL-30B-A3B) and `run_openrouter_inference.py` (Gemini) are the model
  clients. Local models use greedy decoding at pinned checkpoint revisions.
- `prompt_design.py`, `prompt_design_temporal.py`, `prompt_design_textcal.py`:
  prompt templates and the 3 paraphrase seeds.
- `response_parsing.py`: extracts velocity and calibration numbers from free
  text.
- `run_*_batch*.py`, `run_x*.py`, `run_t1_*.py`, `run_temporal_*.py`:
  resumable batch runners, one per experiment × model (table above).

## `results/` file conventions

- Each record is one model call: `clip`, `session`, the condition (`condition`
  = `base` | α for relabeling; `s` for resampling files; extra `alpha` / `beta`
  in T1 / temporal files), `seed_idx`, `raw_response`, parsed `velocity_mps` /
  `calibration_mm_per_tick`, `true_velocity_mps`,
  `asserted_calibration_mm_per_tick`, and `parse_failed` / `error` flags.
- `_rescored.json`: same records, scored against the corrected calibration.
  `*_ORIGINAL` fields keep the pre-fix values. **Use these, not the base
  files, for E2/X4.**
- `"path"` fields in `e2_selected_clips.json` and `oracle/*.json` are relative
  to the raw-video root (`data/raw_videos/` by default); resolve them with
  `sgf_paths.raw_video()`.

## `slurm_jobs/`

`00_setup_env.sh` creates the conda env. `00_x2` … `09_t1` each run one
experiment on one local model (A100 for Qwen2.5-VL, H200 for Qwen3-VL).
Partition, account and QOS are not hard-coded; pass them to `sbatch`. Submit
from the repo root.
