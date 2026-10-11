# CLAUDE.md

Guidance for AI coding assistants (and humans) working in this repository.
Code and results for the AAAI-27 Alignment Track paper "One Perturbation Is
Not Enough". See README.md for the overview and INDEX.md for the file map.

## What the project does

Tests whether vision-language models ground a droplet-velocity estimate in an
in-frame ruler. The ruler video is perturbed (relabel $R_\alpha$, resample
$S_s$, frame-rate relabel $T_\beta$), each model is queried, and response
ratios plus calibration grounding (CG) are scored against a classical-CV
oracle and the paper's blind-policy bound.

## Key commands

```bash
python3 reconcile.py                                    # verify every paper number (must print ALL CHECKS PASSED)
python3 sgf_metrics.py                                  # theory smoke test
python3 code/model_outputs/rescore_calibration_fix.py   # regenerate *_rescored.json
python3 build_repro_csv.py                              # regenerate results/repro_raw_outputs.csv
python3 scripts/download_data.py [--raw]                # fetch videos into data/
python3 code/oracle/run_batch.py                        # oracle (needs raw video)
python3 code/perturbation/render_conditions.py          # perturbed / resampled clips (needs raw video)
python3 code/model_outputs/trim_scale_clips.py          # 3 s scale clips for VLM input
sbatch -p <part> -A <acct> slurm_jobs/01_e2_qwen7b.sh   # local-model jobs; submit from repo root
```

These run from any working directory: scripts resolve paths from `__file__`.
After touching anything under `results/`, the scoring code, or
`sgf_metrics.py`, rerun the rescore → build CSV → reconcile chain. The
rebuilt CSV and `_rescored` files must stay byte-identical unless you meant to
change them.

## Layout and conventions

- **All paths come from `code/sgf_paths.py`.** Never hard-code absolute paths.
  Scripts bootstrap with
  `sys.path.insert(0, <code dir>); import sgf_paths` and use
  `sgf_paths.RESULTS_DIR`, `MANIFEST_PATH`, `FALL_CLIP_DIR`,
  `TRIMMED_SCALE_DIR`, `PERTURBED_CLIP_DIR`, `RESAMPLED_CLIP_DIR` and
  `raw_video(rel_path)`. Env overrides: `SGF_DATA_DIR`, `SGF_RAW_VIDEO_DIR`,
  `SGF_RESULTS_DIR`.
- `results/`: small, committed per-item JSONs (the paper's evidence).
  `data/`: large videos, git-ignored, hosted on Hugging Face. Never commit
  videos; GitHub is not the store for them.
- `"path"` fields in manifest/oracle JSONs are **relative to the raw-video
  root** (e.g. `02182026/cainhcg1.mp4`). Always pass them through
  `sgf_paths.raw_video()` before opening.
- Batch runners are resumable: they reload their output JSON, skip completed
  `(clip, condition, seed)` keys, and drop entries for clips no longer in the
  manifest. Keep that behavior in new runners.
- Sibling modules import each other flat (`from prepare_clips import ...`)
  after `sys.path` inserts of `code/model_outputs`, `code/oracle`,
  `code/perturbation`. `sgf_paths.add_code_dirs_to_path()` does all of them.
- Model checkpoints are pinned (`MODEL_REVISION` in the inference clients;
  override via `SGF_QWEN25VL_REVISION` / `SGF_QWEN3VL_REVISION`). Decoding is
  greedy, and Gemini runs at temperature 0.
- Secrets come only from the environment (`OPENROUTER_API_KEY`, `HF_TOKEN`).
  Never write keys into files; `.env` is git-ignored.

## Pitfalls

- **Calibration constant.** Session 02182026's ruler is 1/16 inch =
  **1.5875 mm/tick**, not 1 mm. It is defined in `droplet_oracle.py`,
  `prompt_design.py` and each batch runner (`# CORRECTED`). If you change it
  in one place, change it everywhere.
- **Use `_rescored.json` for E2/X4.** The base `e2_*_results.json` /
  `x4_*_results.json` keep pre-correction scoring fields for audit.
  `build_repro_csv.py` and `reconcile.py` already read the `_rescored` files.
- **Join on `family`.** In `repro_raw_outputs.csv`, the main-eval, temporal
  and T1 "base" rows are different prompts on the same `(clip_id, seed)`.
  Join perturbed rows to base rows on `(model, clip_id, seed, family)`.
- **Clustering.** 27 clips × 3 seeds × conditions are not independent
  samples. Cluster-bootstrap over clips (`sgf_metrics`), and report n = 27
  clusters, not 81 or 243.
- **Legacy code.** E3/E5 and the `run_e4_*` analyses use the old 12-clip,
  two-session design (`05052026`). They carry no claim in the paper; don't
  "fix" them toward the current design without saying so.
- Some summary JSONs (`e1_positive_control.json`,
  `e4/e4_decoupling_27clip.json`, `*_metrics_summary.json`) were computed
  interactively and have no generating script. Do not overwrite them.
- `scale v.mp4` (with a space) is the scale-reference recording used. The
  similarly named `scale.mp4` in the same session is unused.
- The `reconcile.py` expected values are transcribed from the paper. If the
  paper changes, update them there rather than loosening tolerances.
