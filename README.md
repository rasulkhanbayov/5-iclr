# One Perturbation Is Not Enough

Code, per-item model outputs and reconciliation for **"One Perturbation Is Not
Enough: Identifiability and Blind Baselines for Behavioral AI Evaluation"**
(AAAI-27 Alignment Track).

Rasul Khanbayov, Mariam Sohail, Ahmed Abdala, Hasan Kurban

> New here? [INDEX.md](INDEX.md) maps every file to what it does and which
> paper result it produces.

## What this is

A behavioral evaluation perturbs an input and reads the induced change in a
model's output to certify that the model uses that input. This repo
implements and evaluates a perturbation-based test of whether
vision-language models (VLMs) actually ground a physical-quantity estimate (a
droplet's pre-impact velocity, from high-speed video) in an in-frame
calibration reference (a ruler), rather than reporting a plausible number
regardless of what the reference shows.

Three models are evaluated on 27 clips under three perturbation relations:
relabeling the ruler ($R_\alpha$), resampling the whole frame ($S_s$) and
relabeling the frame rate ($T_\beta$), plus an instructed-constant control and
a text-calibration positive control.

| Model | Where it ran |
|---|---|
| Qwen2.5-VL-7B-Instruct | local, 1x A100 80GB |
| Qwen3-VL-30B-A3B-Instruct | local, 1x H200 |
| Gemini 3.1 Pro Preview | OpenRouter API |

## Quick start: reproduce the paper's numbers (CPU, seconds, no data download)

```bash
git clone https://github.com/rasulkhanbayov/5-iclr.git && cd 5-iclr
pip install numpy==2.2.6

python3 reconcile.py      # recompute every Table 1 / Appendix B.3 number from the
                          # raw per-item outputs; prints MATCH/MISMATCH per cell
python3 sgf_metrics.py    # smoke test: every arithmetic identity / closed-form
                          # bound in the theory section
```

`reconcile.py` ends with `ALL CHECKS PASSED` (65 checks). To rebuild its
input CSV from the per-item JSONs first:

```bash
python3 code/model_outputs/rescore_calibration_fix.py   # *_rescored.json (calibration fix)
python3 build_repro_csv.py                              # results/repro_raw_outputs.csv
python3 reconcile.py
```

Both steps regenerate the committed files byte-for-byte.

## Full pipeline: rerun from raw video

### 1. Environment

```bash
bash slurm_jobs/00_setup_env.sh        # conda env "sgf", Python 3.10, pinned versions
# or manually:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Paper runs used Ubuntu 22.04, `torch 2.6.0+cu124`, `transformers 5.15.0`,
`qwen-vl-utils 0.0.14`. The model checkpoint revisions are pinned in code
(Qwen2.5-VL `cc594898…`, Qwen3-VL `9c4b90e1…`; see Appendix B.4).

For Gemini: `export OPENROUTER_API_KEY=...` (see [.env.example](.env.example)).
No key is stored in this repo.

### 2. Data

Videos are on Hugging Face at
[huggingface.co/datasets/rasulkhanbayov/sgf](https://huggingface.co/datasets/rasulkhanbayov/sgf), not in git:

```bash
python scripts/download_data.py          # derived clips, ~150 MB: enough to rerun all model inference
python scripts/download_data.py --raw    # + raw 02182026 session, ~4.4 GB: oracle, clip extraction, rendering
```

Files land in `data/` (git-ignored). Override locations with `SGF_DATA_DIR`,
`SGF_RAW_VIDEO_DIR`, `SGF_RESULTS_DIR`; all paths are defined in
[code/sgf_paths.py](code/sgf_paths.py).

### 3. Run

Each step's output already exists in `results/` or `data/derived/`, so you
can start at any step.

```bash
# (a) ground truth + clip selection (CPU, needs --raw)
python3 code/oracle/run_batch.py                    # -> results/oracle/oracle_batch_results.json
python3 code/model_outputs/build_e2_manifest.py     # -> results/e2_selected_clips.json (27 clips)

# (b) perturbed inputs (CPU, needs --raw; or use the downloaded derived clips)
python3 code/perturbation/render_conditions.py      # -> data/derived/{perturbed,resampled,fall}_clips
python3 code/model_outputs/trim_scale_clips.py      # -> data/derived/trimmed_scale_clips

# (c) model inference: Slurm (pass your own partition/account/QOS), from repo root
sbatch -p <partition> -A <account> slurm_jobs/01_e2_qwen7b.sh
#   ... see the table below; Gemini runs are plain python (API)

# (d) rescore, rebuild, reconcile
python3 code/model_outputs/rescore_calibration_fix.py
python3 build_repro_csv.py
python3 reconcile.py
```

| Paper result | Qwen2.5-VL-7B (A100) | Qwen3-VL-30B (H200) | Gemini 3.1 Pro (API) | Output |
|---|---|---|---|---|
| Main evaluation $R_\alpha$ (Table 1: CG, $r_R$, Acc@τ) | `slurm_jobs/01_e2_qwen7b.sh` | `04_x1_qwen3vl30b_H200.sh` | `run_e2_batch_frontier.py` | `results/e2/` |
| Resampling control $S_s$ (Table 1: $r_S$) | `02_x4_resampling_qwen7b.sh` | `05_x4_qwen3vl30b_H200.sh` | `run_x4_resampling_frontier.py` | `results/e4_resampling/` |
| Temporal relabeling $T_\beta$ (Table 1: $r_T$) | `07_temporal_relabeling_qwen7b.sh` | not run | not run | `results/e2/temporal_qwen_results.json` |
| Instructed-constant control (Table 1) | `03_x10_instructed_constant_qwen7b.sh` | `06_x10_qwen3vl30b_H200.sh` | `run_x10_instructed_constant_frontier.py` | `results/x10/` |
| Text-calibration positive control (main text, App. B.3) | `08_t1_textcal_qwen7b.sh` | `09_t1_textcal_qwen3vl30b_H200.sh` | `MODEL=gemini python3 run_t1_textcal_{relabeling,resampling}.py` | `results/t1/` |
| Legibility / temporal sufficiency check | `00_x2_legibility_qwen7b.sh` | n/a | n/a | `results/x2/` |

Gemini scripts live in `code/model_outputs/`. Run them from anywhere:
`python3 code/model_outputs/run_e2_batch_frontier.py`. Every batch runner
resumes from its output file, so a killed job can simply be resubmitted.
**Runtime:** each local-model job took roughly 25–45 min on one A100 80GB
or H200 (model load included). Gemini spend was $20.85 in total across all
experiments.

## A calibration correction, and what it changed

Partway through this project, the evaluation session's printed ruler was
found to be graduated in 1/16 inch (1.5875 mm per tick), not the 1 mm
originally assumed. This was confirmed by direct pixel measurement of its
alternating long/short tick marks. A free-fall physics cross-check
($v=\sqrt{2gh}$, whose prediction does not depend on the pixel calibration)
corroborates it independently. The correction changes every quantity scored
against the true calibration or the true oracle velocity. See the paper's
Appendix B.2.

- **Main evaluation and resampling control**: the prompt never stated the
  calibration (the model reads the ruler video itself), so the raw responses
  are unchanged and only the *scoring* changes.
  `code/model_outputs/rescore_calibration_fix.py` writes
  `results/e2/*_rescored.json` and `results/e4_resampling/*_rescored.json`
  next to the originals; `_ORIGINAL`-suffixed fields keep the pre-fix values.
  **`reconcile.py` and `build_repro_csv.py` read the `_rescored` files**, and
  any new analysis should too.
- **Temporal relabeling, instructed-constant control, text-calibration
  positive control**: these state the calibration as text, so they were
  rerun in full against the corrected value. `results/e2/temporal_qwen_results.json`,
  `results/x10/` and `results/t1/` are post-correction. The pre-correction
  files are kept in `results/pre_calibration_fix_backup/` for audit.
- **Legibility check**: no rerun was needed, but the true-value field in
  `results/x2/x2_results.json` was corrected, which reverses that check's
  conclusion.

The corrected constant is in `code/oracle/droplet_oracle.py`
(`SESSION_CALIBRATION_PX_PER_MM`), `code/model_outputs/prompt_design.py`
(`INSTRUCTED_CONSTANT_CALIBRATION_MM_PER_TICK`), and every batch runner that
uses it (marked `# CORRECTED`).

## A note on the CSV schema

`results/repro_raw_outputs.csv` has a `family` column that distinguishes
elicitations sharing the same `(clip_id, seed)` key but coming from
different, independently run prompts. The main evaluation's "base"
condition, temporal relabeling's text-based "base" and the positive
control's text-based "base" are three distinct model responses to three
different prompts on the same clip. Any analysis joining a perturbed row to
"its" base row must match on `family` as well as `(model, clip_id, seed)`;
otherwise it silently averages unrelated elicitations.

## License

Code: MIT, see [LICENSE](LICENSE). The video dataset on Hugging Face carries
its own license (see its dataset card).

## Citation

```bibtex
@inproceedings{khanbayov2027perturbation,
  title     = {One Perturbation Is Not Enough: Identifiability and Blind Baselines for Behavioral AI Evaluation},
  author    = {Khanbayov, Rasul and Sohail, Mariam and Abdala, Ahmed and Kurban, Hasan},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (Alignment Track)},
  year      = {2027},
  note      = {arXiv:XXXX.XXXXX},
}
```

<!-- TODO(authors): add the arXiv ID once assigned. -->
