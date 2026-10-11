#!/bin/bash -l
#SBATCH -J sgf_t1_textcal_qwen3vl
#SBATCH -o t1_textcal_qwen3vl_%j.out
#SBATCH -e t1_textcal_qwen3vl_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -t 12:00:00

# T1 positive control, Qwen3-VL-30B-A3B-Instruct: text-
# calibration R_alpha (243 calls) then S_s (162 calls). Same model-loading
# path already verified correct by the completed E2/X4/X10 runs on this
# model -- no smoke test gate needed.
#
# CLUSTER SETTINGS: partition/account/QOS are site-specific and are NOT set
# here -- pass them at submit time from the repo root, e.g.
#   sbatch -p <gpu-partition> -A <account> -q <qos> slurm_jobs/09_t1_textcal_qwen3vl30b_H200.sh

set -euo pipefail
source "${CONDA_DIR:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate sgf
module load cuda12.6/toolkit/12.6.2 2>/dev/null || true

# Submit from the repo root (or set SGF_REPO_ROOT); data dir follows code/sgf_paths.py.
REPO_ROOT="${SGF_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA="${SGF_DATA_DIR:-$REPO_ROOT/data}"

if [ ! -f "$REPO_ROOT/results/e2_selected_clips.json" ]; then
    echo "ERROR: e2_selected_clips.json does not exist. Run build_e2_manifest.py first." >&2
    exit 1
fi

cd "$REPO_ROOT/code/model_outputs"
MODEL=qwen3vl python3 run_t1_textcal_relabeling.py
MODEL=qwen3vl python3 run_t1_textcal_resampling.py

echo "== T1 (text-calibration positive control), Qwen3-VL-30B-A3B, done. Results at $REPO_ROOT/results/t1/ =="
