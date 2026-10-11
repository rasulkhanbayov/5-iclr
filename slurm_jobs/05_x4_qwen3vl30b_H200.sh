#!/bin/bash -l
#SBATCH -J sgf_x4_qwen3vl30b
#SBATCH -o x4_qwen3vl30b_%j.out
#SBATCH -e x4_qwen3vl30b_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -t 8:00:00

# X4: resampling negative control for Qwen3-VL-30B-A3B. Same
# model class/loading path already verified correct by the completed X1/E2
# run (job 374849, 243/243 items, 0 errors, 0 parse failures) -- no smoke
# test gate needed here, this reuses the same run_qwen3vl_inference.py.
#
# 162 items (27 clips x 2 resample levels x 3 seeds), same cost/runtime
# profile as the E2 run (243 items took ~40 minutes total on H200 including
# ~5 min model load), so this should finish in well under the 8h limit.
#
# CLUSTER SETTINGS: partition/account/QOS are site-specific and are NOT set
# here -- pass them at submit time from the repo root, e.g.
#   sbatch -p <gpu-partition> -A <account> -q <qos> slurm_jobs/05_x4_qwen3vl30b_H200.sh

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
python3 run_x4_resampling_qwen3vl.py

echo "== X4 (resampling, open-weight) done. Results at $REPO_ROOT/results/e4_resampling/x4_qwen3vl_results.json =="
