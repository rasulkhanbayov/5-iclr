#!/bin/bash -l
#SBATCH -J sgf_x1_qwen3vl30b
#SBATCH -o x1_qwen3vl30b_%j.out
#SBATCH -e x1_qwen3vl30b_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -t 16:00:00

# X1: current-generation open-weight model half of "one frontier
# proprietary AND one current open-weight model". Qwen3-VL-30B-A3B is a MoE
# model needing 80GB-class memory -- H200, not A100.
#
# VERIFIED 2026-08-13: the model-loading class (AutoModelForImageTextToText)
# is confirmed correct -- a smoke test on a ~16GB card, too
# small for this 30B-A3B model) got past model instantiation and failed only
# on CUDA OOM during weight loading, not on a class/import error. That is
# real evidence the code path works; the interactive [y/N] confirmation gate
# that used to guard this script is removed below since it silently no-ops
# under sbatch (no stdin attached, job exits immediately, as happened on the
# first two submission attempts -- confirmed via the .out files, which just
# printed the gate's own warning text and nothing else).
#
# CLUSTER SETTINGS: partition/account/QOS are site-specific and are NOT set
# here -- pass them at submit time from the repo root, e.g.
#   sbatch -p <gpu-partition> -A <account> -q <qos> slurm_jobs/04_x1_qwen3vl30b_H200.sh

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
python3 run_e2_batch_qwen3vl.py

echo "== X1 (open-weight) done. Results at $REPO_ROOT/results/e2/e2_qwen3vl_results.json =="
