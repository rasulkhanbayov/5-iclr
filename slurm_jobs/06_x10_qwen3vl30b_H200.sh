#!/bin/bash -l
#SBATCH -J sgf_x10_qwen3vl30b
#SBATCH -o x10_qwen3vl30b_%j.out
#SBATCH -e x10_qwen3vl30b_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -t 4:00:00

# X10: instructed-constant control for Qwen3-VL-30B-A3B. Same
# model class/loading path already verified correct by the completed X1/E2
# (job 374849) and X4 (job 374873) runs -- no smoke test gate needed, reuses
# the same run_qwen3vl_inference.py.
#
# 81 items (27 clips x 3 conditions x 1 seed, no seed sweep), smallest of the
# three Qwen3-VL batches -- should finish well under the 4h limit (E2's 243
# items took ~40 min total including ~5 min model load; this is 1/3 the size).
#
# CLUSTER SETTINGS: partition/account/QOS are site-specific and are NOT set
# here -- pass them at submit time from the repo root, e.g.
#   sbatch -p <gpu-partition> -A <account> -q <qos> slurm_jobs/06_x10_qwen3vl30b_H200.sh

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
if [ ! -d "$DATA/derived/fall_clips" ] || [ -z "$(ls -A "$DATA/derived/fall_clips" 2>/dev/null)" ]; then
    echo "ERROR: data/derived/fall_clips/ is empty. Run run_e2_batch_qwen3vl.py first so fall clips are extracted." >&2
    exit 1
fi

cd "$REPO_ROOT/code/model_outputs"
python3 run_x10_instructed_constant_qwen3vl.py

echo "== X10 (instructed-constant, open-weight) done. Results at $REPO_ROOT/results/x10/x10_qwen3vl_results.json =="
