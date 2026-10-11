#!/bin/bash -l
#SBATCH -J sgf_temporal_qwen7b
#SBATCH -o temporal_qwen7b_%j.out
#SBATCH -e temporal_qwen7b_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -t 8:00:00

# T_beta (temporal relabeling), Qwen2.5-VL-7B-Instruct -- the temporal
# analogue of E2's spatial R_alpha, run per the appendix's Section A.6 flag.
# 27 clips x 3 conditions x 3 seeds = 243 calls, single-video (no
# scale-reference clip needed -- calibration and the asserted frame rate are
# both given in text; see prompt_design_temporal.py). Resumable, same
# stale-entry guard as run_e2_batch.py.
#
# CLUSTER SETTINGS: partition/account/QOS are site-specific and are NOT set
# here -- pass them at submit time from the repo root, e.g.
#   sbatch -p <gpu-partition> -A <account> -q <qos> slurm_jobs/07_temporal_relabeling_qwen7b.sh
#
# Run 00_setup_env.sh first if the "sgf" conda env does not exist yet.

set -euo pipefail
source "${CONDA_DIR:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate sgf
module load cuda12.1/toolkit/12.1.1 2>/dev/null || true

# Submit from the repo root (or set SGF_REPO_ROOT); data dir follows code/sgf_paths.py.
REPO_ROOT="${SGF_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA="${SGF_DATA_DIR:-$REPO_ROOT/data}"

if [ ! -f "$REPO_ROOT/results/e2_selected_clips.json" ]; then
    echo "ERROR: e2_selected_clips.json does not exist. Run build_e2_manifest.py first" >&2
    echo "  (needs results/oracle/oracle_batch_results.json -- run oracle/run_batch.py first if missing)." >&2
    exit 1
fi

cd "$REPO_ROOT/code/model_outputs"
python3 run_temporal_relabeling.py

echo "== T_beta (temporal relabeling) done. Results at $REPO_ROOT/results/e2/temporal_qwen_results.json =="
