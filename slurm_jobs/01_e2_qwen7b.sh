#!/bin/bash -l
#SBATCH -J sgf_e2_qwen7b
#SBATCH -o e2_qwen7b_%j.out
#SBATCH -e e2_qwen7b_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -t 12:00:00

# E2 main run, Qwen2.5-VL-7B-Instruct, on the 27 high-confidence 02182026
# clips. 27 clips x 3 conditions x 3 seeds = 243 calls, resumable (run_e2_batch.py skips already-completed
# (clip, condition, seed) keys on restart, so a preempted/killed job can
# just be resubmitted).
#
# CLUSTER SETTINGS: partition/account/QOS are site-specific and are NOT set
# here -- pass them at submit time from the repo root, e.g.
#   sbatch -p <gpu-partition> -A <account> -q <qos> slurm_jobs/01_e2_qwen7b.sh
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
for cond_file in \
    "$DATA/derived/trimmed_scale_clips/02182026_scalev_base_trim.mp4" \
    "$DATA/derived/trimmed_scale_clips/02182026_scalev_alpha0.5_trim.mp4" \
    "$DATA/derived/trimmed_scale_clips/02182026_scalev_alpha2_trim.mp4"; do
    if [ ! -f "$cond_file" ]; then
        echo "ERROR: $cond_file does not exist. Run code/perturbation/render_conditions.py then trim_scale_clips.py, or scripts/download_data.py --derived." >&2
        exit 1
    fi
done

cd "$REPO_ROOT/code/model_outputs"
python3 run_e2_batch.py

echo "== E2 done. Results at $REPO_ROOT/results/e2/e2_qwen_results.json =="
