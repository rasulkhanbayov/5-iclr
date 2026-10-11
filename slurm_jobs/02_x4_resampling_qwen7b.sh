#!/bin/bash -l
#SBATCH -J sgf_x4_qwen7b
#SBATCH -o x4_qwen7b_%j.out
#SBATCH -e x4_qwen7b_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -t 12:00:00

# X4: resampling negative control (S_s), required by Theorem 2/Corollary 2 --
# see run_x4_resampling.py's module docstring. 27 clips x 2 resample levels
# x 3 seeds = 162 calls. Run AFTER 01_e2_qwen7b.sh has extracted fall clips
# (this job reuses data/derived/fall_clips/*, does not re-extract them) --
# actually run_x4_resampling.py reads the RESAMPLED droplet clips directly
# (data/derived/resampled_clips/*_fall_resample_s*.mp4), which must already exist
# from code/perturbation/render_conditions.py; it does not depend on E2 having run first, only
# on the same e2_selected_clips.json manifest and the resampled clips being
# pre-rendered.
#
# CLUSTER SETTINGS: partition/account/QOS are site-specific and are NOT set
# here -- pass them at submit time from the repo root, e.g.
#   sbatch -p <gpu-partition> -A <account> -q <qos> slurm_jobs/02_x4_resampling_qwen7b.sh

set -euo pipefail
source "${CONDA_DIR:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate sgf
module load cuda12.1/toolkit/12.1.1 2>/dev/null || true

# Submit from the repo root (or set SGF_REPO_ROOT); data dir follows code/sgf_paths.py.
REPO_ROOT="${SGF_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA="${SGF_DATA_DIR:-$REPO_ROOT/data}"

if [ ! -f "$REPO_ROOT/results/e2_selected_clips.json" ]; then
    echo "ERROR: e2_selected_clips.json does not exist. Run build_e2_manifest.py first." >&2
    exit 1
fi
for cond_file in \
    "$DATA/derived/trimmed_scale_clips/02182026_scalev_resample_s0.5_trim.mp4" \
    "$DATA/derived/trimmed_scale_clips/02182026_scalev_resample_s2_trim.mp4"; do
    if [ ! -f "$cond_file" ]; then
        echo "ERROR: $cond_file does not exist. Run code/perturbation/render_conditions.py then trim_scale_clips.py, or scripts/download_data.py --derived." >&2
        exit 1
    fi
done
if [ ! -d "$DATA/derived/resampled_clips" ] || [ -z "$(ls -A "$DATA/derived/resampled_clips" 2>/dev/null)" ]; then
    echo "ERROR: data/derived/resampled_clips/ is empty. Run code/perturbation/render_conditions.py first" >&2
    echo "  (extract fall clips with prepare_clips.py / run_e2_batch.py before resampling them)." >&2
    exit 1
fi

cd "$REPO_ROOT/code/model_outputs"
python3 run_x4_resampling.py

echo "== X4 done. Results at $REPO_ROOT/results/e4_resampling/x4_qwen_results.json =="
