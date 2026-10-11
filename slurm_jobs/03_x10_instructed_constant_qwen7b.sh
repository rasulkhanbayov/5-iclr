#!/bin/bash -l
#SBATCH -J sgf_x10_qwen7b
#SBATCH -o x10_qwen7b_%j.out
#SBATCH -e x10_qwen7b_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -t 04:00:00

# X10: instructed-constant control, added for the AAAI-27 Alignment Track --
# see run_x10_instructed_constant.py's module docstring. 27 clips x 3
# conditions x 1 seed = 81 calls. Depends on fall clips already being
# extracted (run 01_e2_qwen7b.sh first, or run prepare_clips.py directly) --
# this job does NOT extract them itself.
#
# CLUSTER SETTINGS: partition/account/QOS are site-specific and are NOT set
# here -- pass them at submit time from the repo root, e.g.
#   sbatch -p <gpu-partition> -A <account> -q <qos> slurm_jobs/03_x10_instructed_constant_qwen7b.sh

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
if [ ! -d "$DATA/derived/fall_clips" ] || [ -z "$(ls -A "$DATA/derived/fall_clips" 2>/dev/null)" ]; then
    echo "ERROR: data/derived/fall_clips/ is empty. Run 01_e2_qwen7b.sh first (or prepare_clips.py" >&2
    echo "  directly) to extract fall clips before running X10." >&2
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
python3 run_x10_instructed_constant.py

echo "== X10 done. Results at $REPO_ROOT/results/x10/x10_qwen_results.json =="
