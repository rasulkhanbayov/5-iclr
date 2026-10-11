#!/bin/bash -l
#SBATCH -J sgf_x2_legibility
#SBATCH -o x2_legibility_%j.out
#SBATCH -e x2_legibility_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -t 02:00:00

# X2 (DECISIVE): input-legibility and temporal-sufficiency
# control. Numbered 00_ (before 01_e2) deliberately -- per the runbook,
# "Run it first," since it can invalidate the headline result before any
# other experiment's compute is spent. Cheap: 2 static-image calls (native
# resolution + the resolution the video path actually delivers) plus
# instrumentation over the 27 already-selected clips, no new video renders.
#
# Depends on:
#   - fall clips already extracted (temporal-sufficiency instrumentation
#     needs something to measure -- if data/derived/fall_clips/ is empty, this
#     job still runs the static-legibility checks but SKIPs every clip's
#     temporal check, see run_x2_legibility.py's per-clip guard);
#   - the trimmed base-condition scale clip existing (needed to determine
#     what resolution the video path delivers -- see
#     get_video_path_resolution() in run_x2_legibility.py). Without it, the
#     video-path-resolution legibility check is skipped but native-resolution
#     and temporal checks still run.
# Running 01_e2_qwen7b.sh first (which extracts fall clips as a side effect)
# and trim_scale_clips.py before that satisfies both -- extraction/trimming
# need no model calls, so this ordering costs nothing in practice.
#
# CLUSTER SETTINGS: partition/account/QOS are site-specific and are NOT set
# here -- pass them at submit time from the repo root, e.g.
#   sbatch -p <gpu-partition> -A <account> -q <qos> slurm_jobs/00_x2_legibility_qwen7b.sh

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
if [ ! -f "$DATA/derived/trimmed_scale_clips/02182026_scalev_base_trim.mp4" ]; then
    echo "WARNING: trimmed base scale clip missing -- the video-path-resolution" >&2
    echo "  legibility check will be skipped (native-resolution and temporal" >&2
    echo "  checks still run). Run trim_scale_clips.py first to get both." >&2
fi

cd "$REPO_ROOT/code/model_outputs"
python3 run_x2_legibility.py

echo "== X2 done. Results at $REPO_ROOT/results/x2/x2_results.json =="
echo "== READ BOTH STATIC-LEGIBILITY RESPONSES BEFORE RUNNING ANYTHING ELSE. =="
echo "== If native passes but the video-path-resolution check does not, the =="
echo "== honest paper is about preprocessing, not grounding -- see =="
echo "== run_x2_legibility.py's docstring. =="
