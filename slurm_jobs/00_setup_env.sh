#!/bin/bash
# One-time environment setup for the GPU jobs in this directory. Run it
# interactively on a machine that shares a home directory with the cluster's
# compute nodes (NOT on a login node that kills long-running processes).
# Run it from the repo root:  bash slurm_jobs/00_setup_env.sh
#
# Creates a conda env "sgf" (Python 3.10) with the exact package versions the
# paper's local-model runs used (see requirements.txt). The sbatch scripts in
# this directory activate that env by name.
#
# Safe to re-run: the conda install step is skipped if already present, and
# the env-create step is skipped if the env already exists.
set -euo pipefail

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="sgf"
REPO_ROOT="${SGF_REPO_ROOT:-$(pwd)}"

if [ ! -f "$REPO_ROOT/requirements.txt" ]; then
    echo "ERROR: $REPO_ROOT/requirements.txt not found -- run from the repo root or set SGF_REPO_ROOT." >&2
    exit 1
fi

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "== conda not found at $CONDA_DIR -- installing Miniconda =="
    TMP_INSTALLER="$HOME/miniconda_installer.sh"
    curl -fsSL -o "$TMP_INSTALLER" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "$TMP_INSTALLER" -b -p "$CONDA_DIR"
    rm -f "$TMP_INSTALLER"
    echo "== Miniconda installed to $CONDA_DIR =="
else
    echo "== conda already installed at $CONDA_DIR =="
fi

source "$CONDA_DIR/etc/profile.d/conda.sh"

# Newer Anaconda default channels (pkgs/main, pkgs/r) require Terms of
# Service acceptance before any env can be created or solved against them.
# Accept once, non-interactively, so env creation below does not fail with
# CondaToSNonInteractiveError. Harmless no-op if already accepted.
echo "== accepting Anaconda channel Terms of Service (one-time, account-wide) =="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "== creating conda env ${ENV_NAME} (python 3.10) if missing =="
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

echo "== installing packages (this takes a few minutes, mostly torch) =="
pip install --upgrade pip
# torch 2.6.0 was never published on the cu121 wheel index (tops out at
# 2.5.1+cu121); 2.6.0 shipped on cu124 instead. The cu124 wheel bundles its
# own CUDA 12.4 runtime, so it works on both the A100 (CUDA 12.1 toolkit) and
# H200 (CUDA 12.6 toolkit) nodes the paper's runs used, regardless of which
# system toolkit module is loaded.
pip install "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124
pip install -r "$REPO_ROOT/requirements.txt"

echo
echo "== done. =="
echo "IMPORTANT: run 'conda init bash' once and re-login, OR always source"
echo "  $CONDA_DIR/etc/profile.d/conda.sh"
echo "before 'conda activate ${ENV_NAME}' in any NEW shell (including inside"
echo "sbatch scripts) -- the sbatch scripts in this directory already do this."
echo
echo "Verify GPU visibility (the real check happens inside an sbatch job on a"
echo "GPU compute node):"
echo "  conda activate ${ENV_NAME} && python3 -c 'import torch; print(torch.cuda.is_available())'"
