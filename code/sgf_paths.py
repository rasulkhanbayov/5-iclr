"""
Single source of truth for every filesystem path used by the pipeline.

Every script under code/ imports this module instead of hard-coding paths,
so the repository runs from wherever it is cloned. Defaults follow the
repository layout:

    <repo>/results/                 committed per-item result JSONs (small)
    <repo>/data/raw_videos/         raw droplet recordings (Hugging Face dataset)
    <repo>/data/derived/            intermediate clips rendered from the raw
                                    videos (Hugging Face, or regenerate them)

Override any of them with environment variables (see .env.example):

    SGF_DATA_DIR       parent of raw_videos/ and derived/  (default <repo>/data)
    SGF_RAW_VIDEO_DIR  raw recordings root                 (default $SGF_DATA_DIR/raw_videos)
    SGF_RESULTS_DIR    result JSON root                    (default <repo>/results)

Raw-video paths stored in result/manifest JSONs (e.g. "02182026/cainhcg1.mp4")
are relative to SGF_RAW_VIDEO_DIR; resolve them with raw_video().
"""
import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CODE_DIR)

RESULTS_DIR = os.environ.get("SGF_RESULTS_DIR", os.path.join(REPO_ROOT, "results"))
DATA_DIR = os.environ.get("SGF_DATA_DIR", os.path.join(REPO_ROOT, "data"))
RAW_VIDEO_DIR = os.environ.get("SGF_RAW_VIDEO_DIR", os.path.join(DATA_DIR, "raw_videos"))
DERIVED_DIR = os.path.join(DATA_DIR, "derived")

# Intermediate clips (all regenerable from RAW_VIDEO_DIR, also mirrored on HF).
FALL_CLIP_DIR = os.path.join(DERIVED_DIR, "fall_clips")                  # prepare_clips.py
TRIMMED_SCALE_DIR = os.path.join(DERIVED_DIR, "trimmed_scale_clips")     # trim_scale_clips.py
PERTURBED_CLIP_DIR = os.path.join(DERIVED_DIR, "perturbed_clips")        # perturb_renderer.py
RESAMPLED_CLIP_DIR = os.path.join(DERIVED_DIR, "resampled_clips")        # resample_renderer.py
ENHANCED_FALL_CLIP_DIR = os.path.join(DERIVED_DIR, "fall_clips_enhanced")  # legacy E3 only

# The 27-clip evaluation manifest shared by every experiment.
MANIFEST_PATH = os.path.join(RESULTS_DIR, "e2_selected_clips.json")

# Default Hugging Face model cache (scripts only set it if HF_HOME is unset).
HF_CACHE_DIR = os.path.join(os.path.expanduser("~"), "sgf_hf_cache")

# Hugging Face dataset repo holding raw_videos/ and derived/ (see HF_UPLOAD.md).
HF_DATASET_REPO = os.environ.get("SGF_HF_DATASET", "rasulkhanbayov/sgf")


def raw_video(path):
    """Resolve a raw-video path from a manifest/result JSON against RAW_VIDEO_DIR."""
    return path if os.path.isabs(path) else os.path.join(RAW_VIDEO_DIR, path)


def add_code_dirs_to_path():
    """Make the sibling code packages and the repo root importable."""
    for d in (REPO_ROOT, CODE_DIR,
              os.path.join(CODE_DIR, "model_outputs"),
              os.path.join(CODE_DIR, "oracle"),
              os.path.join(CODE_DIR, "perturbation")):
        if d not in sys.path:
            sys.path.insert(0, d)
