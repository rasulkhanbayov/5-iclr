"""
Download the droplet-impact dataset from Hugging Face into data/.

    python scripts/download_data.py              # derived clips only (~150 MB)
    python scripts/download_data.py --raw        # + raw 02182026 session (~4.4 GB)
    python scripts/download_data.py --all        # + every raw session on the Hub

What you need for what:
  * reconcile.py / sgf_metrics.py ............ nothing (results/ is committed)
  * rerun model inference (E2/X4/X10/T1/T_beta) derived clips (default)
  * rerun the CV oracle, re-extract or re-render clips, X2 native frame
                                               --raw

Files land in data/raw_videos/ and data/derived/, the layout code/sgf_paths.py
expects. Set SGF_HF_DATASET=<user>/<repo> to override the dataset repo id and
SGF_DATA_DIR to download somewhere other than <repo>/data.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import sgf_paths  # noqa: E402

# Raw sessions: 02182026 is the evaluation session; 13.5cm_speed/05122026 is
# only used by the oracle batch (low-confidence clips, excluded from E2);
# 05052026 is only used by the superseded E3/E5 scripts.
MAIN_SESSION = "02182026"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", action="store_true", help="also download the raw 02182026 evaluation session")
    ap.add_argument("--all", action="store_true", help="download everything in the dataset repo")
    ap.add_argument("--no-derived", action="store_true", help="skip the derived clips")
    ap.add_argument("--repo", default=sgf_paths.HF_DATASET_REPO, help="HF dataset repo id")
    args = ap.parse_args()

    if args.repo.startswith("<"):
        sys.exit("ERROR: the Hugging Face dataset repo id is still a placeholder. Pass "
                 "--repo <user>/<name> or set SGF_HF_DATASET (see HF_UPLOAD.md).")

    from huggingface_hub import snapshot_download

    if args.all:
        patterns = None
    else:
        patterns = ["README.md"]
        if not args.no_derived:
            patterns.append("derived/**")
        if args.raw:
            patterns += [f"raw_videos/{MAIN_SESSION}/*.mp4", f"raw_videos/{MAIN_SESSION}/*.txt"]

    os.makedirs(sgf_paths.DATA_DIR, exist_ok=True)
    path = snapshot_download(repo_id=args.repo, repo_type="dataset",
                             local_dir=sgf_paths.DATA_DIR, allow_patterns=patterns)
    print(f"downloaded {args.repo} -> {path}")
    if sgf_paths.RAW_VIDEO_DIR != os.path.join(sgf_paths.DATA_DIR, "raw_videos"):
        print(f"NOTE: SGF_RAW_VIDEO_DIR={sgf_paths.RAW_VIDEO_DIR} overrides data/raw_videos/")


if __name__ == "__main__":
    main()
