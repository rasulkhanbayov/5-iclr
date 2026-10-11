"""
Authors-only: publish the dataset to Hugging Face (see HF_UPLOAD.md).

    huggingface-cli login            # or export HF_TOKEN=...
    python scripts/upload_to_hf.py --repo <HF_USERNAME>/<DATASET_REPO> \
        --raw-src /path/to/drops_videos [--sessions 02182026 13.5cm_speed/05122026] [--private]

Uploads, into one dataset repo:
  README.md                 <- scripts/hf_dataset_card.md
  derived/**                <- data/derived/ (fall, trimmed-scale, perturbed, resampled clips)
  raw_videos/<session>/*    <- top-level *.mp4 / *.txt of each session folder under --raw-src

Only top-level files of each session folder are uploaded (the Frames/,
__Frames/, Spreading/ sub-folders of the lab's working copy are not used by
this project). Use --dry-run to print the file list without uploading.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import sgf_paths  # noqa: E402

DEFAULT_SESSIONS = ["02182026"]
RAW_EXTS = (".mp4", ".txt")
SKIP_NAMES = {"scale.mp4"}  # unused second scale recording in 02182026 ("scale v.mp4" is the one used)


def raw_files(raw_src, session):
    folder = os.path.join(raw_src, session)
    if not os.path.isdir(folder):
        sys.exit(f"ERROR: {folder} does not exist")
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and name.endswith(RAW_EXTS) and not (session == "02182026" and name in SKIP_NAMES):
            yield p, f"raw_videos/{session}/{name}"


def derived_files():
    for root, _, files in os.walk(sgf_paths.DERIVED_DIR):
        for name in sorted(files):
            if name.endswith(".mp4"):
                p = os.path.join(root, name)
                yield p, "derived/" + os.path.relpath(p, sgf_paths.DERIVED_DIR).replace(os.sep, "/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="<HF_USERNAME>/<DATASET_REPO>")
    ap.add_argument("--raw-src", required=True, help="folder that contains the session folders")
    ap.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    ap.add_argument("--skip-derived", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    card = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_dataset_card.md")
    batches = {"README": [(card, "README.md")]}
    if not args.skip_derived:
        batches["derived"] = list(derived_files())
        if not batches["derived"]:
            sys.exit(f"ERROR: no clips under {sgf_paths.DERIVED_DIR}")
    for s in args.sessions:
        batches[s] = list(raw_files(args.raw_src, s))

    total = 0
    for name, files in batches.items():
        size = sum(os.path.getsize(p) for p, _ in files)
        total += size
        print(f"{name}: {len(files)} files, {size / 1e9:.2f} GB")
        if args.dry_run:
            for p, r in files:
                print(f"    {r}")
    print(f"TOTAL {total / 1e9:.2f} GB")
    if args.dry_run:
        return

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    for name, files in batches.items():
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=p) for p, r in files]
        api.create_commit(repo_id=args.repo, repo_type="dataset", operations=ops,
                          commit_message=f"Add {name} ({len(ops)} files)")
        print(f"committed {name}")
    print(f"done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
