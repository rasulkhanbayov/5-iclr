# Publishing the video data to Hugging Face (authors' checklist)

Videos are too large for git, so they go to **one Hugging Face dataset
repo**. `scripts/download_data.py` pulls them back into `data/`, the layout
`code/sgf_paths.py` expects. Everything else (code, per-item results,
examples) is in this GitHub repo.

## What goes to Hugging Face

| HF path | Local source | Files | Size | Needed for |
|---|---|---|---|---|
| `README.md` | `scripts/hf_dataset_card.md` | 1 | — | dataset card |
| `derived/fall_clips/` | `data/derived/fall_clips/` | 27 | 12.5 MB | all model inference |
| `derived/trimmed_scale_clips/` | `data/derived/trimmed_scale_clips/` | 5 | 3.0 MB | all model inference |
| `derived/perturbed_clips/` | `data/derived/perturbed_clips/` | 2 | 88.7 MB | trimming / audit of $R_\alpha$ |
| `derived/resampled_clips/` | `data/derived/resampled_clips/` | 56 | 43.0 MB | X4 / T1 resampling |
| `raw_videos/02182026/` | `<RAW_SRC>/02182026/*.mp4, *.txt` (excl. unused `scale.mp4`) | 32 | 4.31 GB | oracle, clip extraction, rendering, X2 |
| *(optional)* `raw_videos/13.5cm_speed/05122026/` | `<RAW_SRC>/13.5cm_speed/05122026/` | 19 | 7.50 GB | oracle batch only (low-confidence clips, not evaluated) |
| *(optional)* `raw_videos/05052026/` | `<RAW_SRC>/05052026/` | 18 | 3.63 GB | legacy E3/E5 scripts only |

**Default upload: ~4.46 GB** (card + derived + 02182026). With both optional
sessions, ~15.6 GB.

`<RAW_SRC>` is the lab's local copy of the recordings: the folder containing
`02182026/`, `13.5cm_speed/` and `05052026/`.
`data/derived/` (git-ignored) holds a local copy of every derived clip, ready
to upload.

## Steps

1. **Decide**: the **data license** in `scripts/hf_dataset_card.md` (currently
   `other`), and whether to include the optional sessions. The dataset repo is
   `rasulkhanbayov/sgf` (already set in `code/sgf_paths.py`).
2. Log in: `huggingface-cli login` (or `export HF_TOKEN=...`; a write token).
3. Dry run (prints every file and the total size, uploads nothing):
   ```bash
   python scripts/upload_to_hf.py --repo rasulkhanbayov/sgf \
       --raw-src <RAW_SRC> --dry-run
   ```
4. Upload. Add `--private` to stage it privately first, and
   `--sessions 02182026 13.5cm_speed/05122026 05052026` to include the
   optional sessions:
   ```bash
   python scripts/upload_to_hf.py --repo rasulkhanbayov/sgf \
       --raw-src <RAW_SRC>
   ```
5. Verify from a fresh clone:
   ```bash
   python scripts/download_data.py --raw
   python code/oracle/validate_clip2.py data/raw_videos/02182026/cainhcg1.mp4 02182026
   # expect v_pre_impact_mps ~= 1.0834 (matches results/e2_selected_clips.json)
   ```

## Not uploaded (by design)

- The session folders' `Frames/`, `__Frames/` and `Spreading/` sub-folders, and
  `02182026/Results.csv`: these belong to the lab's separate spreading
  analysis and are not used by this code.
- Hugging Face model weights: they are pulled from the Hub at the pinned
  revisions on first run.
