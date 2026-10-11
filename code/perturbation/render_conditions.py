"""
Render every perturbed/resampled clip the main experiments consume, from the
raw 02182026 recordings. Driver for perturb_renderer.py (R_alpha) and
resample_renderer.py (S_s); the original clips were rendered by calling those
functions interactively, with exactly these names and magnitudes.

Outputs (under data/derived/, see code/sgf_paths.py):
  perturbed_clips/02182026_relabel_alpha{0.5,2}.mp4      scale clip, relabeled
  resampled_clips/02182026_resample_s{0.5,2}.mp4         scale clip, resampled
  resampled_clips/<clip>_fall_resample_s{0.5,2}.mp4      each manifest fall clip
  fall_clips/<clip>_fall.mp4                             extracted if missing

Run afterwards: code/model_outputs/trim_scale_clips.py (trims the scale
clips to 3s for VLM input).

Usage: python3 code/perturbation/render_conditions.py [--skip-existing]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgf_paths  # noqa: E402  (code/sgf_paths.py: all repo paths)
sgf_paths.add_code_dirs_to_path()

from perturb_renderer import SESSION_LABEL_CONFIG, render_relabeled_scale_video  # noqa: E402
from resample_renderer import render_resampled_video  # noqa: E402
from prepare_clips import extract_fall_clip  # noqa: E402

SESSION = "02182026"
ALPHAS = [0.5, 2]  # R_alpha magnitudes (main evaluation)
SCALES = [0.5, 2]  # S_s magnitudes (resampling control)


def _run(out_path, fn, skip_existing):
    if skip_existing and os.path.exists(out_path):
        print(f"skip (exists) {out_path}", flush=True)
        return
    meta = fn()
    print(f"wrote {out_path}: {meta.get('n_frames')} frames", flush=True)


def main():
    skip = "--skip-existing" in sys.argv
    for d in (sgf_paths.PERTURBED_CLIP_DIR, sgf_paths.RESAMPLED_CLIP_DIR, sgf_paths.FALL_CLIP_DIR):
        os.makedirs(d, exist_ok=True)

    scale_clip = SESSION_LABEL_CONFIG[SESSION]["scale_clip"]
    if not os.path.exists(scale_clip):
        sys.exit(f"ERROR: {scale_clip} not found. Download the raw videos first "
                 f"(scripts/download_data.py --raw) or set SGF_RAW_VIDEO_DIR.")

    # Scale-reference clip: relabeled (R_alpha) and resampled (S_s) variants.
    for a in ALPHAS:
        out = os.path.join(sgf_paths.PERTURBED_CLIP_DIR, f"{SESSION}_relabel_alpha{a}.mp4")
        _run(out, lambda a=a, out=out: render_relabeled_scale_video(SESSION, a, out), skip)
    for s in SCALES:
        out = os.path.join(sgf_paths.RESAMPLED_CLIP_DIR, f"{SESSION}_resample_s{s}.mp4")
        _run(out, lambda s=s, out=out: render_resampled_video(scale_clip, out, s), skip)

    # Droplet fall clips: extract (if needed), then resample each by s.
    with open(sgf_paths.MANIFEST_PATH) as f:
        clips = json.load(f)
    for c in clips:
        name = os.path.basename(c["path"]).replace(".mp4", "")
        fall = os.path.join(sgf_paths.FALL_CLIP_DIR, f"{name}_fall.mp4")
        if not os.path.exists(fall):
            info = extract_fall_clip(sgf_paths.raw_video(c["path"]), fall, output_fps=15)
            if "error" in info:
                print(f"SKIP {name}: {info['error']}", flush=True)
                continue
        for s in SCALES:
            out = os.path.join(sgf_paths.RESAMPLED_CLIP_DIR, f"{name}_fall_resample_s{s}.mp4")
            _run(out, lambda s=s, out=out: render_resampled_video(fall, out, s), skip)


if __name__ == "__main__":
    main()
