---
license: cc-by-4.0
pretty_name: Droplet-impact videos for "One Perturbation Is Not Enough"
tags:
  - video
  - physics
  - vision-language-models
  - evaluation
---

# Droplet-impact videos

High-speed recordings of liquid droplets falling onto a superhydrophobic
surface, used in **"One Perturbation Is Not Enough: Identifiability and Blind
Baselines for Behavioral AI Evaluation"** (AAAI-27 Alignment Track) to test
whether vision-language models ground a velocity estimate in an in-frame
calibration reference.

Code, per-item model outputs and the reconciliation script:
**https://github.com/rasulkhanbayov/5-iclr**

## Layout

```
raw_videos/
  02182026/                 evaluation session (all model experiments)
    *.mp4                   one droplet per recording, 1280x512, ~2996 fps
    scale v.mp4             static scale-reference (ruler) recording
    samples.txt             sample key (surfactant / concentration codes)
  13.5cm_speed/05122026/    (optional) higher-drop session; oracle only
derived/                    clips rendered from raw_videos/ by the code repo
  fall_clips/               fall+impact window of each of the 27 evaluated clips
  trimmed_scale_clips/      3 s scale-reference clips (base, relabeled, resampled)
  perturbed_clips/          relabeled scale clips (R_alpha, alpha in {0.5, 2})
  resampled_clips/          resampled scale and fall clips (S_s, s in {0.5, 2})
```

## Key facts

- Ruler in session 02182026 is graduated in **1/16 inch (1.5875 mm) per
  tick**, not 1 mm; see the paper's Appendix B.2.
- Ground-truth pre-impact velocities come from a classical CV oracle
  (background subtraction + free-fall-plateau fit), not a learned model.
  Per-clip values are in the code repo's `results/e2_selected_clips.json`.
- The 27 evaluated clips are every high-confidence oracle track in 02182026.

## Download

```bash
git clone https://github.com/rasulkhanbayov/5-iclr.git && cd 5-iclr
python scripts/download_data.py          # derived clips (~150 MB)
python scripts/download_data.py --raw    # + raw 02182026 session (~4.4 GB)
```

## Citation

```bibtex
@inproceedings{khanbayov2027perturbation,
  title     = {One Perturbation Is Not Enough: Identifiability and Blind Baselines for Behavioral AI Evaluation},
  author    = {Khanbayov, Rasul and Sohail, Mariam and Abdala, Ahmed and Kurban, Hasan},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (Alignment Track)},
  year      = {2027},
}
```
