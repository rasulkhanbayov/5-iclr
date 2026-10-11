# Dataset examples

A small set of real frames and short clips from the evaluation session
(02182026), included so a reader can see the raw material without
downloading the full dataset (raw video is on Hugging Face; fetch it with
`python scripts/download_data.py --raw`, see the main README and
Appendix B.2 of the paper).

## videos/

Each fall clip is trimmed to its motion window (a few dozen frames around
the droplet's fall and impact) and re-encoded at a modest bitrate; the
untrimmed session recordings are ~100-500MB each at the original capture
bitrate and are not included here.

- `cainhcg1_fall_02182026.mp4` -- one of the 27 clips used in the main
  evaluation (E2/X4/X10/T1/T_beta). True pre-impact velocity 1.083 m/s.
- `caonly2_fall_02182026.mp4` -- a second evaluation clip from a different
  naming family, included to show the clip-to-clip variation in droplet
  size/position within the frame.
- `cainlsds1_fall_02182026.mp4` -- a third evaluation clip, again from a
  distinct naming family.
- `scale_reference_02182026.mp4` -- the static scale-reference clip shown
  to the model alongside every droplet clip (trimmed to 2s; the ruler does
  not move across the session, so the full recording carries no additional
  information). This is the clip whose 1/16-inch graduation was originally
  misidentified as 1mm; see main paper Appendix B.2 for the correction.

## images/

Individual frames extracted from the videos above, at native resolution
(1280x512), unannotated.

- `cainhcg1_frame392_midfall.png` / `cainhcg1_frame397_impact.png` -- the
  same clip 5 frames (1.7ms at the true 2996fps capture rate) apart, showing
  the droplet as a sphere mid-fall and flattened at impact.
- `caonly2_frame280_midfall.png` / `caonly2_frame288_impact.png` -- the same
  pair for a second clip.
- `cainlsds1_frame241_midfall.png` -- mid-fall frame for a third clip.
- `scale_reference_frame10.png` -- a frame from the scale-reference clip
  showing the ruler's "12" label and four adjacent tick marks. Note the
  ticks are NOT uniform length (three short, one long) -- this is the
  visual signature of 1/16-inch graduation, confirmed by direct pixel
  measurement (Appendix B.2), not the uniform 1mm spacing originally
  assumed.

## Provenance

All frames/clips are drawn directly from the session used in model
evaluation (02182026); no synthetic or illustrative content. Frame indices
and true velocities match the values in `../results/oracle/oracle_batch_results.json`
and `../results/e2_selected_clips.json`.
