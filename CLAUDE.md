# SGF (Scale-Grounded Faithfulness) — ICLR Submission

## What this is
ICLR D&B-style submission introducing **Scale-Grounded Faithfulness (SGF)**: a label-free
protocol for testing whether a VLM's reported physical measurement from video is actually
anchored to the in-frame scale reference, vs. guessed from a learned prior. Testbed: droplet
impact videos with an in-frame ruler/comb calibration target and a known-but-off-frame
hidden-prior trap (release height).

Read `README.md` (student-facing overview, claims, integrity checklist) and `sgf_runbook.txt`
(experiment order E1-E5, flag-back triggers) before making any changes to the paper or protocol.
`main.tex` has red `\torun{}` and blue `\result{}` macros marking what's still placeholder.

## Environment
- GPU: 1x A100 80GB (idle unless a model is loaded).
- **Root disk (`/`) is nearly full — 93% used, ~7GB free.** Never install large packages,
  download model weights, or write bulk output there.
- **`/data` is on the SAME root filesystem as `/`** (not a separate mount) — don't treat it as
  spacious. It holds a pre-existing venv (`/data/venv`, has `torch`+CUDA, `opencv-python-headless`)
  and a SAM2 checkpoint (`/data/checkpoints/sam2.1_hiera_large.pt`) from a separate, unrelated,
  still-ongoing project — do not touch or reuse that project's data.
- **`/ephemeral` is the real large volume** (676GB free). All SGF working files, model caches,
  and outputs go under `/ephemeral/sgf_work/`.
- Python env: activate with `source /data/venv/bin/activate`. Has `torch`, `transformers`,
  `accelerate`, `qwen-vl-utils`, `opencv-python-headless` installed (transformers/accelerate/
  qwen-vl-utils were added this session — they were missing initially).
- `HF_HOME` should point to `/ephemeral/sgf_work/hf_cache`, not the pre-existing `/data/hf_cache`
  default (same root-disk problem). Set explicitly per-command if `~/.bashrc`'s export doesn't
  propagate (non-interactive shells skip `.bashrc`'s early-exit guard).
- When running batch/parallel CV jobs: cap OpenCV+BLAS threading per worker
  (`cv2.setNumThreads(1)`, `OPENBLAS_NUM_THREADS=1` etc. as env vars set at shell level before
  the process starts) or many-worker parallelism oversubscribes the 28-core box badly.

## Dataset scope — which sessions/clips are actually usable
Source videos: `/ephemeral/videos/{02182026,03242026_particlesonlypreparedinsurfactant,
05052026,05172026,new_videos}/`.

**Valid for SGF (3 sessions):**
| Session | Scale-ref clip | Target type | Calibration | Frame rate |
|---|---|---|---|---|
| `02182026` | `scale v.mp4` (NOT `scale.mp4`) | ruler, 1mm ticks | 104.52 px/mm | 2996 fps |
| `05052026` | `scale2.mp4` (NOT `scale.mp4`) | ruler, 1mm ticks | 66.19 px/mm | 2996 fps |
| `new_videos/05122026` | `scale2.mp4` | pin comb, 1/16"=1.5875mm pitch | 42.99 px/mm | 2996 fps |

**Excluded:**
- `new_videos/05112026` — its `scale.mp4` is corrupted (moov atom missing, confirmed unreadable
  by OpenCV and ffmpeg). Orphans 8 droplet clips.
- `05172026` — no scale video, and its `experiment details.txt` is a byte-identical duplicate of
  `new_videos/experiment details.txt` (describes 05/11-05/12, not this session). No verified
  provenance.
- `03242026_particlesonlypreparedinsurfactant` — no scale video anywhere in the folder.
- `new_videos/05122026/cain0.028tx3.mp4` — corrupted (same moov-atom issue), excluded but rest
  of that session is fine.

**Critical, easy-to-get-wrong facts:**
- Every video's container metadata reports **60fps** — this is WRONG. True capture rate is
  **2996fps** (confirmed by the user). Using 60fps makes velocities ~50x too slow.
- `scale.mp4` is NOT reliably the calibration clip — verify visually per session before trusting
  a filename convention. Two calibration target styles exist (printed ruler vs. pin comb) and
  need different pixel-measurement logic.
- Drop height varies by session and the written documentation is NOT always correct:
  - `05052026`: 6.5cm (documented, confirmed by physics match: measured 1.130 m/s vs.
    predicted 1.129 m/s).
  - `02182026`: undocumented, but consistent ~0.680 m/s cluster across 22 clips (shorter than
    6.5cm; height itself never independently confirmed, just internally consistent).
  - `new_videos/05122026`: documented as 13.5cm in `experiment details.txt` — **this is WRONG**,
    confirmed by the user. Actual height matches 6.5cm (measured ~1.073 m/s cluster).

Full detail and reasoning trail: see the Claude memory files `project_sgf_dataset_scope.md`,
`project_sgf_frame_rate.md`, `project_sgf_oracle_batch_results.md`.

## CV Oracle (built from scratch — do NOT reuse the pre-existing sam2/cv CSVs in
`/ephemeral/videos/05052026/*_sam2.csv` / `*_results.csv` / `compare_summary.csv` — those belong
to a separate, unrelated, still-in-progress project and the two trackers in them disagree badly
with each other; ignore them entirely)

Code: `/ephemeral/sgf_work/oracle/droplet_oracle.py`. Key pieces:
- `SESSION_CALIBRATION_PX_PER_MM` — the 3 measured calibration values above.
- `TRUE_FPS = 2996.0`.
- `detect_droplet_blobs` — background-subtraction blob detector.
- `find_motion_window` — **two-pass**: sequential coarse scan (reads every frame, scores every
  Nth) to locate the fall+impact region cheaply, then a small seek-based fine scan to pin down
  exact start/end. Do NOT go back to seeking (`cap.set(POS_FRAMES)`) for the coarse pass — on
  long clips (some have 5,000-12,000+ frames) that made the coarse scan itself the bottleneck
  (minutes per clip).
- `link_primary_droplet` — greedily links blobs frame-to-frame by nearest centroid. Seeds from
  the first non-static blob found by scanning ALL blobs in ALL frames (not just single-blob
  frames — some clips have a persistent static artifact, e.g. a dust speck, that coexists with
  the real droplet in every frame the droplet is visible, so single-blob-only seeding can never
  find it). See `_is_static_artifact`.
- `compute_pre_impact_velocity` — finds the free-fall plateau just before impact (NOT a
  whole-window average, which underestimates due to gravity acceleration; NOT the frames right
  at the trajectory peak either, which are already inside the contact/deceleration cushion and
  underestimate the other way). Falls back to a `low_confidence: True` plain average when the
  track is too short to see the impact signature at all (a real, valid measurement, just noisier
  and unverified against a deceleration cutoff — this affects most of `new_videos/05122026`,
  where the droplet is only trackable for 3-4 frames before detection is lost).

Batch runner: `run_batch.py` (supports `--skip-file <path>` to resume after a kill, since one
run oversubscribed the CPU badly and had to be restarted with `cv2.setNumThreads(1)` + a fixed
sequential coarse scan). Full results in `/ephemeral/sgf_work/oracle/results/oracle_batch_results.json`
(62 entries, one per non-scale clip across the 3 valid sessions).

**Oracle validation results** (cross-checked against free-fall physics, v = sqrt(2gh)):
- `05052026`: 12/17 valid, mean 1.130 m/s — matches 6.5cm prediction almost exactly. Most
  trustworthy session.
- `02182026`: 22/30 valid, mean 0.680 m/s — internally consistent, undocumented height.
- `new_videos/05122026`: 14/15 valid (after fixing the static-artifact + short-track bugs above),
  mean 1.073 m/s, std 0.022 — tight cluster, confirmed 6.5cm not the documented 13.5cm.

**Known remaining gaps** (not yet individually root-caused): 3 genuinely empty captures in
`02182026` (`cainhcg5`, `cainhcg3`, `cainhtx3` — no droplet visible anywhere in the clip, likely
a missed dispenser trigger), ~10 clips across sessions still return `v=None` from linking/
plateau-detection edge cases not yet dug into individually.

## Perturbation renderer (Phase 2)
Code: `/ephemeral/sgf_work/perturbation/perturb_renderer.py`. Edits ONLY the scale-reference clip
(never the droplet clip). Ruler-style sessions (02182026, 05052026): inpaint out the printed
number once (ruler is static all clip — reuse the same patch for every frame, since per-frame
`cv2.inpaint` costs ~1.3s/call and made a naive version take 20+ min for one clip), draw a new
number at the same position. Comb-style session (new_videos/05122026, no printed numerals):
add a text banner overlay stating the new pin pitch instead, in the empty space above the pins.

**Relabeling formula: `new_label = alpha * original_label`** (e.g. alpha=0.5 on "30" -> "15").

**CRITICAL unit trap:** the paper's `c` is mm-per-pixel; the oracle's `SESSION_CALIBRATION_PX_PER_MM`
is pixels-per-mm (the reciprocal). When `c -> alpha*c` (paper's convention), the oracle's
`px_per_mm -> px_per_mm / alpha` (divide, NOT multiply — multiplying inverts the whole SFS sign).
Verified by hand (runbook's mandatory pre-scale check): using a real oracle measurement and the
correct `px_per_mm/alpha` substitution, a simulated faithful re-measurer's velocity ratio under
`c` vs `alpha*c` came out to exactly `alpha`, giving SFS=1.0 as expected. Full derivation in
Claude memory `project_sgf_perturbation_renderer.md` — don't re-derive this from scratch, the
sign is easy to get backwards.

Rendered perturbed scale clips (alpha in {0.5, 2}) go in `/ephemeral/sgf_work/perturbed_clips/`.

## Experiment plan status (per sgf_runbook.txt's E1-E5 order)
- **Phase 0 (env setup):** done.
- **Phase 1 (oracle build + validation):** done — see above.
- **Phase 2 (perturbation renderer):** done — see above. Hand-verification against the SFS
  metric passed (SFS=1.0 for a simulated faithful re-measurer).
- **Phase 3 (E1 positive control):** DONE, PASSED. Fed 48 real oracle velocity measurements
  through `sgf_metrics.py`'s `scale_faithfulness_score` as (a) a faithful re-measurer
  (`oracle*alpha`) and (b) a fixed-prior ~1m/s emitter. Result: SFS=1.0 (CI [1.0,1.0]) for (a),
  SFS=0.032 (CI [0.018,0.048]) for (b), paired sign test 48/48 p=7e-15. Confirmed robust across
  alpha in {0.5,2} and 5 random seeds. Report: `/ephemeral/sgf_work/results/e1/e1_positive_control.json`.
  This was the blocking gate — E2 (real model runs) can now proceed.
- **Phase 4 (E2 main table):** DONE for Qwen2.5-VL-7B-Instruct (no API keys available in this
  environment; user confirmed open-weight-only scope). 108 calls: 12 clips (6 from 02182026, 6
  from 05052026 — new_videos/05122026 excluded, all its oracle ground truth is low_confidence)
  x 3 conditions (base, alpha=0.5, alpha=2) x 3 prompt seeds. Results:
  **Acc@10% = 0.0**, **CG = 0.222** (CI [0.148, 0.306]), **SFS = 0.131 (alpha=0.5) / 0.107
  (alpha=2)**. Raw + computed metrics in `/ephemeral/sgf_work/results/e2/`. Found (and ruled out
  as a pipeline bug) that the model's answers vary little across genuinely different droplet
  clips within a condition group — it CAN perceive/describe both videos correctly when asked
  directly, but doesn't reliably perform the compositional measurement task. See
  `project_sgf_e2_results.md` / `project_sgf_e2_finding.md` in memory for full detail. To add
  another model (API or larger open-weight), reuse `run_e2_batch.py` / `run_qwen_inference.py`
  as templates.
- **Phase 5 (E3 hidden-prior trap):** DONE (v3, root-caused). v1's HPH=0.0 was uninterpretable —
  the model said "UNKNOWN" even with NO video attached, so it was a content-independent default,
  not a real decline. Fixed by pairing the hidden-prior question with a CONTROL question in the
  same call (`HIDDEN_PRIOR_WITH_CONTROL_PROMPT` in `prompt_design.py`): "does the droplet impact
  the surface?" — always true in this dataset, trivially visible. Only trust a decline on items
  where the model answers the control correctly.
  **v2/v3 result: HPH = 0.0 on the 7/12 items with a control-verified reliable video read.** The
  other 5/12 (5 of 6 from 05052026, only 1 from 02182026) got the control WRONG.
  **Investigated across three rounds; final root cause confirmed by direct frame comparison, NOT
  a pipeline bug.** Hand-labeled all 12 clips (per user direction to stop CV-tuning a multi-droplet
  detector after 6 failed rounds — see `project_sgf_e3_results.md` for that abandoned effort) and
  found only 1/12 clips (`cainhcg 0.08 b.mp4`) is genuinely multi-droplet. Built and validated a
  per-clip settle-based extraction-window fix (`tune_extraction_window.py`, replacing the fixed
  `pre_margin=15/post_margin=15` in `prepare_clips.py`) and reran as v3 with longer, more-settled
  clips (78-135 frames vs ~40-70 originally) — **the control-answer pattern came back byte-for-byte
  identical to v2**, refuting rebound-timing as the cause. Direct frame-by-frame comparison then
  showed the real cause: 05052026's camera framing compresses the impact surface into a thin,
  heavily vignette-darkened strip at the very bottom frame edge (vs. 02182026's surface sitting
  comfortably in a well-lit middle-lower third), making "does it touch the surface" genuinely hard
  to verify from that footage regardless of timing. **This is a property of the raw footage/camera
  setup, not something further pipeline tuning can fix** — report as a dataset limitation alongside
  HPH, and treat as a legitimate finding about VLM robustness to low-contrast/edge-of-frame
  evidence rather than a defect to engineer away. Full detail in memory `project_sgf_e3_results.md`.
  Results: `/ephemeral/sgf_work/results/e3/e3v3_qwen_results.json` + `e3v3_metrics_summary.json`
  (v2 kept for comparison, same numbers).
  **UPDATE (2026-07-06), improvement plan A2: partially fixed the 05052026 framing issue.** Global
  contrast fixes (linear stretch, CLAHE) barely helped — the whole frame has low dynamic range
  (peak pixel ~90/255), not just the surface strip. What worked: crop tightly to the impact region
  (via the oracle's own track-linking) and contrast-stretch only that local crop, per frame.
  Rerunning the control question on the 5 enhanced clips recovered **1/5** (`0.08cg3.mp4`) — HPH
  coverage improves to **8/12** (was 7/12). Of the remaining 4: 1 is the confirmed multi-droplet
  clip (`cainhcg 0.08 b.mp4`, unfixable by contrast since the ambiguity is about which droplet, not
  visibility); the other 3 still fail despite visibly clear enhanced crops, plausibly because the
  necessarily small crop (~440x420px) gets downsampled by the model's default video preprocessing
  in a way that undoes the local contrast gain — not confirmed, reported as an open question, not
  chased further (avoided repeating the earlier multi-droplet-detector many-rounds-diminishing-
  returns pattern). **main.tex updated**: E3 discussion gets a new "partially recoverable by
  post-processing" paragraph; Limitations' E3 paragraph now states 8/12. Code:
  `enhance_e3_clips.py`, `run_e3_enhanced_retest.py`. Results: `fall_clips_enhanced/`,
  `e3_enhanced_retest_results.json`. Full detail: memory `project_sgf_e3_a2_enhancement.md`.
  **UPDATE (2026-07-06), improvement plan B2: root-caused E2's calibration-grounding (CG) session
  asymmetry (Qwen CG=0.0 on 05052026 vs 0.44 on 02182026) — CONFIRMED DIFFERENT cause from E3's,
  not the same one.** E3's cause was footage visibility; this is a task-comprehension error:
  Qwen literally echoes the ruler's PRINTED NUMBER as its calibration answer instead of computing
  tick spacing. Deterministic, not noisy: 18/18 base-condition 05052026 answers are exactly 30.0
  or 3.0 (ruler prints "30", true answer always 1.0); at alpha=0.5 (relabeled "15"), 18/18 answers
  are exactly 3.75=15/4. On 02182026 (ruler prints "12"), wrong answers are small plausible
  mm-per-tick GUESSES (0.5/1.0) that sometimes coincidentally match the true value — a
  qualitatively different error. Gemini shows a much smaller session gap (0.43 vs 0.38),
  consistent with this being Qwen-specific label-echoing, not a footage property. **main.tex
  updated**: new paragraph after the E2 table calling this "the clearest single illustration...
  of the behavior SGF is designed to catch." Code: `analyze_b2_cg_asymmetry.py`. Full detail:
  memory `project_sgf_b2_cg_asymmetry.md`.
- **Phase 6 (E4 decoupling analysis):** DONE, with an important caveat. Runbook objective:
  correlate per-item Acc@10% with per-item SFS across E2's 36 (clip x seed) items.
  **The literal question is unanswerable at the 10% threshold** — Acc@10%_hit is 0 for all 36/36
  items (matches E2's Acc@10%=0.0), a constant with zero variance, so correlation against it is
  mathematically undefined (NaN), not "no correlation." Substituted continuous relative error
  (negated) as the closest computable stand-in for "accuracy": **r(-rel_err, SFS) = -0.48, 95% CI
  [-0.71, -0.03]** (excludes zero) — a significant NEGATIVE relationship, i.e. neither the
  hypothesized "decoupled" (r~0) story nor simple positive coupling.
  **This correlation is CONFOUNDED, not a clean result** — cross-checked against the E2
  near-constant-response finding and confirmed `base_velocity_mps` only takes 3 discrete values
  across all 36 items, driven by `seed_idx` almost independent of clip content (seed 1 always
  outputs 3.5 m/s regardless of clip). The scatter plot
  (`/ephemeral/sgf_work/results/e4/e4_scatter.png`) shows SFS clustering into discrete bands
  aligned with seed color, not with relative error continuously — both axes are likely driven by
  the same seed-dependent non-responsiveness, not a genuine accuracy/faithfulness tradeoff. Per
  the runbook's honesty rule, do NOT report r=-0.48 as clean support for "coupled" or "decoupled"
  — report that the decoupling question couldn't be meaningfully tested for this model, as a
  further symptom of E2's non-responsiveness finding, not an independent result. Should be
  re-run once a second model is added (this analysis only has one model to correlate "across").
  Code: `/ephemeral/sgf_work/model_outputs/run_e4_decoupling.py`. Results:
  `/ephemeral/sgf_work/results/e4/e4_decoupling.json` + `e4_scatter.png`.
  **UPDATE (2026-07-06): second model added via OpenRouter, re-ran E2+E4.** User provided an
  OpenRouter API key ($31.30 budget) specifically to close the single-model gap. Added
  `google/gemini-2.5-flash-lite` (cost-checked first: ~$0.0003/call, full 108-call E2 batch cost
  only $0.0324 total). **Result: does NOT resolve the degeneracy, generalizes it** — Acc@10%=0 for
  Gemini too (36/36) and for the 72-item pool (72/72). Gemini's failure mode is related but
  distinct from Qwen's: on 05052026 it reports calibration=1.0 across ALL conditions (complete
  non-response), on 02182026 it never once reports the correct alpha=2 value (2.0) even though the
  on-screen "24" label was confirmed legible by direct frame extraction — both models default to a
  physically-plausible prior instead of reading the digits, just different priors. Pooled r(-rel_err,
  SFS) = -0.460 (CI [-0.664,-0.094], excludes 0); Gemini-alone r=-0.218 (CI [-0.671,0.063],
  INCLUDES zero, not significant alone). Same confound caveat applies to Gemini too. **Secondary
  finding**: Gemini is much more accurate at looser tolerance (Acc@30%=0.333 vs Qwen's 0.028)
  despite both being 0 at Acc@10%. **main.tex updated accordingly** (abstract, contributions,
  E2/E4 protocol + results, Limitations) — paper now reports a genuine two-model finding, not a
  single-model limitation. Code: `run_openrouter_inference.py`, `run_e2_batch_openrouter.py`,
  `run_e4_multimodel.py`. Results: `e2_gemini_flash_lite_results.json`,
  `e4_multimodel_decoupling.json`. Full detail: memory `project_sgf_openrouter_e2_e4_results.md`.
  **UPDATE (2026-07-06), improvement plan A1: checked whether the Acc@30% lead actually rescues
  the decoupling analysis at a looser tolerance — it does NOT.** Swept tau from 10% to 100% on the
  same already-collected E2 data (zero new API calls). Correlation becomes computable at
  tau>=15-30% but is NEGATIVE at every tolerance where measurable (Gemini r=-0.415 at tau=30%, CI
  excludes 0; pooled r ranges -0.10 to -0.25 across tau in [15%,75%]) — never resembles the
  originally hypothesized decoupled pattern. Worse, confirmed Gemini's Acc@30% "hits" are ALL from
  one session (05052026) whose true velocities happen to sit near Gemini's favorite constant
  output — the same near-constant-palette confound, not real accuracy gain. **main.tex updated**:
  abstract, E4 discussion, and the "reconciling headline claim" paragraph now state this directly
  instead of leaving it as an open lead. Code: `run_e4_tau_sweep.py`. Results: `e4_tau_sweep.json`.
  Full detail: memory `project_sgf_e4_tau_sweep.md`.
- **Phase 7 (E5 robustness):** DONE. Objective: SFS stability across alpha, ruler orientation
  (horizontal vs vertical), and prompt phrasing. Scoped to 2 representative clips
  (water.mp4/02182026, 0.08cg2.mp4/05052026) x alpha in {0.25, 0.5, 2, 4} x orientation
  {original, synthetically rotated} x 3 prompt seeds = 48 items (60 calls incl. base), per
  user-agreed scope — E5 tests stability, not per-clip coverage (already covered by E2).
  Built a new synthetic ruler-rotation mode in `perturb_renderer.py`
  (`rotate_orientation` param) since both real sessions physically photograph a vertical ruler
  rotated in-frame — no naturally-occurring horizontal-ruler footage exists, so this rotates the
  same real ruler 90°. Caught and fixed a real bug along the way: an initial attempt pre-rotated
  the label before rotating the whole frame, which got the digit's rotation right but its
  position wrong (clipped/cut-off labels). Fixed by erasing the original digit in original
  coordinates, rotating the erased frame, then drawing the new label fresh directly in the
  rotated frame's own coordinates.
  **Result: SFS stable (low) across orientation (range 0.046) and prompt seed (range 0.087) —
  genuine robustness evidence.** Alpha axis shows a much bigger apparent range (0.331, driven by
  an alpha=4 spike to SFS=0.396) but **this is a coincidental artifact, not a real effect** — the
  model's velocities are drawn from a tiny fixed palette (0.3, 0.5, 0.6, 0.84, 1.5, 1.875, 3.5
  m/s) barely tied to the actual displayed calibration (same pathology as E2/E4). At alpha=4
  specifically, every single item (both clips, both orientations, all seeds) happened to land on
  the exact same (base=0.5, scaled=1.5) pair — a content-independent 3x ratio that coincidentally
  sits close to alpha=4's ideal target, scoring high by luck; other alphas' pairs land far from
  their own targets. Do NOT report the alpha spread as "more faithful under strong perturbation."
  Code: `/ephemeral/sgf_work/model_outputs/run_e5_batch.py` + `run_e5_analysis.py`. Results:
  `/ephemeral/sgf_work/results/e5/e5_qwen_results.json` + `e5_stability_table.json`.
  **UPDATE (2026-07-06), improvement plan B1: broadened E5 from 2 to all 12 E2 clips (360 calls,
  zero new API cost, local Qwen only).** Overall SFS (0.184) and CG (0.132) closely matched the
  2-clip sample, and orientation/seed stability held up on 6x more data. **But revealed a new,
  much bigger instability axis unavailable with only 2 clips: SFS by session ranges from 0.058
  (all six 02182026 clips, IDENTICAL value, not just similar) to 0.311 (all six 05052026 clips) —
  range 0.253, bigger than the already-flagged alpha=4 artifact.** Root cause confirmed directly:
  02182026 outputs 1.5 m/s in 34-36/36 items at EVERY alpha tested (essentially zero response to
  any perturbation strength, mechanically flooring its SFS); 05052026 shows more incidental spread
  under perturbation that coincidentally scores higher, not genuine faithfulness — same
  "near-constant-palette landing near a target by luck" phenomenon as the alpha artifact, now
  clip/session-dependent instead of alpha-dependent. Does NOT contradict the orientation/seed
  stability findings (still real, strengthened by more data) but shows clip/session choice
  matters more than any condition tested. **main.tex updated**: E5 discussion gets a new
  "broadened to all 12 clips" paragraph, Table 4 gets a new session row, Limitations' dataset-
  scale paragraph now cites this finding. Code: `run_e5_batch_b1_full12.py`,
  `run_e5_analysis_full12.py`. Results: `e5_qwen_results_full12.json`,
  `e5_stability_table_full12.json`. Full detail: memory `project_sgf_b1_full12_e5.md`.
- **Phase 8 (write-up):** DONE. Reconciled every `\torun{}`/`\result{}` placeholder in `main.tex`
  with the real E1-E5 numbers. **Key framing decision (user-approved)**: reframed from the
  originally hypothesized "decoupled" headline to an honest single-model null-result narrative —
  only one open-weight model was evaluated (no API keys available), Acc@10%=0 for every E2 item
  leaves E4's correlation with no accuracy variance to run against (mathematically undefined, not
  "decoupled"), and the substitute correlation is confounded by the same non-responsiveness
  documented in E2. Rewrote: abstract, intro contributions, dataset section (real session/clip
  counts replacing the placeholder), Experimental Protocol (E1-E5 described as actually run, not
  as originally planned), Results and Discussion (written from scratch — 3 tables, full E1-E5
  writeup, explicit reconciliation of the headline claim), and Limitations (expanded to 6
  paragraphs led by the single-model/no-API gap). Left the `sgf.bib` `% VERIFY` author-list
  placeholders and Table 1's `[UNVERIFIED]` novelty tag untouched — both are explicitly flagged in
  README.md as human-only steps (fabricating them would itself be an integrity violation).
  Compiles cleanly with pdflatex (3 passes, 12 pages, no undefined refs/citations). Fixed one real
  bug found during compilation: a literal `°` unicode character silently failed to render in math
  mode under this document's font setup — replaced with `90^{\circ}`. Full detail in memory
  `project_sgf_writeup.md`.
  **Remaining before submission (per README's integrity checklist, human-only)**: arXiv novelty
  re-confirmation, bib author-list fill-in, human mock reviews.

## Working conventions
- All large/temp files go under `/ephemeral/sgf_work/`, never in this repo folder or `/data`.
- Before trusting any pixel measurement (calibration, frame rate, drop height), verify against
  ground truth or ask — this codebase has repeatedly had wrong assumptions baked into filenames
  and documentation (see Dataset Scope section above). Don't silently trust a filename or a
  session's own notes file without a physics or visual cross-check.
- Long-running CV jobs (motion scanning, batch oracle runs) should run backgrounded with
  incremental result writes, since some clips are large enough that a full run takes many
  minutes and killing/resuming is common.
