# SGF Project — Full Knowledge Dump

Written 2026-07-31 as a complete-state snapshot, prompted by `/ephemeral` (the
676GB working volume this project used all session) being wiped. This file
exists so that everything learned across the project — not just what's in
`main.tex` — survives even if this project folder or the Claude memory store
is later lost. See the "Reproducibility audit" section at the end for exactly
what is and isn't recoverable from what remains.

## 1. What this project is

An ICLR Datasets & Benchmarks submission introducing **Scale-Grounded
Faithfulness (SGF)**: a label-free protocol for testing whether a VLM's
reported physical measurement from video is genuinely anchored to an in-frame
scale reference, or guessed from a learned prior. Testbed: high-speed droplet-
impact videos with an in-frame ruler/comb calibration target and a known-but-
off-frame hidden-prior trap (release height).

Core mechanism: relabel the in-frame scale so its asserted calibration changes
by a known factor α, while the object's actual pixel motion is untouched. A
faithful measurer's reported quantity must then change by α^d (d = the
quantity's dimension in length, d=1 for velocity) — a closed-form, judge-free
target response (Proposition 1 in `main.tex`).

Three reported quantities: **SFS** (Scale-Faithfulness Score, response to the
perturbation), **CG** (Calibration-Grounding, did it read the displayed scale
correctly, independent of the final number), **HPH** (Hidden-Prior
Hallucination, does it invent an off-frame quantity instead of declining).

## 2. Files in this folder (what's here now)

- `main.tex` / `main.pdf` — the paper. Compiles cleanly with `pdflatex`
  (2 passes), 15 pages, no undefined refs/citations as of the last edit.
- `sgf.bib` — references. Every entry has a `% VERIFY author list` placeholder
  — **never filled in**, deliberately; this is flagged in `README.md` as a
  human-only step, and fabricating author lists would be an integrity
  violation.
- `sgf_metrics.py` — the metric library (SFS, CG, HPH, bootstrap CI, paired
  sign test). numpy-only, self-contained, has a synthetic smoke test
  (`python3 sgf_metrics.py`) that verifies the metrics separate archetypal
  faithful/prior-driven/partial behaviors. Confirmed still runs correctly
  (2026-07-31, using `/data/venv`, not the system Python which lacks numpy).
- `README.md` — student-facing overview: claims, integrity checklist, how to
  build.
- `sgf_runbook.txt` — the E1–E5 experiment order and flag-back triggers this
  whole project executed against.
- `CLAUDE.md` — the running project-state tracker maintained across the whole
  session; has a phase-by-phase log of Phases 0–8 plus improvement-plan
  Phases A/B (all done) and C (blocked, see below).
- `code/` — a curated copy of the actual pipeline code and every results JSON
  from `/ephemeral/sgf_work/`, organized as `oracle/`, `perturbation/`,
  `model_outputs/`, `results/e1..e5/`. This is the part that was deliberately
  synced from the ephemeral working directory into git so it would survive
  volume loss. Large binary artifacts (raw videos, rendered perturbed clips,
  the HF model cache) were deliberately **excluded** — see Section 5.
- `.env` — **contains a live OpenRouter API key** (`sk-or-v1-...`), gitignored,
  never committed. Balance as of last check: ~$681 remaining of a $700 limit,
  usage this session so far ~$18.34.
- `.gitignore` — excludes LaTeX build junk, `__pycache__`, `.claude/` (local
  session state), `sgf_submission.zip` (a stale pre-project bundle), and `.env`.
- `sgf_submission.zip` — a **stale**, pre-project build from 2026-06-16,
  predates all the real experimental work. Not in git. Safe to delete
  independently of everything else in this file.
- `__pycache__/` — compiled bytecode cache, not meaningful, gitignored.

## 3. What was done, phase by phase

**Phase 0 (env setup):** done. Environment: 1x A100 80GB GPU, root disk (`/`)
93% full (only ~7GB free, never write bulk data there), `/data` is a separate
mount with a pre-existing venv (`torch`+CUDA, `opencv-python-headless`) and an
unrelated project's SAM2 checkpoint (do not touch), `/ephemeral` was the real
676GB working volume (now wiped, see Section 5).

**Phase 1 (oracle):** built `droplet_oracle.py` from scratch — background-
subtraction blob detection, two-pass motion-window localization, a droplet-
track linker robust to static artifacts (dust specks) and multi-droplet
frames, and a free-fall-plateau velocity estimator (not a whole-window average,
which underestimates due to gravity; not the frames at the trajectory peak,
which are already in the contact/deceleration cushion). True capture rate is
**2996fps** — container metadata falsely reports 60fps; this was independently
confirmed and is critical (using 60fps makes velocities ~50x too slow).
Batch-ran across all valid sessions: 62 attempted clips, 48 valid (77%), 34
high-confidence (verified deceleration signature) + 14 lower-confidence
(short-track plain averages). Every oracle value cross-checked against
free-fall physics (v=√(2gh)).

**Dataset scope:** 5 recorded sessions existed; only 3 are valid.
- `02182026` (`scale v.mp4`, NOT `scale.mp4` — verify visually, don't trust
  filename convention): printed ruler, 1mm ticks, 104.52 px/mm, undocumented
  drop height (internally consistent ~0.680 m/s cluster across 22 clips).
- `05052026` (`scale2.mp4`): printed ruler, 1mm ticks, 66.19 px/mm, drop height
  6.5cm (documented AND physics-confirmed: measured 1.130 m/s vs predicted
  1.129 m/s).
- `new_videos/05122026` (`scale2.mp4`): pin comb, 1/16"=1.5875mm pitch, 42.99
  px/mm, documented height 13.5cm is **WRONG** (confirmed by user) — actual
  height matches 6.5cm (measured ~1.073 m/s cluster). Excluded from model
  evaluation because its oracle ground truth is predominantly low-confidence
  (droplet too close to impact when entering frame).
- **Excluded**: `new_videos/05112026` (corrupted `scale.mp4`, unreadable moov
  atom, confirmed by both OpenCV and ffmpeg, orphans 8 droplet clips);
  `05172026` (no scale video, its notes file is a byte-identical duplicate of
  another session's, describing the wrong dates — no verified provenance);
  `03242026_particlesonlypreparedinsurfactant` (no scale video at all).

**Phase 2 (perturbation renderer):** `perturb_renderer.py`. Relabels the
printed ruler number (inpaint old digit, draw new one) for ruler sessions, or
adds a text banner for the pin-comb session (nothing printed to inpaint).
**Critical unit trap, verified by hand**: the paper's `c` is mm-per-pixel; the
oracle's calibration constant is pixels-per-mm (the reciprocal). When
`c → alpha*c`, the oracle math needs `px_per_mm → px_per_mm/alpha` (divide,
NOT multiply — multiplying inverts the whole SFS sign). Hand-verified: a
simulated faithful re-measurer's velocity ratio under `c` vs `alpha*c` came out
to exactly `alpha`, giving SFS=1.0 as expected.

Also built (later, for E5) a **synthetic ruler-rotation mode**
(`rotate_orientation` param): both real sessions physically photograph a
*vertical* ruler rotated in-frame — no naturally-occurring horizontal-ruler
footage exists — so this rotates the same real ruler 90°. Real bug caught and
fixed: an initial attempt pre-rotated the label before rotating the whole
frame (digit rotation right, position wrong/clipped). Fixed by erasing the
original digit in original coordinates, rotating the erased frame, then
drawing the new label fresh directly in the rotated frame's own coordinates.

**Phase 3 (E1, positive control):** PASSED. Fed 48 real oracle velocities
through (a) a simulated perfectly-faithful re-measurer (`oracle*alpha`) and
(b) a fixed ~1 m/s prior-emitter. Result: SFS=1.000 (CI [1.000,1.000]) for (a),
SFS=0.032 (CI [0.018,0.048]) for (b), paired sign test 48/48, p=7.1e-15.
Confirmed robust across α∈{0.5,2} and 5 random seeds. This was the blocking
gate before trusting any real model result.

**Phase 4 (E2, main table):** 108 calls: 12 clips (6 each from 02182026 and
05052026) × 3 conditions (base, α=0.5, α=2) × 3 prompt seeds, on
**Qwen2.5-VL-7B-Instruct** (local, via `transformers`). Result: **Acc@10% =
0.0**, **CG = 0.222** (CI [0.148,0.306]), **SFS = 0.131** (α=0.5) / **0.107**
(α=2). Investigated and ruled out as a pipeline bug: the model CAN perceive
and describe both videos correctly when asked directly, but its answers within
a condition group draw from a small set of near-constant values (0.5, 1.5,
3.5 m/s) correlated more with which prompt seed was used than with clip
content — this is a real model behavior, not a bug.

Video pipeline notes: Qwen2.5-VL samples video at ~2fps by default; the
fall+impact event at 2996fps is only ~20-40 frames, so clips were extracted
(motion window ± margin) and re-encoded at 15fps so the model could actually
see the motion stretched into watchable time. Fixed a real `av`/`torchvision`
crash (`AttributeError: module 'av' has no attribute 'AVError'`) by installing
`decord` (preferred automatically by `qwen_vl_utils` once present). Fixed a
`transformers`/`qwen_vl_utils` version mismatch where `video_kwargs['fps']`
came back as a list but the processor wanted a scalar
(`video_kwargs.pop("fps", None)` before passing through).

**Phase 5 (E3, hidden-prior trap), v1→v2→v3:**
- v1: asked only the hidden-prior question (off-frame drop height). Got
  HPH=0.0 on all 12 items — looked like perfect faithfulness. **Retracted**
  after the user asked to scrutinize the suspiciously-uniform result: tested
  the identical prompt with **no video attached at all** and got the identical
  "UNKNOWN" decline. The response was a content-independent default, not a
  video-grounded judgment.
- v2: redesigned to pair the hidden-prior question with a **control question**
  in the same call ("does the droplet impact the surface?" — always true,
  trivially visible). A decline only counts toward HPH if the control was
  answered correctly (proof of engagement with that item's video). Result:
  HPH=0.0 trustworthy on **7 of 12 items**; the other 5 (4 from 05052026, 1
  from 02182026) failed the control and are excluded as uninterpretable.
- Root-cause investigation across 3 rounds: **not** a multi-droplet confound
  (hand-labeled all 12 clips, only 1 — `cainhcg 0.08 b.mp4` — is genuinely
  multi-droplet); **not** an extraction-window timing issue (built and
  validated a per-clip motion-settling detector — `tune_extraction_window.py`
  — replacing the fixed pre/post margin, re-extracted all 12 clips with
  genuinely longer/more-settled windows, reran as v3: the control-answer
  pattern came back byte-for-byte identical to v2, refuting timing as the
  cause). Actual cause, confirmed by direct frame comparison: 05052026's
  camera framing compresses the impact surface into a thin, heavily
  vignette-darkened strip at the extreme bottom of frame, vs 02182026's
  surface sitting in a well-lit middle region.
- **Improvement A2** (2026-07-06): partial fix. Global contrast fixes (linear
  stretch, CLAHE) barely helped — the whole frame has low dynamic range
  (peak pixel ~90/255), not just the surface strip. What worked: crop tightly
  to the impact region (reusing the oracle's own track-linking) and
  contrast-stretch only that local crop, per frame. Recovered 1 of the 5
  failing items (`0.08cg3.mp4`) — **coverage now 8/12**. Of the remaining 4:
  1 is the confirmed multi-droplet clip (unfixable by contrast, the ambiguity
  is about which droplet); 3 still fail despite visibly clear enhanced crops
  — plausibly the necessarily-small crop (~440×420px) gets downsampled by the
  model's default video preprocessing in a way that undoes the local contrast
  gain, but this was **not confirmed**, reported as an open question, not
  chased further (explicitly to avoid repeating the multi-droplet-detector
  diminishing-returns pattern below).

**Abandoned effort, for the record:** before hand-labeling, 5-6 rounds were
spent trying to build a general CV multi-droplet detector
(`detect_multi_droplet.py`, still in the repo but **not used** — every
approach traded one false-positive/negative mode for another: background
subtraction missed resting droplets baked into the bg model; raw-frame
detection over-flagged vignette corners (near-identical brightness stats to a
real droplet, mean~20/std~10 both); edge-contrast filtering fixed vignette
false-positives but broke on genuinely blurry real droplets; position-
persistence clustering inverted the problem (vignette corners are MORE
positionally stable than anything else). User explicitly said "stop tuning,
hand-label instead" — correct call, hand-labeling 12 clips was cheap and
resolved it immediately (only 1/12 genuinely multi-droplet).

**Phase 6 (E4, decoupling analysis):** The runbook's literal ask — correlate
per-item Acc@10% with per-item SFS — is **mathematically degenerate**:
Acc@10%_hit is exactly 0 for all 36 E2 items (matches E2's headline number), a
zero-variance constant, so correlation against it is undefined, not "no
correlation." Substituted continuous relative error (negated) as the closest
computable stand-in: r=-0.481 (CI [-0.708,-0.029], excludes zero) — but this is
**confounded**, not clean: cross-checked against E2's near-constant-response
finding and confirmed `base_velocity_mps` takes only 3 discrete values across
all 36 items, driven by `seed_idx` almost independent of clip content. Per the
runbook's own honesty rule, reported as "the decoupling question couldn't be
meaningfully tested for this model," not as evidence for either "coupled" or
"decoupled."

**Phase 7 (E5, robustness):** α∈{0.25,0.5,2,4} × orientation (real vs
synthetic-rotated) × 3 prompt seeds, scoped to 2 representative clips
(48 items, 60 calls incl. base). Result: **SFS stable across orientation**
(range 0.046) and **prompt seed** (range 0.087) — genuine robustness evidence.
α axis *appears* unstable (range 0.331, spike to 0.396 at α=4) but this is a
**coincidental artifact**: at α=4 specifically every single item (both clips,
both orientations, all seeds) happened to land on the identical (base=0.5,
scaled=1.5) pair — a content-independent 3x ratio that coincidentally sits
near α=4's ideal target and scores well by luck.

**Improvement B1** (2026-07-06): broadened E5 to **all 12 E2 clips** (360
calls, zero extra API cost, local Qwen only). Orientation/seed stability
confirmed on 6x more data. But revealed a **new, much bigger instability
axis**: SFS by session ranges from **0.058** (all six 02182026 clips —
*identical* value, not just similar) to **0.311** (all six 05052026 clips) —
range 0.253, bigger than the already-flagged α artifact. Root cause: 02182026
outputs 1.5 m/s in 34-36/36 items at **every** α tested (essentially zero
perturbation response, mechanically flooring its SFS); 05052026 shows more
incidental spread that coincidentally scores higher, not genuine faithfulness
— same "near-constant-palette landing near a target by luck" phenomenon as the
α=4 artifact, now clip/session-dependent.

**Phase 8 (write-up):** reconciled every `\torun{}`/`\result{}` placeholder
in `main.tex` with real numbers. **Key framing decision (user-approved)**:
reframed from the originally-hypothesized "decoupled" headline to an honest
single-model (later two-model) null-result narrative. Fixed one real LaTeX
bug: a literal `°` unicode character silently failed to render in math mode
under this document's font setup — replaced with `90^{\circ}`.

**Second model, OpenRouter (2026-07-06):** user provided an OpenRouter key
($31.30 budget at the time) specifically to close the single-model gap. Added
`google/gemini-2.5-flash-lite`. Cost-checked first (~$0.0003/call; full
108-call batch cost only **$0.0324 total**). **Result: does NOT resolve the
degeneracy, generalizes it** — Acc@10%=0 for Gemini too (36/36) and for the
72-item pool (72/72). Gemini's failure mode is related but distinct from
Qwen's: on 05052026 it reports calibration=1.0 across ALL conditions (complete
non-response); on 02182026 it **never once** reports the correct α=2 value
(2.0) even though the on-screen "24" label was confirmed legible by direct
frame extraction. Pooled r(-rel_err, SFS) = -0.460 (CI [-0.664,-0.094],
excludes 0); Gemini-alone r=-0.218 (CI [-0.671,0.063], includes zero, not
significant alone). Secondary finding: Gemini is much more accurate at looser
tolerance (Acc@30%=0.333 vs Qwen's 0.028) despite both being 0 at Acc@10%.

**Improvement A1** (2026-07-06): swept the accuracy tolerance τ from 10% to
100% on the same already-collected E2 data (zero new calls). Correlation
becomes computable at τ≥15-30% but is **negative at every tolerance where
measurable** (Gemini r=-0.415 at τ=30%, CI excludes 0; pooled r ranges -0.10
to -0.25 across τ∈[15%,75%]) — never resembles the originally-hypothesized
decoupled pattern. Worse, confirmed Gemini's Acc@30% "hits" are **all from one
session** (05052026) whose true velocities happen to sit near Gemini's
favorite constant output — the same near-constant-palette confound, not real
accuracy gain.

**Improvement B2** (2026-07-06): root-caused E2's CG session asymmetry
(Qwen CG=0.0 on 05052026 vs 0.44 on 02182026) — **confirmed a different cause
from E3's** (footage visibility), not the same one. Qwen literally echoes the
ruler's **printed label** as its calibration answer instead of computing tick
spacing: base condition (ruler prints "30", true answer always 1.0) →
18/18 answers are exactly 30.0 or 3.0; α=0.5 (relabeled "15") → 18/18 answers
are exactly 3.75=15/4. On 02182026 (ruler prints "12"), wrong answers are
small plausible mm-per-tick *guesses* (0.5/1.0) that sometimes coincidentally
match the true value — qualitatively different. Gemini shows a much smaller
session gap (0.43 vs 0.38), consistent with this being Qwen-specific
label-echoing, not a footage property.

Both A1/A2 and B1/B2 were folded into `main.tex` (abstract, E2/E4/E5
discussion sections, Limitations) with full numbers and caveats. Everything
above through B2 is **DONE and reflected in the committed `main.tex`/`main.pdf`**.

## 4. Phase C — started, not completed, currently on hold

Goal: add a third model (frontier cross-family, C1) and/or a within-family
capability-tier comparison (C2, e.g. Qwen 7B vs 72B) to strengthen E4 further.

**What was tried and failed, with root causes (important — don't repeat
these attempts blindly):**
- **Claude Sonnet 4.5 via OpenRouter**: rejected the video call outright.
  Anthropic's API `image` content type only accepts `image/jpeg`, `image/png`,
  `image/gif`, `image/webp` — **no video support at all**, unlike Gemini which
  happens to accept `video/mp4` through the same OpenRouter field. Confirmed
  by direct error message across multiple backing providers (Azure, Bedrock,
  Anthropic direct) — not a fluke of one provider route.
- **GPT-5 via OpenRouter**: same rejection, explicit error: "Expected a
  base64-encoded data URL with an image MIME type... but got unsupported MIME
  type 'video/mp4'."
- **Qwen2.5-VL-72B-Instruct via OpenRouter**: OpenRouter's hosted endpoint's
  `input_modalities` is `["text","image"]` only — **no video** — confirmed via
  the models API metadata, not just a failed call. Providers backing that
  OpenRouter route (Parasail, Nebius) either fail to decode the base64 as an
  image or reject it as not a valid string. Running Qwen-72B would require
  ~140GB+ of local weights, downloaded and hosted locally like the 7B model
  was — a much bigger disk/time operation than this session was scoped for,
  and now moot since `/ephemeral` (where it would have had to live) is gone.
- **Gemini 2.5 Pro via OpenRouter**: this ONE actually works (confirmed with a
  live call). But it's a "thinking" model — burns ~1000 tokens on internal
  reasoning before the final answer. The original client's default
  `max_tokens=256` **silently truncated every response** to a bare
  "CALIBRATION_MM_PER_TICK" with no value — this looked like a parse failure
  but was actually a token-budget bug. Fixed in `run_openrouter_inference.py`
  (now takes a `max_tokens` param; callers using a thinking model must pass a
  larger value, e.g. 1024). Confirmed with a corrected call: full valid
  answer, cal=1.0 (correct), v=1.3 (true=1.164, reasonably close). Cost ~
  $0.011-0.0114/call, ~$1.23 for a full 108-call batch — cheap.

**Where this was interrupted:** the user was asked whether to (a) add Gemini
2.5 Pro as a same-family capability-tier test (Flash-Lite vs Pro, both
Gemini — not the originally-envisioned cross-family C1, but a real, working
capability-tier comparison), (b) build a frame-sequence adapter so Claude/
GPT-5 could be tested via discrete image frames instead of video (more
engineering, changes what the model perceives vs. every other model tested),
or (c) stop and report C1/C2 as blocked by tooling limitations. **The user
interrupted before answering** and redirected to the current
knowledge-dump/audit task instead. Phase C is paused here, not abandoned —
picking it back up should start from "which of (a)/(b)/(c) do you want."

**Also unresolved as of the interruption:** whether the earlier B1/B2
improvements needed a final push to GitHub — they were committed locally
(commit `36b9ea8`, message "Fold improvement plan B1/B2 into the paper") but
a fresh GitHub token was still needed to push at the time. **Check
`git log` and `git status` against the remote before assuming this is
synced** — see Section 5's git audit for current state as of this file being
written.

## 5. Reproducibility audit — what survives, what doesn't

**Trigger for this audit**: `/ephemeral` (the 676GB working volume used all
session for raw videos, the HF model cache, rendered perturbed clips, and most
intermediate files) has been **completely wiped** — confirmed by directly
checking (`ls /ephemeral/sgf_work` → "No such file or directory"; `df -h`
shows `/ephemeral` at 28KB used out of 738GB, i.e. freshly re-mounted empty).
Nothing under `/ephemeral/sgf_work/` can be recovered. This includes:

- **All raw droplet-impact video recordings** (`/ephemeral/videos/{02182026,
  05052026,new_videos,...}/*.mp4`) — the actual experimental footage this
  entire paper is built on. **Confirmed via filesystem search: no copy exists
  anywhere else on this machine** (checked this project folder, `/data`, and
  broadly under `/home`). If the original recordings aren't backed up
  somewhere off this machine (e.g. wherever they were recorded/transferred
  from originally), **they are gone**, full stop. This is the single most
  important fact in this document.
- The HF model cache (`/ephemeral/sgf_work/hf_cache/`, ~16GB) — re-downloadable
  from HuggingFace, not a real loss, just a re-download.
- All rendered perturbed scale clips, extracted fall clips, trimmed scale
  clips, contrast-enhanced clips — all regenerable **only if the raw source
  videos still exist somewhere**; otherwise gone along with them.
- Every intermediate debugging PNG/frame-extraction image generated during
  this session's investigations (contact sheets, contrast-comparison images,
  ROI checks, etc.) — none of these were ever copied into git (correctly, per
  the original scoping decision to keep the repo to code+small-JSON-results
  only). Gone, but these were disposable working artifacts, not results.

**What DOES survive, independent of `/ephemeral`:**

1. **This project folder** (`/home/ubuntu/5-ICLR (work with Dr. Mariam)/`) —
   confirmed present and intact as of this audit.
2. **The git repository**, pushed to `https://github.com/rasulkhanbayov/5-iclr`
   — contains the paper (`main.tex`/`main.pdf`, compiles cleanly, verified
   2026-07-31), the metric library (`sgf_metrics.py`, verified runnable),
   and a curated `code/` tree with every pipeline script and every results
   JSON from E1 through the B1/B2 improvements (60 files tracked as of the
   last check). **This is sufficient to fully substantiate every number and
   claim in the paper** — the results JSONs are self-contained computed
   outputs (bootstrapped means, CIs, raw model responses), not references
   that need re-deriving from video.
   - **git log as of this audit**: `fe523be` (Ignore .env) →
     `36b9ea8` (Fold improvement plan B1/B2 into the paper) →
     `8a05be3` (Add second-model results...) → `9609667` (initial commit).
   - **Push status: needs verification.** `36b9ea8` was committed locally but
     pushing it required a fresh GitHub token that may not have been supplied
     yet at the time of the Phase C interruption. **Run `git log
     origin/main..main` (or compare against `git ls-remote origin main`)
     before relying on the assumption that everything here is pushed.**
     `fe523be` (the `.env` gitignore fix) was made during this audit and is
     almost certainly not yet pushed either.
3. **Claude's memory store**
   (`/home/ubuntu/.claude/projects/-home-ubuntu-5-ICLR--work-with-Dr--Mariam-/memory/`)
   — 19 files, entirely separate from both this folder and git. Contains
   substantially more investigative detail than what made it into `main.tex`
   (dead ends, root-cause chains, exact numbers behind every claim above).
   Not portable/shareable the way a git repo is, but not at risk from
   deleting this project folder either.
4. **`/data/venv`** — still present (confirmed via `df -h` and a successful
   `sgf_metrics.py` run through it), has `torch`+CUDA, `transformers`,
   `qwen_vl_utils`, `opencv-python-headless`, `decord`, `requests`. Belongs
   partly to an unrelated, still-ongoing project per `CLAUDE.md` — do not
   delete or repurpose its checkpoint (`/data/checkpoints/sam2.1_hiera_large.pt`).

**What is NOT reproducible even with everything above:**
- Re-running the CV oracle, the perturbation renderer, or any of the E2-E5
  model-inference batches **from scratch** — every one of those scripts reads
  from `/ephemeral/videos/...` raw footage, which no longer exists on this
  machine. The scripts themselves are correct and complete (committed), but
  they are not currently *runnable* without first recovering or re-acquiring
  the original video recordings.
- Any further Phase C work (adding more models to E2/E4) is unaffected by this
  — it only needs the already-extracted, already-committed fall clips... wait,
  **no** — the fall clips (re-encoded 15fps droplet clips) and trimmed scale
  clips also lived under `/ephemeral/sgf_work/model_outputs/` and are **also
  gone**. Phase C cannot proceed with the existing pipeline until either the
  raw videos are recovered and clips re-extracted, or new clips are sourced
  some other way. **This is a real blocker for Phase C, separate from and in
  addition to the OpenRouter tooling issues in Section 4.**

## 6. Is this folder safe to delete?

**Short answer: NO, do not delete this folder — but the risk is not about
losing anything currently unique to it.**

The paper, its code, and every substantiating result number are already
pushed to GitHub as of commit `8a05be3` (confirmed multi-commit history on
`origin/main`) — **except** the two most recent local commits (`36b9ea8` the
B1/B2 fold-in, and `fe523be` the `.env` gitignore fix), which have **not yet
been confirmed pushed**. If this folder is deleted before those are pushed,
the B1/B2 paper updates (a real, substantive improvement — the by-clip/
by-session E5 finding and the calibration label-echoing E2 finding) would be
lost and would need to be redone from the memory-store detail in Section 3
above (recoverable, since the memory files capture the numbers, but it would
be rework, not a simple `git pull`).

**Concretely, before deleting this folder:**
1. Confirm `36b9ea8` and `fe523be` are on `origin/main` (needs a fresh GitHub
   token — the last one used should already be revoked per this session's
   security practice).
2. Confirm `.env`'s OpenRouter key either still has its ~$681 balance
   available for future work, or deliberately decide to let it go — it is
   NOT in git and NOT recoverable from anywhere else once this folder is gone.
3. Independently of the delete decision: **the raw video recordings are
   already gone** (see Section 5) — this is unrelated to whether you delete
   this folder, already happened when `/ephemeral` was wiped, and is worth
   flagging to whoever owns the original recordings in case an unwiped copy
   exists elsewhere (e.g. a lab machine, a backup, the instrument's own
   storage).

Once (1) is confirmed, this folder itself becomes fully redundant with the
GitHub repo for the purpose of the paper and its results — but the **Claude
memory store** (Section 5, item 3) is a separate, richer archive that
persists independently and is worth keeping in mind for future sessions on
this project even after this folder is gone.
