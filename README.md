# SGF: Student README

This is your entry point. It explains, in plain terms, what the paper claims, how
to build it, what to run, and what to do before submitting. The detailed
experimental protocol lives in `sgf_runbook.txt`; this file points you there.

## 1. What the paper claims and why it matters

**One sentence:** existing benchmarks check whether a vision-language model (VLM)
reports the *right number* for a physical quantity from video, but a model can
get the right number for the *wrong reason* (guessing a plausible value instead
of reading the in-frame ruler), and we give a label-free way to tell the
difference.

**The gap.** Benchmarks like QuantiPhy, IRIS, and PhysicsMind score the predicted
number against ground truth. None checks whether that number is actually anchored
to the visual scale reference. A model that always says "about 1 m/s" will look
good whenever the true answer is near 1 m/s, even though it never measured
anything. Recent faithfulness work (step-level visual grounding faithfulness,
cross-modal synergy) already argues faithfulness is a separate axis from
accuracy, so we build on that consensus; our one change is that our perturbation
(relabeling the in-frame scale) has an EXACTLY KNOWN correct answer, so we need no human
raters or judge model. Note: a May 2026 paper, PRISM-VL, also says
"measurement-grounded" but means something different (feeding RAW sensor data and
training a model); we cite and disambiguate it in the paper. Re-check both lines
on arXiv before you submit.

**The contribution (the delta over the closest work).** We introduce
**Scale-Grounded Faithfulness (SGF)**: we relabel the in-frame scale so the true
calibration changes by a known factor alpha (we change what the scale asserts,
e.g. swap in a ruler of a different known length, leaving the object's pixel
motion unchanged) and check whether the model's reported velocity changes by
alpha**d, as a real measurer's would. (Do NOT uniformly resample the whole frame
with the ruler's asserted length fixed; that is inert.) Because the correct response is known by
simple arithmetic (Proposition 1 in the paper), we can score faithfulness
**without any human annotation**. We add two more diagnostics: a
calibration-grounding check (did it read the ruler right, separately from getting
the number right?) and a hidden-prior trap (does it hallucinate the off-frame 6.6
cm drop height instead of declining?). Droplets are our first testbed; the
protocol works for any video with a ruler in frame.

**Why this is the strongest publishable cut of your data.** Your droplet videos
alone are too narrow to be "a benchmark" for a top venue, and a generic
"VLMs-estimate-velocity" benchmark is already taken (see the paper's Table 1). But
your data has three things those benchmarks lack and need: an in-frame scale bar,
a known-but-off-frame quantity (the 6.6 cm height) that makes a perfect
hallucination trap, and an exact classical-CV oracle. That combination is exactly
what the SGF protocol needs. So we make the *protocol* the contribution and your
data its first instantiation.

## 2. How to read and build the paper

Files:
- `main.tex`  — the paper.
- `sgf.bib`   — references (fill in the `% VERIFY` author lists).
- `main.bbl`  — precompiled references, so one pdflatex pass works even without bibtex.
- `main.pdf`  — the built paper.

Build (full):
```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
For ACL/ARR or NeurIPS D&B submission, drop the official style files into this
folder and switch the `\documentclass` line as noted at the top of `main.tex`.

Where each claim lives: the reframing is in the Introduction; the protocol and
Proposition 1 are in Section 3; the dataset in Section 4; every experiment to run
is in Section 5 (the red `>>> TO RUN` tags) and the results placeholders are blue.

## 3. The experiments

Read `sgf_runbook.txt`. **Do E1 (the positive control) first** — it confirms the
metric and oracle work before you spend any API budget. Then E2 (main table), E3
(hallucination trap), E4 (the decoupling analysis, your headline), E5 (robustness).

What "promising" looks like: models with high accuracy but low SFS — that is the
finding that accuracy and faithfulness are decoupled, which is the interesting,
citable result. What "negative" looks like: accuracy and SFS move together — then
honestly reframe the paper as showing that, on this testbed, number-accuracy does
track grounding. Either way, **reconcile the abstract and intro to what you
actually measure.**

## 4. The code

`sgf_metrics.py` (numpy only) implements every quantity in the paper: SFS,
calibration grounding, hidden-prior rate, bootstrap CIs, paired sign test.

Run the smoke test:
```
python3 sgf_metrics.py
```
It uses *synthetic* model behaviors (not real results) to show the metric
separates a faithful measurer (SFS near 1) from a prior-driven guesser (SFS near
0). For the real analysis, replace the synthetic arrays with your oracle's
measurements and each model's reported numbers.

## 5. Integrity and next-steps checklist (do all of these)

- [ ] Re-confirm novelty on arXiv the week before submission; resolve the
      `[UNVERIFIED]` flag in Table 1 of the paper.
- [ ] Fill every `% VERIFY` citation in `sgf.bib` with the real author list. Never
      invent author names.
- [ ] Have a human author independently verify Proposition 1 and its proof, and
      confirm the calibration-degree `d` for each quantity.
- [ ] Run the experiments; replace every `[RESULT PLACEHOLDER]` and delete every
      `>>> TO RUN` tag with real numbers.
- [ ] If results are weaker than hoped, walk back the framing honestly.
- [ ] Get 2-3 independent human mock reviews and resolve every issue. This is the
      single highest-value step before submission.
- [ ] You are the only authors. The AI assistant drafted; you verify and own
      everything. Check your venue's current AI-use disclosure policy and comply.

## 6. Honest expectation setting

This process maximizes the probability the research deserves; it cannot guarantee
acceptance. Top venues reject most submissions and reviewing is noisy. The
dominant factor will be whether your experiments come back strong and survive
expert scrutiny — especially E1 (does the metric hold up) and E4 (is the
decoupling real). Get those right and this is a genuinely competitive D&B
submission. Until the numbers exist, the draft is "complete pending validation."
