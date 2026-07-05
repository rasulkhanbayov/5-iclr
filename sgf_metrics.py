"""
sgf_metrics.py
Scale-Grounded Faithfulness (SGF) metrics for quantitative vision-language metrology.

This module implements the quantities defined in the paper:
  - Relative measurement error (RME) and tolerance-band accuracy (Acc@tau)
  - Scale-Faithfulness Score (SFS): does a reported quantity scale with a known
    scale perturbation as a faithful measurer's would?
  - Calibration-Grounding indicator (CG): was the in-frame scale read correctly,
    independent of whether the final number was right?
  - Hidden-Prior Hallucination rate (HPH): did the model assert an off-frame,
    unobservable prior (e.g. drop height) instead of declining?
  - Bootstrap confidence intervals and a paired sign test.

Design notes
------------
* numpy-only. No fabricated experimental outcomes: every function computes from
  inputs you pass in. The __main__ smoke test uses *synthetic* model behaviors
  (a faithful measurer, a prior-driven guesser, a partial measurer) purely to
  demonstrate that the metrics separate these behaviors as the paper claims.
* The CV oracle (optical-flow / blob-tracking ground truth) is assumed to live
  outside this file; here ground-truth values are passed in as arrays.

Author: drafted by an AI assistant; must be independently verified by the human
authors before use in a submission.
"""

from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------------
# Basic accuracy quantities
# ----------------------------------------------------------------------

def relative_error(pred, truth, eps=1e-12):
    """Symmetric-safe relative error |pred-truth|/|truth|."""
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return np.abs(pred - truth) / (np.abs(truth) + eps)


def acc_at_tau(pred, truth, tau=0.10):
    """Fraction of predictions within relative tolerance tau (e.g. 0.10 = 10%)."""
    return float(np.mean(relative_error(pred, truth) <= tau))


def log_accuracy_ratio(pred, truth, eps=1e-12):
    """Order-of-magnitude error: |log10(pred/truth)|. Robust to scale, like ALDE."""
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return np.abs(np.log10((np.abs(pred) + eps) / (np.abs(truth) + eps)))


# ----------------------------------------------------------------------
# Scale-Faithfulness Score (the core contribution)
# ----------------------------------------------------------------------

def scale_response_ratio(pred_base, pred_scaled, alpha):
    """
    The perturbation is a CALIBRATION RELABELING: the in-frame scale is changed to
    assert a different physical length, so the true mm-per-pixel becomes alpha*c,
    while the object's pixel motion is unchanged. A faithful measurer's reported
    quantity then changes by exactly alpha**d (d = calibration-degree; see the
    paper, Proposition 1). Do NOT uniformly resample the whole frame with the
    ruler's asserted length fixed -- that is inert (ideal response 1). We measure
    r = pred_scaled/pred_base and compare to the ideal alpha**d.

    Returns r (array). For a faithful measurer r ~ alpha; for a prior-driven
    guesser r ~ 1 (the number is anchored to a learned prior, not the scale).
    """
    pred_base = np.asarray(pred_base, dtype=float)
    pred_scaled = np.asarray(pred_scaled, dtype=float)
    return pred_scaled / (pred_base + 1e-12)


def scale_faithfulness_score(pred_base, pred_scaled, alpha, d=1):
    """
    SFS in [0,1] roughly. Defined so that:
      - perfect scaling (r == alpha**d)             -> 1
      - no response to the scale change (r == 1)    -> 0
      - over/under response is penalized symmetrically in log space.

    SFS = 1 - |log(r) - d*log(alpha)| / |d*log(alpha)|, clipped to [0,1] per item,
    then averaged. `d` is the calibration-degree of the quantity (d=1 for velocity
    and length at fixed frame rate, d=2 for area), matching Proposition 1 and the
    SFS definition in the paper. Using log space makes the metric symmetric to
    alpha and 1/alpha.
    """
    alpha = float(alpha)
    if np.isclose(alpha, 1.0):
        raise ValueError("alpha must differ from 1 to probe scale faithfulness.")
    r = scale_response_ratio(pred_base, pred_scaled, alpha)
    r = np.clip(r, 1e-9, None)
    target = d * np.log(alpha)
    num = np.abs(np.log(r) - target)
    den = np.abs(target)
    per_item = np.clip(1.0 - num / den, 0.0, 1.0)
    return float(np.mean(per_item)), per_item


# ----------------------------------------------------------------------
# Calibration-grounding and hidden-prior hallucination
# ----------------------------------------------------------------------

def calibration_grounding(reported_mm_per_px, true_mm_per_px, tau=0.10):
    """
    Did the model read the in-frame scale correctly? This is independent of the
    final answer: a model can get the velocity 'right' with a wrong calibration
    by luck/prior. CG = Acc@tau on the reported calibration factor.
    """
    return acc_at_tau(reported_mm_per_px, true_mm_per_px, tau=tau)


def hidden_prior_hallucination(asserted_offscreen_prior):
    """
    The drop height (6.6 cm) is real but OFF-FRAME (not in the camera FOV).
    A faithful model should decline to state it from the video alone.
    asserted_offscreen_prior: bool array, True if the model asserted a numeric
    off-frame prior instead of declining. HPH = mean(True).
    """
    a = np.asarray(asserted_offscreen_prior, dtype=bool)
    return float(np.mean(a))


# ----------------------------------------------------------------------
# Uncertainty: bootstrap CI and paired sign test
# ----------------------------------------------------------------------

def bootstrap_ci(values, stat=np.mean, n_boot=10000, ci=0.95, seed=0):
    """Percentile bootstrap CI for a 1-D array statistic."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = stat(values[idx])
    lo = np.percentile(boots, 100 * (1 - ci) / 2)
    hi = np.percentile(boots, 100 * (1 + ci) / 2)
    return float(stat(values)), float(lo), float(hi)


def paired_sign_test(a, b):
    """
    Two-sided paired sign test on per-item scores a vs b (e.g. SFS of model A vs B
    on the same items). Returns (n_a_better, n_b_better, p_value) via exact binomial.
    numpy-only exact two-sided binomial p-value.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    pos = int(np.sum(diff > 0))
    neg = int(np.sum(diff < 0))
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    # exact two-sided binomial p under p=0.5
    from math import comb
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2.0 * tail)
    return pos, neg, float(p)


# ----------------------------------------------------------------------
# Smoke test: demonstrates metric behavior on SYNTHETIC model archetypes.
# These are not experimental results; they verify the metrics separate
# the three behaviors the paper predicts.
# ----------------------------------------------------------------------

def _synthetic_demo(seed=0):
    rng = np.random.default_rng(seed)
    n = 200
    true_v = rng.uniform(0.5, 2.5, n)          # m/s, CV-oracle ground truth
    true_cal = rng.uniform(0.04, 0.06, n)      # mm per pixel, in-frame ruler
    alpha = 0.5                                 # we shrink the ruler by 2x

    # --- Faithful measurer: reads the ruler, scales correctly, declines off-frame prior
    faith_base = true_v * rng.normal(1.0, 0.05, n)
    faith_scaled = faith_base * alpha * rng.normal(1.0, 0.05, n)
    faith_cal = true_cal * rng.normal(1.0, 0.05, n)
    faith_offscreen = np.zeros(n, dtype=bool)

    # --- Prior-driven guesser: outputs a plausible ~1 m/s prior, ignores the ruler
    guess_base = rng.normal(1.0, 0.15, n)
    guess_scaled = guess_base * rng.normal(1.0, 0.10, n)   # ~no response to scale
    guess_cal = rng.uniform(0.04, 0.06, n)                 # random, ungrounded
    guess_offscreen = rng.random(n) < 0.8                  # often hallucinates 6.6 cm

    # --- Partial measurer: reads ruler but with bias, half-responds to scale
    part_base = true_v * rng.normal(1.0, 0.20, n)
    part_scaled = part_base * (alpha ** 0.5) * rng.normal(1.0, 0.10, n)
    part_cal = true_cal * rng.normal(1.0, 0.12, n)
    part_offscreen = rng.random(n) < 0.3

    rows = []
    for name, pb, ps, cal, off in [
        ("faithful", faith_base, faith_scaled, faith_cal, faith_offscreen),
        ("prior_guesser", guess_base, guess_scaled, guess_cal, guess_offscreen),
        ("partial", part_base, part_scaled, part_cal, part_offscreen),
    ]:
        acc = acc_at_tau(pb, true_v, tau=0.10)
        sfs, per = scale_faithfulness_score(pb, ps, alpha)
        _, lo, hi = bootstrap_ci(per)
        cg = calibration_grounding(cal, true_cal, tau=0.10)
        hph = hidden_prior_hallucination(off)
        rows.append((name, acc, sfs, lo, hi, cg, hph, per))
    return rows, alpha


if __name__ == "__main__":
    rows, alpha = _synthetic_demo()
    print(f"Synthetic demo (scale perturbation alpha={alpha}); SYNTHETIC, not results.\n")
    hdr = f"{'archetype':14s} {'Acc@10%':>8s} {'SFS':>6s} {'SFS 95% CI':>16s} {'CalGround':>10s} {'HiddenPrior':>12s}"
    print(hdr)
    print("-" * len(hdr))
    pers = {}
    for name, acc, sfs, lo, hi, cg, hph, per in rows:
        print(f"{name:14s} {acc:8.2f} {sfs:6.2f} [{lo:5.2f},{hi:5.2f}]   {cg:10.2f} {hph:12.2f}")
        pers[name] = per
    pos, neg, p = paired_sign_test(pers["faithful"], pers["prior_guesser"])
    print(f"\nPaired sign test SFS faithful vs prior_guesser: "
          f"+{pos}/-{neg}, p={p:.2e}")
    # Sanity assertions on the metric's *behavior*, not on any real outcome.
    faith = [r for r in rows if r[0] == "faithful"][0]
    guess = [r for r in rows if r[0] == "prior_guesser"][0]
    assert faith[2] > guess[2], "SFS should rank faithful above prior-guesser"
    assert guess[6] > faith[6], "HPH should be higher for prior-guesser"
    print("\nMetric behavior checks passed: SFS and HPH separate the archetypes.")
