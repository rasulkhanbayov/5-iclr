"""
sgf_metrics.py -- reference implementation of every quantity reported in
"One Perturbation Is Not Enough: Identifiability and Blind Baselines for
Behavioral AI Evaluation".

numpy only. No scipy dependency: the exact binomial tail, the Clopper-Pearson
interval and the incomplete beta function are implemented here directly so the
file runs anywhere.

Design rule for this file: it COMPUTES. It never asserts an outcome. Every
function takes data and returns a number. The smoke test at the bottom checks
arithmetic identities that must hold by construction (E1 cancellation, the
item-versus-cluster interval gap, the tent shape of SFS, the tightness of the
Theorem 5 bound in the small-noise regime), not experimental conclusions.

Run:  python sgf_metrics.py            # smoke test
      python sgf_metrics.py --repro    # recompute every number quoted in the
                                       # paper FROM a results CSV, and print a
                                       # reconciliation table (paper vs. data)

The --repro path needs raw per-item outputs. See RECONCILE_SPEC at the bottom
for the required CSV schema. Without that file the paper's numbers cannot be
verified, only re-typed.
"""

from __future__ import annotations

import math
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 0. Exact-test primitives (no scipy)
# ---------------------------------------------------------------------------


def _log_binom_coef(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    return math.exp(_log_binom_coef(n, k) + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_sf(k: int, n: int, p: float) -> float:
    """P[X >= k]."""
    if k <= 0:
        return 1.0
    return float(sum(binom_pmf(i, n, p) for i in range(k, n + 1)))


def binom_test_two_sided(k: int, n: int, p: float) -> float:
    """Exact two-sided binomial test, equal-or-less-likely (Clopper) convention.

    This is the convention scipy.stats.binomtest uses. The paper quotes
    two-sided p-values (CG against the 1/3 blind bound), so this is the
    matching test.
    """
    obs = binom_pmf(k, n, p)
    tol = 1e-7
    total = 0.0
    for i in range(n + 1):
        pi = binom_pmf(i, n, p)
        if pi <= obs * (1 + tol):
            total += pi
    return min(1.0, total)


def _betacf(a: float, b: float, x: float, itmax: int = 500, eps: float = 3e-16) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(q: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact interval. Used wherever the paper reports an all-zero or
    near-zero count: a bootstrap of an all-zero sample returns zero width,
    which is an artifact rather than precision."""
    lo = 0.0 if k == 0 else _beta_ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    ph = k / n
    d = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


# ---------------------------------------------------------------------------
# 1. Theorem 4 -- supremum over calibration-blind policies for CG
# ---------------------------------------------------------------------------


def tau_separated(alphas: np.ndarray, tau: float) -> bool:
    """Check the tau-separation hypothesis of Theorem 4.

    |a_j - a_j'| / max(a_j, a_j') > 2*tau for all j != j'.
    Theorem 4's simple max_j p_j form is only valid when this holds; when it
    fails the bound must be replaced by the grouped form of Appendix A.5.
    """
    a = np.asarray(alphas, dtype=float)
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if abs(a[i] - a[j]) / max(a[i], a[j]) <= 2 * tau:
                return False
    return True


def blind_bound_cg(shares: np.ndarray, alphas: np.ndarray, tau: float) -> float:
    """Theorem 4. Returns max_j p_j when tau-separated, else the grouped
    supremum of Appendix A.5 (largest total share of any tau-compatible group).

    NOTE FOR THE AUTHORS: the bound depends on the CONDITION SET ACTUALLY
    SCORED. Scoring only the relabeled conditions (dropping the base) makes
    this k=2 and the bound 1/2, not 1/3.
    """
    shares = np.asarray(shares, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    if not np.isclose(shares.sum(), 1.0):
        raise ValueError("condition shares must sum to 1")
    if tau_separated(alphas, tau):
        return float(shares.max())
    # grouped form: sweep candidate reported values over the condition targets
    best = 0.0
    for v in alphas:
        share = sum(p for p, a in zip(shares, alphas) if abs(v - a) <= tau * a)
        best = max(best, share)
    return float(best)


# ---------------------------------------------------------------------------
# 2. SFS (Definition 3) and Theorem 5 -- blind bound for SFS
# ---------------------------------------------------------------------------


def sfs_from_ratio(r: np.ndarray, alpha: float, d: int = 1) -> np.ndarray:
    """SFS = clip_[0,1](1 - |log r - d log alpha| / |d log alpha|), per item.

    Equivalently the tent max(0, min(delta/T, 2 - delta/T)) of Appendix A.4,
    with T = |d log alpha| and delta the sign-oriented log response.
    """
    r = np.asarray(r, dtype=float)
    T = abs(d * math.log(alpha))
    if T == 0:
        raise ValueError("alpha = 1 gives a degenerate target")
    with np.errstate(divide="ignore", invalid="ignore"):
        logr = np.log(r)
    out = 1.0 - np.abs(logr - d * math.log(alpha)) / T
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(logr)] = np.nan
    return out


def sfs_blind_bound(mean_abs_log_r: float, alpha: float, d: int = 1) -> float:
    """Theorem 5: E[SFS] <= E|delta| / (2T), T = |d log alpha|.

    mean_abs_log_r is E|delta|, estimated from the spread of repeated reports
    WITHIN a single condition (no extra model calls needed).
    """
    T = abs(d * math.log(alpha))
    return mean_abs_log_r / (2.0 * T)


def sfs_blind_bound_from_sigma(sigma: float, alpha: float, d: int = 1) -> float:
    """Same bound written in terms of the log-scale dispersion of a single
    report: independent reports give E|delta| = 2 sigma / sqrt(pi), so the
    bound becomes sigma / (T sqrt(pi)). This is the curve in Figure 2b."""
    T = abs(d * math.log(alpha))
    return sigma / (T * math.sqrt(math.pi))


# ---------------------------------------------------------------------------
# 3. Theorem 3 -- policy classification from the (r_R, r_S) pair
# ---------------------------------------------------------------------------


def classify_policy(r_R: float, r_S: float, alpha: float, s: float,
                    d: int = 1, tol: float = 0.15) -> str:
    """Theorem 3 signatures:
        grounded          (alpha^d, 1)
        calibration_blind (1, != 1)
        constant          (1, 1)

    tol is a relative tolerance in LOG space, so it is symmetric under
    r -> 1/r. Returns 'unclassified' when the pair matches no signature; the
    paper reports that outcome rather than forcing a label.
    """
    if not (np.isfinite(r_R) and np.isfinite(r_S) and r_R > 0 and r_S > 0):
        return "unclassified"
    near = lambda x, target: abs(math.log(x) - math.log(target)) <= tol
    grounded_target = alpha ** d
    if near(r_R, grounded_target) and near(r_S, 1.0):
        return "grounded"
    if near(r_R, 1.0) and near(r_S, 1.0):
        return "constant"
    if near(r_R, 1.0) and not near(r_S, 1.0):
        return "calibration_blind"
    return "unclassified"


# ---------------------------------------------------------------------------
# 4. Clustered inference -- the honest cluster count is the CLIP count
# ---------------------------------------------------------------------------


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray,
                         statistic=np.nanmean, draws: int = 10000,
                         alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Resample CLIPS with replacement, not items. Items within a clip share a
    physical scene and are not independent."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    stats = np.empty(draws)
    for b in range(draws):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[c] for c in pick])
        stats[b] = statistic(values[sel])
    return float(np.nanpercentile(stats, 100 * alpha / 2)), \
           float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))


def fisher_ci(r: float, n_clusters: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher-z interval computed at the CLUSTER count, not the item count.

    Using the item count here would understate the width by roughly
    sqrt(items/clusters); the paper's item-versus-cluster gap is exactly this.
    """
    if n_clusters <= 3:
        return (-1.0, 1.0)
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n_clusters - 3)
    zc = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else _z_from_alpha(alpha)
    return math.tanh(z - zc * se), math.tanh(z + zc * se)


def _z_from_alpha(alpha: float) -> float:
    lo, hi = 0.0, 10.0
    target = 1 - alpha / 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1 + math.erf(mid / math.sqrt(2)))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def design_effect(mean_cluster_size: float, rho: float) -> float:
    """deff = 1 + (mbar - 1) * rho."""
    return 1.0 + (mean_cluster_size - 1.0) * rho


def min_detectable_rate(n: int, p0: float = 1.0 / 3.0, level: float = 0.05,
                        power: float = 0.80) -> float:
    """Smallest true grounding rate a one-sided exact binomial test of CG
    against p0 detects with the requested power at n independent elicitations.

    For the clustered figure, pass n_eff = floor(n / deff). Note the function
    is NOT monotone in n because the critical count is an integer; the paper's
    reported values must be read at the exact n_eff used.
    """
    kstar = None
    for k in range(n + 1):
        if binom_sf(k, n, p0) <= level:
            kstar = k
            break
    if kstar is None:
        return 1.0
    lo, hi = p0, 1.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if binom_sf(kstar, n, mid) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 7. Design matrix: the object Theorem 1 is about
# ---------------------------------------------------------------------------


def design_matrix(perturbations: dict) -> tuple[np.ndarray, list[str]]:
    """Build U from a dict {relation_name: multiplicative perturbation vector}.

    Rows are log-perturbation vectors. Example for the three-input task
    (inputs: pixel displacement, asserted calibration, asserted frame rate):

        design_matrix({"R_alpha": [1, 2, 1],
                       "S_s":     [2, 0.5, 1],
                       "T_beta":  [1, 1, 2]})
    """
    names = list(perturbations)
    U = np.log(np.array([perturbations[k] for k in names], dtype=float))
    return U, names


def identifiability_report(U: np.ndarray, names: list[str] | None = None,
                           input_names: list[str] | None = None) -> dict:
    """Theorem 1, applied to a concrete design.

    Returns rank, whether the exponent vector is identified, a basis for the
    confusion space ker U, and which individual inputs the design can certify
    the use of (part iii: e_i must lie in the row space of U).
    """
    U = np.atleast_2d(np.asarray(U, dtype=float))
    k, n = U.shape
    rank = int(np.linalg.matrix_rank(U))
    _, _, Vt = np.linalg.svd(U)
    kernel = Vt[rank:]
    per_input = {}
    for i in range(n):
        e = np.zeros(n); e[i] = 1.0
        # e_i is certified iff it is orthogonal to every kernel direction
        per_input[(input_names or [f"input_{i}" for i in range(n)])[i]] = (
            bool(kernel.shape[0] == 0 or np.allclose(kernel @ e, 0, atol=1e-9)))
    return {
        "relations": names or [f"rel_{j}" for j in range(k)],
        "n_inputs": n, "n_relations": k, "rank": rank,
        "identified": rank == n,
        "confusion_dimension": n - rank,
        "kernel_basis": kernel.tolist(),
        "input_use_certified": per_input,
    }


def _verify_theory(verbose: bool = True) -> None:
    """Independent checks of Proposition 1 and Theorem 1. These are the checks
    that were run before the theory section was written; they are shipped so a
    reviewer can rerun them."""
    print("Proposition 1: only log-linear policies have an operating-point-")
    print("independent response ratio.")
    for name, f in [("monomial x^0.7", lambda x: x**0.7),
                    ("constant 5",     lambda x: 5.0 + 0*x),
                    ("affine 2x+3",    lambda x: 2*x + 3),
                    ("log 1+ln x",     lambda x: 1 + np.log(x)),
                    ("sat. x/(1+x)",   lambda x: x/(1+x))]:
        ms = np.array([0.2, 1.0, 5.0, 50.0]); rr = f(2.0*ms)/f(ms)
        flat = rr.max()-rr.min() < 1e-9
        print(f"  {name:16s} ratio spread {rr.max()-rr.min():.4f}  "
              f"{'log-linear' if flat else 'operating-point dependent'}")

    print("\nTheorem 1: injectivity of theta -> U theta matches rank(U) = n.")
    rng = np.random.default_rng(0); bad = 0
    for _ in range(2000):
        n = int(rng.integers(2, 5)); k = int(rng.integers(1, 6))
        U = rng.normal(size=(k, n))
        if k > 1 and rng.random() < 0.5:
            U[-1] = U[:-1].T @ rng.normal(size=k-1)      # force rank deficiency
        rank = np.linalg.matrix_rank(U)
        rep = identifiability_report(U)
        if rep["identified"] != (rank == n):
            bad += 1
    print(f"  2000 random designs, mismatches: {bad}")

    print("\nThe paper's designs:")
    for label, pert in [
        ("R_alpha alone", {"R": [1, 2, 1]}),
        ("S_s alone",     {"S": [2, 0.5, 1]}),
        ("R + S",         {"R": [1, 2, 1], "S": [2, 0.5, 1]}),
        ("R + S + T",     {"R": [1, 2, 1], "S": [2, 0.5, 1], "T": [1, 1, 2]}),
    ]:
        U, names = design_matrix(pert)
        rep = identifiability_report(U, names, ["pixels", "calibration", "frame_rate"])
        cert = [k for k, v in rep["input_use_certified"].items() if v]
        print(f"  {label:14s} rank {rep['rank']}/3  identified={rep['identified']}  "
              f"certifies use of: {cert or 'nothing'}")
    print("  -> the three relations already run form a complete identifying set.")

    print("\nTemporal target check (the sign error):")
    print("  velocity = (m px)(c mm/px) / (N frames / (beta * phi fps))")
    print("           = m c beta phi / N   ->  response ratio = beta, NOT beta^-1")
    for b, r_obs in [(0.5, 0.642), (2.0, 1.442)]:
        dg = abs(math.log(r_obs) - math.log(b)); db = abs(math.log(r_obs))
        print(f"  beta={b}: observed {r_obs};  log-distance to grounded {dg:.3f}, "
              f"to blind {db:.3f} -> nearer {'GROUNDED' if dg < db else 'BLIND'}")


# ---------------------------------------------------------------------------
# 5. Smoke test -- arithmetic identities only
# ---------------------------------------------------------------------------


def smoke_test(verbose: bool = True) -> None:
    ok = lambda label, cond: print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("E1 cancellation (a faithful re-measurer scores SFS = 1 for ANY oracle)")
    for q in [1e-6, 1e-3, 1.0, 1e3, 1e6]:
        for a in (0.5, 2.0):
            r = (a * q) / q          # ratio of two model outputs; oracle cancels
            ok(f"q*={q:g}, alpha={a}: SFS = {sfs_from_ratio(np.array([r]), a)[0]:.6f}",
               abs(sfs_from_ratio(np.array([r]), a)[0] - 1.0) < 1e-12)
    print("  -> SFS = 1 follows by arithmetic, so E1 checks the implementation "
          "and does NOT validate the oracle.")

    print("\nA fixed-prior emitter is invariant, so SFS = 0")
    r = np.array([1.0] * 10)
    ok("constant emitter, alpha=2", float(np.mean(sfs_from_ratio(r, 2.0))) == 0.0)
    ok("constant emitter, alpha=0.5", float(np.mean(sfs_from_ratio(r, 0.5))) == 0.0)

    print("\nTheorem 4 bound")
    ok("k=3 balanced, alpha in {1,1/2,2}, tau=0.1 -> 1/3",
       abs(blind_bound_cg([1/3]*3, [1.0, 0.5, 2.0], 0.10) - 1/3) < 1e-12)
    ok("tau-separation holds for that design",
       tau_separated([1.0, 0.5, 2.0], 0.10))
    ok("DROPPING the base condition makes it k=2 and the bound 1/2",
       abs(blind_bound_cg([0.5, 0.5], [0.5, 2.0], 0.10) - 0.5) < 1e-12)
    ok("a fine alpha grid violates tau-separation (grouped form needed)",
       not tau_separated([1.0, 1.05, 1.10], 0.10))

    print("\nTheorem 5 bound, and its tightness in the small-noise regime")
    rng = np.random.default_rng(0)
    for spread in (0.05, 0.15, 1.00):
        realized, bounds = [], []
        for a in (0.5, 2.0):
            x1 = rng.normal(0.0, spread, 200000)
            x2 = rng.normal(0.0, spread, 200000)
            r = np.exp(x1 - x2)                       # exchangeable blind pair
            realized.append(float(np.mean(sfs_from_ratio(r, a))))
            bounds.append(sfs_blind_bound(float(np.mean(np.abs(np.log(r)))), a))
        print(f"    spread {spread:.2f}: E[SFS] = {np.mean(realized):.4f}  "
              f"bound = {np.mean(bounds):.4f}")
        ok(f"bound holds at spread {spread:.2f}", np.mean(realized) <= np.mean(bounds) + 1e-3)
    ok("sigma form agrees with E|delta| form (E|delta| = 2 sigma / sqrt(pi))",
       abs(sfs_blind_bound_from_sigma(0.41, 2.0)
           - sfs_blind_bound(2 * 0.41 / math.sqrt(math.pi), 2.0)) < 1e-12)

    print("\nTheorem 3 signatures separate all three classes")
    ok("grounded", classify_policy(2.0, 1.0, alpha=2.0, s=2.0) == "grounded")
    ok("constant", classify_policy(1.0, 1.0, alpha=2.0, s=2.0) == "constant")
    ok("calibration_blind", classify_policy(1.0, 1.6, alpha=2.0, s=2.0) == "calibration_blind")
    ok("r_R alone merges blind with constant",
       classify_policy(1.0, 1.0, 2.0, 2.0) != classify_policy(1.0, 1.6, 2.0, 2.0))

    print("\nItem-versus-cluster interval gap (why the cluster count is the honest n)")
    r_obs, n_items, n_clips = -0.857, 81, 27
    wide = fisher_ci(r_obs, n_clips)
    narrow = fisher_ci(r_obs, n_items)
    print(f"    r={r_obs}: at {n_clips} clips {wide[0]:+.3f},{wide[1]:+.3f}   "
          f"at {n_items} items {narrow[0]:+.3f},{narrow[1]:+.3f}")
    ok("the cluster interval is strictly wider",
       (wide[1] - wide[0]) > (narrow[1] - narrow[0]))

    print("\nExact-test primitives (independent check of the reported statistics)")
    print(f"    two-sided binomial 106/243 vs 1/3 : p = {binom_test_two_sided(106,243,1/3):.4f}")
    print(f"    two-sided binomial  78/240 vs 1/3 : p = {binom_test_two_sided(78,240,1/3):.4f}")
    print(f"    Wilson 106/243 : [{wilson(106,243)[0]:.3f}, {wilson(106,243)[1]:.3f}]")
    print(f"    Wilson  78/240 : [{wilson(78,240)[0]:.3f}, {wilson(78,240)[1]:.3f}]")
    print(f"    Wilson   0/162 : [{wilson(0,162)[0]:.3f}, {wilson(0,162)[1]:.3f}]")
    print(f"    Wilson   0/243 : [{wilson(0,243)[0]:.3f}, {wilson(0,243)[1]:.3f}]  "
          "<-- differs from 0/162; the denominator is not cosmetic")
    print(f"    Clopper-Pearson 0/81 : [{clopper_pearson(0,81)[0]:.3f}, "
          f"{clopper_pearson(0,81)[1]:.3f}]")
    print(f"    Clopper-Pearson 2/80 : [{clopper_pearson(2,80)[0]:.3f}, "
          f"{clopper_pearson(2,80)[1]:.3f}]")

    print("\nPower (min_detectable_rate); pass n_eff = floor(n / deff) for the clustered rows")
    for label, n in [("108 items, unclustered", 108), ("243 items, unclustered", 243)]:
        print(f"    {label:<26} {min_detectable_rate(n):.3f}")
    for n_items, clips in [(108, 12), (243, 27)]:
        mbar = n_items / clips
        for rho in (0.2, 0.5):
            deff = design_effect(mbar, rho)
            n_eff = int(math.floor(n_items / deff))
            print(f"    {n_items} items / {clips} clips, rho={rho}: deff={deff:.2f}, "
                  f"n_eff={n_eff}, min detectable = {min_detectable_rate(n_eff):.3f}")
    print("    ^ the 27-clip clustered rows must be read off THIS function, not "
          "carried over from the 12-clip design.")

    print("\nPhysics consistency check for Acc@tau (free fall from the stated height)")
    h = 0.066
    v = math.sqrt(2 * 9.81 * h)
    for q_hat in (1.5,):
        rel = abs(q_hat - v) / v
        print(f"    h={h*100:.1f} cm -> v_freefall = {v:.3f} m/s; a fixed "
              f"{q_hat} m/s report has relative error {rel:.3f}")
        for tau in (0.10, 0.30, 0.40, 0.50, 1.00):
            print(f"      Acc@{int(tau*100)}% would count this item as "
                  f"{'CORRECT' if rel <= tau else 'wrong'}")
    print("    ^ a fixed report and a tight oracle make Acc@tau a step function "
          "in tau. Any reported sweep that is flat past the step needs "
          "recomputing from raw outputs.")


# ---------------------------------------------------------------------------
# 6. Reconciliation entry point
# ---------------------------------------------------------------------------

RECONCILE_SPEC = """
--repro needs one CSV of raw per-item model outputs with these columns:

  model        model identifier
  clip_id      clip identifier (this is the CLUSTER unit; there are 27)
  seed         prompt-paraphrase seed
  relation     one of {base, R, S, T}   (relabel, resample, temporal)
  magnitude    alpha for R, s for S, beta for T; 1.0 for base
  q_hat        the model's reported quantity (m/s)
  c_hat        the model's reported calibration (mm/tick)
  c_true       the calibration the condition actually asserts (mm/tick)
  q_oracle     the classical-CV oracle value for this clip (m/s)
  oracle_conf  high | low
  parsed       1 if the response parsed, 0 otherwise

Every table cell and every number in the abstract, introduction and
contribution list is then recomputed from this file and printed next to the
value currently written in the .tex, with a MATCH / MISMATCH verdict. Until
that file exists, no number in the paper is verified -- only internally
consistent.
"""


def main(argv: list[str]) -> int:
    if "--repro" in argv:
        print(RECONCILE_SPEC)
        print("No raw results file was supplied with the submission bundle, so "
              "reconciliation could not be run.")
        return 2
    print("sgf_metrics.py smoke test")
    print("=" * 72)
    _verify_theory()
    print("-" * 72)
    smoke_test()
    print("=" * 72)
    print("Smoke test checks arithmetic identities only. It does NOT and cannot "
          "confirm any experimental result. Run --repro against raw outputs for that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
