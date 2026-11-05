"""experiments.falsification -- Paper A falsification gates (G02 S4/S8).

Status: IMPLEMENTED by I05, 2026-10-21 (replaces the G09 skeleton).

Spec source (read-only): ``N1000_INIT_2026-10-21/exec/G02_log.md`` S8
(backbone-slope interaction test F(2,6); Benjamini-Hochberg in the
Mardia Gaussianity aggregation; Condition-(S) checks) and S4 (the -1/16
falsification check: both pooled bootstrap CIs -- 12-pretrained and
16-config -- must CONTAIN -1/8 = -0.125 and EXCLUDE -1/16 = -0.0625;
the spread-corrected -(1-nu_bar)/8 ~ -0.089 exclusion is recorded as an
informational ungated row). The appendix falsification target is at
L2836-2839 of the paper TeX (per the N26 brief); its numbers are QUOTED
CONTEXT, never asserted.

Modules: the fit pipeline (per-backbone OLS, pooled common-slope with
backbone intercepts, 1,000-resample bootstrap; D1 unit = configuration
stratified by backbone) is imported from the I05 sibling
``experiments/slope_fits`` -- both files ship together (hard import,
drift-recorded).

GATES IMPLEMENTED
-----------------
G1  F(2,6) backbone-slope interaction (S8): common-slope+intercepts
    (4 params, 8 resid df) vs per-backbone slopes+intercepts (6 params,
    6 resid df) on the 12 pretrained configurations; F via scipy, p =
    scipy.stats.f.sf(F, 2, 6). Pooling admissible on NON-rejection
    (p >= 0.05) with the power caveat recorded (largest pairwise slope
    difference + the diff the design could detect at power ~0.8).
G2  The -1/16 falsification check (S4): per population (12 and 16):
    CI contains -0.125 AND excludes -0.0625. Headline PASS needs both
    populations on both sub-conditions. Informational: excludes -0.089.
G3  Benjamini-Hochberg helper over the Mardia-style p-value family
    (S8): per-class Mardia skewness (chi2) and kurtosis (normal) tests,
    BH-corrected ACROSS CLASSES within each test family; a class passes
    iff BOTH corrected p > 0.05; the recorded quantity is the fraction
    of classes passing both (paper typical range 75-95%, quoted context
    only). Synthetic self-check: Gaussian null classes keep a high pass
    fraction (false-positive rate <= 0.10) while lognormal-contaminated
    classes are BH-rejected (detection >= 0.8).

PLANTED-TRUTH DEMONSTRATION (the self-check of the gates): synthetic
populations with planted slopes -0.30 / -0.125 / -0.0625 / -0.089 are
pushed through the full G2 pipeline; the recorded assertion is that the
gate DECISION equals the pre-declared expected decision for each truth
(only -0.125 is consistent with the 1/8 theory; -0.0625 trips the
exclusion; -0.089 trips containment; -0.30 is far from both). Gate
mechanics verified, no paper number asserted.

STORAGE LAW (owner): one small JSON under ``repo/results/runs/A/`` per
run (G08-S2 envelope) + the INDEX row; markdown summary to stdout
(embedded in ``exec/I05_log.md``). No downloads, no OneDrive paths.

Self-check: ``python experiments/falsification.py --selfcheck`` (seconds).
Full suite: ``python experiments/falsification.py`` (~1 min).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.stats

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # allow direct-script execution
    sys.path.insert(0, str(_REPO_ROOT))

DRIFT_NOTES: List[str] = []

from experiments import slope_fits as sf  # I05 sibling (hard import)

DRIFT_NOTES.append("experiments.slope_fits imported (I05 sibling; the fit "
                   "pipeline is owned there).")
DRIFT_NOTES.extend(sf.DRIFT_NOTES)

NEG_1_8 = sf.NEG_1_8          # -0.125, operational Cor 3.8 constant
NEG_1_16 = -0.0625            # Condition-(S) worst case
SPREAD_CORRECTED = -0.089     # -(1-nu_bar)/8, quoted context

PAPER_CONTEXT_QUOTED: Dict[str, Any] = {
    "never_asserted": True,
    "F_interaction": (2.14, 0.199, "(2,6) df"),
    "largest_pairwise_slope_diff": 0.042,   # ViT vs DINOv2, undetected
    "ci_targets": {"contains": NEG_1_8, "excludes": [NEG_1_16,
                                                     SPREAD_CORRECTED]},
    "pooled_12": (-0.115, [-0.128, -0.102], 0.97),
    "pooled_16": (-0.119, [-0.134, -0.105], 0.96),
    "mardia_pass_typical": [0.75, 0.95],
    "mardia_formulas": "(n/6)b_{1,d} ~ chi2_{d(d+1)(d+2)/6}; "
                       "sqrt(n/(8d(d+2)))(b_{2,d}-d(d+2)) ~ N(0,1)",
}


# ---------------------------------------------------------------------------
# Benjamini-Hochberg helper (G02 S8; used in the Mardia aggregation)
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals: Sequence[float],
                       alpha: float = 0.05) -> Dict[str, Any]:
    """BH step-up corrected p-values + rejections at ``alpha``.

    adj_p_(i) = min_{j >= i} min(1, n/j * p_(j)) over the ascending sort;
    rejected iff adj_p <= alpha (equivalent to the step-up procedure).
    """
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order] * n / np.arange(1, n + 1)
    adj_ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.minimum(adj_ranked, 1.0)
    return dict(adjusted=[float(x) for x in adj],
                rejected=[bool(x) for x in (adj <= alpha)],
                n=int(n), alpha=float(alpha),
                n_rejected=int((adj <= alpha).sum()),
                method="Benjamini-Hochberg step-up (1995)")


# ---------------------------------------------------------------------------
# Mardia per-class tests + the BH aggregation (S8 Gaussianity check)
# ---------------------------------------------------------------------------

def mardia_test(X: np.ndarray) -> Dict[str, Any]:
    """Mardia skewness/kurtosis (G02 S8 formulas, quoted above):
    b_{1,d} = n^-2 sum_ij ((x_i-m)' S^{-1} (x_j-m))^3,
    b_{2,d} = n^-1 sum_i ((x_i-m)' S^{-1} (x_i-m))^2, S = MLE covariance."""
    X = np.asarray(X, float)
    n, d = X.shape
    Z = X - X.mean(0)
    S = Z.T @ Z / n
    A = Z @ np.linalg.solve(S, Z.T)  # n x n quadratic-form matrix
    b1d = float((A ** 3).sum() / n ** 2)
    b2d = float((np.diag(A) ** 2).sum() / n)
    df = d * (d + 1) * (d + 2) // 6
    skew_stat = n / 6.0 * b1d
    skew_p = float(scipy.stats.chi2.sf(skew_stat, df))
    kurt_stat = float(np.sqrt(n / (8.0 * d * (d + 2)))
                      * (b2d - d * (d + 2)))
    kurt_p = float(2.0 * scipy.stats.norm.sf(abs(kurt_stat)))
    return dict(b1d=b1d, b2d=b2d, skew_stat=skew_stat, skew_df=int(df),
                skew_p=skew_p, kurt_stat=kurt_stat, kurt_p=kurt_p, n=n, d=d)


def mardia_family_gaussianity(class_samples: Sequence[np.ndarray],
                              alpha: float = 0.05) -> Dict[str, Any]:
    """Fraction of classes passing BOTH Mardia tests at ``alpha``,
    BH-corrected ACROSS CLASSES within each test family (S8 aggregation).
    ``class_samples[k]`` is the (n_k x d) feature matrix of class k."""
    skew_p, kurt_p = [], []
    for X in class_samples:
        m = mardia_test(X)
        skew_p.append(m["skew_p"])
        kurt_p.append(m["kurt_p"])
    bh_s = benjamini_hochberg(skew_p, alpha)
    bh_k = benjamini_hochberg(kurt_p, alpha)
    pass_both = [not (rs or rk) for rs, rk in zip(bh_s["rejected"],
                                                  bh_k["rejected"])]
    return dict(n_classes=len(class_samples), alpha=alpha,
                skew_p=skew_p, kurt_p=kurt_p,
                skew_adj=bh_s["adjusted"], kurt_adj=bh_k["adjusted"],
                skew_rejected=bh_s["rejected"], kurt_rejected=bh_k["rejected"],
                pass_both=pass_both,
                fraction_pass_both=float(np.mean(pass_both)),
                fraction_pass_skew=float(1.0 - np.mean(bh_s["rejected"])),
                fraction_pass_kurt=float(1.0 - np.mean(bh_k["rejected"])))


def make_mardia_family(n_null: int = 90, n_contam: int = 10,
                       n: int = 400, d: int = 20, frac: float = 0.05,
                       purpose: str = "mardia"
                       ) -> Tuple[List[np.ndarray], List[str]]:
    """Synthetic Mardia family: Gaussian null classes + lognormal-contaminated
    classes (5% of rows, scale 1.0). Synthetic-proxy construction; the
    paper's real per-class features come from the backbone runs (store)."""
    rng = sf._rng(f"i05:{purpose}")
    samples, kinds = [], []
    for k in range(n_null + n_contam):
        X = rng.standard_normal((n, d))
        if k >= n_null:
            idx = rng.choice(n, size=max(1, int(frac * n)), replace=False)
            X[idx] = rng.lognormal(0.0, 1.0, (len(idx), d))
            kinds.append("contaminated")
        else:
            kinds.append("null")
        samples.append(X)
    return samples, kinds


# ---------------------------------------------------------------------------
# G1: F(2,6) backbone-slope interaction test (S8)
# ---------------------------------------------------------------------------

def f_interaction_test(records: Sequence[sf.ConfigRecord]
                       ) -> Dict[str, Any]:
    """Reduced = common slope + backbone intercepts (p params, resid df
    n-p); Full = per-backbone slopes + intercepts (2g params, resid df
    n-2g). On the canonical 12-config/3-backbone population this is the
    spec's F(2, 6). scipy.stats.f.sf; non-rejection => pooling admissible,
    with the power caveat recorded."""
    rows = sf.pretrained_only(list(records))
    g = len({r.backbone for r in rows})
    n = len(rows)
    x = np.asarray([r.gamma_min_hat for r in rows], float)
    y = np.asarray([r.log_tc_gap for r in rows], float)
    groups = sorted({r.backbone for r in rows})
    # reduced design: [dummies | x]
    Xr = np.zeros((n, g + 1))
    for i, r in enumerate(rows):
        Xr[i, groups.index(r.backbone)] = 1.0
    Xr[:, -1] = x
    # full design: [dummies | dummies*x]
    Xf = np.zeros((n, 2 * g))
    for i, r in enumerate(rows):
        j = groups.index(r.backbone)
        Xf[i, j] = 1.0
        Xf[i, g + j] = x[i]
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    bf, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    ssr_r = float(((y - Xr @ br) ** 2).sum())
    ssr_f = float(((y - Xf @ bf) ** 2).sum())
    df1 = int(Xf.shape[1] - Xr.shape[1])       # g-1 (= 2 on the 12-config fit)
    df2 = int(n - Xf.shape[1])                 # n-2g (= 6 on the 12-config fit)
    F = max(((ssr_r - ssr_f) / df1) / (ssr_f / df2), 0.0) if ssr_f > 0 \
        else np.inf  # clamp: zero-noise round-off can make SSR_r - SSR_f < 0
    p = float(scipy.stats.f.sf(F, df1, df2))
    per_bb = sf.per_backbone_ols(rows)
    sl = {bb: per_bb[bb]["slope"] for bb in per_bb}
    bbs = sorted(sl)
    pairwise = {f"{a}|{b}": abs(sl[a] - sl[b])
                for i, a in enumerate(bbs) for b in bbs[i + 1:]}
    largest = max(pairwise, key=lambda k: pairwise[k])
    # smallest per-backbone slope difference this design flags at alpha=0.05
    # (approx: F_crit * df1 * sigma2_full / mean within-backbone Sxx)
    sigma2_f = ssr_f / df2
    denom = float(np.mean([
        float(np.sum((x[np.array([i for i, r in enumerate(rows)
                                  if r.backbone == bb])]
                      - x[np.array([i for i, r in enumerate(rows)
                                    if r.backbone == bb])].mean()) ** 2))
        for bb in bbs]))
    diff_detect = (float(np.sqrt(scipy.stats.f.isf(0.05, df1, df2) * df1
                                 * sigma2_f / denom))
                   if denom > 0 else float("nan"))
    return dict(F=float(F), df1=df1, df2=df2, p=p,
                ssr_reduced=ssr_r, ssr_full=ssr_f, n=n, groups=groups,
                per_backbone_slopes=sl, pairwise_abs_diffs=pairwise,
                largest_pairwise=(largest, pairwise[largest]),
                detectable_diff=diff_detect,
                power_caveat="F(2,6) non-rejection is weak evidence: with 4 "
                             "points/backbone only slope differences above "
                             "~detectable_diff are reliably flagged (G02 S8 "
                             "records the paper's own 0.042 undetected diff).")


# ---------------------------------------------------------------------------
# G2: the -1/16 falsification check (S4) on pipeline outputs
# ---------------------------------------------------------------------------

def ci_gates(populations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate the S4 falsification conditions on pooled-slope bootstrap
    CIs. ``populations`` maps population label -> dict with ci_lo/ci_hi
    (and slope/r2). Headline PASS iff EVERY population CI contains -1/8
    and excludes -1/16. The -0.089 spread-corrected exclusion is recorded
    informational (ungated; its nu_bar ~ 0.29 is a paper-quoted input)."""
    assertions: List[Dict[str, Any]] = []
    overall = True
    for pop in sorted(populations):
        ci = populations[pop]
        lo, hi = ci["ci_lo"], ci["ci_hi"]
        contains = bool(lo <= NEG_1_8 <= hi)
        excludes16 = not bool(lo <= NEG_1_16 <= hi)
        excl89 = not bool(lo <= SPREAD_CORRECTED <= hi)
        overall = overall and contains and excludes16
        assertions.append(dict(
            id=f"G2[{pop}]", description="pooled bootstrap CI contains -1/8 "
            "and excludes -1/16 (S4 falsification check)",
            arm=pop, status="PASS" if (contains and excludes16) else "FAIL",
            value=f"slope {ci.get('slope', float('nan')):+.4f} CI "
                  f"[{lo:+.4f}, {hi:+.4f}]: contains -1/8 {contains}, "
                  f"excludes -1/16 {excludes16}",
            threshold="lo <= -0.125 <= hi AND NOT lo <= -0.0625 <= hi",
            diagnosis="",
            details=dict(excludes_spread_corrected_0p089=excl89,
                         note="-0.089 exclusion recorded informational "
                              "(ungated); quoted context, never asserted.")))
    return dict(status="PASS" if overall else "FAIL", assertions=assertions)


def run_full_pipeline(records: Sequence[sf.ConfigRecord], B: int = 1000,
                      purpose: str = "fals") -> Dict[str, Dict[str, Any]]:
    """Pooled fits + bootstrap CIs for the 12 and 16 populations, the exact
    pipeline whose CIs the G2 gates consume."""
    fit12 = sf.pooled_common_slope_fit(records, population="12")
    fit16 = sf.pooled_common_slope_fit(records, population="16",
                                       anchor_mode="own_intercept")
    boot12 = sf.bootstrap_pooled_slope(records, B=B, population="12",
                                       purpose=f"{purpose}-12")
    boot16 = sf.bootstrap_pooled_slope(records, B=B, population="16",
                                       purpose=f"{purpose}-16")
    return {"12": dict(slope=fit12["slope"], r2=fit12["r2"],
                       ci_lo=boot12["ci_lo"], ci_hi=boot12["ci_hi"]),
            "16": dict(slope=fit16["slope"], r2=fit16["r2"],
                       ci_lo=boot16["ci_lo"], ci_hi=boot16["ci_hi"])}


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

def _assert(assertions, aid, desc, arm, ok, value, threshold,
            diagnosis="", details=None):
    assertions.append(dict(id=aid, description=desc, arm=arm,
                           status="PASS" if ok else "FAIL", value=value,
                           threshold=threshold, diagnosis=diagnosis,
                           details=details or {}))


def run_g1_interaction(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """G1 on planted populations: homogeneous common slope => pooling
    admissible (non-rejection); planted per-backbone heterogeneity
    (paper-context magnitudes -0.141/-0.099/-0.118 as PLANTS) =>
    rejection. F/df/p recorded; nothing asserts F=2.14/p=0.199."""
    t0 = time.time()
    assertions: List[Dict[str, Any]] = []
    noise = cfg["g1_noise_sd"]
    homog = sf.make_synthetic_population(-0.115, noise_sd=noise,
                                         purpose="g1-homog")
    r_hom = f_interaction_test(homog)
    ok_h = r_hom["p"] >= 0.05
    _assert(assertions, "G1.1", "homogeneous planted common slope -0.115: "
            "non-rejection (pooling admissible; single realization)",
            "homogeneous", ok_h,
            f"F({r_hom['df1']},{r_hom['df2']}) = {r_hom['F']:.3f}, "
            f"p = {r_hom['p']:.4f}", "p >= 0.05",
            diagnosis="" if ok_h else
            "single-draw type-I event (5% by construction); record as-is")
    slopes = {"ViT-IN21K": -0.141, "DINOv2": -0.099, "CLIP": -0.118}
    het = []
    for bb, sl in slopes.items():
        for ds, g in zip(sf.DATASETS, sf.SYNTHETIC_GAMMA_GRID[bb]):
            rng = sf._rng(f"i05:g1-het:{bb}|{ds}")
            het.append(sf.ConfigRecord(
                backbone=bb, dataset=ds, anchor=False, gamma_min_hat=float(g),
                log_tc_gap=float(sl * g + rng.normal(0, 0.02))))
    r_het = f_interaction_test(het)
    ok_d = r_het["p"] < 0.05
    _assert(assertions, "G1.2", "planted heterogeneity at paper-context "
            "magnitudes (-0.141/-0.099/-0.118 PLANTED, quoted never "
            "asserted; noise sd 0.02): rejection", "heterogeneous", ok_d,
            f"F({r_het['df1']},{r_het['df2']}) = {r_het['F']:.2f}, "
            f"p = {r_het['p']:.2e}", "p < 0.05", diagnosis="")
    _assert(assertions, "G1.3", "df identity matches the spec: F(2,6) on the "
            "12-config/3-backbone population",
            "df-check",
            (r_hom["df1"], r_hom["df2"]) == (2, 6),
            f"df = ({r_hom['df1']}, {r_hom['df2']})", "(2, 6)")
    status = "PASS" if all(a["status"] == "PASS" for a in assertions) else "FAIL"
    return dict(suite="G1-interaction", theorem="S8 backbone-slope "
                "homogeneity (pooling admissibility)", status=status,
                runtime_s=round(time.time() - t0, 2), assertions=assertions,
                homogeneous=r_hom, heterogeneous=r_het,
                quoted_context=PAPER_CONTEXT_QUOTED)


def run_g2_planted(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """G2 planted-truth demonstration: gate DECISION must equal the
    pre-declared expected decision for each planted slope. Only a truth at
    the operational -1/8 passes; -1/16 trips the exclusion; -0.089 trips
    containment; -0.30 fails both sub-conditions."""
    t0 = time.time()
    assertions: List[Dict[str, Any]] = []
    expected = {-0.30: "FAIL", -0.125: "PASS", -0.0625: "FAIL",
                -0.089: "FAIL"}
    per_plant = {}
    for planted in (-0.30, -0.125, -0.0625, -0.089):
        recs = sf.make_synthetic_population(planted, noise_sd=0.02,
                                            purpose="g2")
        pops = run_full_pipeline(recs, B=cfg["g2_bootstrap_B"],
                                 purpose=f"g2-{planted}")
        gates = ci_gates(pops)
        ok = gates["status"] == expected[planted]
        _assert(assertions, f"G2[planted={planted:+.4f}]",
                f"planted truth {planted:+.4f}: gate decision "
                f"{gates['status']} equals expected {expected[planted]}",
                "planted", ok,
                "; ".join(f"{pop}: slope {pops[pop]['slope']:+.4f} CI "
                          f"[{pops[pop]['ci_lo']:+.4f}, "
                          f"{pops[pop]['ci_hi']:+.4f}]"
                          for pop in ("12", "16")),
                f"decision == {expected[planted]}")
        per_plant[f"{planted:+.4f}"] = dict(populations=pops,
                                            gate_status=gates["status"])
    # sub-condition resolution on the two task-pinned truths:
    recs625 = sf.make_synthetic_population(-0.0625, noise_sd=0.02,
                                           purpose="g2b")
    pops625 = run_full_pipeline(recs625, B=cfg["g2_bootstrap_B"],
                                purpose="g2-625-sub")
    lo, hi = pops625["12"]["ci_lo"], pops625["12"]["ci_hi"]
    _assert(assertions, "G2[trip=-1/16]",
            "planted -1/16 truth: the CI CONTAINS -0.0625 (the exclusion "
            "sub-gate is what trips)", "planted",
            bool(lo <= NEG_1_16 <= hi),
            f"12-config CI [{lo:+.4f}, {hi:+.4f}] contains -0.0625: "
            f"{bool(lo <= NEG_1_16 <= hi)}",
            "lo <= -0.0625 <= hi (containment expected)")
    recs125 = sf.make_synthetic_population(-0.125, noise_sd=0.02,
                                           purpose="g2c")
    pops125 = run_full_pipeline(recs125, B=cfg["g2_bootstrap_B"],
                                purpose="g2-125-sub")
    lo, hi = pops125["12"]["ci_lo"], pops125["12"]["ci_hi"]
    _assert(assertions, "G2[trip=-1/8]",
            "planted -1/8 truth: the CI CONTAINS -0.125 and EXCLUDES "
            "-0.0625 (both sub-conditions hold)", "planted",
            bool(lo <= NEG_1_8 <= hi and not (lo <= NEG_1_16 <= hi)),
            f"12-config CI [{lo:+.4f}, {hi:+.4f}]",
            "contains -0.125 AND excludes -0.0625")
    status = "PASS" if all(a["status"] == "PASS" for a in assertions) else "FAIL"
    return dict(suite="G2-planted", theorem="S4 -1/16 falsification check "
                "(gate mechanics on planted truths)", status=status,
                runtime_s=round(time.time() - t0, 2), assertions=assertions,
                per_plant=per_plant)


def run_g3_mardia(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """G3 BH over the Mardia family (S8): null classes keep passing
    (false-positive rate <= 0.10), contaminated classes get BH-rejected
    (detection >= 0.8); pass-both fraction recorded against the paper's
    quoted typical 75-95% band (context only)."""
    t0 = time.time()
    assertions: List[Dict[str, Any]] = []
    samples, kinds = make_mardia_family(
        n_null=cfg["g3_n_null"], n_contam=cfg["g3_n_contam"],
        n=cfg["g3_n"], d=cfg["g3_d"])
    fam = mardia_family_gaussianity(samples)
    null_idx = [i for i, k in enumerate(kinds) if k == "null"]
    cont_idx = [i for i, k in enumerate(kinds) if k == "contaminated"]
    fp_rate = float(np.mean([not fam["pass_both"][i] for i in null_idx]))
    detect = float(np.mean([not fam["pass_both"][i] for i in cont_idx]))
    _assert(assertions, "G3.1", "BH-corrected Mardia family: Gaussian null "
            "classes false-positive rate <= 0.10", "null",
            fp_rate <= 0.10, f"false-positive rate {fp_rate:.3f} "
            f"({int(round(fp_rate * len(null_idx)))}/{len(null_idx)} classes)",
            "<= 0.10")
    _assert(assertions, "G3.2", "lognormal-contaminated classes "
            "BH-rejected (pass-both fails) at >= 0.8", "contaminated",
            detect >= 0.8, f"detection {detect:.2f} "
            f"({int(round(detect * len(cont_idx)))}/{len(cont_idx)})",
            ">= 0.80")
    frac = fam["fraction_pass_both"]
    in_band = 0.60 <= frac <= 1.0  # generous synthetic band; context 75-95%
    _assert(assertions, "G3.3", "pass-both fraction recorded in a sane "
            "synthetic band (paper typical 75-95% QUOTED context only; "
            "with 10% contaminated classes the synthetic mix cannot hit "
            "the paper band by construction)", "family",
            in_band, f"fraction_pass_both {frac:.3f} "
            f"(null {np.mean([fam['pass_both'][i] for i in null_idx]):.3f}, "
            f"contaminated {np.mean([fam['pass_both'][i] for i in cont_idx]):.3f})",
            "0.60 <= fraction <= 1.0")
    # BH helper identities
    fam_probe = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.50], alpha=0.05)
    mono = all(a >= b - 1e-15 for a, b in zip(fam_probe["adjusted"][1:],
                                              fam_probe["adjusted"][:-1]))
    _assert(assertions, "G3.4", "BH helper identities: adjusted >= raw, "
            "monotone in the sorted order", "helper",
            mono and all(a >= p - 1e-15 for a, p in
                         zip(fam_probe["adjusted"], [0.01, 0.02, 0.03,
                                                     0.04, 0.50])),
            f"adjusted {fam_probe['adjusted']}", "adj >= raw and sorted-monotone")
    status = "PASS" if all(a["status"] == "PASS" for a in assertions) else "FAIL"
    return dict(suite="G3-mardia-BH", theorem="S8 Gaussianity check "
                "(Mardia x BH across classes)", status=status,
                runtime_s=round(time.time() - t0, 2), assertions=assertions,
                family_summary=dict(fraction_pass_both=frac,
                                    fp_rate_null=fp_rate,
                                    detection_contaminated=detect,
                                    n_classes=len(samples)),
                quoted_context=dict(mardia_pass_typical=
                                    PAPER_CONTEXT_QUOTED["mardia_pass_typical"]))


DEFAULT_CFG = dict(
    g1_noise_sd=0.05, g2_bootstrap_B=1000,
    g3_n_null=90, g3_n_contam=10, g3_n=400, g3_d=20,
)

_SUITES = {"G1": run_g1_interaction, "G2": run_g2_planted,
           "G3": run_g3_mardia}


# ---------------------------------------------------------------------------
# Writer (G08-S2 envelope) + markdown + CLI
# ---------------------------------------------------------------------------

_SUITE = "falsification"


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def write_run_json(record: Dict[str, Any], cfg: Dict[str, Any],
                   started_utc: str, out_root: Path,
                   run_name: str = "synthetic") -> Path:
    run_dir = out_root / "runs" / "A" / _SUITE
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"A_{_SUITE}_{run_name}"
    payload = dict(
        run_id=run_id, paper="A", experiment=_SUITE,
        provenance="synthetic", created_utc=_utcnow(),
        executed_utc=started_utc,
        schema="G08-S2-proposal (G01 not yet reconciled; see G08 log S7.1)",
        config=cfg, seeds=[0], metrics=record,
        env=dict(python=platform.python_version(), numpy=np.__version__,
                 scipy=scipy.__version__,
                 machine=platform.machine(), platform=platform.platform()),
        code_version=_file_sha(Path(__file__)),
        inputs={"slope_fits": _file_sha(
                    _REPO_ROOT / "experiments" / "slope_fits.py")},
        derived_tables=[],
        drift_notes=DRIFT_NOTES,
    )
    path = run_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=1, default=str),
                    encoding="utf-8")
    index_path = out_root / "INDEX.json"
    index: Dict[str, Any] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index = {}
    runs = [r for r in index.setdefault("runs", [])
            if r.get("run_id") != run_id]
    runs.append(dict(run_id=run_id, path=str(path.relative_to(out_root)),
                     provenance="synthetic", paper="A", experiment=_SUITE,
                     suite_status=record["status"],
                     created_utc=payload["created_utc"]))
    index["runs"] = runs
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return path


def render_markdown(records: List[Dict[str, Any]], cfg: Dict[str, Any],
                    started_utc: str) -> str:
    lines = [
        "# falsification gates (I05) -- run summary", "",
        f"Run: {started_utc} | module: experiments/falsification.py (I05) | "
        f"provenance: synthetic | pipeline: experiments.slope_fits (I05)",
        "",
        "Quoted context (never asserted): F(2,6)=2.14 p=0.199; largest "
        "undetected pairwise slope diff 0.042; CIs contain -1/8, exclude "
        "-1/16 and -0.089; Mardia pass 75-95%.",
        "",
    ]
    for rec in records:
        lines.append(f"## {rec['suite']} -- {rec.get('theorem', '')}: "
                     f"**{rec['status']}** ({rec['runtime_s']}s)")
        lines.append("")
        lines.append("| id | arm | status | value | threshold |")
        lines.append("|----|-----|--------|-------|-----------|")
        for a in rec["assertions"]:
            val = str(a["value"])
            if len(val) > 110:
                val = val[:107] + "..."
            lines.append(f"| {a['id']} | {a['arm']} | {a['status']} | "
                         f"{val} | {a['threshold']} |")
        lines.append("")
    lines.append("## Drift notes")
    lines.append("")
    for note in DRIFT_NOTES:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--suites", default="G1,G2,G3")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--out-root", default=str(_REPO_ROOT / "results"))
    p.add_argument("--run-name", default="synthetic")
    args = p.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    cfg = dict(DEFAULT_CFG)
    started = _utcnow()
    t_all = time.time()
    records = []
    for name in [s.strip().upper() for s in args.suites.split(",") if s.strip()]:
        if name not in _SUITES:
            print(f"unknown suite {name}; skipping", file=sys.stderr)
            continue
        print(f"[i05-f] running suite {name} ...", flush=True)
        rec = _SUITES[name](cfg)
        records.append(rec)
        print(f"[i05-f] {name}: {rec['status']} in {rec['runtime_s']}s",
              flush=True)
    aggregate = dict(
        suite="G1+G2+G3", status="PASS" if all(
            r["status"] == "PASS" for r in records) else "FAIL",
        runtime_s=round(time.time() - t_all, 2),
        assertions=[a for r in records for a in r["assertions"]],
        suites=records, paper_context_quoted=PAPER_CONTEXT_QUOTED,
        decisions=["gate semantics per G02 S4/S8; planted truths -0.30/"
                   "-0.125/-0.0625/-0.089 with pre-declared expected "
                   "decisions; -0.089 exclusion informational (ungated)"],
    )
    path = write_run_json(aggregate, cfg, started, Path(args.out_root),
                          run_name=args.run_name)
    print(render_markdown(records, cfg, started))
    print(f"[i05-f] JSON -> {path}")
    print(f"[i05-f] total wall {time.time() - t_all:.1f}s")
    return 0 if aggregate["status"] == "PASS" else 1


def _selfcheck() -> int:  # pragma: no cover - fast identities, no writes
    # 1. BH step-up on a textbook family
    r = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.50], alpha=0.05)
    assert r["adjusted"][0] <= 0.05 and r["adjusted"][4] > 0.05
    assert all(a >= p - 1e-15 for a, p in zip(r["adjusted"],
                                              [0.01, 0.02, 0.03, 0.04, 0.50]))
    print(f"PASS BH helper on textbook family: adjusted "
          f"{[round(x, 4) for x in r['adjusted']]}, "
          f"rejected {r['n_rejected']}/5")
    # 2. BH rejections == step-up k* rule
    rng = sf._rng("i05:f-selfcheck")
    pv = rng.uniform(size=200)
    r2 = benjamini_hochberg(pv, alpha=0.05)
    order = np.argsort(pv)
    kstar = 0
    for i in range(len(pv)):
        if pv[order[i]] <= (i + 1) / len(pv) * 0.05:
            kstar = i + 1
    assert sum(r2["rejected"]) == kstar, (sum(r2["rejected"]), kstar)
    print(f"PASS BH adjusted-p rejections == step-up k* rule "
          f"({kstar}/200 uniform nulls)")
    # 3. Mardia on exact Gaussian: p's uniform-ish, test stats finite
    X = sf._rng("i05:f-mardia").standard_normal((500, 10))
    m = mardia_test(X)
    assert 0.0 < m["skew_p"] < 1.0 and 0.0 < m["kurt_p"] < 1.0
    assert m["skew_df"] == 10 * 11 * 12 // 6
    print(f"PASS Mardia test on exact Gaussian: skew p {m['skew_p']:.3f}, "
          f"kurt p {m['kurt_p']:.3f}, df {m['skew_df']} (d=10)")
    # 4. Mardia skew detects planted lognormal (d=10, n=500)
    rngm = sf._rng("i05:f-mardia-c")
    Xc = rngm.standard_normal((500, 10))
    Xc[:100] = rngm.lognormal(0, 1.0, (100, 10))
    mc = mardia_test(Xc)
    assert mc["skew_p"] < 1e-6, mc["skew_p"]
    print(f"PASS Mardia skew rejects contaminated class "
          f"(p {mc['skew_p']:.2e})")
    # 5. F-test closed forms: zero-noise homogeneity => F == 0; perfect
    #    separation => huge F, tiny p; df (2, 6)
    recs = sf.make_synthetic_population(-0.115, noise_sd=0.0,
                                        purpose="f-selfcheck")
    r0 = f_interaction_test(recs)
    assert abs(r0["F"]) < 1e-9 and (r0["df1"], r0["df2"]) == (2, 6)
    print(f"PASS F(2,6) interaction: zero-noise homogeneous F = "
          f"{r0['F']:.1e}, df = (2, 6)")
    # 6. CI gate logic table (no pipeline): direct interval probes
    probe = ci_gates({"12": dict(slope=-0.115, ci_lo=-0.128, ci_hi=-0.102),
                      "16": dict(slope=-0.119, ci_lo=-0.134, ci_hi=-0.105)})
    assert probe["status"] == "PASS"  # the quoted-context intervals satisfy
    probe2 = ci_gates({"12": dict(slope=-0.06, ci_lo=-0.064, ci_hi=-0.056),
                       "16": dict(slope=-0.06, ci_lo=-0.064, ci_hi=-0.056)})
    assert probe2["status"] == "FAIL"
    print("PASS CI gate logic: contain(-1/8)+exclude(-1/16) decision table")
    print(f"drift notes: {DRIFT_NOTES}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
