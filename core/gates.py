"""core.gates -- statistical tolerance gates for the synthetic verification suites.

Two kinds of assertion live in the suites and must not share a tolerance:

  MECHANISM    an algebraic or population-level identity (closed-form head, gauge, exact
               confusion integral).  Tolerance is numerical (1e-9 relative) and a miss is a
               MECHANISM-FAIL: the code or the theorem is wrong.
  CALIBRATION  a Monte-Carlo or finite-sample estimate compared with a population value.
               The gate is a z-score against the estimate's own standard error; a miss is a
               CALIBRATION-EXCURSION whose size (in standard errors) is reported, never a
               bare "FAIL" at a hand-picked absolute tolerance.

The helpers here implement the calibration gates:

  z_battery        mean |z| against the half-normal expectation 0.798, max |z| against the
                   N-sample order-statistic bound, and a KS test of the z's against N(0, 1)
  binomial_tail    exact two-sided binomial p-value for a low-count cell (MC confusion counts)
  order_stat_bound the expected maximum of N standard normals (Blom approximation), the
                   natural threshold for "largest excursion in a battery of N comparisons"
  one_sided_band   for inequalities that hold in population (a bound), the gate only fires on
                   violations in the forbidden direction, by more than k standard errors
  planted_false    power check: the same gate must FIRE on a deliberately wrong population
                   value shifted by ``shift`` standard errors; a gate that cannot see a
                   planted error has no power and its PASS carries no information
  verdict          the two verdict words and a short machine-readable reason
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np
from scipy import stats as sps

HALF_NORMAL_MEAN = float(np.sqrt(2 / np.pi))   # E|Z| for Z ~ N(0,1) = 0.7979


def order_stat_bound(n: int, quantile: float = 0.99) -> float:
    """Upper ``quantile`` of max_i |Z_i| over n independent standard normals."""
    # P(max|Z| <= t) = (2 Phi(t) - 1)^n
    return float(sps.norm.ppf(0.5 + 0.5 * quantile ** (1.0 / max(n, 1))))


def z_scores(estimates: Sequence[float], truths: Sequence[float], ses: Sequence[float]) -> np.ndarray:
    e = np.asarray(estimates, float); t = np.asarray(truths, float); s = np.asarray(ses, float)
    return (e - t) / np.where(s > 0, s, np.inf)


@dataclass
class GateResult:
    ok: bool
    verdict: str                      # PASS | CALIBRATION-EXCURSION | MECHANISM-FAIL
    value: Dict[str, float] = field(default_factory=dict)
    threshold: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict:
        return dict(ok=self.ok, verdict=self.verdict, value=self.value, threshold=self.threshold, reason=self.reason)


def z_battery(estimates, truths, ses, *, quantile: float = 0.99, mean_tol_se: float = 3.0) -> GateResult:
    """Battery gate for N calibration comparisons with known standard errors.

    Fires when (a) mean|z| exceeds 0.798 by more than ``mean_tol_se`` of its own standard error
    (sd of |Z| is 0.603, so se = 0.603/sqrt(N)), or (b) max|z| exceeds the ``quantile`` order-statistic
    bound for N normals, or (c) the KS test of z against N(0,1) rejects at 1 - quantile."""
    z = z_scores(estimates, truths, ses); z = z[np.isfinite(z)]; n = z.size
    if n == 0:
        return GateResult(False, "MECHANISM-FAIL", reason="no finite z-scores (zero standard errors?)")
    mean_abs = float(np.abs(z).mean()); max_abs = float(np.abs(z).max())
    bound = order_stat_bound(n, quantile)
    mean_bound = HALF_NORMAL_MEAN + mean_tol_se * 0.6028 / np.sqrt(n)
    ks_p = float(sps.kstest(z, "norm").pvalue) if n >= 5 else 1.0
    fails = []
    if mean_abs > mean_bound: fails.append(f"mean|z| {mean_abs:.2f} > {mean_bound:.2f}")
    if max_abs > bound: fails.append(f"max|z| {max_abs:.2f} > {bound:.2f}")
    if ks_p < 1 - quantile: fails.append(f"KS p {ks_p:.3g} < {1 - quantile:.3g}")
    ok = not fails
    return GateResult(ok, "PASS" if ok else "CALIBRATION-EXCURSION",
                      value=dict(n=n, mean_abs_z=mean_abs, max_abs_z=max_abs, ks_p=ks_p),
                      threshold=dict(mean_abs_z=mean_bound, max_abs_z=bound, ks_p=1 - quantile),
                      reason="; ".join(fails) if fails else f"{n} comparisons consistent with N(0,1) errors")


def binomial_tail(count: int, n: int, p0: float) -> float:
    """Exact two-sided p-value of observing ``count`` successes in n Bernoulli(p0) trials."""
    return float(sps.binomtest(int(count), int(n), float(p0)).pvalue)


def binomial_battery(counts, ns, p0s, *, alpha: float = 0.01) -> GateResult:
    """Low-count MC cells (confusion probabilities near 0): exact binomial tails with a Bonferroni-
    corrected level, which replaces the absolute tolerances that are meaningless when p0*n is small."""
    ps = np.array([binomial_tail(c, n, p) for c, n, p in zip(counts, ns, p0s)])
    m = len(ps); level = alpha / max(m, 1)
    bad = np.flatnonzero(ps < level)
    ok = bad.size == 0
    return GateResult(ok, "PASS" if ok else "CALIBRATION-EXCURSION",
                      value=dict(n_cells=m, min_p=float(ps.min()) if m else 1.0, n_fail=int(bad.size)),
                      threshold=dict(bonferroni_level=level),
                      reason=("" if ok else f"{bad.size} cell(s) below the Bonferroni level {level:.2e}: indices {bad.tolist()[:8]}"))


def one_sided_band(estimates, bounds, ses, *, direction: str = "below", k: float = 3.0) -> GateResult:
    """Population inequality estimate <= bound (direction='below') or >= bound ('above'); fires only on
    violations in the forbidden direction by more than k standard errors."""
    e = np.asarray(estimates, float); b = np.asarray(bounds, float); s = np.asarray(ses, float)
    excess = (e - b) if direction == "below" else (b - e)
    z = excess / np.where(s > 0, s, np.inf)
    worst = float(np.nanmax(z)) if z.size else 0.0
    ok = worst <= k
    return GateResult(ok, "PASS" if ok else "CALIBRATION-EXCURSION", value=dict(worst_violation_se=worst, n=int(z.size)),
                      threshold=dict(k_se=k), reason="" if ok else f"inequality violated by {worst:.1f} standard errors")


def mechanism(value: float, reference: float, *, rtol: float = 1e-9, atol: float = 1e-12) -> GateResult:
    """Algebraic identity gate: |value - reference| <= atol + rtol |reference|."""
    dev = abs(float(value) - float(reference)); tol = atol + rtol * abs(float(reference))
    ok = dev <= tol
    return GateResult(ok, "PASS" if ok else "MECHANISM-FAIL", value=dict(deviation=dev), threshold=dict(tolerance=tol),
                      reason="" if ok else f"identity violated by {dev:.3e} (tolerance {tol:.1e})")


def planted_false(gate, estimates, truths, ses, *, shift_se: float = 4.0, **kw) -> GateResult:
    """Power check for a calibration gate: shift every population value by ``shift_se`` standard errors and
    require the gate to fire.  Returns PASS when the gate has power (fires on the planted error)."""
    t = np.asarray(truths, float) + shift_se * np.asarray(ses, float)
    r = gate(estimates, t, ses, **kw)
    ok = not r.ok
    return GateResult(ok, "PASS" if ok else "MECHANISM-FAIL", value=dict(planted_shift_se=shift_se),
                      reason=("gate fires on a planted error" if ok else "gate did not fire on a planted error: no power"))


# ---- verdict vocabulary (ruling of 2026-10-22, docs/VERDICTS.md) ------------------------------------
# The verdict unit is the CLAIM.  Three words are printed:
#   MECHANISM-FAIL         the mechanism the claim asserts is not exhibited: an exact arm misses its numerical
#                          tolerance, a finite-sample arm misses by more than the battery's family-wise bound,
#                          or the spec design cannot exhibit the claim at any sample size ("[design]")
#   CALIBRATION-EXCURSION  the mechanism is exhibited and one gate number is missed within the battery's bound
#   PAPER-QUESTION         a deterministic finding about a printed number (no Monte Carlo, no gate to recalibrate);
#                          the question is carried to the owner's Q-list and the measured value is printed
PAPER_QUESTIONS = {"E4.2c", "A23a", "A26"}
DESIGN_WORDS = ("unidentified", "non-identifiable", "nonidentifiable", "not identifiable", "n-insensitive", "saturated",
                "floor artifact", "sits at/below", "at any n", "tail-limited", "overlap-mass-limited")
Z_MECHANISM = 5.0   # a finite-sample miss beyond this many standard errors is not a calibration matter
# Suite records whose spec geometry is non-identifiable at any sample size for every assertion in the record
# (B_synthetic_bias_E1: median within-task Mahalanobis about 48, condition number about 180; the n = 2e5 anchor
# reproduces the same misses).  Every failing spec arm of such a record is a [design] mechanism failure, including
# arms whose own diagnosis line only names the downstream symptom.
DESIGN_RECORDS = {"B_synthetic_bias_E1"}


def claim_of(assertion_id) -> str:
    """The claim an assertion row belongs to: the id up to its first arm bracket (E5.2[spec|exact] -> E5.2)."""
    s = str(assertion_id)
    return s.split("[", 1)[0].strip()


def is_spec_arm(assertion: dict) -> bool:
    text = (str(assertion.get("id", "")) + " " + str(assertion.get("arm", ""))).lower()
    return "spec" in text and "variant" not in text


def is_variant_arm(assertion: dict) -> bool:
    return "variant" in (str(assertion.get("id", "")) + " " + str(assertion.get("arm", ""))).lower()


def _z_of(assertion: dict):
    """A reported z-score, when the suite wrote one into value or diagnosis ("z = 9.94", "z=0.579")."""
    import re
    text = " ".join(str(assertion.get(k, "")) for k in ("value", "diagnosis", "desc"))
    m = re.findall(r"\bz\s*=\s*(-?\d+(?:\.\d+)?)", text)
    return max(abs(float(v)) for v in m) if m else None


def tier_of(assertion: dict) -> str:
    """MECHANISM for arms exact in all their inputs (closed-form / population / algebraic); CALIBRATION for arms
    with sampling anywhere in their inputs: Monte-Carlo, empirical, estimation, bootstrap, and exact functionals
    of a finite-sample object (a trained or plug-in head whose value carries a confidence interval)."""
    arm = str(assertion.get("arm", "")).lower(); thr = str(assertion.get("threshold", "")).lower()
    desc = str(assertion.get("desc", assertion.get("description", ""))).lower()
    value = str(assertion.get("value", "")).lower()
    sampled = ("mc", "monte", "empirical", "estimat", "bootstrap", "sample", "trained", "n=", "seeds", "protocol")
    if any(w in arm for w in sampled):
        return "CALIBRATION"
    if "ci" in value.replace("ci_", "ci") and ("95%" in value or "ci [" in value or "ci=" in value):
        return "CALIBRATION"          # a confidence interval means the quantity is an estimate, whatever the arm is called
    if any(w in desc for w in ("lam(n)", "plug-in", "trained", "finite-sample", "finite sample")):
        return "CALIBRATION"
    text = " ".join([arm, thr, desc])
    if any(w in text for w in ("exact", "algebra", "closed-form", "population", "identity", "1e-9", "1e-10", "1e-12", "1e-8")):
        return "MECHANISM"
    return "CALIBRATION"


def verdict(assertion: dict) -> str:
    """The arm's own verdict word (the claim-level word is ``claim_verdict``)."""
    if assertion.get("status", "PASS") == "PASS":
        return "PASS"
    if claim_of(assertion.get("id")) in PAPER_QUESTIONS:
        return "PAPER-QUESTION"
    if tier_of(assertion) == "MECHANISM":
        return "MECHANISM-FAIL"
    diag = str(assertion.get("diagnosis", "")).lower()
    rec = str(assertion.get("run", assertion.get("experiment", "")))
    if is_spec_arm(assertion) and (any(w in diag for w in DESIGN_WORDS) or any(d in rec for d in DESIGN_RECORDS)):
        return "MECHANISM-FAIL [design]"
    z = _z_of(assertion)
    if z is not None and z > Z_MECHANISM:
        return "MECHANISM-FAIL"
    return "CALIBRATION-EXCURSION"


def claim_verdict(arms: list) -> str:
    """One word per claim from its arms' words: PASS if every arm passes; PAPER-QUESTION for the deterministic
    printed-number findings; MECHANISM-FAIL if any arm's word is one (a confirming Monte-Carlo arm inherits the
    claim's word; a non-identifiable spec design is a mechanism failure of the demonstration, qualified
    [design], with the variant arm's result printed beside it); CALIBRATION-EXCURSION otherwise."""
    words = [verdict(a) for a in arms]
    if all(w == "PASS" for w in words):
        return "PASS"
    if any(w == "PAPER-QUESTION" for w in words):
        return "PAPER-QUESTION"
    if any(w.startswith("MECHANISM-FAIL [design]") for w in words):
        rescue = [a for a in arms if is_variant_arm(a)]
        if rescue:
            tag = ", ".join(f"{str(a.get('id'))}: {verdict(a)}" for a in rescue)
            return f"MECHANISM-FAIL [design; variant: {tag}]"
        return "MECHANISM-FAIL [design]"
    if any(w == "MECHANISM-FAIL" for w in words):
        return "MECHANISM-FAIL"
    return "CALIBRATION-EXCURSION"
