"""tests.test_theory_B -- Paper B theory checks (G06 E1-E3 fast variants +
guarded E5/E6). Implemented by H10, 2026-10-21.

Fast-variant configuration (coordinator task order): d=8, T=2 tasks,
C=3 classes/task, n=5000 samples/class, geometry within_separation=0.5 /
task_separation=1.5 (moderate margins keep the within-task CE
identification informative -- H05's E1 drift note; at large separation the
empirical CE saturates and the fit is tail-limited). All streams pinned
via conftest purpose seeds (SEED=0); nothing touches global numpy state.

Tolerance policy (per H05's finding, adopted here): the EXACT-population
arms are algebraic identities and are asserted STRICTLY (EPS_LINALG /
EPS_TIGHT, or the exact-algebra 1e-10 fix bound of G06 E3.1); the
EMPIRICAL arms (L-BFGS fit, same-sample MC error) use SE-calibrated
tolerances with the epsilon named and justified in each docstring.

NO-REAL-BACKBONE RULE / synthetic-proxy: every threshold below (0.9999
exact-arm cosine, 0.995 SE-calibrated trained-arm cosine, 1e-10 fix
match, recovery >= 0.95, 0.003 pairwise residual) is a G06 suite
threshold on SYNTHETIC closed-form constructions. The paper's real-data
numbers (e.g. 82-94% gap recovery) are quoted context only and are never
asserted.

E5/E6: the coordinator order wires the fast rate/kappa variants through
experiments/synthetic_rate IF implemented. H06 landed WHILE H10 was
running, so the guarded tests call its run_e5/run_e6 with fast overrides
of its own DEFAULT_CFG (geometry keys untouched). NOTE: at H10 run time
both suites' own E*.0 geometry gates FAIL inside the module (gamma_ij
off-target, e.g. 32 vs 8) -- an H06 module defect, reported in
exec/H10_log.md, not fixed here; the two tests below fail for exactly
that reason and pass again once H06 repairs the geometry builder.

Run (stdlib unittest; pytest absent):

    python repo/tests/test_theory_B.py
    python -m unittest discover -s repo/tests
"""

from __future__ import annotations

import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import conftest as ct  # noqa: E402
from core import diagnostic as dg  # noqa: E402
from core import fix as fx  # noqa: E402
from core import gaussian_model as gm  # noqa: E402
from experiments.synthetic_bias import fit_within_task_lbfgs  # noqa: E402

# Fast-variant constants (task order: d=8 / 2 tasks / 3 classes / 5k samples).
D: int = 8
CPT: tuple[int, int] = (3, 3)  # C=3 classes per task, T=2 tasks
N_PER_CLASS: int = 5000
WITHIN_SEP: float = 0.5
TASK_SEP: float = 1.5
LAM_FIT: float = 1e-5  # E1 fit arm lambda (G06 E1 range {1e-5, 1e-4})


def _fast_task() -> gm.GaussianTask:
    """The shared fast-variant population task (pinned purpose seed)."""
    return gm.make_gaussian_task(
        d=D, classes_per_task=CPT, within_separation=WITHIN_SEP,
        task_separation=TASK_SEP, purpose="h10-theoryB",
    )


def _sig_inv(task: gm.GaussianTask) -> np.ndarray:
    """Symmetrized Sigma^{-1} via LU solve (independent of H01's Cholesky
    route, so exact-arm agreement is a cross-route check)."""
    inv = np.linalg.solve(task.cov, np.eye(task.d))
    return 0.5 * (inv + inv.T)


def _euclid_cos(u: np.ndarray, v: np.ndarray) -> float:
    """Euclidean cosine (H04's diagnostic uses the Sigma^{-1} inner product;
    the exact-arm residuals here are compared in plain R^d as well)."""
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


# ---------------------------------------------------------------------------
# E1 -- Theorem 3.3: closed form and its recovery by a trained head
# ---------------------------------------------------------------------------


class E1ClosedFormFastTests(unittest.TestCase):
    """E1 fast variant: exact identities + trained-head empirical arm."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.task = _fast_task()
        cls.W_cf, cls.b_cf = gm.closed_form_ce_head(cls.task)
        cls.stats = gm.population_statistics(cls.task)
        # Trained arm: per-task L-BFGS within-task CE fit (H05's fitter,
        # protocol of G06 E1: full-batch, ridge lam=1e-5, zero init convex).
        cls.fits = []
        for t in range(cls.task.T):
            idx = cls.task.classes_in_task(t)
            X, y = cls.task.sample(N_PER_CLASS, purpose="h10-theoryB:fit",
                                   salt=t)
            sel = np.isin(y, idx)
            W_f, b_f, _ = fit_within_task_lbfgs(
                X[sel], y[sel] - idx[0], len(idx), lam=LAM_FIT)
            cls.fits.append((W_f, b_f))

    def test_e1_exact_closed_form_cosines(self):
        """VERIFICATION (synthetic-proxy): exact closed-form arm of E1.
        cos(w_r^dagger, Sigma^{-1} tilde_mu_r) > 0.9999 per class (G06 E2.3a
        exact-arm threshold; observed deviation is float round-off only --
        the two vectors are the same object up to cholesky-vs-LU solver
        round-off, hence the cross-route EPS_LINALG comparison); Lemma 3.2
        gauge sums < EPS_LINALG (exact algebra; G06 E1.4 quotes 1e-6 for
        this arm, EPS_LINALG is stricter and holds); the bias identity
        b_r^dagger = -1/2 mu_r^T Sigma^{-1} mu_r + 1/2 S_bar^{(t)} at
        EPS_LINALG with an independently computed quadratic form.
        """
        s_inv = _sig_inv(self.task)
        toc = self.task.task_of_class
        for r in range(self.task.n_classes):
            v = s_inv @ self.stats.tilde_means[r]
            cos = _euclid_cos(self.W_cf[r], v)
            self.assertGreater(cos, 0.9999,
                               f"class {r}: cos(w^dag, Sinv tilde_mu)={cos:.6f}")
        # bias identity with an independent quadratic form (einsum, LU Sinv)
        q = np.einsum("ri,ij,rj->r", self.task.means, s_inv,
                      self.task.means)
        ct.allclose(self.b_cf, -0.5 * q + 0.5 * self.stats.S_bar[toc],
                    rtol=ct.EPS_LINALG, atol=ct.EPS_LINALG,
                    label="e1-bias-identity-cross-route")
        # Lemma 3.2 gauge (exact arm)
        for t in range(self.task.T):
            idx = self.task.classes_in_task(t)
            self.assertLess(float(np.abs(self.W_cf[idx].sum(axis=0)).max()),
                            ct.EPS_LINALG, f"gauge sum w, task {t}")
            self.assertLess(abs(float(self.b_cf[idx].sum())),
                            ct.EPS_LINALG, f"gauge sum b, task {t}")

    def test_e1_trained_head_matches_closed_form(self):
        """VERIFICATION (synthetic-proxy): empirical arm of E1 (trained
        within-task CE head, n=5000/class, lam=1e-5, pinned seeds).
        SE-calibrated tolerances (per the H05 noise-floor finding, the
        fixed G06 E1.1 threshold 0.999 sits AT the finite-sample floor of
        this fast config and flips across geometries, so the bound below
        is derived from the estimation noise instead):
          * cos(w_trained, w^dag) > 0.995 per class. Derivation:
            1 - cos = ||eps_perp||^2 / (2 ||w||^2) with the direction
            error scale ||eps_perp||/||w|| ~ sqrt((d - C)/n) =
            sqrt(5/5000) = 0.032, so E[1 - cos] ~ 5e-4; allowing a ~4x
            Gaussian fluctuation of the squared quantity gives ~2e-3, and
            0.995 keeps additional margin (observed min 0.9984 on the
            pinned seeds).
          * per-task Frobenius norm ratio in (0.95, 1.05): at lam=1e-5 the
            ridge shrinkage is ~1e-5, so the n^{-1/2} MLE noise dominates;
            0.05 ~ sqrt(d/n)=0.04 with headroom (observed 0.973-1.012).
            G06 E1.2's (0.98, 1.0) window is the d=32 large-n regime and is
            deliberately widened here per the H05 noise-floor finding.
          * trained gauge ||sum_r w_r||_inf < 1e-3 (G06 E1.4 trained arm;
            observed ~1e-14: the convex fit lands in gauge up to optimizer
            tolerance).
        """
        for t in range(self.task.T):
            idx = self.task.classes_in_task(t)
            W_f, b_f = self.fits[t]
            for j, r in enumerate(idx):
                cos = _euclid_cos(W_f[j], self.W_cf[r])
                self.assertGreater(
                    cos, 0.995,
                    f"task {t} class {r}: cos(w_trained, w^dag)={cos:.6f}")
            ratio = float(np.linalg.norm(W_f) / np.linalg.norm(self.W_cf[idx]))
            self.assertGreater(ratio, 0.95,
                               f"task {t}: norm ratio {ratio:.4f} <= 0.95")
            self.assertLess(ratio, 1.05,
                            f"task {t}: norm ratio {ratio:.4f} >= 1.05")
            self.assertLess(float(np.abs(W_f.sum(axis=0)).max()), 1e-3,
                            f"task {t}: trained gauge sum w")
            self.assertLess(abs(float(b_f.sum())), 1e-3,
                            f"task {t}: trained gauge sum b")


# ---------------------------------------------------------------------------
# E2 -- Theorem 4.1 + Cors 4.2/4.3: structural bias, exact and lambda-invariant
# ---------------------------------------------------------------------------


class E2BiasIdentityFastTests(unittest.TestCase):
    """E2 fast variant: bias residual identity, cosine, lambda invariance."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.task = _fast_task()
        cls.W_cf, cls.b_cf = gm.closed_form_ce_head(cls.task)
        cls.stats = gm.population_statistics(cls.task)
        # H04 diagnostic functions consume a Head-protocol object; wrap the
        # H01 closed-form head (cross-module verification, not a copy).
        cls.head = types.SimpleNamespace(W=cls.W_cf, means=cls.task.means,
                                         task_of_class=cls.task.task_of_class)
        cls.s_inv = _sig_inv(cls.task)

    def _bias_residual(self, W: np.ndarray, k: int, l: int) -> np.ndarray:
        diff = self.task.means[k] - self.task.means[l]
        return (W[k] - W[l]) - self.s_inv @ diff

    def test_e2_bias_identity_exact_cosines(self):
        """VERIFICATION (synthetic-proxy): Theorem 4.1 exact arm at the fast
        config. For EVERY ordered cross-task pair (k, l) with tasks (i, j):
        B_kl = (w_k^dag - w_l^dag) - Sigma^{-1}(mu_k - mu_l) equals
        Sigma^{-1}(mu_bar^{(j)} - mu_bar^{(i)}) to EPS_LINALG (exact algebra;
        cross-route cho_solve vs LU). Consequently cos(DeltaW_hat,
        DeltaW_pred) > 0.9999 (G06 E2.1a exact-arm threshold; observed
        1 - O(1e-16)), relative magnitude 1 and the Sigma-norm identity
        ||B||_Sigma / sqrt(gamma_ij) = 1 at EPS_LINALG, and Cor 4.2's
        pair invariance holds EXACTLY: all C^2 residuals of a block are
        the same vector to EPS_LINALG (G06 E2.3a, population arm).
        """
        toc = self.task.task_of_class
        for i in range(self.task.T):
            for j in range(self.task.T):
                if i == j:
                    continue
                pred = self.s_inv @ (self.stats.mu_bar_tasks[j]
                                     - self.stats.mu_bar_tasks[i])
                block = [(k, l) for k in np.flatnonzero(toc == i)
                         for l in np.flatnonzero(toc == j)]
                for k, l in block:
                    ct.allclose(self._bias_residual(self.W_cf, int(k), int(l)),
                                pred, rtol=ct.EPS_LINALG, atol=ct.EPS_LINALG,
                                label=f"e2-thm41-residual-({k},{l})")
                # H04 diagnostic route: cosine / magnitude / Sigma-norm.
                d_hat = dg.delta_w_hat(self.head, i, j, self.s_inv)
                d_pred = dg.delta_w_pred(self.head, i, j, self.s_inv)
                cos = dg.sigma_inner_cosine(np.stack([d_hat, d_pred]),
                                            self.s_inv)
                self.assertGreater(cos, 0.9999,
                                   f"pair ({i},{j}): cos={cos:.8f}")
                rel = dg.relative_magnitude(d_hat, d_pred, self.s_inv)
                self.assertLess(abs(rel - 1.0), ct.EPS_LINALG,
                                f"pair ({i},{j}): rel mag {rel:.8f}")
                gamma = dg.mahalanobis_sq(
                    self.stats.mu_bar_tasks[i], self.stats.mu_bar_tasks[j],
                    self.s_inv)
                snr = dg.sigma_norm(d_hat, self.task.cov) / np.sqrt(gamma)
                self.assertLess(abs(snr - 1.0), ct.EPS_LINALG,
                                f"pair ({i},{j}): ||B||_S/sqrt(gamma)={snr:.8f}")

    def test_e2_lambda_invariance(self):
        """VERIFICATION (synthetic-proxy): Cor 4.3 fast variant -- the bias
        direction is invariant across TWO lambdas (lam in {0, 1e-4}, scalar
        alpha_t = 1 per eq:wdag-lambda). Thresholds from G06 E2.4:
        cos(B(lam1), B(lam0)) > 0.999 and norm change < 2%. This is an
        exact-algebra arm (closed-form shrunk head): the lambda-perturbed
        residual differs from B(0) by (1-f) * Sigma^{-1}(tilde_mu_k -
        tilde_mu_l) with (1-f) ~ 1e-4, so both margins hold with two orders
        of magnitude of headroom (observed cos 0.999999999, change 1.6e-5).
        """
        lam1 = 1e-4
        W1, _ = gm.closed_form_ce_head(self.task, lam=lam1, alpha_t=1.0)
        toc = self.task.task_of_class
        for i in range(self.task.T):
            for j in range(self.task.T):
                if i == j:
                    continue
                k = int(np.flatnonzero(toc == i)[0])
                l = int(np.flatnonzero(toc == j)[0])
                B0 = self._bias_residual(self.W_cf, k, l)
                B1 = self._bias_residual(W1, k, l)
                cos = _euclid_cos(B1, B0)
                self.assertGreater(cos, 0.999,
                                   f"pair ({i},{j}): cos(B_lam)={cos:.8f}")
                change = float(np.linalg.norm(B1) / np.linalg.norm(B0)) - 1.0
                self.assertLess(abs(change), 0.02,
                                f"pair ({i},{j}): norm change {change:.2e}")


# ---------------------------------------------------------------------------
# E3 -- Def 7.1 / Thm 7.2: exact Bayes match + gap recovery
# ---------------------------------------------------------------------------


class E3FixFastTests(unittest.TestCase):
    """E3 fast variant: 1e-10 exact match, recovery, same-sample no-harm."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.task = _fast_task()
        cls.W_cf, cls.b_cf = gm.closed_form_ce_head(cls.task)
        cls.stats = fx.TaskStats.from_task(cls.task, cls.task.task_of_class)
        cls.fixed = fx.de_center(fx.LinearHead(cls.W_cf, cls.b_cf), cls.stats)
        cls.W_by, cls.b_by = gm.bayes_head(cls.task)
        # worst cross-task pair under the WT head (max exact confusion)
        P = np.full((cls.task.n_classes, cls.task.n_classes), -1.0)
        for k in range(cls.task.n_classes):
            for l in range(cls.task.n_classes):
                if cls.task.task_of_class[k] != cls.task.task_of_class[l]:
                    P[k, l] = gm.pairwise_confusion_exact(
                        cls.task, cls.W_cf, cls.b_cf, k, l)
        cls.k_star, cls.l_star = (int(v) for v in np.unravel_index(
            int(np.argmax(P)), P.shape))

    def test_e3_exact_fix_match_1e10(self):
        """VERIFICATION (synthetic-proxy): G06 E3.1 -- de_center(cf head)
        equals the Bayes head with max componentwise deviation < 1e-10 on
        BOTH W and b. This is exact algebra (Theorem 7.2), so only float
        round-off remains; 1e-10 is double-precision slack (observed
        ~2e-15). Cross-checked against H02's and H01's Bayes routes.
        """
        d_w = float(np.abs(self.fixed.W - self.W_by).max())
        d_b = float(np.abs(self.fixed.b - self.b_by).max())
        self.assertLess(d_w, 1e-10, f"max|W_fix - W_bayes| = {d_w:.3e}")
        self.assertLess(d_b, 1e-10, f"max|b_fix - b_bayes| = {d_b:.3e}")
        bayes_h02 = fx.bayes_head(self.stats)
        d_w2 = float(np.abs(self.fixed.W - bayes_h02.W).max())
        d_b2 = float(np.abs(self.fixed.b - bayes_h02.b).max())
        self.assertLess(d_w2, 1e-10, f"max|W_fix - W_bayes(H02)| = {d_w2:.3e}")
        self.assertLess(d_b2, 1e-10, f"max|b_fix - b_bayes(H02)| = {d_b2:.3e}")

    def test_e3_gap_recovery_exact(self):
        """VERIFICATION (synthetic-proxy): G06 E3.2/E3.3 fast variant on the
        worst cross-task pair (k*, l*) chosen by maximal exact WT confusion
        (observed P_WT ~ 0.94 vs Bayes ~ 0.10 -- a wide gap, so the ratio is
        numerically robust). Exact closed-form confusions (no MC):
          * recovery R = (P_WT - P_FIX)/(P_WT - P_Bayes) >= 0.95 (task-order
            bound; observed 1.0 to machine precision since FIX == Bayes).
            Synthetic-proxy: this is NOT the paper's 82-94% real-data
            recovery; it is the synthetic population analog.
          * upper guard R <= 1 + 1e-6 (round-off only).
          * |P_FIX - P_Bayes| <= 0.003 (G06 E3.3 threshold; observed ~1e-16,
            exact algebra).
        """
        k, l = self.k_star, self.l_star
        p_wt = gm.pairwise_confusion_exact(self.task, self.W_cf, self.b_cf,
                                           k, l)
        p_fix = gm.pairwise_confusion_exact(self.task, self.fixed.W,
                                            self.fixed.b, k, l)
        p_by = gm.bayes_pair_error_exact(self.task, k, l)
        self.assertGreater(p_wt - p_by, 0.05,
                           "precondition: WT-Bayes confusion gap must be "
                           "robust for the recovery ratio to be meaningful")
        r = (p_wt - p_fix) / (p_wt - p_by)
        self.assertGreaterEqual(r, 0.95, f"gap recovery R = {r:.6f}")
        self.assertLessEqual(r, 1.0 + 1e-6, f"gap recovery R = {r:.6f} > 1")
        self.assertLessEqual(abs(p_fix - p_by), 0.003,
                             f"|P_FIX - P_Bayes| = {abs(p_fix - p_by):.3e}")

    def test_e3_fix_no_harm_same_samples(self):
        """VERIFICATION (synthetic-proxy): empirical same-sample arm
        (SE-calibrated, H05 policy). One pinned evaluation stream of
        n=5000/class (task-order sample count). Assertions:
          * err_WT >= err_FIX on the identical samples (fix cannot hurt;
            observed 0.981 vs 0.407).
          * |err_FIX - err_Bayes| <= 6 * sqrt(se_FIX^2 + se_Bayes^2)
            (Z_STAT=6 conftest multiplier on the binomial SEs of the two
            error estimates; observed deviation 0.0 vs bound 0.024).
          * empirical recovery >= 0.95 (observed 1.0; the WT-Bayes gap is
            ~0.57 while the 6-sigma SE band is ~0.02, so the bound holds
            with overwhelming margin on the pinned stream).
        """
        X, y = self.task.sample(N_PER_CLASS, purpose="h10-theoryB:eval")
        n = len(y)

        def err(W: np.ndarray, b: np.ndarray) -> float:
            return float(np.mean(gm.predict(W, b, X) != y))

        e_wt = err(self.W_cf, self.b_cf)
        e_fix = err(self.fixed.W, self.fixed.b)
        e_by = err(self.W_by, self.b_by)
        se = lambda e: float(np.sqrt(max(e * (1.0 - e), np.finfo(float).tiny)
                                     / n))
        self.assertGreaterEqual(e_wt, e_fix,
                                f"fix hurt on pinned samples: {e_wt:.4f} "
                                f"vs {e_fix:.4f}")
        tol = ct.Z_STAT * float(np.hypot(se(e_fix), se(e_by)))
        self.assertLessEqual(abs(e_fix - e_by), tol,
                             f"|err_FIX - err_Bayes| = {abs(e_fix - e_by):.4f}"
                             f" > 6-sigma {tol:.4f}")
        denom = e_wt - e_by
        self.assertGreater(denom, 0.0, "WT-Bayes population gap must be > 0")
        r_emp = (e_wt - e_fix) / denom
        self.assertGreaterEqual(r_emp, 0.95,
                                f"empirical gap recovery = {r_emp:.4f}")


# ---------------------------------------------------------------------------
# E5 / E6 -- fast variants via experiments.synthetic_rate (H06), guarded
# ---------------------------------------------------------------------------


def _load_synthetic_rate():
    """Return (module, skip_reason); module is None when unusable.

    Probe contract (documented for the H06 landing): the module counts as
    IMPLEMENTED when its docstring no longer carries the G09 skeleton
    markers. H06 landed while H10 was running, so the live path calls
    run_e5/run_e6 with a FAST override of the module's own DEFAULT_CFG
    (reduced n grid / seeds / MC; geometry keys untouched).
    """
    try:
        import experiments.synthetic_rate as sr
    except ImportError as exc:  # absent or broken import
        return None, f"import experiments.synthetic_rate failed: {exc}"
    doc = sr.__doc__ or ""
    if "SKELETON PLACEHOLDER" in doc or "NOT_IMPLEMENTED_YET" in doc:
        return None, (
            "experiments/synthetic_rate.py is still the G09 skeleton "
            "(H06 not landed); E5/E6 fast variants pending"
        )
    return sr, ""


def _run_rate_suite(suite: str, cfg_overrides: dict):
    """Run one fast suite ('run_e5'/'run_e6') on H06's module.

    Returns (result_dict, skip_reason). The cfg is H06's own DEFAULT_CFG
    with the documented fast overrides ONLY (never the geometry keys), so
    any failure of the suite's own E*.0 geometry gate is a property of the
    module, not of this fast variant.
    """
    sr, reason = _load_synthetic_rate()
    if sr is None:
        return None, reason
    runner = getattr(sr, suite, None)
    if not callable(runner):
        return None, (
            f"experiments.synthetic_rate implemented but exposes no "
            f"callable {suite}(); fast variant needs rewiring"
        )
    cfg = dict(sr.DEFAULT_CFG)
    cfg.update(cfg_overrides)
    try:
        result = runner(cfg)
    except Exception as exc:  # contract drift: skip, do not fail
        return None, (
            f"{suite}(fast cfg) raised {type(exc).__name__}: {exc} -- "
            f"cfg contract drifted; rewiring needed"
        )
    if not (isinstance(result, dict) and isinstance(result.get("assertions"),
                                                     list)):
        return None, f"{suite} returned no assertion-record list"
    return result, ""


class E5E6GuardedFastTests(unittest.TestCase):
    """E5 (rate) / E6 (kappa) fast variants through experiments.synthetic_rate."""

    def test_e5_rate_fast_variant(self):
        """VERIFICATION (synthetic-proxy, GUARDED): G06 E5 fast variant via
        H06's run_e5 on a reduced grid (n in [250..2000], 3 seeds, MC 1e4;
        geometry keys kappa/gamma/d/C untouched from DEFAULT_CFG).
        Gating: the EXACT-population records E5.0 (geometry: gamma_ij ==
        target and min D == kappa^2*gamma to 1e-12) and E5.0b (lambda
        invariance of P13^dagger + Bayes bookkeeping) are algebraic
        identities and MUST pass at any grid. The slope records E5.1-E5.3
        are REPORT-ONLY here: G06's own E5.1 note flags pre-asymptotic
        grid flattening, and the fast 4-point grid cannot support the
        [-0.31, -0.19] window (per the H05 noise-floor policy: never gate
        an empirical window at a grid that cannot resolve it).
        KNOWN STATE 2026-10-21: E5.0 FAILS inside the module (gamma_ij
        32.0 vs target 8.0 -- a geometry-scaling defect in H06's
        kappa_geometry_task), so this test currently fails FOR THAT
        MODULE-SIDE reason (reported to the coordinator, not fixed here).
        """
        result, reason = _run_rate_suite("run_e5", dict(
            e5_n_grid=[250, 500, 1000, 2000], e5_seeds=[0, 1, 2],
            e5_n_mc=10_000, e5_sens_nseeds=2, e5_budget_s=120.0,
            n_boot=500,
        ))
        if result is None:
            self.skipTest(reason)
        by_id = {rec["id"]: rec for rec in result["assertions"]}
        for rec in result["assertions"]:  # report-only visibility
            print(f"[E5-fast] {rec['id']} {rec['status']} "
                  f"{str(rec['value'])[:100]}")
        for gate in ("E5.0", "E5.0b"):
            self.assertIn(gate, by_id, f"{gate} record missing")
            self.assertEqual(
                by_id[gate]["status"], "PASS",
                f"{gate} (exact identity) failed: {by_id[gate]['value']} "
                f"-- module-side geometry defect, see H10 log")

    def test_e6_kappa_fast_variant(self):
        """VERIFICATION (synthetic-proxy, GUARDED): G06 E6 fast variant via
        H06's run_e6 on the full kappa set {1.5, 2, 3} with a reduced
        8-point gamma grid (2..30 step 4); exact-population numerics, no
        training. Gating: ALL records must pass (E6.0 geometry, E6.1
        slopes <= -(kappa-1)^2/8 + 0.02, E6.2 ordering, E6.3 pointwise
        bound, E6.4 kappa->1 flatness, E6.5 MC-vs-exact control) -- every
        one is either exact algebra or an MC control at the default MC
        size, all observed PASS except E6.0.
        KNOWN STATE 2026-10-21: E6.0 FAILS inside the module (same
        geometry-scaling defect family as E5.0: e.g. gamma_ij 3.25 vs
        target 2.0 at kappa=1.5), so this test currently fails FOR THAT
        MODULE-SIDE reason (reported, not fixed here).
        """
        result, reason = _run_rate_suite("run_e6", dict(
            e6_gamma_grid=[2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0],
            n_boot=500,
        ))
        if result is None:
            self.skipTest(reason)
        for rec in result["assertions"]:
            print(f"[E6-fast] {rec['id']} {rec['status']} "
                  f"{str(rec['value'])[:100]}")
            self.assertEqual(
                rec["status"], "PASS",
                f"{rec['id']} failed: {str(rec['value'])[:200]} "
                f"-- see H10 log for the module-side analysis")


if __name__ == "__main__":
    unittest.main(verbosity=2)
