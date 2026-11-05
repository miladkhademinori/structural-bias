"""tests.test_core -- unit tests for core/ modules under test by H10.

Coverage (coordinator task order, H10):
  * closed-form head identities: eq:wdag-lambda shrinkage formula; the
    Bayes <-> CF-head relation at lambda -> 0+ (H01 core.gaussian_model);
  * LDA >= NCM ordering on a separated synthetic task (H03 core.lda_ncm);
  * diagnostic vs known-offset truth, cos > 0.999 (H04 core.diagnostic);
  * fix round-trip: de-centered head == Bayes population head to 1e-9
    (H02 core.fix, Def 7.1 / Theorem 7.2);
  * determinism: bit-identity across reruns (EPS_EXACT tier).

Framework: tests/conftest.py (G10) -- stdlib unittest, SHA-256 purpose
seeds (SEED=0), tolerance helpers, NO-REAL-BACKBONE rule. Every test
declares its tier in its docstring. Run:

    python repo/tests/test_core.py
    python -m unittest discover -s repo/tests

Per H05's finding, exact identities are asserted STRICTLY (machine
precision via the conftest epsilons) and no empirical/MC arm appears in
this file (those live in test_theory_B.py with SE-calibrated tolerances).
Synthetic-proxy label: every number below is a fact of a synthetic
Gaussian construction; no paper measurement is asserted.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# conftest lives next to this file; repo root hosts the core/ package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import conftest as ct  # noqa: E402
from core import diagnostic as dg  # noqa: E402
from core import fix as fx  # noqa: E402
from core import gaussian_model as gm  # noqa: E402
from core import lda_ncm as ln  # noqa: E402


def _cil_task(purpose: str) -> gm.GaussianTask:
    """Shared small population CIL task: d=8, T=2, C=3 per task."""
    return gm.make_gaussian_task(
        d=8, classes_per_task=(3, 3), within_separation=1.5,
        task_separation=2.0, purpose=purpose,
    )


def _sig_inv(task: gm.GaussianTask) -> np.ndarray:
    """Symmetrized Sigma^{-1} via LU solve (independent route from H01's
    Cholesky solves, so agreement is a real cross-check, not a tautology)."""
    inv = np.linalg.solve(task.cov, np.eye(task.d))
    return 0.5 * (inv + inv.T)


# ---------------------------------------------------------------------------
# 1. Closed-form head identities (H01 core.gaussian_model)
# ---------------------------------------------------------------------------


class ClosedFormHeadIdentityTests(unittest.TestCase):
    """Shrinkage formula and the Bayes relation at lambda -> 0+."""

    def test_shrinkage_formula_exact(self):
        """UNIT: eq:wdag-lambda -- W(lam, alpha) == (1+lam*alpha_t)^{-1} W(0)
        per task, on BOTH w and b (same float64 product => bit-identical;
        EPS_TIGHT guard), plus the alpha_t error contracts.
        """
        task = _cil_task("h10-core-shrinkage")
        lam, alpha = 0.1, np.array([2.0, 3.0])
        W0, b0 = gm.closed_form_ce_head(task)
        Ws, bs = gm.closed_form_ce_head(task, lam=lam, alpha_t=alpha)
        factor = gm.shrinkage_factor(lam, alpha)[task.task_of_class]
        ct.allclose(Ws, W0 * factor[:, None], rtol=0.0, atol=ct.EPS_TIGHT,
                    label="shrinkage-W-same-ops")
        ct.allclose(bs, b0 * factor, rtol=0.0, atol=ct.EPS_TIGHT,
                    label="shrinkage-b-same-ops")
        # the factor itself is the documented rational form (float64 round-off)
        ct.allclose(gm.shrinkage_factor(lam, alpha), 1.0 / (1.0 + lam * alpha),
                    rtol=0.0, atol=ct.EPS_TIGHT, label="shrinkage-factor-form")
        # error contracts: lam < 0, missing alpha_t at lam > 0, and
        # 1 + lam*alpha <= 0 must all refuse (ValueError, never fabricate).
        with self.assertRaises(ValueError):
            gm.shrinkage_factor(-1.0, 1.0)
        with self.assertRaises(ValueError):
            gm.closed_form_ce_head(task, lam=lam)  # alpha_t=None
        with self.assertRaises(ValueError):
            gm.shrinkage_factor(1.0, -2.0)  # 1 + lam*alpha = -1

    def test_bayes_head_lambda_zero_relation(self):
        """UNIT: at lambda -> 0+ the CF head satisfies the exact offset
        identity w_bayes - w_cf = Sigma^{-1} mu_bar^{(t)} and
        b_bayes - b_cf = -0.5 * S_bar^{(t)} (rem:full-vs-within), and the
        shrunk head converges to the unshrunk one with the analytic bound
        |W(lam) - W(0)| <= lam * |W(0)| elementwise.
        """
        task = _cil_task("h10-core-bayes-lam0")
        stats = gm.population_statistics(task)
        s_inv = _sig_inv(task)
        W_cf, b_cf = gm.closed_form_ce_head(task)
        W_by, b_by = gm.bayes_head(task)
        # offset identity, cross-route (H01 cho_solve vs test LU solve):
        # cholesky/lu round-off => EPS_LINALG.
        ct.allclose(W_by - W_cf, stats.mu_bar_tasks[task.task_of_class] @ s_inv,
                    rtol=ct.EPS_LINALG, atol=ct.EPS_LINALG,
                    label="bayes-minus-cf-eq-Sinv-mubar")
        ct.allclose(b_by - b_cf, -0.5 * stats.S_bar[task.task_of_class],
                    rtol=ct.EPS_LINALG, atol=ct.EPS_LINALG,
                    label="bayes-minus-cf-bias-eq-minus-half-Sbar")
        # lambda -> 0+ continuity: |Ws - W0| elementwise <=
        # (lam*alpha/(1+lam*alpha)) * |W0| up to one product round-off, so
        # rtol = lam*(1 + tiny) + atol EPS_TIGHT is the documented bound.
        lam = 1e-9
        Ws, bs = gm.closed_form_ce_head(task, lam=lam, alpha_t=1.0)
        ct.allclose(Ws, W_cf, rtol=lam * 1.000001, atol=ct.EPS_TIGHT,
                    label="lam-to-zero-W-continuity(rtol=lam)")
        ct.allclose(bs, b_cf, rtol=lam * 1.000001, atol=ct.EPS_TIGHT,
                    label="lam-to-zero-b-continuity(rtol=lam)")
        # canonical vs global-centroid gauge: identical argmax on pinned
        # samples; logits differ by a per-sample constant (EPS_LINALG).
        X, _ = task.sample(500, purpose="h10-core-bayes-lam0:gauge")
        W_by_c, _ = gm.bayes_head(task, canonical=False)
        self.assertTrue(np.array_equal(gm.predict(W_by, b_by, X),
                                       gm.predict(W_by_c, b_by, X)),
                        "canonical vs centered gauge must agree on argmax")
        shift = gm.logits(W_by, b_by, X) - gm.logits(W_by_c, b_by, X)
        dev = float(np.abs(shift - shift[:, :1]).max())
        self.assertLess(dev, ct.EPS_LINALG,
                        f"gauge logit shift not constant across classes "
                        f"(max dev {dev:.3e})")

    def test_gauge_and_ST_decomposition(self):
        """UNIT: Lemma 3.2 gauge (per-task sums of w and b vanish) and the
        eq:S-T-decomp identity S_bar = T_bar + ||mu_bar^{(t)}||^2_Sinv,
        exact algebra => EPS_LINALG (einsum vs explicit quadratic form).
        """
        task = _cil_task("h10-core-gauge")
        stats = gm.population_statistics(task)
        W, b = gm.closed_form_ce_head(task)
        s_inv = _sig_inv(task)
        for t in range(task.T):
            idx = task.classes_in_task(t)
            self.assertLess(float(np.abs(W[idx].sum(axis=0)).max()),
                            ct.EPS_LINALG, f"gauge sum_r w_r != 0, task {t}")
            self.assertLess(abs(float(b[idx].sum())), ct.EPS_LINALG,
                            f"gauge sum_r b_r != 0, task {t}")
            self.assertLess(
                float(np.abs(stats.tilde_means[idx].sum(axis=0)).max()),
                ct.EPS_LINALG, f"sum_r tilde_mu_r != 0, task {t}")
        cen = np.einsum("ti,ij,tj->t", stats.mu_bar_tasks, s_inv,
                        stats.mu_bar_tasks)
        ct.allclose(stats.S_bar, stats.T_bar + cen, rtol=ct.EPS_LINALG,
                    atol=ct.EPS_LINALG, label="eq:S-T-decomp")


# ---------------------------------------------------------------------------
# 2. LDA >= NCM ordering on a separated synthetic task (H03)
# ---------------------------------------------------------------------------


class LdaNcmOrderingTests(unittest.TestCase):
    """LDA (shared-covariance discriminant) must not lose to NCM (Sigma=I)."""

    def test_lda_geq_ncm_separated(self):
        """UNIT: on the conftest separated Gaussian task (anisotropic shared
        covariance -- precondition asserted, else the ordering is vacuous),
        the LDA head's test accuracy is >= the NCM head's. Deterministic
        (pinned purpose seeds); observed gap +0.057 at this config.
        """
        task = ct.make_gaussian_task(d=8, n_classes=3, separation=3.0,
                                     n_per_class=600,
                                     purpose="h10-lda-ncm")
        # precondition: covariance is genuinely anisotropic (else LDA==NCM).
        self.assertFalse(np.allclose(task.cov, np.eye(task.d), atol=1e-3),
                         "test precondition: cov must be anisotropic")
        n_tr = 300
        Xtr = np.vstack([task.X[task.y == k][:n_tr] for k in range(3)])
        ytr = np.concatenate([np.full(n_tr, k) for k in range(3)])
        Xte = np.vstack([task.X[task.y == k][n_tr:] for k in range(3)])
        yte = np.concatenate([np.full(600 - n_tr, k) for k in range(3)])
        lda = ln.fit_lda_head(Xtr, ytr)
        ncm = ln.fit_ncm_head(Xtr, ytr)
        acc_lda = ln.evaluate_head(lda, Xte, yte)
        acc_ncm = ln.evaluate_head(ncm, Xte, yte)
        self.assertGreaterEqual(acc_lda, acc_ncm,
                                f"LDA {acc_lda:.4f} < NCM {acc_ncm:.4f}")
        # NCM is LDA with the true cov replaced by I: on this construction
        # NCM must actually err somewhere (else the test is toothless).
        self.assertLess(acc_ncm, 1.0, "NCM should not be perfect here")


# ---------------------------------------------------------------------------
# 3. Diagnostic vs known-offset truth (H04)
# ---------------------------------------------------------------------------


class DiagnosticKnownTruthTests(unittest.TestCase):
    """core.diagnostic against the synthetic checkpoint's exact truth."""

    def test_diagnostic_known_offset_cosine(self):
        """VERIFICATION (synthetic-proxy): population-arm synthetic
        checkpoint (exact CF head, true Sigma): cos(DeltaW_hat, DeltaW_pred)
        > 0.999 for every task pair (task-order threshold; observed 1 - O(
        1e-15) since the identity is exact algebra), relative magnitude 1
        and the expected-offset direction match at EPS_LINALG, and the
        construction's min kappa == kappa_target exactly.
        Epsilons: EPS_LINALG (eigh/solve round-off in the construction);
        the > 0.999 cosine threshold is the task-order bound, not an
        epsilon.
        """
        seed = ct.derive_seed("h10-diag-known-truth")
        ckpt = dg.make_synthetic_checkpoint(
            n_tasks=3, n_classes_per_task=2, d=8, kappa_target=2.0, seed=seed
        )
        self.assertEqual(len(ckpt.expected_offset), 6)  # T(T-1) ordered pairs
        all_pair_minima = []
        for (i, j), offset in ckpt.expected_offset.items():
            d_hat = dg.delta_w_hat(ckpt, i, j, ckpt.Sigma_inv)
            d_pred = dg.delta_w_pred(ckpt, i, j, ckpt.Sigma_inv)
            cos = dg.sigma_inner_cosine(np.stack([d_hat, d_pred]),
                                        ckpt.Sigma_inv)
            self.assertGreater(cos, 0.999,
                               f"cos(DeltaW_hat, DeltaW_pred) pair ({i},{j})")
            rel = dg.relative_magnitude(d_hat, d_pred, ckpt.Sigma_inv)
            self.assertLess(abs(rel - 1.0), ct.EPS_LINALG,
                            f"relative magnitude pair ({i},{j})")
            # against the construction's own expected offset direction
            cos_true = float(
                (d_hat @ ckpt.Sigma_inv @ offset)
                / np.sqrt((d_hat @ ckpt.Sigma_inv @ d_hat)
                          * (offset @ ckpt.Sigma_inv @ offset))
            )
            self.assertGreater(cos_true, 0.999,
                               f"cos(DeltaW_hat, expected_offset) ({i},{j})")
            kap = dg.kappa_block(ckpt.mu_hat, ckpt.task_of_class,
                                 ckpt.Sigma_inv, i, j)
            all_pair_minima.append(float(kap.min()))
            # per pair: the construction only guarantees kappa >= target
            # (equality at the WIDEST task pair); round-off slack EPS_LINALG.
            self.assertGreaterEqual(
                float(kap.min()), ckpt.kappa_target - ct.EPS_LINALG,
                f"pair ({i},{j}) kappa below construction target")
        # the GLOBAL min equals kappa_target exactly (widest pair, eigh
        # round-off only) -- Assumption 6.1 construction guarantee.
        self.assertLess(abs(min(all_pair_minima) - ckpt.kappa_target),
                        ct.EPS_LINALG, "global min kappa != kappa_target")


# ---------------------------------------------------------------------------
# 4. Fix round-trip (H02): de-centered == Bayes population head
# ---------------------------------------------------------------------------


class FixRoundTripTests(unittest.TestCase):
    """Def 7.1 / Theorem 7.2 on the population arm."""

    def test_decenter_equals_bayes_population(self):
        """UNIT: de_center(cf_optimal_head) == Bayes head to 1e-9 (task-order
        bound; observed ~2e-15). Rationale: the match is exact algebra, so
        only cholesky/solve round-off remains; EPS_LINALG=1e-9 already
        covers it and 1e-9 is the asserted bound. Cross-checked against
        BOTH H02's bayes_head and H01's independent route, and the drift
        guard cf_optimal_head == H01 closed_form_ce_head holds first.
        """
        task = _cil_task("h10-core-fixroundtrip")
        stats = fx.TaskStats.from_task(task, task.task_of_class)
        cf_h02 = fx.cf_optimal_head(stats)
        W_h01, b_h01 = gm.closed_form_ce_head(task)
        ct.allclose(cf_h02.W, W_h01, rtol=ct.EPS_LINALG, atol=ct.EPS_LINALG,
                    label="drift-cf-head-H02-vs-H01")
        ct.allclose(cf_h02.b, b_h01, rtol=ct.EPS_LINALG, atol=ct.EPS_LINALG,
                    label="drift-cf-bias-H02-vs-H01")
        fixed = fx.de_center(cf_h02, stats)
        bayes_h02 = fx.bayes_head(stats)
        d_w = float(np.abs(fixed.W - bayes_h02.W).max())
        d_b = float(np.abs(fixed.b - bayes_h02.b).max())
        self.assertLess(d_w, 1e-9, f"max|W_fix - W_bayes| = {d_w:.3e}")
        self.assertLess(d_b, 1e-9, f"max|b_fix - b_bayes| = {d_b:.3e}")
        # independent route: H01's Bayes head must agree too.
        W_by, b_by = gm.bayes_head(task)
        ct.allclose(fixed.W, W_by, rtol=0.0, atol=1e-9,
                    label="fix-vs-H01-bayes-W")
        ct.allclose(fixed.b, b_by, rtol=0.0, atol=1e-9,
                    label="fix-vs-H01-bayes-b")

    def test_fix_within_task_discriminators_unchanged(self):
        """UNIT: Theorem 7.2 second half -- within-task weight and bias
        DIFFERENCES are unchanged by the fix (the per-task additions cancel
        within a task). Floating-point subtraction reordering => EPS_LINALG.
        """
        task = _cil_task("h10-core-fix-within")
        stats = fx.TaskStats.from_task(task, task.task_of_class)
        cf = fx.cf_optimal_head(stats)
        fixed = fx.de_center(cf, stats)
        for t in range(task.T):
            idx = task.classes_in_task(t)
            dW_before = cf.W[idx][:, None, :] - cf.W[idx][None, :, :]
            dW_after = fixed.W[idx][:, None, :] - fixed.W[idx][None, :, :]
            ct.allclose(dW_after, dW_before, rtol=ct.EPS_LINALG,
                        atol=ct.EPS_LINALG,
                        label=f"within-task-dW-unchanged-task{t}")
            db_before = cf.b[idx][:, None] - cf.b[idx][None, :]
            db_after = fixed.b[idx][:, None] - fixed.b[idx][None, :]
            ct.allclose(db_after, db_before, rtol=ct.EPS_LINALG,
                        atol=ct.EPS_LINALG,
                        label=f"within-task-db-unchanged-task{t}")

    def test_taskstats_drift_vs_population_statistics(self):
        """UNIT: H02 TaskStats (mu_bar, S_bar) must equal H01's
        population_statistics on the same task (cholesky vs einsum routes,
        EPS_LINALG), and from_class_stats with equal blocks must reproduce
        the default contiguous task partition.
        """
        task = _cil_task("h10-core-taskstats")
        stats = fx.TaskStats.from_task(task, task.task_of_class)
        pop = gm.population_statistics(task)
        ct.allclose(stats.mu_bar, pop.mu_bar_tasks, rtol=ct.EPS_LINALG,
                    atol=ct.EPS_LINALG, label="TaskStats-mu_bar-vs-H01")
        ct.allclose(stats.S_bar, pop.S_bar, rtol=ct.EPS_LINALG,
                    atol=ct.EPS_LINALG, label="TaskStats-S_bar-vs-H01")
        default = fx.TaskStats.from_class_stats(task.means, task.cov)
        np.testing.assert_array_equal(default.task_of_class,
                                      task.task_of_class)


# ---------------------------------------------------------------------------
# 5. Determinism: bit-identity across reruns (EPS_EXACT)
# ---------------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    """Same purpose seeds => bit-identical artifacts across full reruns."""

    def test_bit_identity_across_reruns(self):
        """UNIT: two independent rebuilds of the full local pipeline
        (conftest fixture, H01 task + sampling, H03 LDA fit, H02 fix,
        H04 synthetic checkpoint) produce bit-identical arrays
        (ct.arrays_identical, EPS_EXACT tier).
        """
        def pipeline(run: int):
            out = {}
            gt = ct.make_gaussian_task(d=4, n_classes=3, separation=2.0,
                                       n_per_class=50,
                                       purpose="h10-determinism-ct")
            out["ct-X"], out["ct-y"] = gt.X, gt.y
            task = _cil_task("h10-determinism")
            out["means"], out["cov"] = task.means, task.cov
            X, y = task.sample(64, purpose="h10-determinism:sample")
            out["X"], out["y"] = X, y
            head = ln.fit_lda_head(X, y)
            out["lda-W"], out["lda-b"] = head.weights, head.bias
            stats = fx.TaskStats.from_task(task, task.task_of_class)
            cf = fx.cf_optimal_head(stats)
            fixed = fx.de_center(cf, stats)
            out["fix-W"], out["fix-b"] = fixed.W, fixed.b
            ckpt = dg.make_synthetic_checkpoint(
                n_tasks=2, n_classes_per_task=2, d=6, kappa_target=2.0,
                seed=ct.derive_seed("h10-determinism-ckpt"), noise_scale=0.02,
            )
            out["ckpt-W"], out["ckpt-mu"] = ckpt.W, ckpt.mu_hat
            out["run"] = run  # run tag must NOT leak into the arrays
            return out

        a = pipeline(1)
        b = pipeline(2)
        for key in ("ct-X", "ct-y", "means", "cov", "X", "y", "lda-W",
                    "lda-b", "fix-W", "fix-b", "ckpt-W", "ckpt-mu"):
            ct.arrays_identical(a[key], b[key], label=f"rerun-bit-identity-{key}")


if __name__ == "__main__":
    unittest.main(verbosity=2)


def test_claim_verdict_vocabulary():
    from core import gates
    spec = dict(id="E9.1", arm="spec", status="FAIL", diagnosis="radially unidentified at feasible n", value="min 0.88", threshold="> 0.999")
    var = dict(id="E9.1", arm="variant_wellcond", status="PASS")
    assert gates.verdict(spec) == "MECHANISM-FAIL [design]"
    assert gates.claim_verdict([spec, var]).startswith("MECHANISM-FAIL [design; variant: E9.1: PASS")
    exact_ci = dict(id="E9.2[variant|exact]", arm="exact", status="FAIL", value="slope -0.3873 (95% CI [-0.4198, -0.3510])",
                    threshold="slope <= -0.40", desc="FIX arm (de-centered, lam(n)=lam0 (n/1000)^-1/2)")
    assert gates.tier_of(exact_ci) == "CALIBRATION" and gates.verdict(exact_ci) == "CALIBRATION-EXCURSION"
    big_z = dict(id="E9.3", arm="empirical MC", status="FAIL", value="gap -0.71, z = 9.96", desc="x")
    assert gates.verdict(big_z) == "MECHANISM-FAIL"
    pq = dict(id="A26", arm="", status="FAIL", desc="Pearson r", value=None)
    assert gates.verdict(pq) == "PAPER-QUESTION" and gates.claim_verdict([pq]) == "PAPER-QUESTION"
    exact = dict(id="EX9.9", arm="EXACT", status="FAIL", threshold="1e-9", value="3e-3")
    assert gates.verdict(exact) == "MECHANISM-FAIL"
    assert gates.claim_verdict([dict(id="E9.4", arm="MC", status="PASS")]) == "PASS"
