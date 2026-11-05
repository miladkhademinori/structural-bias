"""tests.test_smoke -- smoke tests for the test harness itself (agent G10).

Tier: UNIT (fast, synthetic) for tests 1-2; VERIFICATION for test 3.
Synthetic-proxy label: none of these tests assert any paper number; all
assertions are against synthetic closed-form ground truth (see the NO
REAL-BACKBONE RULE in conftest.py).

Runs on stdlib unittest (pytest absent, probed 2026-10-21):

    python repo/tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# conftest lives next to this file; make it importable under plain unittest
# and under `python -m unittest discover -s repo/tests`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import conftest as ct  # noqa: E402


class SmokeTests(unittest.TestCase):
    """Three smoke tests: imports, seed determinism, Gaussian fixture sanity."""

    def test_core_imports(self):
        """UNIT: numpy, scipy, sklearn import and perform a trivial op each."""
        import numpy
        import scipy
        import scipy.linalg
        import scipy.stats
        import sklearn.linear_model

        # numpy: basic array math
        self.assertTrue(np.array_equal(np.arange(3) + 1, np.array([1, 2, 3])))
        # scipy: cholesky round-trip at EPS_LINALG
        M = np.array([[4.0, 1.0], [1.0, 3.0]])
        L = scipy.linalg.cholesky(M, lower=True)
        ct.allclose(L @ L.T, M, rtol=ct.EPS_LINALG, label="scipy-cholesky-roundtrip")
        # scipy.stats: a cdf value at a well-known point (exact constant, not a paper number)
        self.assertAlmostEqual(float(scipy.stats.norm.cdf(0.0)), 0.5, places=12)
        # sklearn: fit LogisticRegression on the synthetic Gaussian fixture
        task = ct.make_gaussian_task(d=4, n_classes=3, separation=4.0,
                                     n_per_class=200, purpose="smoke-imports")
        clf = sklearn.linear_model.LogisticRegression(max_iter=500).fit(task.X, task.y)
        self.assertGreaterEqual(clf.score(task.X, task.y), 0.95)
        # versions recorded for the exec log
        print(f"\n[versions] numpy {numpy.__version__} | scipy {scipy.__version__} "
              f"| sklearn {sklearn.__version__}")

    def test_seed_determinism(self):
        """UNIT: SEED=0 policy yields bit-identical replays (EPS_EXACT tier)."""
        t1 = ct.make_gaussian_task(purpose="smoke-determinism")
        t2 = ct.make_gaussian_task(purpose="smoke-determinism")
        self.assertEqual(t1.seed, t2.seed)
        ct.arrays_identical(t1.X, t2.X, label="X-replay")
        ct.arrays_identical(t1.y, t2.y, label="y-replay")
        ct.arrays_identical(t1.means, t2.means, label="means-replay")
        ct.arrays_identical(t1.cov, t2.cov, label="cov-replay")
        # derive_seed: SHA-256 (unsalted), stable across calls, distinct across purposes
        self.assertEqual(ct.derive_seed("alpha"), ct.derive_seed("alpha"))
        self.assertNotEqual(ct.derive_seed("alpha"), ct.derive_seed("beta"))
        # global SEED policy is 0
        self.assertEqual(ct.SEED, 0)
        # raw Generator replay
        a = ct.new_rng(purpose="smoke-rng").standard_normal(64)
        b = ct.new_rng(purpose="smoke-rng").standard_normal(64)
        ct.arrays_identical(a, b, label="rng-stream-replay")

    def test_gaussian_fixture_sanity(self):
        """VERIFICATION: empirical mean/cov converge to the true parameters.

        Synthetic closed-form ground truth only (synthetic-proxy rule: no
        paper numbers asserted). Tolerances: finite-sample bounds from
        conftest.mean_tol / conftest.cov_tol with Z_STAT=6, plus EPS_LINALG
        for the exact orthonormal geometry of the class means.
        """
        d, K, sep, n = 5, 3, 4.0, 40000
        task = ct.make_gaussian_task(d=d, n_classes=K, separation=sep,
                                     n_per_class=n, purpose="smoke-sanity")
        # shapes / labels
        self.assertEqual(task.X.shape, (K * n, d))
        self.assertEqual(task.y.shape, (K * n,))
        self.assertEqual(task.cov.shape, (d, d))

        sigma = float(np.sqrt(np.diag(task.cov)).max())
        tol_mean = ct.mean_tol(sigma, n)  # z*sigma/sqrt(n), worst coordinate
        for k in range(K):
            Xk = task.X[task.y == k]
            ct.allclose(Xk.mean(axis=0), task.means[k],
                        rtol=0.0, atol=tol_mean,
                        label=f"empirical-mean-class{k}(atol={tol_mean:.4g})")

        # pooled within-class covariance vs true shared cov
        centered = np.vstack([task.X[task.y == k] - task.means[k] for k in range(K)])
        emp_cov = np.cov(centered.T, ddof=1)
        sigma2 = float((np.outer(np.diag(task.cov), np.diag(task.cov))).max())
        tol_cov = ct.cov_tol(sigma2, n)  # z*sigma2*sqrt(2/(n-1)), conservative
        ct.allclose(emp_cov, task.cov, rtol=0.0, atol=tol_cov,
                    label=f"empirical-cov(atol={tol_cov:.4g})")

        # exact geometry: orthonormal mean directions => pairwise distances
        # equal separation*sqrt(2) up to EPS_LINALG
        for i in range(K):
            for j in range(i + 1, K):
                dist = float(np.linalg.norm(task.means[i] - task.means[j]))
                ct.allclose(dist, sep * np.sqrt(2.0),
                            rtol=ct.EPS_LINALG, label=f"mean-dist-{i}{j}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
