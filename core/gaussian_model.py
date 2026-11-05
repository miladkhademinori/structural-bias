"""core.gaussian_model -- Gaussian class-conditional CIL model with exact heads.

Paper B synthetic model per SPEC 1 of
``N1000_INIT_2026-10-21/exec/G03_log.md``. All TeX anchors below refer to
``working/paper_B/neurips_2026.tex`` (G03's cited source) as ``anchor@line``.

Model (Sec 2, L182): CIL over T tasks with disjoint label sets Y^(t) of size
C_t (ragged allowed, cf. Theorem 5.2's C_1=2, C_2=3 construction), N = sum_t
C_t classes, frozen feature space R^d, uniform global prior 1/N.

Assumption 2.1 (``ass:gauss``@L191-194): phi(x) | y=r ~ N(mu_r, Sigma) with
one shared, full-rank Sigma for every class.

Exact objects implemented here (population level, no sampling error):
  * CF-optimal within-task CE head  (``thm:closedform`` / ``eq:Wstar``@L547,
    ``eq:wdag``@L1178, ``eq:bdag``@L1194) with the first-order lambda
    shrinkage (``eq:wdag-lambda``@L1265-1270).
  * Bayes-optimal head under uniform priors (``rem:full-vs-within``@L560-563;
    canonical form per ``app:notation``@L1060-1061).
  * Per-task population statistics: mu_bar^(t), S_bar^(t), T_bar^(t),
    tilde_mu_r (``app:notation``@L1028-1056, ``eq:S-T-decomp``).
  * Pairwise confusion P_kl in closed form (``eq:pairwise``@L216-219).
  * Sampling with seeds pinned through tests/conftest.py (G10): derive_seed /
    new_rng are imported and reused -- no private seed logic in this module.

SYNTHETIC-ONLY POLICY (mirrors conftest.py): every number produced by this
module is a fact of a synthetic Gaussian construction, never a paper
measurement. Nothing here asserts published benchmark values.

NOTE ON THE NAME: this ``GaussianTask`` is the *population T-task CIL model*
(means + shared SPD covariance + task partition). It is intentionally
distinct from ``conftest.GaussianTask`` (a single-block sampled fixture);
tests importing both should alias, e.g.
``from core.gaussian_model import GaussianTask as CILGaussianTask``.

Self-check: ``python core/gaussian_model.py`` runs exact identity checks, an
erfc-vs-Monte-Carlo cross-validation and a tiny L-BFGS within-task CE fit;
prints PASS lines. Runs in seconds (numpy/scipy only).

Status: IMPLEMENTED by H01, 2026-10-21 (replaces G09 skeleton).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import scipy.linalg
import scipy.special

# ---------------------------------------------------------------------------
# conftest reuse (G10 test framework): derive_seed / new_rng / tolerances
# ---------------------------------------------------------------------------


def _load_conftest():  # pragma: no cover - trivial import shim
    """Import repo/tests/conftest.py (owner-mandated reuse, no duplication).

    Works (a) when the tests dir is already on sys.path (unittest runners,
    per test_smoke.py), (b) when imported as a bare module from anywhere,
    by locating <repo>/tests relative to this file.
    """
    try:
        import conftest as _ct
        return _ct
    except ImportError:
        pass
    tests_dir = str(Path(__file__).resolve().parents[1] / "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import conftest as _ct  # noqa: E402
    return _ct


_conftest = _load_conftest()
#: SHA-256 purpose-seeded child-seed derivation (owner policy SEED=0).
derive_seed = _conftest.derive_seed
#: Fresh pinned numpy Generator factory (never touches global np.random).
new_rng = _conftest.new_rng
#: Linear-algebra round-off tolerance (cholesky/solve), see conftest.
EPS_LINALG = _conftest.EPS_LINALG
#: Pure float64 round-off tolerance, see conftest.
EPS_TIGHT = _conftest.EPS_TIGHT


# ---------------------------------------------------------------------------
# Linear-algebra helpers (private; margin.py owns the gamma/D/kappa products)
# ---------------------------------------------------------------------------


def _solve_sigma(cov: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return Sigma^{-1} B via Cholesky (cov SPD, symmetric)."""
    fac = scipy.linalg.cho_factor(cov, lower=True, check_finite=True)
    return scipy.linalg.cho_solve(fac, B, check_finite=True)


def _sigma_inv(cov: np.ndarray) -> np.ndarray:
    """Full Sigma^{-1} (d is small in every synthetic use)."""
    return _solve_sigma(cov, np.eye(cov.shape[0]))


# ---------------------------------------------------------------------------
# GaussianTask: the population T-task CIL model (Assumption 2.1)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GaussianTask:
    """Population Gaussian CIL model under ``ass:gauss``@L191-194.

    Fields
    ------
    means : (N, d) float64 -- class means mu_r, global class order (class r
        belongs to task ``task_of_class[r]``; class blocks are contiguous and
        tasks are numbered 0..T-1 in block order).
    cov : (d, d) float64 -- the shared SPD covariance Sigma (identical for
        every class; full rank by construction/validation).
    task_of_class : (N,) int64 -- task index of each class (non-decreasing,
        forming a contiguous partition of the N classes).
    seed, purpose -- provenance of the construction (conftest seed policy).

    Properties: ``d``, ``n_classes`` (=N), ``T``, ``task_sizes`` (tuple,
    ragged allowed). Use :meth:`classes_in_task` / :func:`sample`.
    """

    means: np.ndarray
    cov: np.ndarray
    task_of_class: np.ndarray
    seed: int
    purpose: str

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=float)
        cov = np.asarray(self.cov, dtype=float)
        toc = np.asarray(self.task_of_class, dtype=np.int64)
        if means.ndim != 2 or means.shape[0] < 1:
            raise ValueError(f"means must be (N>=1, d), got {means.shape}")
        N, d = means.shape
        if cov.shape != (d, d):
            raise ValueError(f"cov must be ({d},{d}), got {cov.shape}")
        if toc.shape != (N,):
            raise ValueError(f"task_of_class must be ({N},), got {toc.shape}")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(cov)):
            raise ValueError("means/cov must be finite")
        if not np.allclose(cov, cov.T, rtol=0.0, atol=EPS_LINALG * np.abs(cov).max()):
            raise ValueError("cov must be symmetric")
        try:
            scipy.linalg.cholesky(cov, lower=True)
        except np.linalg.LinAlgError as exc:  # pragma: no cover
            raise ValueError("cov must be SPD (full rank, ass:gauss)") from exc
        if np.any(np.diff(toc) < 0):
            raise ValueError("task_of_class must be non-decreasing "
                             "(contiguous per-task class blocks)")
        if toc[0] != 0 or toc[-1] != toc.max():
            raise ValueError("task ids must start at 0 and be contiguous")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "cov", cov)
        object.__setattr__(self, "task_of_class", toc)

    # -- geometry bookkeeping ------------------------------------------------
    @property
    def d(self) -> int:
        """Feature dimension."""
        return int(self.means.shape[1])

    @property
    def n_classes(self) -> int:
        """Total number of classes N = sum_t C_t."""
        return int(self.means.shape[0])

    @property
    def T(self) -> int:
        """Number of tasks."""
        return int(self.task_of_class[-1]) + 1

    @property
    def task_sizes(self) -> Tuple[int, ...]:
        """C_t per task (ragged allowed)."""
        counts = np.bincount(self.task_of_class, minlength=self.T)
        return tuple(int(c) for c in counts)

    def classes_in_task(self, t: int) -> np.ndarray:
        """Global class indices of task t (contiguous block)."""
        return np.flatnonzero(self.task_of_class == t)

    # -- sampling (pinned seeds via conftest) ---------------------------------
    def sample(
        self,
        n_per_class: int,
        *,
        seed: int | None = None,
        purpose: str | None = None,
        salt: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw class-conditioned samples; see :func:`sample`."""
        return sample(self, n_per_class, seed=seed, purpose=purpose, salt=salt)


def make_gaussian_task(
    *,
    d: int = 6,
    classes_per_task: Sequence[int] = (3, 3),
    within_separation: float = 3.0,
    task_separation: float = 4.0,
    means: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    seed: int | None = None,
    purpose: str = "cil-gaussian-model",
) -> GaussianTask:
    """Build a synthetic population GaussianTask (``ass:gauss``@L191-194).

    Random geometry (when ``means``/``cov`` are not supplied):
      * shared SPD ``cov = A A^T / d + 0.05 I`` (diag ~ O(1)), same style as
        conftest.make_gaussian_task;
      * per-task class means ``mu_r = centroid_t + within_separation * u_r``
        where the u_r are orthonormal (QR) when C_t <= d, so within-task
        pairwise Euclidean mean distance is exactly
        ``within_separation * sqrt(2)`` in that case;
      * task centroids sit on orthonormal directions scaled by
        ``task_separation`` (exact inter-centroid distance
        ``task_separation * sqrt(2)`` when T <= d). NOTE: the *actual*
        mu_bar^(t) also absorbs the O(within_separation/C_t) mean of the u_r;
        all downstream code uses the actual mu_bar^(t) computed from
        ``means`` -- never the intended centroid.
      * pass explicit ``means`` (N, d) for exact constructions (e.g. the
        Theorem 5.1/5.2 §J configurations); ``classes_per_task`` must then
        match ``len(means)``.

    Seeds: ``seed`` or SHA-256 ``derive_seed(purpose, 0)`` (conftest policy,
    SEED=0). Same inputs => bit-identical task.
    """
    sizes = tuple(int(c) for c in classes_per_task)
    if not sizes or any(c < 1 for c in sizes):
        raise ValueError("classes_per_task must be non-empty with C_t >= 1")
    T = len(sizes)
    if d < 1:
        raise ValueError("d must be >= 1")
    if within_separation <= 0 or task_separation <= 0:
        raise ValueError("separations must be > 0")
    if seed is None:
        seed = derive_seed(purpose)
    rng = new_rng(seed)

    if cov is None:
        A = rng.standard_normal((d, d))
        cov_np = A @ A.T / d + 0.05 * np.eye(d)
    else:
        cov_np = np.array(cov, dtype=float)
        if cov_np.shape != (d, d):
            raise ValueError(f"cov must be ({d},{d}), got {cov_np.shape}")

    N = sum(sizes)
    if means is not None:
        means_np = np.array(means, dtype=float)
        if means_np.shape != (N, d):
            raise ValueError(
                f"explicit means must be ({N}, {d}) to match "
                f"classes_per_task={sizes}, got {means_np.shape}"
            )
    else:
        # Task centroids: orthonormal directions when T <= d.
        if T <= d:
            Q, _ = np.linalg.qr(rng.standard_normal((d, T)))
            centroids = task_separation * Q.T
        else:
            raw = rng.standard_normal((T, d))
            centroids = task_separation * raw / np.linalg.norm(
                raw, axis=1, keepdims=True
            )
        rows = []
        for t, c_t in enumerate(sizes):
            if c_t <= d:
                Q, _ = np.linalg.qr(rng.standard_normal((d, c_t)))
                dirs = np.ascontiguousarray(Q.T)  # rows orthonormal
            else:
                raw = rng.standard_normal((c_t, d))
                dirs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
            rows.append(centroids[t] + within_separation * dirs)
        means_np = np.ascontiguousarray(np.vstack(rows))

    toc = np.repeat(np.arange(T, dtype=np.int64), np.asarray(sizes))
    return GaussianTask(
        means=means_np, cov=cov_np, task_of_class=toc,
        seed=int(seed), purpose=purpose,
    )


def sample(
    task: GaussianTask,
    n_per_class: int,
    *,
    seed: int | None = None,
    purpose: str | None = None,
    salt: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample X ~ N(mu_r, Sigma) per class (shared covariance, ass:gauss).

    Returns (X, y): X (N*n_per_class, d) float64 class-block ordered, y
    int64 in [0, N) -- same layout convention as conftest.make_gaussian_task.

    Seeds are pinned through conftest.new_rng: give ``seed`` explicitly, or a
    ``purpose`` label (default ``f"{task.purpose}:sample"``). Same inputs =>
    bit-identical arrays; distinct purposes/salts => independent streams.
    """
    n = int(n_per_class)
    if n < 1:
        raise ValueError("n_per_class must be >= 1")
    if seed is not None:
        rng = new_rng(seed)
    else:
        rng = new_rng(
            purpose=purpose if purpose is not None else f"{task.purpose}:sample",
            salt=salt,
        )
    L = scipy.linalg.cholesky(task.cov, lower=True)
    chunks = [
        task.means[r] + rng.standard_normal((n, task.d)) @ L.T
        for r in range(task.n_classes)
    ]
    X = np.ascontiguousarray(np.vstack(chunks))
    y = np.ascontiguousarray(
        np.repeat(np.arange(task.n_classes, dtype=np.int64), n)
    )
    return X, y


# ---------------------------------------------------------------------------
# Population task-averaged statistics (app:notation@L1028-1056)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PopulationStatistics:
    """Exact population quantities of the model (``app:notation``@L1028-1056).

    mu_bar_tasks   (T, d): mu_bar^(t) = (1/C_t) sum_{r in Y^(t)} mu_r.
    mu_bar_global  (d,):   mu_bar = (1/N) sum_r mu_r (global centroid).
    tilde_means    (N, d): tilde_mu_r = mu_r - mu_bar^(t(r)); satisfies
                    sum_{r in Y^(t)} tilde_mu_r = 0 exactly.
    class_maha_norms (N,): mu_r^T Sigma^{-1} mu_r.
    S_bar          (T,):   S_bar^(t) = (1/C_t) sum_r mu_r^T Sigma^{-1} mu_r.
    T_bar          (T,):   T_bar^(t) = (1/C_t) sum_r tilde_mu_r^T Sigma^{-1}
                    tilde_mu_r. Identity ``eq:S-T-decomp``:
                    S_bar^(t) = T_bar^(t) + ||mu_bar^(t)||^2_{Sigma^{-1}}.
    """

    mu_bar_tasks: np.ndarray
    mu_bar_global: np.ndarray
    tilde_means: np.ndarray
    class_maha_norms: np.ndarray
    S_bar: np.ndarray
    T_bar: np.ndarray


def population_statistics(task: GaussianTask) -> PopulationStatistics:
    """Compute the exact population statistics of ``task`` (no sampling).

    Cites ``app:notation``@L1028-1056 (tilde_mu, S_bar, T_bar,
    ``eq:S-T-decomp``). Pure linear algebra; float64 exact up to EPS_LINALG.
    """
    sizes = np.asarray(task.task_sizes, dtype=float)
    toc = task.task_of_class
    mu_bar_tasks = np.vstack(
        [task.means[task.task_of_class == t].mean(axis=0) for t in range(task.T)]
    )
    mu_bar_global = task.means.mean(axis=0)
    tilde_means = task.means - mu_bar_tasks[toc]
    sig_inv = _sigma_inv(task.cov)
    class_maha_norms = np.einsum("ri,ij,rj->r", task.means, sig_inv, task.means)
    S_bar = np.bincount(toc, weights=class_maha_norms) / sizes
    T_bar = np.bincount(toc, weights=np.einsum(
        "ri,ij,rj->r", tilde_means, sig_inv, tilde_means)) / sizes
    return PopulationStatistics(
        mu_bar_tasks=np.ascontiguousarray(mu_bar_tasks),
        mu_bar_global=np.ascontiguousarray(mu_bar_global),
        tilde_means=np.ascontiguousarray(tilde_means),
        class_maha_norms=np.ascontiguousarray(class_maha_norms),
        S_bar=np.ascontiguousarray(S_bar),
        T_bar=np.ascontiguousarray(T_bar),
    )


def task_centroids(task: GaussianTask) -> np.ndarray:
    """mu_bar^(t) per task, (T, d) (Sec 2, L182)."""
    return population_statistics(task).mu_bar_tasks


def global_centroid(task: GaussianTask) -> np.ndarray:
    """Global centroid mu_bar, (d,) (Sec 2, L182)."""
    return population_statistics(task).mu_bar_global


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


def shrinkage_factor(lam: float, alpha_t: float | np.ndarray) -> np.ndarray:
    """(1 + lam * alpha_t)^{-1} per ``eq:wdag-lambda``@L1265-1270.

    ``alpha_t``: scalar or per-task array (task-specific, r-independent,
    alpha_t > 0 -- the only characterization the source gives). The factor
    resums the paper's first-order path (1 - lam*alpha_t) + O(lam^2); both w
    and b shrink by it.
    """
    lam = float(lam)
    alpha = np.asarray(alpha_t, dtype=float)
    if lam < 0:
        raise ValueError("lam must be >= 0")
    if lam > 0 and np.any(1.0 + lam * alpha <= 0.0):
        raise ValueError("need 1 + lam*alpha_t > 0")
    return 1.0 / (1.0 + lam * alpha)


def closed_form_ce_head(
    task: GaussianTask,
    lam: float = 0.0,
    alpha_t: float | np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """CF-optimal within-task softmax head (Theorem 3.3, ``eq:Wstar``@L547).

    For every class r with t = t(r) (``eq:wdag``@L1178, ``eq:bdag``@L1194)::

        w_r^dagger = Sigma^{-1} (mu_r - mu_bar^(t)) = Sigma^{-1} tilde_mu_r
        b_r^dagger = -(1/2) mu_r^T Sigma^{-1} mu_r
                     + (1/2 C_t) sum_{s in Y^(t)} mu_s^T Sigma^{-1} mu_s
                   = -(1/2)||mu_r||^2_{Sigma^{-1}} + (1/2) S_bar^(t)

    Gauge (Lemma 3.2 ``lem:gauge``@L540): sum_{r in Y^(t)} w_r = 0 and
    sum_{r in Y^(t)} b_r = 0 hold exactly.

    lam > 0: multiplicative shrinkage of BOTH w_r and b_r by
    (1 + lam*alpha_t)^{-1} per ``eq:wdag-lambda``@L1265-1270 (first order in
    lam; higher-order terms uncharacterized in the source). The source does
    NOT give a closed form for alpha_t (only: > 0, task-specific,
    r-independent, "depends only on the within-task geometry"), so it must be
    supplied by the caller -- scalar (same for all tasks) or (T,) array.
    Passing ``alpha_t=None`` with lam > 0 raises instead of inventing a value.

    Returns (W, b): W (N, d) row r = w_r, b (N,) row r = b_r.
    """
    lam = float(lam)
    if lam < 0:
        raise ValueError("lam must be >= 0")
    stats = population_statistics(task)
    sig_inv = _sigma_inv(task.cov)
    W = stats.tilde_means @ sig_inv          # Sigma^{-1} tilde_mu_r (rows)
    b = -0.5 * stats.class_maha_norms + 0.5 * stats.S_bar[task.task_of_class]
    if lam > 0.0:
        if alpha_t is None:
            raise ValueError(
                "lam > 0 requires alpha_t: eq:wdag-lambda characterizes it "
                "only as task-specific, r-independent, > 0; the source gives "
                "no closed form, and this module does not fabricate one."
            )
        factor = np.asarray(shrinkage_factor(lam, alpha_t), dtype=float)
        if factor.ndim == 0:
            factor = np.full(task.T, float(factor))
        if factor.shape != (task.T,):
            raise ValueError(f"alpha_t must be scalar or ({task.T},)")
        W = W * factor[task.task_of_class, None]
        b = b * factor[task.task_of_class]
    return np.ascontiguousarray(W), np.ascontiguousarray(b)


def bayes_head(
    task: GaussianTask, canonical: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Bayes-optimal head under uniform priors (``rem:full-vs-within``@L560).

    canonical=True (default, ``app:notation``@L1060-1061 classical form)::

        w_r^* = Sigma^{-1} mu_r,          b_r^* = -(1/2)||mu_r||^2_{Sigma^{-1}}

    canonical=False (global-centroid gauge of the full-softmax analogue)::

        w_r^* = Sigma^{-1}(mu_r - mu_bar), b_r^* unchanged,

    which differs from the canonical logits by the class-INdependent shift
    (Sigma^{-1} mu_bar)^T x: identical argmax predictions and softmax
    probabilities. Relation to the CF head (``rem:full-vs-within``):
    w_r^* - w_r^dagger = Sigma^{-1} mu_bar^(t) (canonical gauge).

    Returns (W, b): W (N, d), b (N,).
    """
    sig_inv = _sigma_inv(task.cov)
    if canonical:
        W = task.means @ sig_inv
    else:
        W = (task.means - global_centroid(task)) @ sig_inv
    b = -0.5 * np.einsum(
        "ri,ij,rj->r", task.means, sig_inv, task.means
    )
    return np.ascontiguousarray(W), np.ascontiguousarray(b)


# ---------------------------------------------------------------------------
# Discriminant / decision functions
# ---------------------------------------------------------------------------


def logits(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """ell_r(x) = w_r^T phi(x) + b_r (Sec 2, L197). Returns (n, N)."""
    return np.asarray(X, dtype=float) @ np.asarray(W, dtype=float).T \
        + np.asarray(b, dtype=float)


def predict(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """argmax_r ell_r(x) (ties -> lowest r). Returns (n,) int64 labels."""
    return np.argmax(logits(W, b, X), axis=1).astype(np.int64)


def pairwise_discriminant(
    W: np.ndarray, b: np.ndarray, k: int, l: int
) -> Tuple[np.ndarray, float]:
    """Delta_{kl}(x) = (w_k - w_l)^T x + (b_k - b_l); > 0 => class k wins.

    This is the object whose sign defines the pairwise confusion P_kl of
    ``eq:pairwise``@L216-219. Returns (w, c) with w (d,), c scalar.
    """
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    return W[k] - W[l], float(b[k] - b[l])


def pairwise_confusion_exact(
    task: GaussianTask, W: np.ndarray, b: np.ndarray, k: int, l: int
) -> float:
    """Exact P_kl = Pr_{x~p(.|k)}[ell_l(x) > ell_k(x)] (``eq:pairwise``@L216).

    Under ``ass:gauss`` the score difference Delta_{kl}(x) is Gaussian:
    mean m = (w_k-w_l)^T mu_k + (b_k-b_l), variance
    v = (w_k-w_l)^T Sigma (w_k-w_l), so

        P_kl = Phi(-m/sqrt(v)) = 0.5 * erfc(m / sqrt(2 v))     [exact]

    Degenerate v == 0 (identical sub-heads): step function at m. This is the
    exact population quantity -- no Monte Carlo anywhere in this function.
    """
    w, c = pairwise_discriminant(W, b, k, l)
    m = float(w @ task.means[k]) + c
    v = float(w @ task.cov @ w)
    if v <= 0.0:
        return 0.0 if m > 0 else 1.0
    return float(0.5 * scipy.special.erfc(m / np.sqrt(2.0 * v)))


def bayes_pair_error_exact(task: GaussianTask, k: int, l: int) -> float:
    """Exact Bayes 0-1 error between classes k, l (equal priors).

    With D_kl = ||mu_k - mu_l||^2_{Sigma^{-1}} (``eq:gamma``@L183-188) the
    Bayes discriminator has mean D_kl/2 and variance D_kl, so

        err = Phi(-(1/2) sqrt(D_kl)) = 0.5 * erfc(sqrt(D_kl / 8))   [exact]

    Cross-checked in the self-check against pairwise_confusion_exact on the
    Bayes head (the two algebraic routes must agree to EPS_LINALG).
    """
    diff = task.means[k] - task.means[l]
    D_kl = float(diff @ _solve_sigma(task.cov, diff))
    return float(0.5 * scipy.special.erfc(np.sqrt(D_kl / 8.0)))


# ---------------------------------------------------------------------------
# Evaluation harness: population 0-1 error, high-n Monte Carlo, pinned seeds
# ---------------------------------------------------------------------------


def population_error_mc(
    task: GaussianTask,
    W: np.ndarray,
    b: np.ndarray,
    *,
    n_per_class: int = 50_000,
    seed: int | None = None,
    purpose: str | None = None,
    salt: int = 0,
) -> float:
    """Monte-Carlo population 0-1 error of head (W, b) under the true model.

    Multiclass argmax error has no closed form under Gaussian mixtures, so it
    is estimated on N * n_per_class fresh samples from :func:`sample` (high-n,
    seeds pinned through conftest.new_rng; default purpose
    ``f"{task.purpose}:pop-err"``). Exact pairwise quantities should use
    :func:`pairwise_confusion_exact` / :func:`bayes_pair_error_exact` instead.
    """
    if seed is None and purpose is None:
        purpose = f"{task.purpose}:pop-err"
    X, y = sample(task, n_per_class, seed=seed, purpose=purpose, salt=salt)
    return float(np.mean(predict(W, b, X) != y))


# ---------------------------------------------------------------------------
# Self-check (python core/gaussian_model.py) -- seconds, synthetic-only
# ---------------------------------------------------------------------------


def _selfcheck() -> None:  # pragma: no cover - manual/diagnostic entry point
    import scipy
    import scipy.optimize

    Z = _conftest.Z_STAT  # 6-sigma finite-sample multiplier (conftest policy)
    print(f"[versions] numpy {np.__version__} | scipy {scipy.__version__}")

    # Model: ragged tasks (C_1=3, C_2=2) exercise ragged task_sizes; modest
    # inter-task margin so cross-task confusion is estimable at high n.
    task = make_gaussian_task(
        d=5, classes_per_task=(3, 2), within_separation=3.0,
        task_separation=1.2, purpose="h01-selfcheck",
    )
    stats = population_statistics(task)
    W_cf, b_cf = closed_form_ce_head(task)
    W_bs, b_bs = bayes_head(task)
    W_bs_c, _ = bayes_head(task, canonical=False)

    # 1. Determinism of the pinned-seed generator (conftest policy).
    Xa, ya = task.sample(64)
    Xb, yb = task.sample(64)
    assert np.array_equal(Xa, Xb) and np.array_equal(ya, yb)
    Xc, _ = task.sample(64, salt=1)
    assert not np.array_equal(Xa, Xc)
    print("PASS 1  sample() pinned-seed determinism (same purpose bit-identical; salt differs)")

    # 2. Gauge Lemma 3.2 (lem:gauge): per-task sums of w and b vanish.
    for t in range(task.T):
        idx = task.classes_in_task(t)
        assert np.abs(W_cf[idx].sum(axis=0)).max() < EPS_LINALG
        assert abs(b_cf[idx].sum()) < EPS_LINALG
    print("PASS 2  lem:gauge -- sum_r w_r^dag = 0 and sum_r b_r^dag = 0 per task")

    # 3. eq:S-T-decomp: S_bar - T_bar = ||mu_bar^(t)||^2_{Sigma^{-1}}.
    sig_inv = _sigma_inv(task.cov)
    cen_norm = np.einsum("ti,ij,tj->t", stats.mu_bar_tasks, sig_inv, stats.mu_bar_tasks)
    assert np.abs(stats.S_bar - stats.T_bar - cen_norm).max() < EPS_LINALG
    # tilde means sum to zero per task (app:notation).
    for t in range(task.T):
        assert np.abs(stats.tilde_means[task.classes_in_task(t)].sum(axis=0)).max() < EPS_LINALG
    print("PASS 3  eq:S-T-decomp S_bar = T_bar + ||mu_bar^(t)||^2_Sinv; sum tilde_mu = 0")

    # 4. Theorem 4.1 (eq:bias) exact identity, every cross-task pair.
    for k in range(task.n_classes):
        for l in range(task.n_classes):
            i, j = int(task.task_of_class[k]), int(task.task_of_class[l])
            if i == j:
                continue
            lhs = (W_cf[k] - W_cf[l]) - _solve_sigma(
                task.cov, task.means[k] - task.means[l])
            rhs = _solve_sigma(
                task.cov, stats.mu_bar_tasks[j] - stats.mu_bar_tasks[i])
            assert np.abs(lhs - rhs).max() < EPS_LINALG
    print("PASS 4  thm:bias/eq:bias exact for all cross-task pairs (offset = Sinv(mu_bar^j - mu_bar^i))")

    # 5. eq:thresh-gap: b_k^dag - b_l^dag = (b_k^* - b_l^*) + (1/2)(S^i - S^j).
    for k in range(task.n_classes):
        for l in range(task.n_classes):
            i, j = int(task.task_of_class[k]), int(task.task_of_class[l])
            if i == j:
                continue
            lhs = (b_cf[k] - b_cf[l]) - (b_bs[k] - b_bs[l])
            rhs = 0.5 * (stats.S_bar[i] - stats.S_bar[j])
            assert abs(lhs - rhs) < EPS_LINALG
    print("PASS 5  eq:thresh-gap exact for all cross-task pairs")

    # 6. Bayes head canonical vs global-centroid gauge: identical decisions;
    #    logits differ by the class-independent shift (Sinv mu_bar)^T x.
    X, y = task.sample(20_000, purpose="h01-selfcheck:bayes-gauge")
    assert np.array_equal(predict(W_bs, b_bs, X), predict(W_bs_c, b_bs, X))
    shift = (logits(W_bs, b_bs, X) - logits(W_bs_c, b_bs, X))
    assert np.abs(shift - shift[:, :1]).max() < EPS_LINALG
    print("PASS 6  bayes_head canonical == centered gauge (same argmax; per-sample const logit shift)")

    # 7. Shrinkage (eq:wdag-lambda): factor (1 + lam*alpha_t)^{-1} on w AND b.
    alpha = np.array([2.0, 3.0])
    lam = 0.1
    W_sh, b_sh = closed_form_ce_head(task, lam=lam, alpha_t=alpha)
    fac = shrinkage_factor(lam, alpha)[task.task_of_class]
    assert np.abs(W_sh - W_cf * fac[:, None]).max() < EPS_LINALG
    assert np.abs(b_sh - b_cf * fac).max() < EPS_LINALG
    try:
        closed_form_ce_head(task, lam=lam)  # must refuse to invent alpha_t
        raise AssertionError("lam>0 without alpha_t must raise")
    except ValueError:
        pass
    print("PASS 7  eq:wdag-lambda shrinkage (1+lam*alpha_t)^-1 on w and b; lam>0 w/o alpha_t raises")

    # 8. Exact routes to the Bayes pair error agree (erfc(sqrt(D/8)) vs
    #    Gaussian score difference of the Bayes head).
    k, l = 0, 3  # cross-task pair
    e1 = bayes_pair_error_exact(task, k, l)
    e2 = pairwise_confusion_exact(task, W_bs, b_bs, k, l)
    assert abs(e1 - e2) < EPS_LINALG
    print(f"PASS 8  bayes pair error two exact routes agree: {e1:.12f}")

    # 9. Exact vs high-n Monte Carlo for pairwise confusion under the CF head
    #    (6-sigma tolerance, conftest.Z_STAT; n = 200_000 pinned).
    n_mc = 200_000
    p_exact = pairwise_confusion_exact(task, W_cf, b_cf, k, l)
    # Draw exactly n_mc class-k samples with a pinned conftest generator.
    rng = new_rng(purpose="h01-selfcheck:pair", salt=0)
    L = scipy.linalg.cholesky(task.cov, lower=True)
    Xk = task.means[k] + rng.standard_normal((n_mc, task.d)) @ L.T
    w, c = pairwise_discriminant(W_cf, b_cf, k, l)
    p_mc = float(np.mean((Xk @ w + c) <= 0.0))
    tol = Z * np.sqrt(max(p_exact * (1 - p_exact), 1e-12) / n_mc)
    assert abs(p_mc - p_exact) < tol, (p_mc, p_exact, tol)
    print(f"PASS 9  pairwise confusion exact vs MC: {p_exact:.6f} vs {p_mc:.6f} (|d|={abs(p_mc-p_exact):.2e} < 6-sigma {tol:.2e})")

    # 10. Population 0-1 error: Bayes <= CF head on the SAME pinned samples.
    X, y = task.sample(10_000, purpose="h01-selfcheck:poperr")
    err_bayes = float(np.mean(predict(W_bs, b_bs, X) != y))
    err_cf = float(np.mean(predict(W_cf, b_cf, X) != y))
    assert err_bayes <= err_cf + 1e-12
    print(f"PASS 10 population 0-1 error (n={len(y)}): bayes {err_bayes:.4f} <= cf {err_cf:.4f}")

    # 11. Tiny L-BFGS within-task CE fit vs closed form (synthetic protocol
    #     optimizer, lambda = 1e-3): direction cosine per class ~ 1.
    #     REGIME NOTE: within_separation is deliberately moderate (0.5). At
    #     large separation the empirical CE saturates (~0 loss over a wide
    #     plateau), the L2 term dominates, and the REGULARIZED EMPIRICAL
    #     minimizer is genuinely norm-shrunk (||W_fit||/||W_dag|| -> small):
    #     that is finite-sample + lambda physics, NOT a mismatch with
    #     eq:Wstar, which is the lambda -> 0+ population statement. With
    #     Bayes pair error ~0.3 the CE curvature is informative and the fit
    #     recovers the closed form's direction, gauge and scale.
    ft = make_gaussian_task(
        d=4, classes_per_task=(3,), within_separation=0.5,
        task_separation=1.0, purpose="h01-selfcheck:fit",
    )
    Xf, yf = ft.sample(2_000, purpose="h01-selfcheck:fit")
    n, d = Xf.shape
    C = ft.n_classes
    lam_fit = 1e-3

    def obj(theta: np.ndarray):
        Wv = theta[: C * d].reshape(C, d)
        bv = theta[C * d:]
        S = Xf @ Wv.T + bv
        S = S - S.max(axis=1, keepdims=True)
        lse = np.log(np.exp(S).sum(axis=1))
        loss = float((lse - S[np.arange(n), yf]).mean()
                     + 0.5 * lam_fit * (np.sum(Wv * Wv) + np.sum(bv * bv)))
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        G = P
        G[np.arange(n), yf] -= 1.0
        gW = (Xf.T @ G).T / n + lam_fit * Wv
        gb = G.mean(axis=0) + lam_fit * bv
        return loss, np.concatenate([gW.ravel(), gb])

    res = scipy.optimize.minimize(
        obj, np.zeros(C * d + C), jac=True, method="L-BFGS-B",
        options=dict(maxiter=5000, maxfun=10_000, ftol=1e-16, gtol=1e-12),
    )
    W_fit = res.x[: C * d].reshape(C, d)
    b_fit = res.x[C * d:]
    W_dag, b_dag = closed_form_ce_head(ft)
    cosines = []
    for r in range(C):
        a, bb = W_fit[r], W_dag[r]
        cosines.append(float(a @ bb / (np.linalg.norm(a) * np.linalg.norm(bb))))
    min_cos = min(cosines)
    assert min_cos > 0.99, cosines
    assert abs(b_fit.sum()) < 1e-6            # gauge: sum b = 0 at stationarity
    # 0.1 absorbs the n^{-1/2} estimation noise + lambda=1e-3 shrinkage of the
    # empirical minimizer at n = 2000/class (observed ~0.073 on pinned seeds;
    # b scale is 0.24). Pinned seeds make this deterministic per numpy 2.5.2.
    assert np.abs(b_fit - b_dag).max() < 0.1
    ratio = float(np.linalg.norm(W_fit) / np.linalg.norm(W_dag))
    assert 0.75 < ratio <= 1.0 + 1e-6  # shrinkage-side of 1, never inflation
    print(f"PASS 11 L-BFGS within-task CE fit vs eq:Wstar: cos(w_fit, w^dag) per class = "
          f"{['%.4f' % c for c in cosines]}, min {min_cos:.4f} > 0.99; "
          f"sum(b_fit)={b_fit.sum():.2e}; max|b_fit-b^dag|={np.abs(b_fit-b_dag).max():.3f}; "
          f"||W_fit||/||W^dag||={ratio:.4f}")

    print("ALL SELF-CHECKS PASSED (synthetic constructions only; no paper numbers)")


if __name__ == "__main__":  # pragma: no cover
    _selfcheck()
