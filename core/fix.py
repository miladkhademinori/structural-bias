"""core.fix -- Paper B (12500): post-hoc de-centering of within-task heads.

Status: IMPLEMENTED by H02, 2026-10-21. Replaces the G09 skeleton placeholder
(whose docstring mislabeled this module "Paper A"; the binding architecture is
G01 log section 3: fix.py is B-ONLY, Def 7.1 de-centering + gap recovery).

Spec anchors (all quoted, none re-derived here):
- G03 SPEC 3 (N1000_INIT_2026-10-21/exec/G03_log.md):
  * Definition 7.1 `def:fix` / `eq:decentre` (TeX L670-678):
        w_r^fix := w_r^dagger + Sigma^{-1} mu_bar^{(t(r))}
        b_r^fix := b_r^dagger - (1/2) S_bar^{(t(r))}
    with S_bar^{(t)} = (1/C) sum_{r in Y^(t)} mu_r^T Sigma^{-1} mu_r;
    mu_bar^{(t)}, S_bar^{(t)} estimated at training time from training data
    only; evaluation-time only; two additions per logit.
  * Theorem 7.2 `thm:fix` (L680-688): cross-task discriminators of the fixed
    head equal the Bayes-optimal (uniform-prior LDA) discriminators exactly;
    within-task differences unchanged.
  * Definition 7.4 `def:fix-weights-only` (L716-724): ABLATION ONLY variant
        w_r^wfix := w_r + Sigma^{-1} mu_bar^{(t)}
        b_r^wfix := b_r - (1/2) S_bar^{(t)} - c_hat^{(t)}
    Paper context (quoted, NEVER asserted here): pre-registered control
    `app:weights-only` (L2568-2687) reports residual c_hat^(i) - c_hat^(j) =
    0.014 +/- 0.005, scalar gain +0.09-0.13, factor 38 below the main effect;
    default ruling L706-712 keeps Def 7.1 unmodified.
  * Cost line `rem:fix-cost` (L2339-2342): storage T(d+1) scalars, two
    additions per logit; Sigma^{-1} mu_bar^{(t)} precomputable once per task.
    The oft-quoted "1,925 scalars" is this line instantiated at the 20NG
    config (d=384 at TeX L2719, T=5 per G03 SPEC 9): 5*(384+1) = 1925.
- G06 E3 (working/paper_B/exec/G06_log.md): exact Bayes-match verification
  design; recovery ratio R = (err_WT - err_FIX)/(err_WT - err_Bayes); paper
  context (quoted only): real-data gap recovery 82-94%, mean 90.5% (G03
  SPEC 8, TeX L750-781). Synthetic ratios produced by this module are their
  own numbers and never claims about paper measurements.

Dependencies: numpy/scipy only. core.gaussian_model (owner H01, built
concurrently) is imported GUARDED: nothing in the functional API below
requires it (all inputs are arrays or duck-typed containers); the self-check
uses it when present and falls back to a local builder when not. Drift is
recorded in exec/H02_log.md. core.heads is also still a skeleton, so a
minimal local LinearHead is defined here; _as_head() accepts any object
with .W/.b, so a future heads.py type passes through unchanged.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Protocol

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm as _norm

# --- H01 dependency, guarded (see module docstring + exec/H02_log.md) -------
try:  # package-style import (repo root on sys.path)
    from core import gaussian_model as _gaussian_model
except ImportError:  # pragma: no cover - direct-script execution
    try:  # module-style import (repo/core on sys.path)
        import gaussian_model as _gaussian_model
    except ImportError:
        _gaussian_model = None

GAUSSIAN_MODEL_AVAILABLE = _gaussian_model is not None


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LinearHead:
    """Minimal linear head: logits ell_r(x) = W[r] @ x + b[r].

    Local stand-in until core.heads (SHARED, still skeleton) lands its own
    head type; _as_head() coerces any .W/.b object to this shape.
    """

    W: np.ndarray  # (n_classes, d)
    b: np.ndarray  # (n_classes,)


class _HasWB(Protocol):  # pragma: no cover - structural type
    W: np.ndarray
    b: np.ndarray


def _as_head(head: LinearHead | _HasWB | tuple) -> tuple[np.ndarray, np.ndarray]:
    """Coerce head-like input to (W, b) float64 arrays, copied."""
    if isinstance(head, LinearHead):
        W, b = head.W, head.b
    elif hasattr(head, "W") and hasattr(head, "b"):
        W, b = head.W, head.b
    else:
        try:
            W, b = head  # type: ignore[misc]
        except (TypeError, ValueError):
            raise TypeError(
                f"head must be LinearHead, an object with .W/.b, or (W, b); "
                f"got {type(head)}"
            )
    W = np.array(W, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D (n_classes, d); got shape {W.shape}")
    if b.shape != (W.shape[0],):
        raise ValueError(
            f"b must have shape (n_classes,)=({W.shape[0]},); got {b.shape}"
        )
    return W, b


@dataclasses.dataclass(frozen=True)
class TaskStats:
    """Per-task statistics the fix needs (all training-side quantities).

    means        : (K, d) class means mu_r (estimates in real use).
    Sigma        : (d, d) shared class covariance estimate.
    task_of_class: (K,) int, task index t(r) in [0, T).
    mu_bar       : (T, d) per-task centroids mu_bar^{(t)}.
    S_bar        : (T,) per-task average squared Mahalanobis class norm
                   S_bar^{(t)} = (1/C_t) sum_{r in Y^(t)} mu_r^T Sigma^{-1} mu_r.

    Built via from_class_stats / from_task / from_data. Sigma enters only
    through Cholesky solves (no explicit inverse, per G01 pipeline rule).
    """

    means: np.ndarray
    Sigma: np.ndarray
    task_of_class: np.ndarray
    mu_bar: np.ndarray
    S_bar: np.ndarray

    @property
    def n_classes(self) -> int:
        return self.means.shape[0]

    @property
    def d(self) -> int:
        return self.means.shape[1]

    @property
    def n_tasks(self) -> int:
        return self.mu_bar.shape[0]

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_class_stats(
        cls,
        means: np.ndarray,
        Sigma: np.ndarray,
        task_of_class: np.ndarray | None = None,
    ) -> TaskStats:
        """From class means + shared covariance (+ task assignment).

        task_of_class=None -> contiguous equal-size blocks of K/T classes.
        """
        means = np.array(means, dtype=float, copy=True)
        if means.ndim != 2:
            raise ValueError(f"means must be 2-D; got {means.shape}")
        K, d = means.shape
        Sigma = np.array(Sigma, dtype=float, copy=True)
        if Sigma.shape != (d, d):
            raise ValueError(
                f"Sigma shape {Sigma.shape} does not match means ({K}, {d})"
            )
        Sigma = 0.5 * (Sigma + Sigma.T)  # symmetrize defensively
        if task_of_class is None:
            T = 2 if K >= 2 else 1
            task_of_class = np.arange(K) * T // K
        else:
            task_of_class = np.array(task_of_class, dtype=np.int64, copy=True)
            if task_of_class.shape != (K,):
                raise ValueError(
                    f"task_of_class must have shape ({K},); got {task_of_class.shape}"
                )
            T = int(task_of_class.max()) + 1
            if T < 1 or task_of_class.min() < 0:
                raise ValueError("task_of_class must be non-negative")
        if not np.all(np.bincount(task_of_class, minlength=T) > 0):
            raise ValueError("every task index in [0, T) must own >= 1 class")

        cf = _chol(Sigma)
        # Mahalanobis quadratic forms mu_r^T Sigma^{-1} mu_r, all classes at once.
        sig_inv_means = cho_solve(cf, means.T)  # (d, K)
        quad = np.sum(means * sig_inv_means.T, axis=1)  # (K,)
        mu_bar = np.stack([means[task_of_class == t].mean(axis=0) for t in range(T)])
        S_bar = np.array(
            [quad[task_of_class == t].mean() for t in range(T)]
        )
        return cls(
            means=means,
            Sigma=Sigma,
            task_of_class=task_of_class,
            mu_bar=mu_bar,
            S_bar=S_bar,
        )

    @classmethod
    def from_task(
        cls,
        task: object,
        task_of_class: np.ndarray | None = None,
    ) -> TaskStats:
        """From a GaussianTask-like object exposing .means and .cov.

        Duck-typed against the spec'd core.gaussian_model.GaussianTask
        (H01, concurrent); tests/conftest.GaussianTask (G10) satisfies the
        same attribute contract. Uses the parameters carried by the task:
        for the population arm these are the true generative parameters; an
        estimated-stats arm builds a task from training estimates first.
        """
        means = getattr(task, "means", None)
        cov = getattr(task, "cov", None)
        if means is None or cov is None:
            raise TypeError(
                f"task of type {type(task)} lacks .means/.cov attributes; "
                "the spec'd GaussianTask contract (core.gaussian_model, H01) "
                "requires both"
            )
        if task_of_class is None:
            toc = getattr(task, "task_of_class", None)
            task_of_class = None if toc is None else np.asarray(toc)
        return cls.from_class_stats(means, cov, task_of_class)

    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        task_of_class: np.ndarray | None = None,
        n_classes: int | None = None,
    ) -> TaskStats:
        """Training-side estimation path (paper: training data only, never test).

        Empirical class means + pooled within-class covariance (unbiased,
        per-class dof), the same statistics a trained head's task would
        maintain. Ledoit-Wolf shrinkage lives in the diagnostic/LDA modules
        (G03 SPEC 5); plug a shrunk Sigma through from_class_stats when needed.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=np.int64)
        K = int(y.max()) + 1 if n_classes is None else int(n_classes)
        d = X.shape[1]
        means = np.stack([X[y == k].mean(axis=0) for k in range(K)])
        centered = X - means[y]
        Sigma = centered.T @ centered / max(len(y) - K, 1)
        return cls.from_class_stats(means, Sigma, task_of_class)


def _chol(Sigma: np.ndarray):
    try:
        return cho_factor(Sigma, lower=True)
    except np.linalg.LinAlgError as exc:  # pragma: no cover
        raise ValueError(f"Sigma is not positive definite: {exc}") from exc


# ---------------------------------------------------------------------------
# Definition 7.1 (eq:decentre) and Theorem 7.2
# ---------------------------------------------------------------------------


def de_center(head: LinearHead | _HasWB | tuple, task_stats: TaskStats) -> LinearHead:
    """Def 7.1 post-hoc de-centering (eq:decentre, TeX L670-678).

        w_r^fix = w_r + Sigma^{-1} mu_bar^{(t(r))}          (vector addition)
        b_r^fix = b_r - 0.5 * S_bar^{(t(r))}                (scalar addition)

    Two additions per logit, evaluation-time only; the input head is never
    mutated. By Theorem 7.2 (eq:fix-weight), cross-task discriminators of the
    returned head equal the Bayes (uniform-prior LDA) discriminators exactly
    when the input is the CF-optimal within-task head; within-task
    differences are unchanged (rem:fix-within L2334-2337).
    """
    W, b = _as_head(head)
    _check_compat(W, task_stats)
    cf = _chol(task_stats.Sigma)
    sig_inv_mu_bar = cho_solve(cf, task_stats.mu_bar.T)  # (d, T)
    W_fix = W + sig_inv_mu_bar.T[task_stats.task_of_class]  # (K, d)
    b_fix = b - 0.5 * task_stats.S_bar[task_stats.task_of_class]
    return LinearHead(W_fix, b_fix)


def cf_optimal_head(task_stats: TaskStats) -> LinearHead:
    """Closed-form CF-optimal within-task head (Theorem 3.3, eq:Wstar L547-556).

    Reference constructor for the verification arm (G06 E1/E3 population
    arm); production trained heads come from core.heads. lambda -> 0+ limit:
        w_r^dagger = Sigma^{-1} (mu_r - mu_bar^{(t)})
        b_r^dagger = -0.5 mu_r^T Sigma^{-1} mu_r + 0.5 S_bar^{(t)}
    """
    cf = _chol(task_stats.Sigma)
    centered = task_stats.means - task_stats.mu_bar[task_stats.task_of_class]
    W = cho_solve(cf, centered.T).T  # (K, d)
    quad = np.sum(task_stats.means * cho_solve(cf, task_stats.means.T).T, axis=1)
    b = -0.5 * quad + 0.5 * task_stats.S_bar[task_stats.task_of_class]
    return LinearHead(W, b)


def bayes_head(task_stats: TaskStats) -> LinearHead:
    """Bayes-optimal head under uniform priors, canonical gauge
    (rem:full-vs-within L560-563; app:notation L1060-1061):
        w_r^* = Sigma^{-1} mu_r,   b_r^* = -0.5 ||mu_r||^2_{Sigma^{-1}}.
    """
    cf = _chol(task_stats.Sigma)
    W = cho_solve(cf, task_stats.means.T).T  # (K, d)
    quad = np.sum(task_stats.means * W, axis=1)
    return LinearHead(W, -0.5 * quad)


def _check_compat(W: np.ndarray, task_stats: TaskStats) -> None:
    if W.shape[1] != task_stats.d:
        raise ValueError(
            f"head dimension d={W.shape[1]} does not match task_stats d={task_stats.d}"
        )
    if W.shape[0] != task_stats.n_classes:
        raise ValueError(
            f"head has {W.shape[0]} classes; task_stats has {task_stats.n_classes}"
        )


# ---------------------------------------------------------------------------
# Definition 7.4 (weights-only variant, ABLATION ONLY) + c_hat estimator
# ---------------------------------------------------------------------------


def weights_only_de_center(
    head: LinearHead | _HasWB | tuple,
    c_hat: np.ndarray | float,
    task_stats: TaskStats,
) -> LinearHead:
    """Def 7.4 weights-only-decay variant (eq:decentre-weights-only, L716-724).

        w_r^wfix = w_r + Sigma^{-1} mu_bar^{(t)}
        b_r^wfix = b_r - 0.5 S_bar^{(t)} - c_hat^{(t)}

    STATUS: ABLATION ONLY. The paper's default ruling (L706-712) keeps
    Definition 7.1 unmodified; this variant exists for the pre-registered
    weights-only control (`app:weights-only` L2568-2687). Paper context,
    quoted and never asserted by this module: residual c_hat^(i) -
    c_hat^(j) = 0.014 +/- 0.005; the scalar buys +0.09-0.13, a factor 38
    below the main fix effect. c_hat may be a scalar (broadcast over tasks)
    or a (T,) array. Only differences c_hat^(i) - c_hat^(j) affect argmax
    decisions; a global constant is gauge.
    """
    W, b = _as_head(head)
    _check_compat(W, task_stats)
    cf = _chol(task_stats.Sigma)
    sig_inv_mu_bar = cho_solve(cf, task_stats.mu_bar.T)  # (d, T)
    W_fix = W + sig_inv_mu_bar.T[task_stats.task_of_class]
    c = np.broadcast_to(np.asarray(c_hat, dtype=float), (task_stats.n_tasks,))
    b_fix = b - 0.5 * task_stats.S_bar[task_stats.task_of_class] \
        - c[task_stats.task_of_class]
    return LinearHead(W_fix, b_fix)


def estimate_c_hat(
    head: LinearHead | _HasWB | tuple,
    task_stats: TaskStats,
    *,
    center: bool = False,
) -> np.ndarray:
    """Closed-form least-squares estimate of the Def 7.4 scalar c_hat^(t).

    FLEET DECISION on G03 OWNER-ASK #1 (source gives no estimator, split
    size m_t unstated, TODO OWNER L2614-2616): we DECLARE the per-task
    least-squares threshold-alignment estimator on task statistics. For each
    task t it minimizes over the scalar c:

        sum_{r in Y^(t)} [ (b_r - 0.5 S_bar^{(t)} - c) - b_r^* ]^2 ,
        b_r^* = -0.5 mu_r^T Sigma^{-1} mu_r   (Bayes canonical threshold),

    whose closed form is the per-task MEAN residual:

        c_hat^(t) = (1/C_t) sum_{r in Y^(t)} [ b_r - 0.5 S_bar^{(t)}
                                               + 0.5 mu_r^T Sigma^{-1} mu_r ].

    Properties: (i) exactly 0 on any canonical-gauge CF-optimal head (the
    residual vanishes by Theorem 3.3), so the ablation degrades gracefully
    to Def 7.1; (ii) recovers a per-task bias-gauge shift g^(t) exactly
    (c_hat^(t) = g^(t)), which is the misalignment the weights-only
    training regime leaves in the bias gauge; (iii) uses training-side
    statistics only (class means, covariance, trained biases) -- no
    held-out split, so the paper's m_t^{-1/2} concentration remark applies
    with m_t replaced by the training sample size driving those statistics.
    center=True subtracts the mean (identifiability: only differences
    c_hat^(i) - c_hat^(j) affect decisions).
    """
    W, b = _as_head(head)
    _check_compat(W, task_stats)
    cf = _chol(task_stats.Sigma)
    quad = np.sum(task_stats.means * cho_solve(cf, task_stats.means.T).T, axis=1)
    residual = b - 0.5 * task_stats.S_bar[task_stats.task_of_class] + 0.5 * quad
    c_hat = np.stack(
        [residual[task_stats.task_of_class == t].mean() for t in range(task_stats.n_tasks)]
    )
    if center:
        c_hat = c_hat - c_hat.mean()
    return c_hat


# ---------------------------------------------------------------------------
# Cost line (rem:fix-cost)
# ---------------------------------------------------------------------------


def additions_per_logit(d: int, T: int) -> dict:
    """Evaluation-time cost of the fix (rem:fix-cost, TeX L2339-2342).

    Returns {"storage_scalars": T*(d+1), "additions_per_logit": 2}:
    one d-vector (mu_bar^{(t)}) + one scalar (S_bar^{(t)}) stored per task;
    Sigma^{-1} mu_bar^{(t)} precomputed once per task, so each logit pays
    one vector addition (weights) and one scalar addition (bias).
    Derived example (arithmetic on G03 SPEC 9's 20NG config, d=384, T=5):
    5*(384+1) = 1925 stored scalars.
    """
    d, T = int(d), int(T)
    if d < 1 or T < 1:
        raise ValueError("d and T must be positive integers")
    return {"storage_scalars": T * (d + 1), "additions_per_logit": 2}


# ---------------------------------------------------------------------------
# Risk evaluators (G06 E3 design)
# ---------------------------------------------------------------------------


def population_error(
    head: LinearHead | _HasWB | tuple,
    task_stats: TaskStats,
    *,
    n_mc: int = 100_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Monte-Carlo population 0-1 error of a head under task_stats' model
    (uniform class priors, shared covariance), with a binomial standard error.

    Uses the parameters carried by task_stats (true generative parameters
    for the population arm; training estimates for the estimated arm). G06
    E3 evaluates confusions at 1e6 samples; default here is 1e5/class
    (SE <= ~6.5e-4), adjustable. Synthetic-model evaluator only.
    """
    W, b = _as_head(head)
    _check_compat(W, task_stats)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(task_stats.Sigma)
    K, d = task_stats.n_classes, task_stats.d
    n_total = K * int(n_mc)
    wrong = 0
    for k in range(K):
        Z = rng.standard_normal((int(n_mc), d))
        Xk = task_stats.means[k] + Z @ L.T
        logits = Xk @ W.T + b
        wrong += int(np.count_nonzero(logits.argmax(axis=1) != k))
    err = wrong / n_total
    se = float(np.sqrt(max(err * (1.0 - err), np.finfo(float).tiny) / n_total))
    return float(err), se


def pairwise_confusion_exact(
    head: LinearHead | _HasWB | tuple,
    task_stats: TaskStats,
) -> np.ndarray:
    """Exact closed-form pairwise confusion P[k, l] = Pr_{x~N(mu_k, Sigma)}
    [ell_l(x) > ell_k(x)] for a linear head (eq:pairwise L216-219 with
    Gaussian score): P[k,l] = Phi(-m_kl / s_kl), m_kl = (w_k-w_l)^T mu_k +
    (b_k-b_l), s_kl^2 = (w_k-w_l)^T Sigma (w_k-w_l). No MC, no sampling.
    """
    W, b = _as_head(head)
    _check_compat(W, task_stats)
    D = W[:, None, :] - W[None, :, :]  # (K, K, d)
    DSig = D @ task_stats.Sigma  # (K, K, d)
    var = np.einsum("ijk,ijk->ij", DSig, D)
    mean = np.einsum("kd,kld->kl", task_stats.means, D) + (b[:, None] - b[None, :])
    sd = np.sqrt(np.maximum(var, np.finfo(float).tiny))
    return _norm.cdf(-mean / sd)


def gap_recovery(before: float, after: float, bayes: float) -> float:
    """Fraction of the WT-to-Bayes error gap closed by the fix:
    R = (before - after) / (before - bayes).  Returns nan when the gap is
    non-positive or inputs are non-finite.

    G06 E3 defines this ratio for pairwise confusion; here it is the
    population-error analog. Paper context, quoted only: on real benchmarks
    the paper reports 82-94% (mean 90.5%) gap recovery (G03 SPEC 8, TeX
    L750-781). This function computes synthetic ratios; it never asserts a
    paper number.
    """
    before, after, bayes = float(before), float(after), float(bayes)
    denom = before - bayes
    if not np.isfinite(denom) or denom <= 0.0 or not np.isfinite(after):
        return float("nan")
    return (before - after) / denom


# ---------------------------------------------------------------------------
# Self-check (G06 E3 style, synthetic population arm)
# ---------------------------------------------------------------------------


def _local_gaussian_task(
    *,
    d: int,
    classes_per_task: tuple,
    within_separation: float,
    task_separation: float,
    seed: int,
) -> object:
    """FALLBACK builder, used ONLY when core.gaussian_model (H01) is absent.

    Thin mirror of H01's make_gaussian_task geometry: shared SPD covariance
    A A^T / d + 0.05 I; task centroids on QR-orthonormal directions scaled by
    task_separation; class means centroid_t + within_separation * u_r with
    orthonormal u_r; carries task_of_class so downstream code is unchanged.
    DRIFT NOTE: H01's core.gaussian_model is the authoritative implementation
    (its self-check carries the conftest seed policy); whenever this FALLBACK
    executes instead, the drift is recorded in exec/H02_log.md.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    cov = A @ A.T / d + 0.05 * np.eye(d)
    T = len(classes_per_task)
    Qc, _ = np.linalg.qr(rng.standard_normal((d, T)))
    centroids = task_separation * Qc.T
    rows = []
    for c_t in classes_per_task:
        Qw, _ = np.linalg.qr(rng.standard_normal((d, c_t)))
        rows.append(centroids[len(rows)] + within_separation * Qw.T)
    means = np.vstack(rows)

    class _Task:
        pass

    task = _Task()
    task.means = np.ascontiguousarray(means)
    task.cov = cov
    task.d = d
    task.n_classes = means.shape[0]
    task.task_of_class = np.repeat(
        np.arange(T, dtype=np.int64), np.asarray(classes_per_task)
    )
    task.seed = seed
    return task


def _main() -> int:  # pragma: no cover - self-check transcript
    # package-style retry for direct-script runs (repo root on sys.path)
    global _gaussian_model, GAUSSIAN_MODEL_AVAILABLE
    if _gaussian_model is None:
        repo_root = __file__.replace("\\", "/").rsplit("/core/", 1)[0]
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from core import gaussian_model as _gm_retry

            _gaussian_model = _gm_retry
            GAUSSIAN_MODEL_AVAILABLE = True
        except ImportError:
            pass

    print("=== core.fix self-check (H02) -- synthetic population arm, G06 E3 style ===")
    use_h01 = GAUSSIAN_MODEL_AVAILABLE
    print(f"[task source] {'core.gaussian_model (H01) make_gaussian_task / '
            'closed_form_ce_head / bayes_head' if use_h01 else
            'LOCAL FALLBACK builder (core.gaussian_model unavailable -- DRIFT NOTED)'}")

    # Small CIL task: T=2 tasks x C=3 classes, d=6 (H01 factory when present).
    if use_h01:
        task = _gaussian_model.make_gaussian_task(
            d=6, classes_per_task=(3, 3),
            within_separation=3.0, task_separation=1.5,
            purpose="h02-fix-selfcheck",
        )
    else:
        task = _local_gaussian_task(
            d=6, classes_per_task=(3, 3),
            within_separation=3.0, task_separation=1.5, seed=0,
        )
    task_of_class = np.asarray(task.task_of_class)
    stats = TaskStats.from_task(task, task_of_class)

    # Drift guard: our TaskStats mu_bar/S_bar vs H01's population_statistics.
    drift_mu = drift_s = float("nan")
    if use_h01:
        pop = _gaussian_model.population_statistics(task)
        drift_mu = float(np.max(np.abs(stats.mu_bar - pop.mu_bar_tasks)))
        drift_s = float(np.max(np.abs(stats.S_bar - pop.S_bar)))
        print(f"[drift guard]   max|mu_bar - H01| = {drift_mu:.3e}   "
              f"max|S_bar - H01| = {drift_s:.3e}   (tol 1e-9)")

    # Biased head = CF-optimal within-task head (Thm 3.3). Prefer H01's
    # closed_form_ce_head (cross-module E3 check); ours is cross-checked too.
    if use_h01:
        W_cf, b_cf = _gaussian_model.closed_form_ce_head(task)
        wt = LinearHead(W_cf, b_cf)
        mine = cf_optimal_head(stats)
        xmod = float(max(np.max(np.abs(mine.W - wt.W)), np.abs(mine.b - wt.b).max()))
        print(f"[drift guard]   max|cf_optimal_head - H01 closed_form_ce_head| = "
              f"{xmod:.3e}   (tol 1e-12)")
        W_b, b_b = _gaussian_model.bayes_head(task)
        bayes = LinearHead(W_b, b_b)
    else:
        wt = cf_optimal_head(stats)
        bayes = bayes_head(stats)
    fixed = de_center(wt, stats)

    # (1) Exact Bayes match of the fix (Thm 7.2 / G06 E3 assertion 1).
    dw = float(np.max(np.abs(fixed.W - bayes.W)))
    db = float(np.max(np.abs(fixed.b - bayes.b)))
    print(f"[exact match]   max|W_fix - W_bayes| = {dw:.3e}   "
          f"max|b_fix - b_bayes| = {db:.3e}   (tol 1e-10 per G06 E3)")

    # (2) Population error before / after / Bayes (independent MC streams).
    n_mc = 100_000
    err_wt, se_wt = population_error(wt, stats, n_mc=n_mc, seed=101)
    err_fix, se_fix = population_error(fixed, stats, n_mc=n_mc, seed=202)
    err_bayes, se_bayes = population_error(bayes, stats, n_mc=n_mc, seed=303)
    print(f"[population error] before(WT)  = {err_wt:.5f} +/- {se_wt:.5f}")
    print(f"[population error] after(FIX) = {err_fix:.5f} +/- {se_fix:.5f}")
    print(f"[population error] Bayes      = {err_bayes:.5f} +/- {se_bayes:.5f}")

    # (3) Recovery fraction (synthetic only; paper's 82-94% is quoted context).
    rec = gap_recovery(err_wt, err_fix, err_bayes)
    print(f"[gap recovery]  R = (WT-FIX)/(WT-Bayes) = {rec:.4f}  [synthetic; "
          "paper quotes 82-94% on real benchmarks -- context, not asserted]")

    # (4) Pairwise confusion (WT head) on the WORST cross-task pair (max
    #     exact P): our closed form vs Monte Carlo vs H01's exact route.
    #     Pair selection matters: most cross pairs sit far below MC
    #     resolution; the worst pair is estimable by construction.
    P = pairwise_confusion_exact(wt, stats)
    cross = stats.task_of_class[:, None] != stats.task_of_class[None, :]
    Pc = np.where(cross, P, -1.0)
    k_star, l_star = (int(v) for v in np.unravel_index(int(np.argmax(Pc)), P.shape))
    rng = np.random.default_rng(404)
    L = np.linalg.cholesky(stats.Sigma)
    Z = rng.standard_normal((n_mc, stats.d))
    Xk = stats.means[k_star] + Z @ L.T
    score = (wt.W[k_star] - wt.W[l_star]) @ Xk.T \
        + (wt.b[k_star] - wt.b[l_star])
    p_mc = float(np.mean(score <= 0.0))
    se_p = float(np.sqrt(p_mc * (1.0 - p_mc) / n_mc))
    p_ex = float(P[k_star, l_star])
    # 6-sigma binomial tolerance (conftest Z_STAT policy) with SE taken at
    # the worse of (exact, MC) and a one-count floor for discreteness.
    tol_pair = 6.0 * float(
        np.sqrt(max(p_ex * (1.0 - p_ex), p_mc * (1.0 - p_mc)) / n_mc)
    ) + 1.0 / n_mc
    xpair = float("nan")
    if use_h01:
        xpair = abs(p_ex - _gaussian_model.pairwise_confusion_exact(
            task, wt.W, wt.b, k_star, l_star))
    print(f"[pairwise P_{k_star}{l_star}] exact = {p_ex:.5f}   "
          f"MC = {p_mc:.5f} +/- {se_p:.5f}   (6-sigma tol {tol_pair:.5f})"
          + (f"   |ours - H01| = {xpair:.3e}" if use_h01 else ""))

    # (4b) G06 E3 assertions 1/3 specialized to this pair: after FIX the
    #      pairwise confusion equals the Bayes pair error (both exact routes;
    #      E3 allows <= 0.003 at MC 1e6), and the pairwise recovery ratio.
    P_fix = pairwise_confusion_exact(fixed, stats)
    P_bs = pairwise_confusion_exact(bayes, stats)
    p_fix, p_bs = float(P_fix[k_star, l_star]), float(P_bs[k_star, l_star])
    pair_gap = abs(p_fix - p_bs)
    rec_pair = gap_recovery(p_ex, p_fix, p_bs)
    xbpair = float("nan")
    if use_h01:
        xbpair = abs(p_bs - _gaussian_model.bayes_pair_error_exact(
            task, k_star, l_star))
    print(f"[pairwise FIX]  |P_fix - P_bayes| = {pair_gap:.3e} "
          f"(<= 0.003 per G06 E3.3)   R_pair = {rec_pair:.6f}"
          + (f"   |P_bayes - H01| = {xbpair:.3e}" if use_h01 else ""))

    # (5) Def 7.4 ablation: gauge-shifted head, c_hat estimator, recovery.
    g = np.array([0.030, -0.010])  # per-task bias-gauge drift, size of the
    # paper's quoted residual scale (0.014 +/- 0.005 -- context, not asserted)
    shifted = LinearHead(wt.W.copy(), wt.b + g[stats.task_of_class])
    c_hat = estimate_c_hat(shifted, stats)
    print(f"[Def 7.4 ablation] c_hat raw           = {np.round(c_hat, 6).tolist()}")
    print(f"[Def 7.4 ablation] c_hat^(0)-c_hat^(1) = {c_hat[0] - c_hat[1]:.6f} "
          f"(true gauge diff {g[0] - g[1]:.6f})")
    wfix = weights_only_de_center(shifted, c_hat, stats)
    # Cross-task threshold differences must match Bayes exactly (c_hat closes
    # the gauge); check every cross-task (k, l) pair.
    k_idx, l_idx = np.where(
        stats.task_of_class[:, None] != stats.task_of_class[None, :]
    )
    td_err = float(np.max(np.abs(
        (wfix.b[k_idx] - wfix.b[l_idx]) - (bayes.b[k_idx] - bayes.b[l_idx])
    )))
    w_err = float(np.max(np.abs(wfix.W - bayes.W)))
    print(f"[Def 7.4 ablation] max cross-task |threshold diff - Bayes| = {td_err:.3e}"
          f"   max|W_wfix - W_bayes| = {w_err:.3e}   (tol 1e-12)")

    # (6) Cost line (rem:fix-cost): 20NG instantiation d=384, T=5.
    cost = additions_per_logit(384, 5)
    print(f"[cost line]     storage = {cost['storage_scalars']} scalars = T(d+1) "
          f"with d=384, T=5; additions per logit = {cost['additions_per_logit']}")

    ok = (dw < 1e-10 and db < 1e-10 and 0.9 <= rec <= 1.1
          and abs(p_mc - p_ex) <= tol_pair
          and pair_gap <= 0.003 and 0.99 <= rec_pair <= 1.01
          and abs((c_hat[0] - c_hat[1]) - (g[0] - g[1])) < 1e-12
          and td_err < 1e-12 and w_err < 1e-12
          and cost == {"storage_scalars": 1925, "additions_per_logit": 2}
          and (not use_h01 or (drift_mu < 1e-9 and drift_s < 1e-9
                               and xmod < 1e-12 and xpair < 1e-12
                               and xbpair < 1e-12)))
    print(f"SELF-CHECK {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
