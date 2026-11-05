"""core.diagnostic -- Paper B bias-direction and kappa diagnostics.

Status: IMPLEMENTED by H04 (repo-implementation wave), 2026-10-21, replacing the
G09 skeleton. Spec sources (read-only):
  - N1000_INIT_2026-10-21/exec/G03_log.md SPEC 4 (diagnostic on released
    checkpoints) and SPEC 6 (kappa, Assumption 6.1), extracted from
    working/paper_B/neurips_2026.tex anchors: DeltaW_ij (L2072, L882), cosine in
    the Sigma_hat^{-1} inner product (L2073-2075), relative magnitude + cross-pair
    CV (tab:bias-direction caption L2529), per-pair kappa_hat = sqrt(D_hat/gamma)
    (L1085, L2077), mean aggregation in tab:diagnostic vs min in the body (L879),
    kappa quantiles (tab:kappa-distribution L2148-2164), five-method protocol
    (L2082-2097).
  - N1000_INIT_2026-10-21/exec/G06_log.md E6 tolerances: exact-population
    constructions are algebraic identities, so the self-check asserts them at the
    linear-algebra epsilon tier (1e-9, cholesky/eigh/solve round-off); estimated
    (finite-sample) arms use the E2 estimated-arm thresholds (cos > 0.99) and the
    rate-validation 5% standard for kappa agreement.

Conventions fixed here (documented once, used everywhere):
  - Inner product for the diagnostic cosine: <u, v> := u^T Sigma_hat^{-1} v
    (the paper's "Sigma_hat^{-1}-Mahalanobis inner product", L2073-2075). Note
    this differs from the weight-space Sigma-norm ||w||_Sigma^2 = w^T Sigma w
    (L576) under which the Theorem 4.1 offset has norm sqrt(gamma_ij); both are
    provided (sigma_inner_norm and sigma_norm).
  - gamma_ij lives HERE (eq:gamma, L183-188, shared-Sigma convention:
    gamma_ij = ||mu_bar^(i) - mu_bar^(j)||^2_{Sigma^{-1}}). H01's
    core.gaussian_model landed WITHOUT a gamma_ij export (its own header
    defers the gamma/D/kappa products to core.margin, which is still
    NOT_IMPLEMENTED_YET), so per the reuse rule this module implements
    eq:gamma locally and is the single source until margin.py lands; margin
    should import or re-export from here to avoid a second copy.
  - CV uses the population standard deviation (ddof=0); ddof=1 is available as
    an argument. The paper never states the choice.

Head protocol (duck-typed; core.heads was NOT_IMPLEMENTED_YET at H04 time):
  an object with
    .W              (n_classes, d)  float array of class weight vectors w_r
    .mu_hat         (n_classes, d)  estimated class means (alias: .means)
    .task_of_class  (n_classes,)    int array, task index of each class
                                    (alias: .tasks)
  SyntheticCheckpoint below satisfies it and is the reference implementation.

Five-method released-checkpoint context (G03 SPEC 4): L2P, DualPrompt,
CODA-Prompt, SLCA, RanPAC under the PILOT/RanPAC unified protocol on
Split-CIFAR-100 with google/vit-base-patch16-224-in21k. Real checkpoint URLs
are a TODO OWNER (G03 OWNER-ASK 8); the abstract CheckpointLoader interface
below is where the benchmarks-wave loaders attach. This module asserts no
paper number: every self-check target is synthetic closed-form ground truth
(NO REAL-BACKBONE RULE, tests/conftest.py).
"""

from __future__ import annotations

import abc
import enum
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "FIVE_METHODS",
    "DEFAULT_AGGREGATION",
    "KAPPA_FLAG_THRESHOLD",
    "AggregationStrategy",
    "symmetric_inverse",
    "mahalanobis_sq",
    "gamma_ij",
    "gamma_matrix",
    "task_centroids",
    "sigma_inner_norm",
    "sigma_norm",
    "sigma_inner_cosine",
    "relative_magnitude",
    "cv_cross_pair",
    "kappa",
    "kappa_block",
    "kappa_regime",
    "classes_of",
    "weight_diff",
    "bias_residuals",
    "delta_w_hat",
    "delta_w_pred",
    "AggregationError",
    "aggregate_values",
    "KappaSummary",
    "summarize_kappa",
    "PairDiagnosticRow",
    "MethodDiagnostic",
    "build_diagnostic_table",
    "render_text_table",
    "LoadedCheckpoint",
    "CheckpointLoader",
    "SyntheticCheckpoint",
    "make_synthetic_checkpoint",
    "synthetic_checkpoint_spec",
    "SyntheticCheckpointLoader",
]

#: The five released-checkpoint CIL methods of the paper's diagnostic protocol
#: (G03 SPEC 4). Names only; real artifacts arrive with the benchmarks wave.
FIVE_METHODS: tuple[str, ...] = (
    "L2P",
    "DualPrompt",
    "CODA-Prompt",
    "SLCA",
    "RanPAC",
)

#: Default kappa aggregation. Resolves G03 OWNER-ASK 5 (tab:diagnostic reports
#: the MEAN over pairs, L2077-2080; the body definition, L879, is a MIN): the
#: default strategy is the per-pair MEAN and every summary ALSO reports the MIN
#: under its own key, so both paper sites are reproduced from one object.
DEFAULT_AGGREGATION = "per_pair_mean"

#: tab:diagnostic reports the fraction of pairs with kappa_hat > 1.5 (L2079).
KAPPA_FLAG_THRESHOLD = 1.5

#: Quantile levels of tab:kappa-distribution (min/5th/25th/median/75th/95th/max
#: plus mean), over N cross-task pairs per method.
KAPPA_QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


# ---------------------------------------------------------------------------
# Small linear-algebra helpers
# ---------------------------------------------------------------------------


def symmetric_inverse(a: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric positive-definite matrix, symmetrized.

    Uses np.linalg.solve against the identity (LU, more accurate than inv for
    well-conditioned SPD inputs at these sizes) and returns A^{-1} with the
    symmetric part kept, killing the antisymmetric round-off component.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {a.shape}")
    inv = np.linalg.solve(a, np.eye(a.shape[0]))
    return 0.5 * (inv + inv.T)


def mahalanobis_sq(u, v, sigma_inv: np.ndarray):
    """||u - v||^2_{Sigma^{-1}} = (u-v)^T Sigma^{-1} (u-v).

    Vector inputs return a float; broadcasting (..., d) inputs return the
    element-wise quadratic form via einsum.
    """
    diff = np.asarray(u, dtype=float) - np.asarray(v, dtype=float)
    quad = np.einsum("...i,ij,...j->...", diff, np.asarray(sigma_inv, float), diff)
    return float(quad) if quad.ndim == 0 else quad


def gamma_ij(mu_bar_i: np.ndarray, mu_bar_j: np.ndarray, sigma_inv: np.ndarray) -> float:
    """gamma_ij = ||mu_bar^(i) - mu_bar^(j)||^2_{Sigma^{-1}} (eq:gamma, L183).

    Shared-Sigma convention of Assumption 2.1: one Sigma^{-1} for every class.
    """
    return mahalanobis_sq(mu_bar_i, mu_bar_j, sigma_inv)


def task_centroids(mu_hat: np.ndarray, task_of_class: np.ndarray) -> np.ndarray:
    """(T, d) array of per-task centroids mu_bar^(t) = mean of the task's mu_hat.

    Rows are indexed by task id; task ids must be contiguous in [0, T).
    """
    mu_hat = np.asarray(mu_hat, dtype=float)
    task_of_class = np.asarray(task_of_class, dtype=int)
    n_tasks = int(task_of_class.max()) + 1
    if task_of_class.min() < 0:
        raise ValueError("task ids must be non-negative")
    return np.vstack([mu_hat[task_of_class == t].mean(axis=0) for t in range(n_tasks)])


def gamma_matrix(
    mu_hat: np.ndarray, task_of_class: np.ndarray, sigma_inv: np.ndarray
) -> np.ndarray:
    """(T, T) matrix of gamma_ij values (diagonal 0)."""
    centroids = task_centroids(mu_hat, task_of_class)
    t = centroids.shape[0]
    out = np.zeros((t, t))
    for i in range(t):
        for j in range(t):
            if i != j:
                out[i, j] = gamma_ij(centroids[i], centroids[j], sigma_inv)
    return out


def sigma_inner_norm(w: np.ndarray, sigma_inv: np.ndarray) -> float:
    """||w|| in the diagnostic inner product: sqrt(w^T Sigma^{-1} w)."""
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(max(w @ np.asarray(sigma_inv, dtype=float) @ w, 0.0)))


def sigma_norm(w: np.ndarray, sigma: np.ndarray) -> float:
    """||w||_Sigma = sqrt(w^T Sigma w), the paper's weight-space norm (L576).

    Under this norm ||Sigma^{-1} v||_Sigma = ||v||_{Sigma^{-1}} = sqrt(gamma)
    when v is a centroid difference -- the "bias Sigma-norm = sqrt(gamma_ij)"
    identity of Theorem 4.1."""
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(max(w @ np.asarray(sigma, dtype=float) @ w, 0.0)))


# ---------------------------------------------------------------------------
# Core diagnostic quantities (G03 SPEC 4)
# ---------------------------------------------------------------------------


def _attr(obj: object, names: Sequence[str]) -> object:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(
        f"head/checkpoint object of type {type(obj).__name__} lacks all of "
        f"{tuple(names)}; see the Head protocol in the module docstring"
    )


def _head_weights(head: object) -> np.ndarray:
    w = np.asarray(_attr(head, ("W", "weights")), dtype=float)
    if w.ndim != 2:
        raise ValueError(f"head weights must be (n_classes, d), got {w.shape}")
    return w


def _head_mu_hat(head: object) -> np.ndarray:
    mu = np.asarray(_attr(head, ("mu_hat", "means")), dtype=float)
    if mu.ndim != 2:
        raise ValueError(f"head class means must be (n_classes, d), got {mu.shape}")
    return mu


def _head_task_of_class(head: object) -> np.ndarray:
    t = np.asarray(_attr(head, ("task_of_class", "tasks")), dtype=int).ravel()
    return t


def classes_of(head: object, i: int) -> np.ndarray:
    """Global class indices belonging to task i, in class order."""
    return np.flatnonzero(_head_task_of_class(head) == i)


def weight_diff(head: object, i: int, j: int) -> np.ndarray:
    """Weight differences of the trained head over block (i, j).

    Returns B = (C_i, C_j, d) with B[a, b] = w_k - w_l for the a-th class k of
    task i and the b-th class l of task j (the paper's discriminators
    w_k - w_l, L2072). This is the raw trained-head quantity; pair it with
    Sigma_hat^{-1}(mu_hat_k - mu_hat_l) via bias_residuals to get the
    Theorem 4.1 residual block.
    """
    w = _head_weights(head)
    ki = classes_of(head, i)
    kj = classes_of(head, j)
    if ki.size == 0 or kj.size == 0:
        raise ValueError(f"empty block ({i}, {j}): {ki.size} x {kj.size} classes")
    return w[ki][:, None, :] - w[kj][None, :, :]


def bias_residuals(
    head: object, i: int, j: int, sigma_inv: np.ndarray
) -> np.ndarray:
    """Residual block B_hat_kl = (w_k - w_l) - Sigma_hat^{-1}(mu_hat_k - mu_hat_l).

    Shape (C_i, C_j, d). Theorem 4.1 (+ Cor 4.2) predicts every one of the C^2
    residuals equals Sigma^{-1}(mu_bar^(j) - mu_bar^(i)).
    """
    mu = _head_mu_hat(head)
    sigma_inv = np.asarray(sigma_inv, dtype=float)
    ki = classes_of(head, i)
    kj = classes_of(head, j)
    pred = (mu[ki][:, None, :] - mu[kj][None, :, :]) @ sigma_inv.T
    return weight_diff(head, i, j) - pred


def delta_w_hat(
    head: object, i: int, j: int, sigma_inv: np.ndarray
) -> np.ndarray:
    """Empirical bias direction DeltaW_ij: mean of the C^2 residuals (L2072)."""
    res = bias_residuals(head, i, j, sigma_inv)
    return res.mean(axis=(0, 1))


def delta_w_pred(
    head: object, i: int, j: int, sigma_inv: np.ndarray
) -> np.ndarray:
    """Predicted bias direction Sigma_hat^{-1}(mu_bar_hat^(j) - mu_bar_hat^(i))."""
    mu = _head_mu_hat(head)
    centroids = task_centroids(mu, _head_task_of_class(head))
    return np.asarray(sigma_inv, dtype=float) @ (centroids[j] - centroids[i])


def sigma_inner_cosine(delta: object, sigma_inv: np.ndarray) -> float:
    """Cosine between empirical and predicted bias in the Sigma^{-1} product.

    delta: array-like of shape (2, d) -- row 0 = DeltaW_hat_ij (empirical),
    row 1 = DeltaW_pred_ij (predicted). Equivalently any sequence of two
    d-vectors. Inner product <u, v> = u^T Sigma_hat^{-1} v (L2073-2075).
    Returns NaN when either vector has zero norm (cosine undefined).
    Theory target: 1 (tab:bias-direction).
    """
    arr = np.asarray(delta, dtype=float)
    if arr.shape != (2, arr.shape[-1]) or arr.ndim != 2:
        raise ValueError(f"delta must be stackable to (2, d), got {arr.shape}")
    sigma_inv = np.asarray(sigma_inv, dtype=float)
    u, v = arr[0], arr[1]
    nu2 = float(u @ sigma_inv @ u)
    nv2 = float(v @ sigma_inv @ v)
    if nu2 <= 0.0 or nv2 <= 0.0:
        return float("nan")
    return float((u @ sigma_inv @ v) / math.sqrt(nu2 * nv2))


def relative_magnitude(
    delta_hat: np.ndarray,
    delta_pred: np.ndarray,
    sigma_inv: np.ndarray | None = None,
) -> float:
    """||DeltaW_hat|| / ||DeltaW_pred|| in the diagnostic norm (L2076).

    Sigma^{-1} inner-product norm when sigma_inv is given (consistent with the
    cosine metric), Euclidean norm otherwise. Theory target: 1.
    """
    if sigma_inv is None:
        n_hat = float(np.linalg.norm(delta_hat))
        n_pred = float(np.linalg.norm(delta_pred))
    else:
        sigma_inv = np.asarray(sigma_inv, dtype=float)
        n_hat = sigma_inner_norm(delta_hat, sigma_inv)
        n_pred = sigma_inner_norm(delta_pred, sigma_inv)
    if n_pred <= 0.0:
        return float("nan")
    return n_hat / n_pred


def cv_cross_pair(
    deltas: np.ndarray, sigma_inv: np.ndarray | None = None, ddof: int = 0
) -> float:
    """Cross-pair coefficient of variation of the bias (G03 SPEC 4).

    deltas: (C_i, C_j, d) residual block (bias_residuals output). The bias
    magnitude of each of the C^2 class pairs is measured in the diagnostic
    (Sigma^{-1}) norm when sigma_inv is given, Euclidean norm otherwise; CV =
    std / mean over the C^2 magnitudes, ddof=0 (population) by documented
    choice. Theory target: 0 (Cor 4.2: the residual is pair-invariant).
    Returns NaN when the mean magnitude is 0.
    """
    arr = np.asarray(deltas, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"deltas must be (C_i, C_j, d), got {arr.shape}")
    if sigma_inv is None:
        mags = np.linalg.norm(arr.reshape(-1, arr.shape[-1]), axis=1)
    else:
        sigma_inv = np.asarray(sigma_inv, dtype=float)
        flat = arr.reshape(-1, arr.shape[-1])
        mags = np.sqrt(np.maximum((flat @ sigma_inv * flat).sum(axis=1), 0.0))
    mean = float(mags.mean())
    if mean <= 0.0:
        return float("nan")
    return float(mags.std(ddof=ddof) / mean)


# ---------------------------------------------------------------------------
# Per-pair diagnostic and controls (Claude, 2026-10-22; Q7 of the replication letter)
# ---------------------------------------------------------------------------
#
# GAUGE IDENTITY. The paper's averaged statistic
#     DeltaW_hat_ij = (1/C^2) sum_{k in i, l in j} [(w_k - w_l) - Sigma^{-1}(mu_k - mu_l)]
#                   = (w_bar_i - w_bar_j) - Sigma^{-1}(mu_bar_i - mu_bar_j)
# reduces, under Lemma 3.2 (gauge fix: sum_{r in task} w_r = 0 at every minimizer
# of the L2-regularized within-task softmax, for ANY data), to
#     DeltaW_hat_ij = Sigma^{-1}(mu_bar_j - mu_bar_i) = DeltaW_pred_ij   exactly.
# So cos(DeltaW_hat, DeltaW_pred) = 1 and rel = 1 are identities of the gauge, not
# tests of Theorem 4.1. (Verified on non-Gaussian, heteroskedastic data: averaged
# cos = 1.000000 while per-pair cosines were 0.58 to 0.88.) The content of
# Theorem 4.1 + Corollary 4.2 is that EVERY one of the C^2 residuals equals the
# predicted vector; that is tested per pair below. Theorem 3.3 is tested per class.


def gauge_residual(head: object, i: int) -> float:
    """||sum_{r in task i} w_r|| / mean_r ||w_r||: 0 at the regularized minimizer.

    Large values mean the head was not trained to the Lemma 3.2 gauge (e.g. a
    short AdamW run or an external checkpoint); only then can the averaged
    statistic differ from 1, and then it measures gauge departure, not the theorem.
    """
    w = _head_weights(head)[classes_of(head, i)]
    denom = float(np.mean(np.linalg.norm(w, axis=1)))
    return float(np.linalg.norm(w.sum(axis=0)) / denom) if denom > 0 else float("nan")


def pair_cosines(head: object, i: int, j: int, sigma_inv: np.ndarray) -> np.ndarray:
    """Cosine, in the Sigma^{-1} product, between EACH pair residual and the
    predicted direction. Shape (C_i, C_j). Theory (Cor 4.2): all equal to 1."""
    sigma_inv = np.asarray(sigma_inv, dtype=float)
    res = bias_residuals(head, i, j, sigma_inv)
    pred = delta_w_pred(head, i, j, sigma_inv)
    flat = res.reshape(-1, res.shape[-1])
    num = flat @ sigma_inv @ pred
    den = np.sqrt(np.maximum((flat @ sigma_inv * flat).sum(axis=1), 0.0)
                  * max(float(pred @ sigma_inv @ pred), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(den > 0, num / den, np.nan)
    return c.reshape(res.shape[0], res.shape[1])


def pair_relative_magnitudes(head: object, i: int, j: int, sigma_inv: np.ndarray) -> np.ndarray:
    """||residual_kl|| / ||pred|| per pair, Sigma^{-1} norm. Theory: all 1."""
    sigma_inv = np.asarray(sigma_inv, dtype=float)
    res = bias_residuals(head, i, j, sigma_inv)
    pred = delta_w_pred(head, i, j, sigma_inv)
    flat = res.reshape(-1, res.shape[-1])
    mags = np.sqrt(np.maximum((flat @ sigma_inv * flat).sum(axis=1), 0.0))
    npred = sigma_inner_norm(pred, sigma_inv)
    return (mags / npred if npred > 0 else np.full(mags.shape, np.nan)).reshape(res.shape[0], res.shape[1])


def closed_form_alignment(head: object, sigma_inv: np.ndarray) -> np.ndarray:
    """Theorem 3.3 per class: cos_Sigma(w_k, Sigma^{-1}(mu_k - mu_bar_{t(k)})). Shape (N,).

    This is the quantity the per-pair residual test is sensitive to: the pair
    residual minus the prediction equals [w_k - Sigma^{-1}(mu_k - mu_bar_i)] -
    [w_l - Sigma^{-1}(mu_l - mu_bar_j)], so per-pair cosines are 1 exactly when
    every trained weight equals its closed form.
    """
    w = _head_weights(head)
    mu = _head_mu_hat(head)
    toc = _head_task_of_class(head)
    cents = task_centroids(mu, toc)
    sigma_inv = np.asarray(sigma_inv, dtype=float)
    out = np.empty(w.shape[0])
    for k in range(w.shape[0]):
        v = sigma_inv @ (mu[k] - cents[toc[k]])
        out[k] = sigma_inner_cosine(np.stack([w[k], v]), sigma_inv)
    return out


def permuted_pairing_control(head: object, i: int, j: int, sigma_inv: np.ndarray,
                             rng: np.random.Generator) -> dict:
    """Negative control: compare the pair residuals against the prediction built
    from a RANDOM pair of task centroids (a, b) != (i, j). If the per-pair cosines
    stay near 1 under the wrong prediction, the per-pair metric is saturated too."""
    toc = _head_task_of_class(head)
    T = int(toc.max()) + 1
    others = [(a, b) for a in range(T) for b in range(T) if a != b and (a, b) != (i, j)]
    a, b = others[int(rng.integers(len(others)))]
    sigma_inv = np.asarray(sigma_inv, dtype=float)
    res = bias_residuals(head, i, j, sigma_inv).reshape(-1, _head_weights(head).shape[1])
    pred = delta_w_pred(head, a, b, sigma_inv)
    num = res @ sigma_inv @ pred
    den = np.sqrt(np.maximum((res @ sigma_inv * res).sum(axis=1), 0.0) * max(float(pred @ sigma_inv @ pred), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(den > 0, num / den, np.nan)
    return dict(control_pair=(int(a), int(b)), pair_cos_mean=float(np.nanmean(c)), pair_cos_min=float(np.nanmin(c)))


def random_head_control(head: object, i: int, j: int, sigma_inv: np.ndarray,
                        rng: np.random.Generator) -> dict:
    """Negative control: replace the trained weights by Gaussian noise of the same
    per-class norm (zero-sum within task, so the gauge identity still holds for the
    AVERAGED statistic) and recompute both statistics. Expected: averaged cos stays
    1 (identity), per-pair cosines collapse."""
    w = _head_weights(head)
    toc = _head_task_of_class(head)
    noise = rng.standard_normal(w.shape)
    for t in np.unique(toc):
        m = toc == t
        noise[m] -= noise[m].mean(axis=0)   # keep the Lemma 3.2 gauge
    noise *= (np.linalg.norm(w, axis=1) / np.maximum(np.linalg.norm(noise, axis=1), 1e-300))[:, None]

    class _H:  # minimal head protocol
        W = noise; mu_hat = _head_mu_hat(head); task_of_class = toc
    sigma_inv = np.asarray(sigma_inv, dtype=float)
    dh = delta_w_hat(_H, i, j, sigma_inv); dp = delta_w_pred(_H, i, j, sigma_inv)
    pc = pair_cosines(_H, i, j, sigma_inv)
    return dict(averaged_cos=sigma_inner_cosine(np.stack([dh, dp]), sigma_inv),
                averaged_rel=relative_magnitude(dh, dp, sigma_inv),
                pair_cos_mean=float(np.nanmean(pc)), pair_cos_min=float(np.nanmin(pc)))


def diagnostic_block(head: object, i: int, j: int, sigma_inv: np.ndarray,
                     rng: np.random.Generator | None = None) -> dict:
    """Both statistics for one ordered task pair, labelled.

    'paper_averaged_*' is the paper's metric (iii) and is a gauge identity when
    gauge_residual ~ 0; 'pair_*' is the per-pair test of Theorem 4.1/Cor 4.2.
    """
    sigma_inv = np.asarray(sigma_inv, dtype=float)
    dh = delta_w_hat(head, i, j, sigma_inv); dp = delta_w_pred(head, i, j, sigma_inv)
    pc = pair_cosines(head, i, j, sigma_inv)
    pr = pair_relative_magnitudes(head, i, j, sigma_inv)
    out = dict(
        paper_averaged_cos=sigma_inner_cosine(np.stack([dh, dp]), sigma_inv),
        paper_averaged_rel=relative_magnitude(dh, dp, sigma_inv),
        gauge_residual_i=gauge_residual(head, i), gauge_residual_j=gauge_residual(head, j),
        pair_cos_mean=float(np.nanmean(pc)), pair_cos_min=float(np.nanmin(pc)),
        pair_rel_mean=float(np.nanmean(pr)), pair_rel_min=float(np.nanmin(pr)), pair_rel_max=float(np.nanmax(pr)),
        cv=cv_cross_pair(bias_residuals(head, i, j, sigma_inv), sigma_inv),
    )
    if rng is not None:
        out["control_permuted"] = permuted_pairing_control(head, i, j, sigma_inv, rng)
        out["control_random_head"] = random_head_control(head, i, j, sigma_inv, rng)
    return out


# ---------------------------------------------------------------------------
# kappa (G03 SPEC 6)
# ---------------------------------------------------------------------------


def kappa(d_hat, gamma_hat):
    """kappa_hat = sqrt(D_hat / gamma_hat) (per-pair form, L1085).

    Broadcasts over arrays. Assumption 6.1 requires D_kl >= kappa^2 * gamma_ij
    for every task pair i != j and cross-task class pair (k, l); kappa > 1 is
    the regime where the Theorem 6.2 exponential bound applies, kappa -> 1 is
    the aliasing regime of Corollary 5.3.
    """
    d = np.asarray(d_hat, dtype=float)
    g = np.asarray(gamma_hat, dtype=float)
    if np.any(g <= 0.0):
        raise ValueError("gamma_hat must be > 0 for kappa to be defined")
    if np.any(d < 0.0):
        raise ValueError("d_hat must be >= 0 (squared Mahalanobis distance)")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(d / g)
    return float(out) if out.ndim == 0 else out


def kappa_block(
    mu_hat: np.ndarray,
    task_of_class: np.ndarray,
    sigma_inv: np.ndarray,
    i: int,
    j: int,
) -> np.ndarray:
    """(C_i, C_j) matrix of kappa_hat_{ij,kl} = sqrt(D_hat_kl / gamma_hat_ij).

    D_hat_kl = ||mu_hat_k - mu_hat_l||^2_{Sigma^{-1}} (cross-class distance),
    gamma_hat_ij = ||mu_bar^(i) - mu_bar^(j)||^2_{Sigma^{-1}} (inter-task
    margin); both from the same Sigma_hat (shared-Sigma convention).
    """
    mu_hat = np.asarray(mu_hat, dtype=float)
    sigma_inv = np.asarray(sigma_inv, dtype=float)
    ki = np.flatnonzero(np.asarray(task_of_class, dtype=int) == i)
    kj = np.flatnonzero(np.asarray(task_of_class, dtype=int) == j)
    d_block = kappa(
        mahalanobis_sq(mu_hat[ki][:, None, :], mu_hat[kj][None, :, :], sigma_inv),
        gamma_ij(
            mu_hat[ki].mean(axis=0), mu_hat[kj].mean(axis=0), sigma_inv
        ),
    )
    return np.asarray(d_block, dtype=float)


def kappa_regime(kappa_values) -> dict:
    """Assumption 6.1 regime report for a set of per-pair kappa_hat values.

    Returns dict with:
      min_kappa            -- the body-definition population aggregate (L879)
      holds (bool)         -- min > 1: Theorem 6.2 regime
      margin               -- min_kappa - 1 (0 margin = aliasing boundary)
      frac_below_1         -- aliasing tail (SPEC 6: pure-CE methods leave
                              3-8% of pairs below 1; SLCA the only min > 1)
      frac_above_threshold -- fraction with kappa > KAPPA_FLAG_THRESHOLD (1.5,
                              the tab:diagnostic column)
      mean_kappa           -- the tab:diagnostic aggregate (L2077-2080)
    """
    arr = np.asarray(kappa_values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("kappa_regime needs at least one value")
    return {
        "min_kappa": float(arr.min()),
        "holds": bool(arr.min() > 1.0),
        "margin": float(arr.min() - 1.0),
        "frac_below_1": float(np.mean(arr < 1.0)),
        "frac_above_threshold": float(np.mean(arr > KAPPA_FLAG_THRESHOLD)),
        "mean_kappa": float(arr.mean()),
    }


# ---------------------------------------------------------------------------
# Aggregation strategies (OWNER-ASK 5 resolved explicitly)
# ---------------------------------------------------------------------------


class AggregationStrategy(enum.Enum):
    """Named kappa aggregation strategies.

    The paper uses two sites without reconciling them (G03 OWNER-ASK 5):
    tab:diagnostic takes the MEAN over pairs; the body definition (L879) is a
    MIN; tab:rate-fits per-backbone kappa uses an unstated aggregation. These
    strategies make the choice explicit at every call site. The module default
    is PER_PAIR_MEAN and summarize_kappa always reports the MIN alongside, so
    both paper sites come out of one summary object.
    """

    PER_PAIR_MEAN = "per_pair_mean"
    MIN = "min"
    QUANTILES = "quantiles"


class AggregationError(ValueError):
    """Raised on an unknown aggregation strategy name."""


def _quantile_dict(values: np.ndarray) -> dict:
    qs = np.quantile(values, KAPPA_QUANTILE_LEVELS)
    return {
        "min": float(values.min()),
        "p05": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "n_pairs": int(values.size),
    }


AGGREGATION_STRATEGIES: dict[str, Callable[[np.ndarray], float | dict]] = {
    "per_pair_mean": lambda v: float(np.mean(v)),
    "min": lambda v: float(np.min(v)),
    "quantiles": _quantile_dict,
}


def aggregate_values(values, strategy: str = DEFAULT_AGGREGATION):
    """Aggregate per-pair values under a named strategy.

    Returns float for 'per_pair_mean' / 'min', a distribution dict for
    'quantiles' (min/5th/25th/median/75th/95th/max/mean, tab:kappa-distribution).
    """
    try:
        fn = AGGREGATION_STRATEGIES[strategy]
    except KeyError as exc:
        raise AggregationError(
            f"unknown strategy {strategy!r}; "
            f"known: {sorted(AGGREGATION_STRATEGIES)}"
        ) from exc
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("no values to aggregate")
    return fn(arr)


@dataclass(frozen=True)
class KappaSummary:
    """Full kappa aggregation: strategy value + min + distribution + regime.

    headline       -- value under the requested strategy (float, or the
                      quantile dict when strategy='quantiles')
    strategy       -- the strategy name used
    min            -- always reported (body definition, L879)
    mean           -- always reported (tab:diagnostic definition, L2077)
    distribution   -- tab:kappa-distribution dict
    regime         -- kappa_regime() report (Assumption 6.1 status)
    """

    headline: float | dict
    strategy: str
    min: float
    mean: float
    distribution: dict
    regime: dict

    def to_dict(self) -> dict:
        out = {
            "strategy": self.strategy,
            "headline": self.headline,
            "min": self.min,
            "mean": self.mean,
            "distribution": self.distribution,
            "regime": self.regime,
        }
        return out


def summarize_kappa(values, strategy: str = DEFAULT_AGGREGATION) -> KappaSummary:
    """Aggregate per-pair kappa values with the min always carried along."""
    arr = np.asarray(values, dtype=float).ravel()
    return KappaSummary(
        headline=aggregate_values(arr, strategy),
        strategy=strategy,
        min=float(arr.min()),
        mean=float(arr.mean()),
        distribution=_quantile_dict(arr),
        regime=kappa_regime(arr),
    )


# ---------------------------------------------------------------------------
# Per-pair diagnostic table over method checkpoints (G03 SPEC 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairDiagnosticRow:
    """One (method, task pair) row of the diagnostic table."""

    method: str
    i: int
    j: int
    gamma_hat: float
    cosine: float
    relative_magnitude: float
    cv: float
    bias_sigma_norm: float  # ||DeltaW_hat||_Sigma vs sqrt(gamma) identity
    kappa_min: float
    kappa_mean: float
    kappa_pairs: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "pair": [self.i, self.j],
            "gamma_hat": self.gamma_hat,
            "cosine": self.cosine,
            "relative_magnitude": self.relative_magnitude,
            "cv": self.cv,
            "bias_sigma_norm": self.bias_sigma_norm,
            "kappa_min": self.kappa_min,
            "kappa_mean": self.kappa_mean,
        }


@dataclass(frozen=True)
class MethodDiagnostic:
    """Per-method aggregation of the per-pair rows (tab:diagnostic layout)."""

    method: str
    n_pairs: int
    mean_cosine: float
    mean_relative_magnitude: float
    mean_cv: float
    kappa: KappaSummary
    rows: list[PairDiagnosticRow] = field(repr=False, default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_pairs": self.n_pairs,
            "mean_cosine": self.mean_cosine,
            "mean_relative_magnitude": self.mean_relative_magnitude,
            "mean_cv": self.mean_cv,
            "kappa": self.kappa.to_dict(),
            "rows": [r.to_dict() for r in self.rows],
        }


def build_diagnostic_table(
    checkpoints: Iterable[tuple],
    strategy: str = DEFAULT_AGGREGATION,
    sigma: np.ndarray | None = None,
    ddof: int = 0,
) -> list[MethodDiagnostic]:
    """Build the per-pair diagnostic table over method checkpoints.

    checkpoints: iterable of (name, head, Sigma_inv) tuples. `head` follows the
    module Head protocol (.W, .mu_hat/.means, .task_of_class/.tasks);
    `Sigma_inv` is the estimated inverse covariance used consistently for every
    quantity of that method's rows (means, cosine metric, gamma, D).
    strategy: named kappa aggregation (default per_pair_mean; min always
              reported inside KappaSummary).
    sigma:    optional direct Sigma (bias_sigma_norm identity needs Sigma, not
              Sigma_inv; when omitted it is recomputed by inversion).
    ddof:     passed to cv_cross_pair.

    Returns one MethodDiagnostic per checkpoint (input order), each carrying
    the per-task-pair rows and the method-level aggregation.
    """
    methods: list[MethodDiagnostic] = []
    for entry in checkpoints:
        if len(entry) != 3:
            raise ValueError(
                "checkpoints must be (name, head, Sigma_inv) tuples, got "
                f"{len(entry)}-tuple"
            )
        name, head, sigma_inv = entry
        sigma_inv = np.asarray(sigma_inv, dtype=float)
        sigma_mat = (
            np.asarray(sigma, dtype=float)
            if sigma is not None
            else symmetric_inverse(sigma_inv)
        )
        mu = _head_mu_hat(head)
        t_of_c = _head_task_of_class(head)
        n_tasks = int(t_of_c.max()) + 1
        rows: list[PairDiagnosticRow] = []
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i == j:
                    continue
                res = bias_residuals(head, i, j, sigma_inv)
                d_hat = delta_w_hat(head, i, j, sigma_inv)
                d_pred = delta_w_pred(head, i, j, sigma_inv)
                g = gamma_ij(
                    mu[t_of_c == i].mean(axis=0), mu[t_of_c == j].mean(axis=0),
                    sigma_inv,
                )
                kap = kappa_block(mu, t_of_c, sigma_inv, i, j)
                rows.append(
                    PairDiagnosticRow(
                        method=str(name),
                        i=i,
                        j=j,
                        gamma_hat=g,
                        cosine=sigma_inner_cosine(
                            np.stack([d_hat, d_pred]), sigma_inv
                        ),
                        relative_magnitude=relative_magnitude(
                            d_hat, d_pred, sigma_inv
                        ),
                        cv=cv_cross_pair(res, sigma_inv, ddof=ddof),
                        bias_sigma_norm=sigma_norm(d_hat, sigma_mat),
                        kappa_min=float(kap.min()),
                        kappa_mean=float(kap.mean()),
                        kappa_pairs=kap,
                    )
                )
        flat_kappa = (
            np.concatenate([r.kappa_pairs.ravel() for r in rows])
            if rows
            else np.array([])
        )
        methods.append(
            MethodDiagnostic(
                method=str(name),
                n_pairs=len(rows),
                mean_cosine=float(np.mean([r.cosine for r in rows])),
                mean_relative_magnitude=float(
                    np.mean([r.relative_magnitude for r in rows])
                ),
                mean_cv=float(np.mean([r.cv for r in rows])),
                kappa=summarize_kappa(flat_kappa, strategy=strategy),
                rows=rows,
            )
        )
    return methods


def render_text_table(methods: Sequence[MethodDiagnostic]) -> str:
    """Compact text rendering: method | cos | rel | CV | kappa(strategy/min) |
    frac>1.5 | frac<1 | Assumption 6.1 -- the tab:diagnostic columns this
    module can produce without benchmark artifacts."""
    lines = [
        f"{'method':<18} {'cos':>8} {'rel':>8} {'CV':>8} "
        f"{'kappa_head':>10} {'kappa_min':>9} {'frac>1.5':>8} {'frac<1':>7} {'A6.1':>5}"
    ]
    for m in methods:
        reg = m.kappa.regime
        head_val = (
            f"{m.kappa.headline:10.4f}"
            if isinstance(m.kappa.headline, float)
            else "quantiles"
        )
        lines.append(
            f"{m.method:<18} {m.mean_cosine:8.4f} "
            f"{m.mean_relative_magnitude:8.4f} {m.mean_cv:8.4f} "
            f"{head_val:>10} {m.kappa.min:9.4f} "
            f"{reg['frac_above_threshold']:8.3f} {reg['frac_below_1']:7.3f} "
            f"{str(reg['holds']):>5}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpoint loader interface (abstract; real loaders = benchmarks wave)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedCheckpoint:
    """The (name, head, Sigma_inv) tuple the table builder consumes."""

    name: str
    head: object
    sigma_inv: np.ndarray

    def astuple(self) -> tuple:
        return (self.name, self.head, self.sigma_inv)


class CheckpointLoader(abc.ABC):
    """Abstract source of released-checkpoint diagnostics inputs.

    The five-method protocol (G03 SPEC 4) loads released checkpoints of
    FIVE_METHODS trained under the PILOT/RanPAC unified evaluation on
    Split-CIFAR-100 with google/vit-base-patch16-224-in21k, extracts the trained
    linear head and per-task class means from cached frozen features, and
    estimates Sigma_hat^{-1} (Ledoit-Wolf pooled within-class covariance,
    analytical shrinkage, sklearn.covariance.LedoitWolf). URLs are a TODO OWNER
    (G03 OWNER-ASK 8); concrete loaders arrive with the benchmarks wave and
    register here.
    """

    @abc.abstractmethod
    def names(self) -> tuple[str, ...]:
        """Available checkpoint names."""

    @abc.abstractmethod
    def load(self, name: str) -> LoadedCheckpoint:
        """Load one checkpoint as (name, head, Sigma_inv)."""


# ---------------------------------------------------------------------------
# Synthetic checkpoints (tests + self-check; synthetic-proxy only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticCheckpoint:
    """Synthetic CIL head with closed-form ground truth (Head protocol).

    Satisfies the module Head protocol and additionally carries the exact
    construction so tests can verify diagnostics against known truth
    (Theorem 4.1 offset direction, exact min kappa).

    Fields:
      W, task_of_class  -- the head (CF-optimal closed form by default,
                           Theorem 3.3, plus optional weight noise)
      b                 -- (n_classes,) canonical-gauge biases
      mu_hat, Sigma_inv -- the diagnostic inputs (true or finite-sample
                           estimates depending on arm)
      true_means, Sigma -- ground truth
      true_centroids    -- (T, d) exact task centroids
      expected_offset   -- dict {(i, j): Sigma^{-1}(mu_bar^(j) - mu_bar^(i))}
      kappa_target      -- construction guarantees min kappa = kappa_target
      gamma_true        -- (T, T) exact gamma_ij
    """

    W: np.ndarray
    task_of_class: np.ndarray
    b: np.ndarray
    mu_hat: np.ndarray
    Sigma_inv: np.ndarray
    true_means: np.ndarray
    Sigma: np.ndarray
    true_centroids: np.ndarray
    expected_offset: dict
    kappa_target: float
    gamma_true: np.ndarray
    mu_hat_is_estimate: bool


def _regular_simplex(c: int) -> np.ndarray:
    """(c, c-1) coordinates of a regular simplex on the unit sphere.

    Rows are unit-norm, sum to zero, pairwise inner products -1/(c-1) --
    deterministic (eigendecomposition of the fixed Gram matrix
    G = I - (1/(c-1))(11^T - I), eigenvalues {0, c/(c-1) x (c-1)}). For c=3
    this is the 120-degree triple; the construction generalizes it to any
    C >= 2, which the hardcoded 120-degree pattern could not.
    """
    gram = np.full((c, c), -1.0 / (c - 1))
    np.fill_diagonal(gram, 1.0)
    evals, evecs = np.linalg.eigh(gram)
    # Drop the zero eigenvalue (the all-ones direction); ascending order puts
    # it first.
    return evecs[:, 1:] * np.sqrt(evals[1:])


def make_synthetic_checkpoint(
    *,
    name: str = "synthetic",
    n_tasks: int = 4,
    n_classes_per_task: int = 3,
    d: int = 12,
    kappa_target: float = 2.0,
    seed: int = 0,
    noise_scale: float = 0.0,
    n_train_per_class: int | None = None,
) -> SyntheticCheckpoint:
    """Gaussian CIL construction with known Theorem 4.1 truth and exact kappa.

    Geometry (all directions are mutually Sigma^{-1}-orthogonal by embedding
    whitened coordinates u through Sigma^{1/2} u, since
    <Sigma^{1/2}u, Sigma^{1/2}u'>_{Sigma^{-1}} = u^T u'):
      - Shared Sigma = A A^T / d + 0.05 I (SPD, diag O(1)).
      - Task centroids: T random orthonormal-coordinate directions (rank T-1
        after centering), giving gamma_ij = ||z_i - z_j||^2 of order T.
      - Within-task offsets nu_r: per task a regular-simplex vertex set (unit
        norms, sum_r nu_r = 0, Lem 3.2 gauge), living in the Sigma^{-1}-
        orthogonal complement of the centroid span, so cross-class distances
        split exactly: D_kl = gamma_ij + ||nu_k - nu_l||^2_{Sigma^{-1}}, and
        with unit Sigma^{-1}-norm offsets from different tasks,
        ||nu_k - nu_l||^2 = 2 s^2 for every cross pair.
      - kappa target: min_{ij,kl} kappa_kl^2 = 1 + 2 s^2 / gamma_ij, attained
        at the WIDEST task pair (gamma_max), so s = sqrt((kappa^2 - 1) gamma_max / 2)
        gives min kappa_hat = kappa_target exactly.
    Head: CF-optimal closed form w_r = Sigma^{-1}(mu_r - mu_bar^(t)),
    b_r = -1/2 mu_r^T Sigma^{-1} mu_r + 1/2 S_bar^(t) (Theorem 3.3), optionally
    perturbed by Gaussian weight noise of relative size noise_scale.

    Arms:
      n_train_per_class=None -- population arm: mu_hat = true means,
                                Sigma_inv = true inverse (exact identities).
      n_train_per_class=int  -- estimated arm: mu_hat = per-class sample means,
                                Sigma_inv from the pooled within-class sample
                                covariance (plugin, ddof=1, averaged over
                                classes; the Ledoit-Wolf loader variant belongs
                                to the benchmarks wave).

    Determinism: fixed seed => bit-identical output.
    """
    rng = np.random.default_rng(seed)
    c = n_classes_per_task
    t = n_tasks
    n = t * c

    a = rng.standard_normal((d, d))
    sigma = a @ a.T / d + 0.05 * np.eye(d)
    sigma_inv = symmetric_inverse(sigma)
    evals, evecs = np.linalg.eigh(sigma)
    # Embedding map: whitened coordinates u -> Sigma^{1/2} u, so whitened-space
    # orthogonality and unit norms become Sigma^{-1}-orthogonality and unit
    # Sigma^{-1}-norms of the embedded vectors.
    sigma_half = (evecs * np.sqrt(evals)) @ evecs.T

    # Centroid directions: whitened coords in dims [0, t-1) (rank t-1 after
    # centering), then centered to zero global mean.
    z = rng.standard_normal((t, t - 1 if t > 1 else 1)) if t > 1 else np.zeros((t, 1))
    z = z - z.mean(axis=0, keepdims=True)
    centroid_w = np.zeros((t, d))
    centroid_w[:, : z.shape[1]] = z
    centroids = centroid_w @ sigma_half.T  # rows: mu_bar^(t)

    # Within-task offset directions: a regular simplex per task in its own
    # (c-1)-dim whitened block, dims [t-1, t-1 + t*(c-1)) -- Sigma^{-1}-
    # orthogonal to the centroid span and to every other task's block.
    if c < 2:
        raise ValueError("n_classes_per_task must be >= 2")
    simplex = _regular_simplex(c)  # (c, c-1): unit rows, sum zero
    base = t - 1 if t > 1 else 0
    need = base + t * (c - 1)
    if need > d:
        raise ValueError(
            f"construction needs d >= {need} for T={t}, C={c}; got d={d}"
        )
    offset_w = np.zeros((n, d))
    for task in range(t):
        cls = slice(task * c, (task + 1) * c)
        offset_w[cls, base + task * (c - 1) : base + (task + 1) * (c - 1)] = (
            simplex
        )

    gamma_true = gamma_matrix(centroids, np.arange(t), sigma_inv)
    positive = gamma_true[gamma_true > 0.0]
    if positive.size == 0:
        raise ValueError("degenerate centroid geometry (all gamma_ij = 0)")
    gamma_max = float(positive.max())
    # min kappa over (ij, kl) is attained at the WIDEST task pair:
    # kappa_{ij,kl}^2 = 1 + 2 s^2 / gamma_ij  =>  s = sqrt((k^2-1) gamma_max/2)
    # makes min kappa_hat == kappa_target exactly.
    s = math.sqrt(max(kappa_target**2 - 1.0, 0.0) * gamma_max / 2.0)

    means = np.repeat(centroids, c, axis=0) + s * (offset_w @ sigma_half.T)
    task_of_class = np.repeat(np.arange(t), c)

    # CF-optimal head (Theorem 3.3), canonical gauge.
    w = np.zeros((n, d))
    b = np.zeros(n)
    for task in range(t):
        cls = np.arange(task * c, (task + 1) * c)
        cent = means[cls].mean(axis=0)
        w[cls] = (sigma_inv @ (means[cls] - cent).T).T
        s_bar = float(np.mean([means[r] @ sigma_inv @ means[r] for r in cls]))
        b[cls] = -0.5 * np.einsum(
            "ij,jk,ik->i", means[cls], sigma_inv, means[cls]
        ) + 0.5 * s_bar
    if noise_scale > 0.0:
        scale = float(np.linalg.norm(w, axis=1).mean())
        w = w + noise_scale * scale * rng.standard_normal(w.shape)

    if n_train_per_class is None:
        mu_hat, sigma_inv_hat = means, sigma_inv
        estimated = False
    else:
        chol = np.linalg.cholesky(sigma)
        feats = np.empty((n * n_train_per_class, d))
        for r in range(n):
            z_r = rng.standard_normal((n_train_per_class, d))
            feats[r * n_train_per_class : (r + 1) * n_train_per_class] = (
                means[r] + z_r @ chol.T
            )
        blocks = feats.reshape(n, n_train_per_class, d)
        mu_hat = blocks.mean(axis=1)
        centered = blocks - mu_hat[:, None, :]
        cov_hat = np.einsum("ncd,nce->de", centered, centered) / (
            n * (n_train_per_class - 1)
        )
        sigma_inv_hat = symmetric_inverse(cov_hat)
        estimated = True

    expected = {}
    for i in range(t):
        for j in range(t):
            if i != j:
                expected[(i, j)] = sigma_inv @ (
                    centroids[j] - centroids[i]
                )
    return SyntheticCheckpoint(
        W=w,
        task_of_class=task_of_class,
        b=b,
        mu_hat=mu_hat,
        Sigma_inv=sigma_inv_hat,
        true_means=means,
        Sigma=sigma,
        true_centroids=centroids,
        expected_offset=expected,
        kappa_target=float(kappa_target),
        gamma_true=gamma_true,
        mu_hat_is_estimate=estimated,
    )


def synthetic_checkpoint_spec(**kwargs) -> tuple:
    """(name, head, Sigma_inv) tuple for build_diagnostic_table."""
    if "name" not in kwargs:
        raise ValueError("synthetic_checkpoint_spec requires name=")
    ckpt = make_synthetic_checkpoint(**kwargs)
    return (kwargs["name"], ckpt, ckpt.Sigma_inv)


class SyntheticCheckpointLoader(CheckpointLoader):
    """Loader over named synthetic checkpoints (tests + protocol dry-runs).

    Example: SyntheticCheckpointLoader(FIVE_METHODS, seed=0) yields five
    synthetic-proxy entries so the five-method table runs end-to-end before
    any released checkpoint exists. Asserts no paper number.
    """

    def __init__(self, names: Sequence[str], *, seed: int = 0, **builder_kwargs):
        self._names = tuple(names)
        self._seed = int(seed)
        self._builder_kwargs = dict(builder_kwargs)

    def names(self) -> tuple[str, ...]:
        return self._names

    def load(self, name: str) -> LoadedCheckpoint:
        if name not in self._names:
            raise KeyError(f"unknown synthetic checkpoint {name!r}")
        ckpt = make_synthetic_checkpoint(
            name=name,
            seed=self._seed + self._names.index(name),
            **self._builder_kwargs,
        )
        return LoadedCheckpoint(name=name, head=ckpt, sigma_inv=ckpt.Sigma_inv)


# ---------------------------------------------------------------------------
# Self-check (Theorem 4.1 known truth; run: python core/diagnostic.py)
# ---------------------------------------------------------------------------


def _self_check() -> int:  # noqa: C901 - linear checklist, each step labeled
    """Verify diagnostics against closed-form truth (synthetic-proxy labels).

    Tolerances follow the G10/conftest epsilon tiers and G06 E6 philosophy:
    exact-population constructions are algebraic identities, so 1e-9
    (EPS_LINALG: cholesky/eigh/solve round-off at these condition numbers);
    finite-sample arms use E2 estimated-arm thresholds (cos > 0.99) and the
    rate-validation 5% standard for kappa.
    """
    eps_linalg = 1e-9
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            failures.append(label)

    # --- Population arm: exact identities (Theorem 4.1 + Cor 4.2/4.3) -------
    print("Population arm (closed-form CF head, true Sigma; synthetic-proxy):")
    pop = make_synthetic_checkpoint(
        n_tasks=4, n_classes_per_task=3, d=12, kappa_target=2.0, seed=0
    )
    cosines, rels, cvs, kap_mins = [], [], [], []
    for (i, j), offset in pop.expected_offset.items():
        d_hat = delta_w_hat(pop, i, j, pop.Sigma_inv)
        d_pred = delta_w_pred(pop, i, j, pop.Sigma_inv)
        cos = sigma_inner_cosine(np.stack([d_hat, d_pred]), pop.Sigma_inv)
        rel = relative_magnitude(d_hat, d_pred, pop.Sigma_inv)
        cv = cv_cross_pair(bias_residuals(pop, i, j, pop.Sigma_inv), pop.Sigma_inv)
        g = pop.gamma_true[i, j]
        # Sigma-norm identity: ||DeltaW_hat||_Sigma = sqrt(gamma_ij).
        sig_norm_ratio = sigma_norm(d_hat, pop.Sigma) / math.sqrt(g)
        kap = kappa_block(pop.mu_hat, pop.task_of_class, pop.Sigma_inv, i, j)
        cosines.append(cos)
        rels.append(rel)
        cvs.append(cv)
        kap_mins.append(float(kap.min()))
        if (i, j) == (0, 1):
            check(
                "cos(DeltaW_hat, DeltaW_pred) == 1 (Thm 4.1)",
                abs(cos - 1.0) < eps_linalg,
                f"cos={cos:.12f}",
            )
            check(
                "||DeltaW_hat||/||DeltaW_pred|| == 1",
                abs(rel - 1.0) < eps_linalg,
                f"rel={rel:.12f}",
            )
            check("cross-pair CV == 0 (Cor 4.2)", cv < eps_linalg, f"cv={cv:.3e}")
            check(
                "||DeltaW_hat||_Sigma == sqrt(gamma_ij)",
                abs(sig_norm_ratio - 1.0) < eps_linalg,
                f"ratio={sig_norm_ratio:.12f}, gamma_01={g:.6f}",
            )
    check(
        "all T(T-1) pairs: cos == 1 within 1e-9",
        max(abs(np.asarray(cosines) - 1.0)) < eps_linalg,
        f"min cos={min(cosines):.12f} over {len(cosines)} pairs",
    )
    check(
        "min kappa_hat == kappa_target (Assumption 6.1 construction)",
        abs(min(kap_mins) - 2.0) < eps_linalg,
        f"min over pairs={min(kap_mins):.12f} (target 2.0)",
    )
    reg = kappa_regime(
        np.concatenate(
            [
                kappa_block(
                    pop.mu_hat, pop.task_of_class, pop.Sigma_inv, i, j
                ).ravel()
                for (i, j) in pop.expected_offset
            ]
        )
    )
    check(
        "kappa_regime: holds, zero aliasing tail",
        reg["holds"] and reg["frac_below_1"] == 0.0 and reg["min_kappa"] > 1.0,
        f"min={reg['min_kappa']:.6f}, margin={reg['margin']:.6f}, "
        f"frac<1={reg['frac_below_1']}",
    )
    check(
        "mean aggregation > min aggregation (aggregations differ)",
        reg["mean_kappa"] > reg["min_kappa"],
        f"mean={reg['mean_kappa']:.6f} > min={reg['min_kappa']:.6f}",
    )

    # --- Estimated arm: finite-sample mu_hat, Sigma_hat (E2 thresholds) -----
    print("Estimated arm (n=4000/class plugin estimates; synthetic-proxy):")
    est = make_synthetic_checkpoint(
        n_tasks=4,
        n_classes_per_task=3,
        d=12,
        kappa_target=2.0,
        seed=1,
        n_train_per_class=4000,
    )
    est_cos, est_rel, est_cv, est_kap = [], [], [], []
    for (i, j) in est.expected_offset:
        d_hat = delta_w_hat(est, i, j, est.Sigma_inv)
        d_pred = delta_w_pred(est, i, j, est.Sigma_inv)
        est_cos.append(
            sigma_inner_cosine(np.stack([d_hat, d_pred]), est.Sigma_inv)
        )
        est_rel.append(relative_magnitude(d_hat, d_pred, est.Sigma_inv))
        est_cv.append(
            cv_cross_pair(bias_residuals(est, i, j, est.Sigma_inv), est.Sigma_inv)
        )
        est_kap.append(
            kappa_block(est.mu_hat, est.task_of_class, est.Sigma_inv, i, j).ravel()
        )
    est_kap = np.concatenate(est_kap)
    check(
        "estimated-arm cos > 0.99 (E2 estimated-arm threshold)",
        min(est_cos) > 0.99,
        f"min cos={min(est_cos):.4f} over {len(est_cos)} pairs",
    )
    check(
        "estimated-arm rel magnitude in (0.9, 1.1)",
        all(0.9 < r < 1.1 for r in est_rel),
        f"range=({min(est_rel):.4f}, {max(est_rel):.4f})",
    )
    check(
        "estimated-arm CV < 0.30",
        max(est_cv) < 0.30,
        f"max CV={max(est_cv):.4f}",
    )
    pop_mean_kappa = reg["mean_kappa"]
    check(
        "mean kappa_hat within 5% of population (rate-validation standard)",
        abs(est_kap.mean() - pop_mean_kappa) / pop_mean_kappa < 0.05,
        f"est mean={est_kap.mean():.4f} vs pop mean={pop_mean_kappa:.4f} "
        f"({100 * abs(est_kap.mean() - pop_mean_kappa) / pop_mean_kappa:.2f}%)",
    )

    # --- Table builder + five-method context, structural dry-run ------------
    print("Five-method table dry-run (synthetic proxies, one noisy arm):")
    specs = [
        synthetic_checkpoint_spec(
            name=f"synthetic:{m}", seed=10 + k,
            noise_scale=0.02 if m == "SLCA" else 0.0,
        )
        for k, m in enumerate(FIVE_METHODS)
    ]
    methods = build_diagnostic_table(specs)
    print(render_text_table(methods))
    clean = {m.method: m for m in methods}
    l2p = clean["synthetic:L2P"]
    check(
        "clean synthetic method: cos == 1, CV == 0 through the table",
        abs(l2p.mean_cosine - 1.0) < eps_linalg and l2p.mean_cv < eps_linalg,
        f"cos={l2p.mean_cosine:.12f}, CV={l2p.mean_cv:.3e}",
    )
    check(
        "default aggregation is per-pair mean with min reported",
        l2p.kappa.strategy == "per_pair_mean"
        and isinstance(l2p.kappa.headline, float)
        and l2p.kappa.min == l2p.kappa.distribution["min"],
        f"headline(mean)={l2p.kappa.headline:.6f}, min={l2p.kappa.min:.6f}",
    )
    slca = clean["synthetic:SLCA"]
    check(
        "noisy arm detected: cos < 1 and CV > 0",
        slca.mean_cosine < 1.0 - 1e-6 and slca.mean_cv > 0.0,
        f"cos={slca.mean_cosine:.4f}, CV={slca.mean_cv:.4f}",
    )
    dist = l2p.kappa.distribution
    quantiles = [dist[k] for k in ("min", "p05", "p25", "median", "p75", "p95", "max")]
    check(
        "kappa distribution quantiles monotone (tab:kappa-distribution layout)",
        all(a <= b for a, b in zip(quantiles, quantiles[1:])) and dist["n_pairs"] == 108,
        f"n_pairs={dist['n_pairs']} (T(T-1)=12 ordered pairs x C^2=9), "
        f"min={dist['min']:.4f}, median={dist['median']:.4f}, max={dist['max']:.4f}",
    )
    for strat in ("min", "quantiles"):
        agg = aggregate_values(
            np.concatenate([r.kappa_pairs.ravel() for r in l2p.rows]), strat
        )
        if strat == "min":
            ok = abs(agg - l2p.kappa.min) < eps_linalg
        else:
            ok = abs(agg["mean"] - l2p.kappa.mean) < eps_linalg
        check(f"named strategy '{strat}' callable", ok, f"value={agg if strat == 'min' else agg['mean']:.6f}")
    loader = SyntheticCheckpointLoader(("synthetic:L2P",), seed=10)
    lc = loader.load("synthetic:L2P")
    check(
        "CheckpointLoader interface round-trips into the table",
        len(build_diagnostic_table([(lc.name, lc.head, lc.sigma_inv)])) == 1,
        "1 method loaded and built",
    )

    print()
    if failures:
        print(f"SELF-CHECK FAILED: {len(failures)} failure(s): {failures}")
        return 1
    print("SELF-CHECK PASSED: all diagnostics match closed-form truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
