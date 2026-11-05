"""pipeline.heads -- linear heads on frozen features, every protocol both papers use.

Objectives
----------
within-task softmax (Paper B eq:ce-within; Paper A "diagonal-only joint training"):
    for each task t, mean CE over the task's classes only + (lam/2)(||W_t||^2 + ||b_t||^2)
    The objective is separable across tasks, so sequential training and
    task-balanced joint diagonal training have the same minimizer; we train it
    per task and assemble the head.
full softmax (Paper B eq:ce-full; Paper A "all-blocks joint training"):
    mean CE over all N classes + (lam/2)(||W||^2 + ||b||^2)

Optimizers
----------
lbfgs    full-batch L-BFGS-B on the exact objective (deterministic, converged to gtol);
         the reference optimizer for every closed-form comparison.
adamw    the papers' protocol: AdamW, lr 1e-3, batch 128, cosine schedule to eta_min,
         decoupled weight decay lam on W (and on b when decay_bias), E epochs, no
         augmentation.  Implemented in numpy so it runs anywhere.
clipped  Paper A Algorithm 1: clipped minibatch SGD with eta_t, tau_t from Prop 5.1 and the
         warm-up estimates, task by task on the task's data; ``scope`` selects the diagonal
         (within-task) head of the paper's theory or a class-IL head over the seen classes
         (see clipped_sgd_head).

Corrections and reference heads
-------------------------------
de_center      Definition 7.1 (Paper B): w_r += Sigma^{-1} mu_bar_t, b_r -= S_bar_t / 2
lda_head       shared-covariance LDA / SLDA: w_r = Sigma^{-1} mu_r, b_r = -mu_r^T Sigma^{-1} mu_r / 2
ncm_head       nearest class mean (Sigma = I), as a linear head
fecam_head     per-class-covariance Mahalanobis scores (FeCAM-style, shrunk covariances)
ranpac_head    random projection + ReLU + ridge (RanPAC-style, Gram accumulation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import special

from pipeline import backend as _backend
from pipeline.registry import rng as _rng
from pipeline.stats import ClassStats


# ---------------------------------------------------------------------------
# Loss and gradient
# ---------------------------------------------------------------------------

def _ce_value_grad(theta, X, y, C, lam, decay_bias=True, sample_w=None):
    d = X.shape[1]
    W = theta[: C * d].reshape(C, d); b = theta[C * d:]
    S = X @ W.T + b
    lse = special.logsumexp(S, axis=1)
    n = X.shape[0]
    P = np.exp(S - lse[:, None])
    if sample_w is None:
        loss = float((lse - S[np.arange(n), y]).mean())
        G = P; G[np.arange(n), y] -= 1.0; G /= n
    else:
        loss = float(np.sum(sample_w * (lse - S[np.arange(n), y])))
        G = P; G[np.arange(n), y] -= 1.0; G *= sample_w[:, None]
    loss += 0.5 * lam * (np.sum(W * W) + (np.sum(b * b) if decay_bias else 0.0))
    gW = G.T @ X + lam * W
    gb = G.sum(axis=0) + (lam * b if decay_bias else 0.0)
    return loss, np.concatenate([gW.ravel(), gb])


def fit_lbfgs(X, y, C, lam, decay_bias=True, maxiter=2000, gtol=1e-9, theta0=None, device=None):
    """Exact regularized minimizer by full-batch L-BFGS (scipy on the numpy backend, torch.optim.LBFGS in float64
    on a torch device).  The reference optimizer for every closed-form comparison."""
    d = X.shape[1]
    th0 = np.zeros(C * d + C) if theta0 is None else np.asarray(theta0, np.float64)
    th, info = _backend.lbfgs_ce(th0, X, y, C, lam, decay_bias=decay_bias, maxiter=maxiter, gtol=gtol, device=device)
    return th[: C * d].reshape(C, d), th[C * d:], info


def fit_adamw(X, y, C, lam, *, lr=1e-3, batch=128, epochs=50, eta_min=1e-6, decay_bias=True,
              seed_purpose="adamw", betas=(0.9, 0.999), eps=1e-8, theta0=None, device=None):
    """AdamW with decoupled weight decay (torch semantics: p -= lr*lam*p) and cosine lr, from a zero
    (or given) head.  The epoch permutations come from the seeded numpy Generator; the arithmetic runs
    on the backend selected by ``device`` (see pipeline.backend)."""
    n, d = X.shape
    th = np.zeros(C * d + C) if theta0 is None else np.asarray(theta0, np.float64).copy()
    g = _rng(seed_purpose)
    perms = [g.permutation(n) for _ in range(epochs)]
    th, t = _backend.adamw_ce(th, X, y, C, lam, lr=lr, eta_min=eta_min, perms=perms, batch=batch,
                              decay_bias=decay_bias, betas=betas, eps=eps, device=device)
    W = th[: C * d].reshape(C, d); b = th[C * d:]
    loss, grad = _ce_value_grad(th, X, y, C, lam, decay_bias)
    return W, b, dict(steps=t, max_grad=float(np.abs(grad).max()), fun=float(loss), backend=_backend.resolve_device(device))


# ---------------------------------------------------------------------------
# Head containers
# ---------------------------------------------------------------------------

@dataclass
class Head:
    W: np.ndarray            # [N, d]
    b: np.ndarray            # [N]
    task_of_class: np.ndarray
    info: dict = field(default_factory=dict)

    CHUNK = 16384            # rows per logits block (DomainNet: 176k test rows x 2070 virtual classes)

    def logits(self, X):
        """Full logits matrix; callers with large N x n should use predict / ce_loss, which stream."""
        return np.asarray(X, np.float64) @ self.W.T + self.b

    def _blocks(self, X, classes: Optional[np.ndarray] = None):
        X = np.asarray(X)
        Wt = self.W.T if classes is None else self.W[classes].T
        bb = self.b if classes is None else self.b[classes]
        for s in range(0, len(X), self.CHUNK):
            yield s, np.asarray(X[s:s + self.CHUNK], np.float64) @ Wt + bb

    def predict(self, X, classes: Optional[np.ndarray] = None) -> np.ndarray:
        """argmax over all classes, or over ``classes`` only (returned as class ids), streamed over rows."""
        out = np.empty(len(X), np.int64)
        for s, L in self._blocks(X, classes):
            am = np.argmax(L, axis=1)
            out[s:s + len(am)] = am if classes is None else classes[am]
        return out

    def accuracy(self, X, y, classes: Optional[np.ndarray] = None) -> float:
        return float(np.mean(self.predict(X, classes) == np.asarray(y)))

    def ce_loss(self, X, y, classes: Optional[np.ndarray] = None) -> float:
        """Mean cross-entropy over all classes (or over ``classes``, with y remapped), streamed over rows."""
        y = np.asarray(y, np.int64)
        yy = y if classes is None else np.searchsorted(classes, y)
        tot = 0.0
        for s, L in self._blocks(X, classes):
            yb = yy[s:s + len(L)]
            tot += float(np.sum(special.logsumexp(L, axis=1) - L[np.arange(len(L)), yb]))
        return tot / len(y)

    def task_confusion(self, X, y, i: int, j: int) -> float:
        """|P_ij| estimate: fraction of class-k test samples (k in task i) predicted into task j, summed over the C^2 pairs
        as the paper counts it (sum over k in i of P[pred in j | y=k])."""
        toc = self.task_of_class
        ki = np.flatnonzero(toc == i); kj = np.flatnonzero(toc == j)
        m = np.isin(y, ki)
        if m.sum() == 0:
            return float("nan")
        pred = self.predict(X[m]); ym = np.asarray(y)[m]
        tot = 0.0
        for k in ki:
            mk = ym == k
            if mk.sum():
                tot += float(np.mean(np.isin(pred[mk], kj)))
        return tot

    def copy(self):
        return Head(self.W.copy(), self.b.copy(), self.task_of_class.copy(), dict(self.info))


def train_within_task(X, y, task_of_class, lam=1e-4, optimizer="lbfgs", protocol: dict = None,
                      seed_purpose="wt", decay_bias=True, device=None) -> Head:
    """Within-task softmax, one block per task, assembled into an N-class head."""
    N = task_of_class.size; d = X.shape[1]
    W = np.zeros((N, d)); b = np.zeros(N); infos = []
    for t in range(int(task_of_class.max()) + 1):
        cls = np.flatnonzero(task_of_class == t)
        m = np.isin(y, cls)
        yb = np.searchsorted(cls, y[m])
        if optimizer == "lbfgs":
            Wt, bt, info = fit_lbfgs(X[m], yb, len(cls), lam, decay_bias, device=device)
        else:
            p = dict(protocol or {})
            Wt, bt, info = fit_adamw(X[m], yb, len(cls), lam, lr=p.get("lr", 1e-3), batch=p.get("batch", 128),
                                     epochs=p.get("epochs", 50), eta_min=p.get("eta_min", 1e-6), decay_bias=decay_bias,
                                     seed_purpose=f"{seed_purpose}:t{t}", device=device)
        W[cls] = Wt; b[cls] = bt; infos.append(info)
    return Head(W, b, np.asarray(task_of_class), dict(mode="within", optimizer=optimizer, lam=lam, blocks=infos))


def train_joint(X, y, task_of_class, lam=1e-4, optimizer="lbfgs", protocol: dict = None, seed_purpose="joint",
                decay_bias=True, classes: Optional[np.ndarray] = None, device=None) -> Head:
    """Full N-way softmax on all data (or on ``classes`` only, for class-IL stages)."""
    N = task_of_class.size; d = X.shape[1]
    cls = np.arange(N) if classes is None else np.asarray(classes)
    m = np.isin(y, cls); yb = np.searchsorted(cls, y[m])
    if optimizer == "lbfgs":
        Wc, bc, info = fit_lbfgs(X[m], yb, len(cls), lam, decay_bias, device=device)
    else:
        p = dict(protocol or {})
        Wc, bc, info = fit_adamw(X[m], yb, len(cls), lam, lr=p.get("lr", 1e-3), batch=p.get("batch", 128),
                                 epochs=p.get("epochs", 50), eta_min=p.get("eta_min", 1e-6), decay_bias=decay_bias,
                                 seed_purpose=seed_purpose, device=device)
    W = np.zeros((N, d)); b = np.full(N, -np.inf if classes is not None else 0.0)
    W[cls] = Wc; b[cls] = bc
    return Head(W, b, np.asarray(task_of_class), dict(mode="joint", optimizer=optimizer, lam=lam, fit=info))


# ---------------------------------------------------------------------------
# Corrections and reference heads
# ---------------------------------------------------------------------------

def de_center(head: Head, stats: ClassStats, scale=None) -> Head:
    """Definition 7.1: w_r += Sigma^{-1} mu_bar_t(r); b_r -= S_bar_t(r)/2, from training statistics only.

    ``scale`` (None, a float, or one float per task) multiplies both offsets.  None is the paper's definition
    (scale 1, which presumes the head sits at the population scale of Theorem 3.3).  Passing
    ``stats.head_scales(head.W)`` gives the scale-matched diagnostic variant used in the replication to tell a
    gauge failure from a scale mismatch; it is labelled FIX_c in the results and is not a paper number."""
    cents = stats.centroids
    offs = stats.sigma_inv_apply(cents)                      # [T, d]
    bayes_norm = (stats.whiten(stats.mu) ** 2).sum(1)        # mu_r^T Sigma^{-1} mu_r
    S_bar = np.array([bayes_norm[stats.task_of_class == t].mean() for t in range(stats.T)])
    c = np.ones(stats.T) if scale is None else np.broadcast_to(np.asarray(scale, np.float64), (stats.T,)).copy()
    c = np.where(np.isfinite(c), c, 1.0)
    h = head.copy()
    h.W = h.W + c[stats.task_of_class, None] * offs[stats.task_of_class]
    h.b = h.b - 0.5 * c[stats.task_of_class] * S_bar[stats.task_of_class]
    h.info["mode"] = head.info.get("mode", "") + ("+fix" if scale is None else "+fix_c")
    return h


def estimate_c_hat(head: Head, stats: ClassStats, center: bool = False) -> np.ndarray:
    """Definition 7.4's per-task scalar, in the closed form the camera-ready states (App. weights-only; the fleet's
    core/fix.py::estimate_c_hat): the per-task least-squares threshold-alignment estimator from training statistics,

        c_hat_t = mean_{r in task t} [ b_r - S_bar_t / 2 + mu_r^T Sigma^{-1} mu_r / 2 ],

    the mean residual of the trained thresholds against the canonical ones after the Definition 7.1 shift.  It is
    exactly zero on a canonical-gauge CF-optimal head and recovers a floating per-task bias constant exactly; a
    common scale c on the thresholds drops out of the mean because S_bar_t is the task mean of mu^T Sigma^{-1} mu.
    No held-out data is used.  ``center`` subtracts the mean over tasks (only differences affect decisions)."""
    quad = (stats.whiten(stats.mu) ** 2).sum(1)
    S_bar = np.array([quad[stats.task_of_class == t].mean() for t in range(stats.T)])
    resid = np.asarray(head.b, np.float64) - 0.5 * S_bar[stats.task_of_class] + 0.5 * quad
    c = np.array([resid[stats.task_of_class == t].mean() for t in range(stats.T)])
    return c - c.mean() if center else c


def lda_head(stats: ClassStats) -> Head:
    W = stats.sigma_inv_apply(stats.mu)
    b = -0.5 * (stats.whiten(stats.mu) ** 2).sum(1)
    return Head(W, b, stats.task_of_class, dict(mode="lda"))


def ncm_head(stats: ClassStats) -> Head:
    W = stats.mu.copy(); b = -0.5 * (stats.mu ** 2).sum(1)
    return Head(W, b, stats.task_of_class, dict(mode="ncm"))


class FeCAMHead:
    """Per-class shrunk covariance Mahalanobis classifier (FeCAM-style), on standardized features.

    Each class keeps the Cholesky factor of its shrunk covariance in float32 (N x d x d: 0.7 GB for
    OmniBenchmark's 300 classes at d = 768); prediction solves one triangular system per class."""

    def __init__(self, X, y, N, shrink=1.0):
        X = np.asarray(X, np.float64)
        self.mu = np.zeros((N, X.shape[1])); self.chol = np.empty((N, X.shape[1], X.shape[1]), np.float32)
        for c in range(N):
            Z = X[y == c]; self.mu[c] = Z.mean(0)
            R = Z - self.mu[c]
            S = R.T @ R / max(len(Z) - 1, 1)
            S += shrink * (np.trace(S) / S.shape[0]) * np.eye(S.shape[0])
            self.chol[c] = np.linalg.cholesky(S)

    def predict(self, X, chunk=2048):
        from scipy import linalg
        X = np.asarray(X, np.float64); out = np.empty(len(X), np.int64)
        for s in range(0, len(X), chunk):
            Xb = X[s:s + chunk]
            D = np.stack([(linalg.solve_triangular(np.asarray(L, np.float64), (Xb - m).T, lower=True) ** 2).sum(0)
                          for m, L in zip(self.mu, self.chol)], 1)
            out[s:s + chunk] = np.argmin(D, axis=1)
        return out

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))


class RanPACHead:
    """Random projection to M dims, ReLU, ridge regression on one-hot targets, accumulated as Gram and
    cross-covariance over the task stream (RanPAC-style; no first-session adaptation).  The projection
    matrix comes from the seeded numpy Generator; accumulation and solve run on the backend."""

    def __init__(self, d, N, M=10000, ridge=None, seed_purpose="ranpac", device=None):
        g = _rng(seed_purpose)
        self.N = N; self.M = M; self.ridge = ridge
        self.core = _backend.GramRidge(g.standard_normal((d, M)), N, device=device)

    def update(self, X, y):
        self.core.update(X, y); return self

    def solve(self):
        self.core.solve(self.ridge); return self

    def accuracy(self, X, y):
        return float(np.mean(self.core.predict(X) == y))


# ---------------------------------------------------------------------------
# Paper A Algorithm 1: warm-up diagnostics and clipped SGD on the N-way head
# ---------------------------------------------------------------------------

def warmup_gradient_norms(X, y, N, *, k_warm=1000, batch=128, lr=0.1, seed_purpose="warmup", lam=0.0, device=None):
    """Stage 0: K_warm vanilla minibatch SGD iterations on the full N-way CE from a random head;
    returns gradient norms g_k and the final iterate (Algorithm 1 lines 3-4)."""
    d = X.shape[1]; g = _rng(seed_purpose)
    th = 0.01 * g.standard_normal(N * d + N)
    idx = g.integers(0, len(y), (k_warm, batch))
    all_classes = np.arange(N)
    W, b, norms = _backend.sgd_subset(th[:N * d].reshape(N, d), th[N * d:], X, np.asarray(y), all_classes, idx,
                                      eta=lr, tau=np.inf, lam=lam, clip=False, record_norms=True, device=device)
    return norms, np.concatenate([W.ravel(), b])


def hill_estimate(norms, k=50, n_bootstrap=1000, seed_purpose="hill"):
    """Hill estimator of the tail index on the k largest of ``norms`` (app:alpha-estimation), with a bootstrap CI over
    the k log-ratios.  Finite-k bias is positive and small (core/streaming.py documents it); the CI is the bootstrap one."""
    o = np.sort(np.asarray(norms, np.float64))[::-1]
    k = int(min(k, len(o) - 1))
    top, ref = o[:k], max(float(o[k]), 1e-300)
    lr_ = np.log(np.maximum(top, 1e-300) / ref)
    m0 = float(lr_.mean())
    alpha = 1.0 / m0 if m0 > 0 else float("inf")
    g = _rng(seed_purpose); boots = []
    for _ in range(n_bootstrap):
        m = lr_[g.integers(0, k, k)].mean()
        if m > 0:
            boots.append(1.0 / m)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    return dict(alpha=float(alpha), ci=(float(lo), float(hi)), k=k, boot_mean=float(np.mean(boots)) if boots else float("nan"))


def curvature_probe(X, y, N, theta, lam, n_dirs=10, step=1e-3, seed_purpose="probe"):
    """mu_hat: median finite-difference curvature of the full-batch objective along random directions."""
    g = _rng(seed_purpose); out = []
    for _ in range(n_dirs):
        u = g.standard_normal(theta.size); u /= np.linalg.norm(u)
        _, gp = _ce_value_grad(theta + step * u, X, y, N, lam, True)
        _, gm = _ce_value_grad(theta - step * u, X, y, N, lam, True)
        out.append(float(u @ (gp - gm) / (2 * step)))
    return float(np.median(out)), out


def smoothness_bound(Z, lam: float, chunk: int = 32768) -> float:
    """A rigorous smoothness constant for the l2-regularized softmax cross-entropy on features Z: the Hessian in
    the logits is diag(p) - p p^T, bounded by I/2, so beta <= (1/2) lambda_max(M_2) + lam with M_2 the second-moment
    matrix of the augmented features [z; 1].  Units of a curvature (loss per squared parameter), unlike the maximum
    gradient norm the printed Algorithm 1 uses for beta_hat."""
    Z = np.asarray(Z); n, d = Z.shape; M = np.zeros((d + 1, d + 1))
    for i in range(0, n, chunk):
        A = np.concatenate([Z[i:i + chunk].astype(np.float64), np.ones((min(chunk, n - i), 1))], axis=1)
        M += A.T @ A
    return float(0.5 * np.linalg.eigvalsh(M / n).max() + lam)


def prop51_schedule(K, alpha, sigma, mu, beta, schedule: str = "printed"):
    """Algorithm 1 stage 0, two schedules:

      printed      eta = min{1/beta, (sigma^alpha / (mu^2 K))^(1/alpha)},  tau = (sigma^alpha K / mu)^(1/alpha)
                   (the accepted paper's Proposition 5.3; not dimensionally consistent, see docs/THEORY_NOTES.md,
                   kept so the printed numbers can be regenerated as printed)
      consistent   eta = min{1/(4 beta), 1/(beta sqrt K)},  tau = sigma K^(1/(2 alpha))
                   (the units-consistent schedule under which the in-expectation bound
                   E[Delta_K] <= Delta_0 exp(-(mu/beta) sqrt K) + c_alpha (sigma^2/mu) K^(-(alpha-1)/alpha) holds,
                   c_alpha = 4^(alpha-1) + 2^(alpha/2-1), provided the gradients stay below tau/2)

    The estimated constants are floored at tiny positive values so that a degenerate probe gives a finite, large
    step and threshold, not a NaN."""
    alpha = max(float(alpha), 1e-3); sigma = max(float(sigma), 1e-12); mu = max(float(mu), 1e-12); beta = max(float(beta), 1e-12)
    if schedule == "consistent":
        eta = min(1.0 / (4.0 * beta), 1.0 / (beta * np.sqrt(max(K, 1))))
        tau = sigma * max(K, 1) ** (1.0 / (2.0 * alpha))
    else:
        eta = min(1.0 / beta, (sigma ** alpha / (mu ** 2 * K)) ** (1.0 / alpha))
        tau = (sigma ** alpha * K / mu) ** (1.0 / alpha)
    return float(eta), float(tau)


def clipped_sgd_head(X, y, task_of_class, tasks: Sequence[np.ndarray], *, K, eta, tau, batch=128, lam=1e-4,
                     seed_purpose="clipped", clip=True, scope="diagonal", theta0=None, X_eval=None, y_eval=None, device=None):
    """Paper A Algorithm 1, Stage t head training (lines 13-18): for each task t, K minibatch SGD steps on
    the data of task t with step size eta and clipping threshold tau (clip=False gives vanilla SGD).

    ``scope`` fixes which cross-entropy the stage minimizes, the one point the pseudocode leaves implicit:

      "diagonal"  the CE over the classes of task t only, updating only their rows (the CF-optimal
                  diagonal head of Theorem disc-convergence, whose excess risk is E_CF + E_TC; the task-
                  confusion error of the assembled head is what Theorem relaxed bounds).  This is the
                  reading that matches the paper's theory and its E_CF definition, and is the default.
      "seen"      the CE over all classes seen so far (a class-incremental softmax head trained on the
                  current task's data only), which forgets earlier tasks' discriminators and is the
                  'sequential head' lower bound when unclipped.

    Rows of classes outside the stage's scope keep their values (random init for never-trained rows).
    Returns the final Head and, when (X_eval, y_eval) is given, the accuracy on the classes seen so far
    after every stage."""
    if scope not in ("diagonal", "seen"):
        raise ValueError("scope must be 'diagonal' or 'seen'")
    N = task_of_class.size; d = X.shape[1]; g = _rng(seed_purpose)
    th = (0.01 * g.standard_normal(N * d + N)) if theta0 is None else np.asarray(theta0, np.float64).copy()
    W = th[:N * d].reshape(N, d).copy(); b = th[N * d:].copy()
    seen = np.zeros(0, np.int64); per_stage = []; clipped_steps = 0; total_steps = 0; max_norm = 0.0
    for t, cls in enumerate(tasks):
        cls = np.sort(np.asarray(cls)); seen = np.union1d(seen, cls)
        active = cls if scope == "diagonal" else seen
        m = np.isin(y, cls); Xt, yt = X[m], y[m]
        idx = g.integers(0, len(yt), (K, batch))
        W, b, norms = _backend.sgd_subset(W, b, Xt, np.searchsorted(active, yt), active, idx, eta=eta, tau=tau, lam=lam,
                                          clip=clip, record_norms=True, device=device)
        if norms is not None:
            norms = np.asarray(norms, np.float64); clipped_steps += int((norms > tau).sum()); total_steps += int(norms.size); max_norm = max(max_norm, float(norms.max()))
        if X_eval is not None:
            me = np.isin(y_eval, seen)
            per_stage.append(Head(W, b, np.asarray(task_of_class)).accuracy(X_eval[me], y_eval[me], classes=seen))
    # clip_frac: the fraction of steps whose pre-clipping gradient norm exceeded tau (0 means the threshold never
    # fired and the "clipped" arm ran as plain SGD); max_grad_norm / tau is the condition (G) of docs/THEORY_NOTES.md
    info = dict(mode="clipped_sgd", scope=scope, K=K, eta=eta, tau=tau, clip=clip, backend=_backend.resolve_device(device),
                clip_frac=(clipped_steps / total_steps if total_steps else None), max_grad_norm=max_norm,
                grad_below_half_tau=bool(max_norm <= 0.5 * tau) if np.isfinite(tau) else None)
    return Head(W, b, np.asarray(task_of_class), info), per_stage


def sequential_head_baseline(X, y, task_of_class, tasks, *, K, eta, batch=128, lam=1e-4, seed_purpose="seqft",
                             X_eval=None, y_eval=None, device=None):
    """The 'Sequential (head only)' lower bound: a class-incremental softmax head over the classes seen so
    far, trained by plain SGD on the Stage-t data only, which forgets earlier tasks' discriminators."""
    return clipped_sgd_head(X, y, task_of_class, tasks, K=K, eta=eta, tau=np.inf, batch=batch, lam=lam, scope="seen",
                            seed_purpose=seed_purpose, clip=False, X_eval=X_eval, y_eval=y_eval, device=device)
