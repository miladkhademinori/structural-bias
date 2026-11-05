"""pipeline.stats -- class statistics and the geometric quantities of both papers.

All quantities are computed from TRAINING features only.

Paper A (app:gamma-computation): features are standardized globally (mean and
scalar std over all training features), class means are taken, the pooled
within-class scatter S is Ledoit-Wolf shrunk toward tr(S)/p I, task centroids are
class-mean averages, gamma_ij = (mu_i - mu_j)^T Sigma^{-1} (mu_i - mu_j) via
Cholesky solves, gamma_min = min over task pairs.

Paper B (app:experiments): no standardization; Ledoit-Wolf shrinkage of the
within-class scatter, computed per task on that task's training data and
pooled by averaging across tasks.  gamma_ij, D_kl and kappa follow eq:gamma.

Both estimators are exposed; ``ClassStats.from_features(..., protocol="A"|"B")``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import linalg


def _lw_from_moments(S: np.ndarray, quartic: float, n: int) -> Tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage from the second-moment matrix S = R^T R / n of n centred rows and the quartic
    sum quartic = sum_n ||r_n||^4.  Sigma_hat = (1 - l) S + l mu I with mu = tr(S)/p; the same closed form as
    sklearn.covariance.ledoit_wolf(assume_centered=True), written so that the rows never need to be held at once."""
    p = S.shape[0]
    mu = np.trace(S) / p
    delta = np.sum((S - mu * np.eye(p)) ** 2) / p
    beta = (quartic / n - np.sum(S ** 2)) / (n * p)
    beta = min(beta, delta)
    lam = 0.0 if delta <= 0 else beta / delta
    lam = float(min(max(lam, 0.0), 1.0))
    return (1.0 - lam) * S + lam * mu * np.eye(p), lam


def ledoit_wolf_shrinkage(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of the covariance of centred rows R (reference form; see _lw_from_moments)."""
    R = np.asarray(R, np.float64)
    n = R.shape[0]
    return _lw_from_moments(R.T @ R / n, float(np.sum(np.sum(R * R, axis=1) ** 2)), n)


CHUNK = 32768   # rows per float64 chunk in the streaming passes (32768 x 768 x 8 bytes = 200 MB)


@dataclass
class ClassStats:
    mu: np.ndarray             # [N, d] class means
    sigma: np.ndarray          # [d, d] shrunk pooled within-class covariance
    chol: np.ndarray           # lower Cholesky factor of sigma
    counts: np.ndarray         # [N] training counts
    task_of_class: np.ndarray  # [N]
    T: int
    lam_lw: float              # shrinkage intensity (mean over tasks for protocol B)
    protocol: str
    standardize: Optional[Tuple[np.ndarray, float]] = None   # (mean, std) when protocol A
    sigma_task: Optional[np.ndarray] = None                  # [T, d, d] per-task Ledoit-Wolf covariances

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_features(cls, X: np.ndarray, y: np.ndarray, task_of_class: np.ndarray,
                      protocol: str = "B", n_classes: Optional[int] = None, chunk: int = CHUNK) -> "ClassStats":
        """Streaming construction: X may be float32 and large (DomainNet: 400k x 768); every accumulation is in
        float64 over row chunks, and no full-size copy of X is made."""
        from scipy import sparse
        X = np.asarray(X); y = np.asarray(y, np.int64); toc = np.asarray(task_of_class, np.int64)
        n, d = X.shape
        N = int(n_classes or toc.size); T = int(toc.max()) + 1
        std = None
        if protocol == "A":   # per-dimension centring, one scalar scale (app:gamma-computation step 1)
            m = np.zeros(d)
            for s0 in range(0, n, chunk):
                m += X[s0:s0 + chunk].astype(np.float64).sum(axis=0)
            m /= n
            ss = 0.0
            for s0 in range(0, n, chunk):
                Z = X[s0:s0 + chunk].astype(np.float64) - m; ss += float(np.sum(Z * Z))
            s_ = float(np.sqrt(ss / (n * d)))
            std = (m, s_)

        def block(s0):
            Z = X[s0:s0 + chunk].astype(np.float64)
            return (Z - std[0]) / std[1] if std is not None else Z

        # class means
        sums = np.zeros((N, d)); counts = np.bincount(y, minlength=N).astype(np.int64)
        for s0 in range(0, n, chunk):
            yb = y[s0:s0 + chunk]
            onehot = sparse.csr_matrix((np.ones(len(yb)), (yb, np.arange(len(yb)))), shape=(N, len(yb)))
            sums += onehot @ block(s0)
        mu = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], 0.0)
        # per-task second moments of the centred rows and their quartic sums
        S_t = np.zeros((T, d, d)); q_t = np.zeros(T); n_t = np.zeros(T, np.int64)
        for s0 in range(0, n, chunk):
            yb = y[s0:s0 + chunk]; Rb = block(s0) - mu[yb]; tb = toc[yb]
            for t in np.unique(tb):
                sel = tb == t; Rt = Rb[sel]
                S_t[t] += Rt.T @ Rt; q_t[t] += float(np.sum(np.sum(Rt * Rt, axis=1) ** 2)); n_t[t] += int(sel.sum())
        sigmas, lams = [], []
        for t in range(T):
            if n_t[t] > 1:
                s_t, l_t = _lw_from_moments(S_t[t] / n_t[t], q_t[t], int(n_t[t])); sigmas.append(s_t); lams.append(l_t)
            else:
                sigmas.append(None); lams.append(np.nan)
        if protocol == "A":   # app:gamma-computation: one pooled estimate over all centred rows
            sigma, lam = _lw_from_moments(S_t.sum(0) / n, float(q_t.sum()), n)
        else:                 # Paper B app:experiments: per task, averaged over tasks
            sigma = np.mean([s_ for s_ in sigmas if s_ is not None], axis=0); lam = float(np.nanmean(lams))
        sigma_task = np.stack([s_ if s_ is not None else sigma for s_ in sigmas])
        chol = np.linalg.cholesky(sigma)
        return cls(mu=mu, sigma=sigma, chol=chol, counts=counts, task_of_class=toc, T=T, lam_lw=float(lam),
                   protocol=protocol, standardize=std, sigma_task=sigma_task)

    # ---- Mahalanobis algebra via Cholesky ------------------------------------
    def whiten(self, V: np.ndarray) -> np.ndarray:
        """L^{-1} V^T, so that ||v||^2_{Sigma^{-1}} = ||whiten(v)||^2."""
        V = np.atleast_2d(V)
        return linalg.solve_triangular(self.chol, V.T, lower=True).T

    def sigma_inv_apply(self, V: np.ndarray) -> np.ndarray:
        """Sigma^{-1} v for rows v, via two triangular solves."""
        V = np.atleast_2d(V)
        z = linalg.solve_triangular(self.chol, V.T, lower=True)
        return linalg.solve_triangular(self.chol.T, z, lower=False).T

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply protocol A's standardization to new features (for B the features are returned as given, no copy)."""
        if self.standardize is None:
            return np.asarray(X)
        m, s = self.standardize
        return (np.asarray(X, np.float64) - m) / s

    # ---- geometry -----------------------------------------------------------
    @property
    def centroids(self) -> np.ndarray:
        return np.stack([self.mu[self.task_of_class == t].mean(axis=0) for t in range(self.T)])

    def gamma_matrix(self) -> np.ndarray:
        """gamma_ij = ||mu_bar_i - mu_bar_j||^2_{Sigma^{-1}}, [T, T], zero diagonal."""
        W = self.whiten(self.centroids)
        sq = (W ** 2).sum(1)
        G = sq[:, None] + sq[None, :] - 2 * W @ W.T
        np.fill_diagonal(G, 0.0)
        return np.maximum(G, 0.0)

    def gamma_min(self) -> Tuple[float, Tuple[int, int]]:
        G = self.gamma_matrix()
        iu = np.triu_indices(self.T, 1)
        k = int(np.argmin(G[iu]))
        return float(G[iu][k]), (int(iu[0][k]), int(iu[1][k]))

    def gamma_matrix_pairwise(self) -> np.ndarray:
        """Paper A eq:gamma as written in the main text: gamma_ij = delta^T ((Sigma^(i) + Sigma^(j))/2)^{-1} delta with the
        per-task (shrunk) within-class covariances; app:gamma-computation uses one pooled covariance instead, which
        is ``gamma_matrix``.  Both are reported; the two coincide under the shared-covariance assumption."""
        cents = self.centroids; G = np.zeros((self.T, self.T))
        for i in range(self.T):
            for j in range(i + 1, self.T):
                L = np.linalg.cholesky(0.5 * (self.sigma_task[i] + self.sigma_task[j]))
                z = linalg.solve_triangular(L, cents[i] - cents[j], lower=True)
                G[i, j] = G[j, i] = float(z @ z)
        return G

    def gamma_min_pairwise(self) -> float:
        G = self.gamma_matrix_pairwise(); iu = np.triu_indices(self.T, 1)
        return float(G[iu].min())

    def gamma_min_unshrunk(self, X: np.ndarray, y: np.ndarray, chunk: int = CHUNK) -> Optional[float]:
        """Paper A app:gamma-computation cross-check: gamma_min with the unshrunk pooled within-class covariance
        (None when it is singular).  Call with the same training features the statistics were built from."""
        y = np.asarray(y, np.int64); d = self.mu.shape[1]; S = np.zeros((d, d)); n = len(y)
        for s0 in range(0, n, chunk):
            R = self.transform(X[s0:s0 + chunk]).astype(np.float64) - self.mu[y[s0:s0 + chunk]]
            S += R.T @ R
        S /= n
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return None
        Wc = linalg.solve_triangular(L, self.centroids.T, lower=True).T
        sq = (Wc ** 2).sum(1); G = sq[:, None] + sq[None, :] - 2 * Wc @ Wc.T
        iu = np.triu_indices(self.T, 1)
        return float(np.maximum(G[iu], 0).min())

    def D_matrix(self) -> np.ndarray:
        """D_kl = ||mu_k - mu_l||^2_{Sigma^{-1}}, [N, N]."""
        W = self.whiten(self.mu)
        sq = (W ** 2).sum(1)
        D = sq[:, None] + sq[None, :] - 2 * W @ W.T
        np.fill_diagonal(D, 0.0)
        return np.maximum(D, 0.0)

    def kappa_pairs(self) -> List[dict]:
        """kappa_{ij,kl} = sqrt(D_kl / gamma_ij) for every cross-task class pair (Paper B eq. kappa)."""
        G, D = self.gamma_matrix(), self.D_matrix()
        toc = self.task_of_class
        out = []
        for i in range(self.T):
            for j in range(self.T):
                if i == j or G[i, j] <= 0:
                    continue
                for k in np.flatnonzero(toc == i):
                    for l in np.flatnonzero(toc == j):
                        out.append(dict(i=i, j=j, k=int(k), l=int(l), gamma=float(G[i, j]), D=float(D[k, l]),
                                        kappa=float(np.sqrt(D[k, l] / G[i, j]))))
        return out

    def kappa_min(self) -> float:
        return float(min(p["kappa"] for p in self.kappa_pairs()))

    # ---- Paper A Condition (S): spread ratio, metric mismatch, residual -----------
    def lambda_bar(self, X=None, y=None, chunk: int = CHUNK) -> dict:
        """The isotropic Loewner bound Sigma_bar = lambda_bar I of app:offdiag-proof Step 3.
        task level: max_t lambda_max(Sigma^(t)) (the submitted definition);
        class level: max_r lambda_max(Sigma_r) (the W5-5 correction), computed from the raw rows when X, y are
        given (largest singular value of each class's centred block, streamed one class at a time)."""
        out = dict(task=float(max(np.linalg.eigvalsh(S).max() for S in self.sigma_task)))
        if X is not None:
            y = np.asarray(y, np.int64); lam_cls = 0.0
            for c in range(self.mu.shape[0]):
                idx = np.flatnonzero(y == c)
                if idx.size < 2:
                    continue
                R = self.transform(X[idx]).astype(np.float64) - self.mu[c]
                lam_cls = max(lam_cls, float(np.linalg.norm(R, 2) ** 2) / idx.size)
            out["class"] = lam_cls
        return out

    def spread_star(self, t: int, lam_bar: float) -> float:
        """max_k ||mu_k - mu^(t)||_* with ||v||_* = ||v|| / sqrt(lam_bar) (the isotropic metric of eq:nu-def)."""
        cls = np.flatnonzero(self.task_of_class == t)
        return float(np.sqrt(((self.mu[cls] - self.centroids[t]) ** 2).sum(1)).max() / np.sqrt(lam_bar))

    def nu_ij(self, i: int, j: int, lam_bar: Optional[float] = None, G: Optional[np.ndarray] = None) -> float:
        """Paper A eq:nu-def: nu_ij = 2 (max_k ||mu_k - mu^(i)||_* + max_l ||mu_l - mu^(j)||_*) / sqrt(gamma_ij),
        spreads in the isotropic ||.||_* metric of Sigma_bar (task-level lambda_bar unless given)."""
        G = self.gamma_matrix() if G is None else G
        lb = self.lambda_bar()["task"] if lam_bar is None else lam_bar
        return float(2.0 * (self.spread_star(i, lb) + self.spread_star(j, lb)) / np.sqrt(G[i, j]))

    def nu_ij_pooled(self, i: int, j: int) -> float:
        """The same ratio with spreads measured in the pooled Sigma^{-1} metric (not the paper's definition; kept
        because it is the metric gamma_ij itself uses, so the two bracket the mismatch)."""
        G = self.gamma_matrix(); cents = self.centroids
        def spread(t):
            cls = np.flatnonzero(self.task_of_class == t)
            return float(np.sqrt((self.whiten(self.mu[cls] - cents[t]) ** 2).sum(1)).max())
        return float((2 * spread(i) + 2 * spread(j)) / np.sqrt(G[i, j]))

    def condition_S_pair_metric(self, i: int, j: int, X=None, y=None) -> dict:
        """Step 3 of Lemma 3.5 redone in the task-pair metric M_ij = (Sigma^(i) + Sigma^(j))/2 that defines gamma_ij
        in the main text (docs/THEORY_NOTES.md, section 2): no isotropic detour, hence no delta clause.

          nu_pair     2 (max_k ||mu_k - mu^(i)||_{M} + max_l ||mu_l - mu^(j)||_{M}) / sqrt(gamma^M_ij)
          kappa_cls   max over classes k of the two tasks of lambda_max(Sigma^(t)^{-1/2} Sigma_k Sigma^(t)^{-1/2}),
                      the class-to-task relative Loewner constant (1 under homoscedastic classes); needs the rows
                      X, y for the per-class covariances (Ledoit-Wolf shrunk); None when they are not given
          exponent    (1 - nu_pair) / (8 kappa_cls), the exponent of the per-class-pair bound
                      P_e^{kl} <= exp(-(1 - nu_pair) gamma_ij / (8 kappa_cls)) when nu_pair <= 1/2
          clause      nu_pair <= 1/2
        """
        M = 0.5 * (self.sigma_task[i] + self.sigma_task[j]); L = np.linalg.cholesky(M)
        cents = self.centroids
        def nrm(v):
            return float(np.linalg.norm(linalg.solve_triangular(L, v, lower=True)))
        g = nrm(cents[i] - cents[j]) ** 2
        S = sum(max(nrm(self.mu[k] - cents[t]) for k in np.flatnonzero(self.task_of_class == t)) for t in (i, j))
        nu = 2.0 * S / np.sqrt(g) if g > 0 else float("inf")
        out = dict(gamma_pair=g, nu_pair=nu, clause_pair=bool(nu <= 0.5), kappa_cls=None, exponent_pair=None)
        if X is not None:
            y = np.asarray(y, np.int64); kap = 0.0
            for t in (i, j):
                Lt = np.linalg.cholesky(self.sigma_task[t])
                for k in np.flatnonzero(self.task_of_class == t):
                    idx = np.flatnonzero(y == k)
                    if idx.size < 3:
                        continue
                    Rk = self.transform(X[idx]).astype(np.float64) - self.mu[k]
                    Sk, _ = ledoit_wolf_shrinkage(Rk)
                    Z = linalg.solve_triangular(Lt, Sk, lower=True); Z = linalg.solve_triangular(Lt, Z.T, lower=True)
                    kap = max(kap, float(np.linalg.eigvalsh(0.5 * (Z + Z.T)).max()))
            out["kappa_cls"] = kap
            out["exponent_pair"] = float((1.0 - nu) / (8.0 * kap)) if (kap > 0 and nu <= 0.5) else 0.0
        return out

    def condition_S(self, X=None, y=None) -> dict:
        """tab:residual-check row at the worst task pair, with the two clauses of the camera-ready Condition (S):

          nu_ij <= 1/2                                   (within-task spread ratio, eq:nu-def, ||.||_* metric)
          delta_ij <= (1/2 - nu_ij) gamma_ij             (metric mismatch, eq:mismatch-def)

        delta_ij = gamma_ij - g_ij with g_ij = ||mu^(i) - mu^(j)||^2 / lambda_bar the centroid separation in the
        isotropic metric; xi_ij <= delta_ij + gamma_ij nu_ij is the residual of eq:xi-def and a_ij = xi_ij/gamma_ij
        the exponent fraction of Lemma 3.5 ((1 - a_ij)/8).  Reported for the task-level lambda_bar and, when the
        rows are given, for the class-level one; ``clause_holds`` is the camera-ready condition, ``rhs`` the older
        trace criterion of eq:gamma-vs-spread, and ``frac_pairs_clause`` the fraction of ordered task pairs where
        both clauses hold.  Also reported, for the pooled-metric route that needs no mismatch clause: nu_pooled
        (spreads in the Sigma^{-1} metric), kappa_sigma = max_t lambda_max(Sigma^{-1/2} Sigma^(t) Sigma^{-1/2}) (the
        relative Loewner constant, 1 under shared covariance), and clause_holds_pooled = sqrt(kappa_sigma) nu_pooled
        <= 1/2."""
        gmin, (i, j) = self.gamma_min()
        G = self.gamma_matrix(); cents = self.centroids
        C = int((self.task_of_class == i).sum())
        sig_min = float(np.linalg.eigvalsh(self.sigma).min())

        def tr_between(t):
            cls = np.flatnonzero(self.task_of_class == t)
            d = self.mu[cls] - cents[t]
            return float((d ** 2).sum() / len(cls))

        rhs = 32.0 * C * (tr_between(i) + tr_between(j)) / sig_min
        # relative Loewner constant of the task covariances against the pooled one: Sigma^(t) <= kappa_sigma Sigma
        # (kappa_sigma = 1 under exactly shared covariance); the dimensionless replacement for the isotropic lambda_bar
        kap = 0.0
        for S in self.sigma_task:
            Z = linalg.solve_triangular(self.chol, S, lower=True)
            Z = linalg.solve_triangular(self.chol, Z.T, lower=True)
            kap = max(kap, float(np.linalg.eigvalsh(0.5 * (Z + Z.T)).max()))
        nu_pooled = self.nu_ij_pooled(i, j)
        out = dict(gamma_min=gmin, pair=(i, j), rhs=rhs, sigma_min=sig_min, nu_pooled=nu_pooled, kappa_sigma=kap,
                   nu_pooled_scaled=float(np.sqrt(kap) * nu_pooled), clause_holds_pooled=bool(np.sqrt(kap) * nu_pooled <= 0.5))
        out.update(self.condition_S_pair_metric(i, j, X, y))
        lbs = self.lambda_bar(X, y)
        for level, lb in lbs.items():
            nu = self.nu_ij(i, j, lb, G)
            g = float(((cents[i] - cents[j]) ** 2).sum() / lb)
            delta = gmin - g                      # eq:mismatch-def, signed (negative means the isotropic metric separates the centroids more)
            xi_ub = delta + gmin * nu             # eq:xi-def upper bound on the residual
            holds = bool(nu <= 0.5 and delta <= (0.5 - nu) * gmin)
            # fraction of ordered task pairs on which both clauses hold
            ok = 0; tot = 0
            for a in range(self.T):
                for b in range(self.T):
                    if a == b:
                        continue
                    nu_ab = self.nu_ij(a, b, lb, G); d_ab = G[a, b] - float(((cents[a] - cents[b]) ** 2).sum() / lb)
                    ok += int(nu_ab <= 0.5 and d_ab <= (0.5 - nu_ab) * G[a, b]); tot += 1
            tag = "" if level == "task" else "_class"
            out.update({f"nu{tag}": nu, f"delta{tag}": delta, f"delta_over_gamma{tag}": delta / gmin if gmin > 0 else float("nan"),
                        f"a_ij{tag}": xi_ub / gmin if gmin > 0 else float("nan"), f"lambda_bar{tag}": lb,
                        f"clause_holds{tag}": holds, f"frac_pairs_clause{tag}": ok / tot if tot else float("nan")})
        return out

    # ---- Paper B bias direction (per pair and the paper's averaged statistic) ---
    def bias_direction(self, W: np.ndarray, i: int, j: int) -> dict:
        """Theorem 4.1 diagnostics for trained weights W [N, d] on the ordered block (i, j), in the Sigma^{-1} inner
        product (app:experiments metric iii).  Residual of the class pair (k in i, l in j):
        r_kl = (w_k - w_l) - Sigma^{-1}(mu_k - mu_l); prediction Sigma^{-1}(mu_bar_j - mu_bar_i).

        Returns the paper's C^2-averaged statistic (an identity when the within-task weights sum to zero, see
        core.diagnostic) and the per-pair statistics.  Computed from the C_i x C_j Gram matrices, never from a
        [C_i, C_j, d] tensor (DomainNet has C = 345)."""
        toc = self.task_of_class
        ki, kj = np.flatnonzero(toc == i), np.flatnonzero(toc == j)
        pred_vec = self.sigma_inv_apply(self.centroids[j] - self.centroids[i])[0]
        W = np.asarray(W, np.float64)
        bayes = self.sigma_inv_apply(self.mu)                 # Sigma^{-1} mu_k rows
        A = self.whiten(W[ki] - bayes[ki]); B = self.whiten(W[kj] - bayes[kj]); Wp = self.whiten(pred_vec)[0]
        npred = float(np.linalg.norm(Wp))
        dots = (A @ Wp)[:, None] - (B @ Wp)[None, :]                                        # <r_kl, pred>
        mags = np.sqrt(np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T, 0.0))
        pair_cos = dots / np.maximum(mags * npred, 1e-300)
        Wa = A.mean(0) - B.mean(0)                                                          # whitened C^2 average
        gauge = [float(np.linalg.norm(W[np.flatnonzero(toc == t)].sum(0)) / np.linalg.norm(W[np.flatnonzero(toc == t)], axis=1).mean()) for t in (i, j)]
        return dict(
            paper_averaged_cos=float(Wa @ Wp / max(np.linalg.norm(Wa) * npred, 1e-300)),
            paper_averaged_rel=float(np.linalg.norm(Wa) / max(npred, 1e-300)),
            pair_cos_mean=float(pair_cos.mean()), pair_cos_min=float(pair_cos.min()),
            pair_rel_mean=float((mags / npred).mean()),
            cv=float(mags.std() / mags.mean()) if mags.mean() > 0 else float("nan"),
            gauge_residual=float(max(gauge)),
        )

    def gauge_residuals(self, W: np.ndarray) -> np.ndarray:
        """Per task: ||sum_{r in task} w_r|| / mean_r ||w_r||.  Zero at any interior minimizer of the within-task
        CE with decay on the weights (the summed first-order conditions give lambda sum_r w_r = 0, Lemma 3.2); a
        nonzero value marks a head off the gauge (unconverged, trained by an optimizer whose path does not preserve
        the sum, or not a softmax-CE head), and is the one-line screen that must precede any bias-direction cosine."""
        W = np.asarray(W, np.float64); out = np.zeros(self.T)
        for t in range(self.T):
            k = np.flatnonzero(self.task_of_class == t)
            out[t] = float(np.linalg.norm(W[k].sum(0)) / max(np.linalg.norm(W[k], axis=1).mean(), 1e-300))
        return out

    def head_scales(self, W: np.ndarray) -> np.ndarray:
        """Per-task least-squares scale c_t of the trained within-task weights on the CF-optimal directions
        Sigma^{-1}(mu_r - mu_bar_t) (Euclidean inner product over the task's C rows).  c_t = 1 at the population
        head of Theorem 3.3, which is the scale Definition 7.1 assumes; a weakly regularized finite-sample
        minimizer on separable data sits well below it (its margins are set by log(1/lambda), not by the
        Mahalanobis distances), an under-trained AdamW head below that."""
        W = np.asarray(W, np.float64); toc = self.task_of_class
        target = self.sigma_inv_apply(self.mu - self.centroids[toc])
        cs = np.full(self.T, np.nan)
        for t in range(self.T):
            k = np.flatnonzero(toc == t)
            A = target[k].ravel(); den = float(A @ A)
            if den > 0:
                cs[t] = float(A @ W[k].ravel()) / den
        return cs

    def head_scale(self, W: np.ndarray) -> float:
        cs = self.head_scales(W)
        return float(np.nanmean(cs)) if np.isfinite(cs).any() else float("nan")

    def bias_direction_all(self, W: np.ndarray) -> dict:
        rows = [self.bias_direction(W, i, j) for i in range(self.T) for j in range(self.T) if i != j]
        keys = rows[0].keys()
        return {k: float(np.mean([r[k] for r in rows])) for k in keys} | {"pair_cos_min": float(min(r["pair_cos_min"] for r in rows)), "n_blocks": len(rows)}
