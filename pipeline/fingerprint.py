"""pipeline.fingerprint -- which estimator convention reproduces the printed gamma_min, nu and RHS?

  python replicate.py fingerprint [--backbones ...] [--datasets ...] [--seeds 0]

Feature caches only, minutes on CPU, no training. For every cached Paper A configuration the module computes
gamma_min under a grid of estimator conventions (shrinkage, ridge, standardization, normalization, total
covariance, subsampled covariance, pair-average covariance, diagonal and Euclidean metrics), under several
class-ordering generators, and nu / RHS under a grid of formula conventions (numerator statistic, normalizer,
factor, pair set, denominator), and compares each cell with the values printed in the accepted paper
(`registry.PRINTED`). A convention is a candidate only if ONE setting fits all configurations at once; the
report ranks conventions by their worst-case log-ratio to the printed value across configurations and lists
the per-backbone mediators (eigenvalue scale, Ledoit-Wolf intensity, feature norm) that decide the direction
of each convention's effect. Writes `results/FINGERPRINT.md` and `.json`. The decision tree that orders
these tests is `FINGERPRINT_DECISION_TREE.md` in the owner's zone.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import linalg

from pipeline import registry as R
from pipeline.features import cache_exists, load_pair
from pipeline.stats import _lw_from_moments

RIDGE_ABS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
RIDGE_REL = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)

PRINTED_NU = {("vit_in21k", "cifar100"): 0.51, ("vit_in21k", "imagenet_r"): 0.73, ("vit_in21k", "cub200"): 0.42, ("vit_in21k", "omnibenchmark"): 0.49,
              ("dinov2_b14", "cifar100"): 0.13, ("dinov2_b14", "imagenet_r"): 0.15, ("dinov2_b14", "cub200"): 0.12, ("dinov2_b14", "omnibenchmark"): 0.13,
              ("clip_b16", "cifar100"): 0.23, ("clip_b16", "imagenet_r"): 0.26, ("clip_b16", "cub200"): 0.17, ("clip_b16", "omnibenchmark"): 0.19}
PRINTED_RHS = {("vit_in21k", "cifar100"): 8.4, ("vit_in21k", "imagenet_r"): 7.6, ("vit_in21k", "cub200"): 10.2, ("vit_in21k", "omnibenchmark"): 9.4,
               ("dinov2_b14", "cifar100"): 12.2, ("dinov2_b14", "imagenet_r"): 11.6, ("dinov2_b14", "cub200"): 12.6, ("dinov2_b14", "omnibenchmark"): 11.8,
               ("clip_b16", "cifar100"): 10.4, ("clip_b16", "imagenet_r"): 9.8, ("clip_b16", "cub200"): 10.8, ("clip_b16", "omnibenchmark"): 10.2}


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def _row_layernorm(Z: np.ndarray) -> np.ndarray:
    """Per-row layer normalization: centre and scale each row across its coordinates (the printed Step 1, read literally)."""
    mu = Z.mean(axis=1, keepdims=True); sd = Z.std(axis=1, keepdims=True)
    return (Z - mu) / np.maximum(sd, 1e-12)


class Moments:
    """Class sums, pooled within-class scatter, total scatter, per-class top eigenvalue; optionally on
    L2-normalized rows and on a per-class subsample. One streaming pass per variant of the rows."""

    def __init__(self, X, y, n_classes: int, chunk: int = 32768, normalize: bool = False, subsample: Optional[int] = None, seed_purpose: str = "fp",
                 rowln: bool = False):
        X = np.asarray(X); y = np.asarray(y, np.int64); n, d = X.shape
        self.rowln = rowln
        if subsample:
            g = R.rng(seed_purpose); keep = np.zeros(n, bool)
            for c in range(n_classes):
                idx = np.flatnonzero(y == c)
                if idx.size:
                    keep[g.choice(idx, min(subsample, idx.size), replace=False)] = True
            X, y = X[keep], y[keep]; n = X.shape[0]
        self.n, self.d, self.N = n, d, n_classes
        sums = np.zeros((n_classes, d)); counts = np.zeros(n_classes); S2 = np.zeros((d, d))
        for i in range(0, n, chunk):
            Z = X[i:i + chunk].astype(np.float64)
            if normalize:
                Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
            if rowln:
                Z = _row_layernorm(Z)
            yb = y[i:i + chunk]
            np.add.at(sums, yb, Z); counts += np.bincount(yb, minlength=n_classes)
            S2 += Z.T @ Z
        self.counts = counts; self.mu = sums / np.maximum(counts, 1)[:, None]
        self.gmean = sums.sum(0) / n
        self.S_tot = S2 / n - np.outer(self.gmean, self.gmean)
        self.S_w = (S2 - (self.mu * counts[:, None]).T @ self.mu) / n
        # quartic term for Ledoit-Wolf on the within-class scatter needs a second pass over centred rows
        q = 0.0
        for i in range(0, n, chunk):
            Z = X[i:i + chunk].astype(np.float64)
            if normalize:
                Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
            if rowln:
                Z = _row_layernorm(Z)
            Rz = Z - self.mu[y[i:i + chunk]]
            q += float(np.sum(np.sum(Rz * Rz, axis=1) ** 2))
        self.quart_w = q
        self.X, self.y, self.normalize = X, y, normalize

    def lw(self):
        return _lw_from_moments(self.S_w, self.quart_w, self.n)

    def lw_scaled(self, scale: np.ndarray, chunk: int = 32768):
        """Ledoit-Wolf on the within-class scatter of rows divided per dimension by ``scale`` (exact intensity: the
        quartic term is recomputed on the scaled centred rows). Returns (S_lw_scaled, rho)."""
        scale = np.asarray(scale, np.float64); q = 0.0
        for i in range(0, self.n, chunk):
            Z = self.X[i:i + chunk].astype(np.float64)
            if self.normalize:
                Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
            if self.rowln:
                Z = _row_layernorm(Z)
            Rz = (Z - self.mu[self.y[i:i + chunk]]) / scale
            q += float(np.sum(np.sum(Rz * Rz, axis=1) ** 2))
        return _lw_from_moments(self.S_w / np.outer(scale, scale), q, self.n)

    def class_lambda_max(self, classes=None, chunk: int = 32768) -> np.ndarray:
        out = np.zeros(self.N)
        for c in (range(self.N) if classes is None else classes):
            idx = np.flatnonzero(self.y == c)
            if idx.size < 2:
                continue
            Z = self.X[idx].astype(np.float64)
            if self.normalize:
                Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
            if self.rowln:
                Z = _row_layernorm(Z)
            Z = Z - self.mu[c]
            out[c] = float(np.linalg.norm(Z, 2) ** 2) / idx.size
        return out

    def task_scatter(self, toc: np.ndarray, t: int):
        """Pooled within-class scatter of the classes of task t (unshrunk), from a pass over its rows."""
        cls = np.flatnonzero(toc == t); m = np.isin(self.y, cls); Z = self.X[m].astype(np.float64)
        if self.normalize:
            Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        if self.rowln:
            Z = _row_layernorm(Z)
        Rz = Z - self.mu[self.y[m]]
        return Rz.T @ Rz / max(Rz.shape[0], 1), float(np.sum(np.sum(Rz * Rz, axis=1) ** 2)), Rz.shape[0]

    def class_scatter_lw(self, c: int):
        """Ledoit-Wolf covariance of one class from its own rows (the literal per-class Sigma_r of the definition)."""
        idx = np.flatnonzero(self.y == c); Z = self.X[idx].astype(np.float64)
        if self.normalize:
            Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        if self.rowln:
            Z = _row_layernorm(Z)
        Rz = Z - self.mu[c]
        return _lw_from_moments(Rz.T @ Rz / max(Rz.shape[0], 1), float(np.sum(np.sum(Rz * Rz, axis=1) ** 2)), Rz.shape[0])

    def task_rows(self, toc: np.ndarray, t: int):
        """Rows of task t (after the row maps), with their labels."""
        cls = np.flatnonzero(toc == t); m = np.isin(self.y, cls); Z = self.X[m].astype(np.float64)
        if self.normalize:
            Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        if self.rowln:
            Z = _row_layernorm(Z)
        return Z, self.y[m]


# ---------------------------------------------------------------------------
# gamma under estimator conventions
# ---------------------------------------------------------------------------

def _gamma_matrix(cents: np.ndarray, Sinv_apply) -> np.ndarray:
    T = cents.shape[0]; G = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            dmu = cents[i] - cents[j]; G[i, j] = G[j, i] = float(dmu @ Sinv_apply(dmu))
    return G


def _solver(S: np.ndarray):
    L = np.linalg.cholesky(S)
    return lambda v: linalg.cho_solve((L, True), v)


def gamma_variants(M: Moments, Mn: Moments, Ms: Moments, toc: np.ndarray, Ml: Optional[Moments] = None, order: Optional[np.ndarray] = None,
                   dataset: Optional[str] = None) -> Dict[str, dict]:
    """gamma_min (and its pair) under each estimator convention for the ordering ``toc``.  ``Ml`` carries the
    row-layernormed moments (convention "rowln_*"); ``order`` and ``dataset`` enable the task-count variants."""
    T = int(toc.max()) + 1
    cents = np.array([M.mu[toc == t].mean(0) for t in range(T)])
    out = {}
    S_lw, rho = M.lw()
    tr_d = float(np.trace(M.S_w) / M.d)

    def add(name, G, note=""):
        iu = np.triu_indices(T, 1); k = int(np.argmin(G[iu])); i, j = iu[0][k], iu[1][k]
        out[name] = dict(gamma_min=float(G[i, j]), pair=[int(i), int(j)], gamma_mean=float(G[iu].mean()), G=G.tolist(), note=note)

    add("lw_pooled (ours)", _gamma_matrix(cents, _solver(S_lw)), f"Ledoit-Wolf rho {rho:.4f}")
    add("sample_pooled", _gamma_matrix(cents, _solver(M.S_w + 1e-10 * tr_d * np.eye(M.d))), "unshrunk within-class scatter")
    for eps in RIDGE_ABS:
        add(f"ridge_abs eps={eps:g}", _gamma_matrix(cents, _solver(M.S_w + eps * np.eye(M.d))), "S_w + eps I, eps absolute")
    for r in RIDGE_REL:
        add(f"ridge_rel r={r:g}", _gamma_matrix(cents, _solver(M.S_w + r * tr_d * np.eye(M.d))), "S_w + r (tr S/d) I")
    # Ledoit-Wolf on the total (not class-centred) covariance, and the unshrunk total covariance
    S_tot_lw, rho_t = _lw_from_moments(M.S_tot, M.quart_w, M.n)   # quartic of centred rows is the within one; intensity only approximate
    add("lw_total", _gamma_matrix(cents, _solver(S_tot_lw)), f"total covariance (between-class scatter included), rho about {rho_t:.4f}")
    add("sample_total", _gamma_matrix(cents, _solver(M.S_tot + 1e-10 * tr_d * np.eye(M.d))), "unshrunk total covariance")
    # per-dimension z-scoring (by within-class std) before Ledoit-Wolf
    dsd = np.sqrt(np.maximum(np.diag(M.S_w), 1e-300)); Sz = M.S_w / np.outer(dsd, dsd)
    Sz_lw, rho_z = _lw_from_moments(Sz, M.quart_w / (dsd.mean() ** 4), M.n)
    cz = cents / dsd
    add("zscore_lw", _gamma_matrix(cz, _solver(Sz_lw)), f"dims z-scored by within-class std, then LW (rho about {rho_z:.4f})")
    add("zscore_sample", _gamma_matrix(cz, _solver(Sz + 1e-10 * np.eye(M.d))), "dims z-scored, unshrunk")
    # the rebuttal-era re-measurement pipeline's convention (artifact_pipeline.py, 2026-08-01): per-dimension
    # standardization by the TOTAL std of the training features, class-centred pooled Ledoit-Wolf, + 1e-6 I
    tsd = np.sqrt(np.maximum(np.diag(M.S_tot), 1e-300))
    St_lw, rho_t2 = M.lw_scaled(tsd); ct = cents / tsd
    add("zscore_total_lw", _gamma_matrix(ct, _solver(St_lw)), f"dims z-scored by TOTAL std (ancestor pipeline), then LW (rho {rho_t2:.4f})")
    add("ancestor_aug2026", _gamma_matrix(ct, _solver(St_lw + 1e-6 * np.eye(M.d))), "zscore_total_lw + 1e-6 I ridge: artifact_pipeline.py of 2026-08-01 exactly")
    # diagonal covariance and Euclidean
    add("diag", _gamma_matrix(cents, lambda v: v / np.maximum(np.diag(M.S_w), 1e-300)), "independent dims (variance only)")
    add("euclid", _gamma_matrix(cents, lambda v: v), "||delta mu||^2, no covariance (scale-dependent)")
    add("euclid_over_mean_var", _gamma_matrix(cents, lambda v: v / tr_d), "||delta mu||^2 / (tr S_w / d)")
    add("euclid_over_lambda_max", _gamma_matrix(cents, lambda v: v / float(np.linalg.eigvalsh(S_lw).max())), "||delta mu||^2 / lambda_max (the isotropic metric)")
    # L2-normalized rows
    if Mn is not None:
        Sn_lw, rho_n = Mn.lw(); cn = np.array([Mn.mu[toc == t].mean(0) for t in range(T)])
        add("l2norm_lw", _gamma_matrix(cn, _solver(Sn_lw)), f"rows L2-normalized before statistics (rho {rho_n:.4f})")
        add("l2norm_sample", _gamma_matrix(cn, _solver(Mn.S_w + 1e-10 * np.trace(Mn.S_w) / Mn.d * np.eye(Mn.d))), "rows L2-normalized, unshrunk")
    # covariance from a per-class subsample (means from the subsample too)
    if Ms is not None:
        Ss_lw, rho_s = Ms.lw(); cs = np.array([Ms.mu[toc == t].mean(0) for t in range(T)])
        add("subsample_lw", _gamma_matrix(cs, _solver(Ss_lw)), f"{Ms.n} rows, LW rho {rho_s:.4f}")
        add("subsample_sample", _gamma_matrix(cs, _solver(Ms.S_w + 1e-6 * np.trace(Ms.S_w) / Ms.d * np.eye(Ms.d))), f"{Ms.n} rows, unshrunk (near-singular; tiny ridge)")
    # eq. 1: pair-average task covariances, each Ledoit-Wolf shrunk
    St = {}
    for t in range(T):
        S, q, n = M.task_scatter(toc, t); St[t] = _lw_from_moments(S, q, n)[0]
    G = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            dmu = cents[i] - cents[j]; G[i, j] = G[j, i] = float(dmu @ _solver(0.5 * (St[i] + St[j]))(dmu))
    add("eq1_pair_lw", G, "main-text eq. 1: ((Sigma^(i)+Sigma^(j))/2)^-1, task covariances LW-shrunk")
    # eq. 1 with the literal per-class Sigma_r (each class Ledoit-Wolf shrunk on its own rows, averaged per task):
    # heavy shrinkage when classes have few rows (CUB-200: about 30 per class)
    Sc = {}
    rows_per_class = []
    for t in range(T):
        cls = np.flatnonzero(toc == t); mats = []
        for c in cls:
            if int(M.counts[c]) >= 2:
                Sm, rho_c = M.class_scatter_lw(int(c)); mats.append(Sm); rows_per_class.append(int(M.counts[c]))
        Sc[t] = np.mean(mats, axis=0)
    G = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            dmu = cents[i] - cents[j]; G[i, j] = G[j, i] = float(dmu @ _solver(0.5 * (Sc[i] + Sc[j]))(dmu))
    add("eq1_pair_classlw", G, f"eq. 1 with per-class LW covariances averaged per task (median {int(np.median(rows_per_class)) if rows_per_class else 0} rows per class)")
    # eq. 1 with unshrunk task covariances plus a relative ridge (singular when a task has fewer rows than dims)
    G = np.zeros((T, T)); St_raw = {t: M.task_scatter(toc, t)[0] for t in range(T)}
    for i in range(T):
        for j in range(i + 1, T):
            dmu = cents[i] - cents[j]; Sij = 0.5 * (St_raw[i] + St_raw[j])
            G[i, j] = G[j, i] = float(dmu @ _solver(Sij + 1e-3 * np.trace(Sij) / M.d * np.eye(M.d))(dmu))
    add("eq1_pair_sample_ridge1e-3", G, "eq. 1 with unshrunk task covariances + 1e-3 (tr/d) ridge")
    # per-task standardization (the fleet's Q4 variant): each task's rows divided by that task's own scalar std
    # ("scalar") or per-dimension std ("zscore"), no centring, and the two tasks' standardized statistics combined
    # as eq. 1 combines them, without undoing the per-task scaling (the pipeline accident this variant tests for)
    for mode in ("scalar", "zscore"):
        Sst, cst = {}, {}
        for t in range(T):
            Z, yt = M.task_rows(toc, t)
            sd = np.full(M.d, float(Z.std())) if mode == "scalar" else np.maximum(Z.std(0), 1e-12)
            Zs = Z / sd; uy = np.unique(yt); mus = np.array([Zs[yt == c].mean(0) for c in uy]); Rz = Zs - mus[np.searchsorted(uy, yt)]
            Sst[t] = _lw_from_moments(Rz.T @ Rz / max(Rz.shape[0], 1), float(np.sum(np.sum(Rz * Rz, axis=1) ** 2)), Rz.shape[0])[0]
            cst[t] = cents[t] / sd
        G = np.zeros((T, T))
        for i in range(T):
            for j in range(i + 1, T):
                dmu = cst[i] - cst[j]; G[i, j] = G[j, i] = float(dmu @ _solver(0.5 * (Sst[i] + Sst[j]))(dmu))
        add(f"pertask_std_{mode}_eq1", G, f"per-task {mode} standardization (no centring) before LW task covariances; eq. 1 combination in mixed scales")
    # per-row layer normalization before the pooled statistics (the printed Step 1 read literally)
    if Ml is not None:
        Sl_lw, rho_l = Ml.lw(); cl = np.array([Ml.mu[toc == t].mean(0) for t in range(T)])
        add("rowln_lw", _gamma_matrix(cl, _solver(Sl_lw)), f"rows layer-normalized (centre and scale across dims) before statistics (rho {rho_l:.4f})")
        add("rowln_sample", _gamma_matrix(cl, _solver(Ml.S_w + 1e-10 * np.trace(Ml.S_w) / Ml.d * np.eye(Ml.d))), "rows layer-normalized, unshrunk")
    # task-count variants: the same ordering cut into 5 or 20 tasks (the CUB question: a different task count moves gamma_min)
    if order is not None and dataset is not None:
        N = len(order)
        for T2 in (5, 20):
            if N % T2 == 0 and T2 != T:
                C2 = N // T2; toc2 = np.empty(N, np.int64); toc2[order] = np.arange(N) // C2
                c2 = np.array([M.mu[toc2 == t].mean(0) for t in range(T2)])
                G2 = _gamma_matrix(c2, _solver(S_lw)); iu = np.triu_indices(T2, 1); k = int(np.argmin(G2[iu]))
                out[f"lw_pooled T={T2}"] = dict(gamma_min=float(G2[iu][k]), pair=[int(iu[0][k]), int(iu[1][k])], gamma_mean=float(G2[iu].mean()), G=None,
                                               note=f"ours with the ordering cut into {T2} tasks of {C2} classes")
    out["_mediators"] = dict(d=int(M.d), n=int(M.n), tr_S_over_d=tr_d, lambda_max_w=float(np.linalg.eigvalsh(S_lw).max()),
                             lambda_min_w=float(np.linalg.eigvalsh(S_lw).min()), lambda_max_unshrunk=float(np.linalg.eigvalsh(M.S_w).max()),
                             lambda_min_unshrunk=float(np.linalg.eigvalsh(M.S_w).min()), lw_rho=float(rho),
                             mean_row_norm=float(np.sqrt(np.trace(M.S_tot) + M.gmean @ M.gmean)), total_lambda_max=float(np.linalg.eigvalsh(M.S_tot).max()))
    return out


def orderings(dataset: str, seed: int) -> Dict[str, np.ndarray]:
    N = R.DATASETS[dataset].n_classes
    out = {"ours (sha256 rng)": R.class_ordering(dataset, seed), "canonical": np.arange(N),
           "np.random.RandomState(seed)": np.random.RandomState(seed).permutation(N),
           "np.random.default_rng(seed)": np.random.default_rng(seed).permutation(N),
           "np.random.seed(seed); shuffle": None}
    rs = np.random.RandomState(seed); a = np.arange(N); rs.shuffle(a); out["np.random.seed(seed); shuffle"] = a
    try:
        import torch
        g = torch.Generator().manual_seed(seed); out["torch.randperm(seed)"] = torch.randperm(N, generator=g).numpy()
    except Exception:
        pass
    try:
        import random
        rr = random.Random(seed); b = list(range(N)); rr.shuffle(b); out["random.Random(seed).shuffle"] = np.array(b)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# nu and RHS under formula conventions
# ---------------------------------------------------------------------------

def nu_rhs_variants(M: Moments, toc: np.ndarray, G_ours: np.ndarray, printed_gamma: Optional[float], class_lmax: np.ndarray) -> Dict[str, dict]:
    T = int(toc.max()) + 1; C = int((toc == 0).sum())
    cents = np.array([M.mu[toc == t].mean(0) for t in range(T)])
    S_lw, _ = M.lw(); ev = np.linalg.eigvalsh(S_lw)
    St = {t: _lw_from_moments(*M.task_scatter(toc, t))[0] for t in range(T)}
    task_lmax = {t: float(np.linalg.eigvalsh(St[t]).max()) for t in range(T)}
    norm_src = {"task_within_max (paper)": lambda i, j: max(task_lmax[i], task_lmax[j]), "task_within_max_all": lambda i, j: max(task_lmax.values()),
                "class_max_all": lambda i, j: float(class_lmax.max()), "pooled_within_max": lambda i, j: float(ev.max()),
                "total_max": lambda i, j: float(np.linalg.eigvalsh(M.S_tot).max()), "mean_eig_pooled (tr S/d)": lambda i, j: float(ev.mean()),
                "min_eig_pooled": lambda i, j: float(ev.min())}
    offs = {t: M.mu[toc == t] - cents[t] for t in range(T)}
    num_src = {"max": lambda t: float(np.sqrt((offs[t] ** 2).sum(1)).max()), "mean": lambda t: float(np.sqrt((offs[t] ** 2).sum(1)).mean()),
               "rms": lambda t: float(np.sqrt((offs[t] ** 2).sum(1).mean()))}
    iu = np.triu_indices(T, 1); k = int(np.argmin(G_ours[iu])); i0, j0 = int(iu[0][k]), int(iu[1][k])
    pairs = {"worst_pair": [(i0, j0)], "mean_over_pairs": [(int(a), int(b)) for a, b in zip(*iu)]}
    out = {}
    for nk, nf in norm_src.items():
        for sk, sf in num_src.items():
            for fac in (2.0, 1.0):
                for pk, pl in pairs.items():
                    for dk in ("gamma_ours", "gamma_printed"):
                        vals = []
                        for (i, j) in pl:
                            lam = nf(i, j); S = (sf(i) + sf(j)) / np.sqrt(lam)
                            g = G_ours[i, j] if dk == "gamma_ours" else (printed_gamma if printed_gamma else np.nan)
                            vals.append(fac * S / np.sqrt(g) if g and g > 0 else np.nan)
                        out[f"nu | norm={nk} | stat={sk} | factor={fac:g} | {pk} | {dk}"] = float(np.nanmean(vals))
    # RHS = 32 C (tr_i + tr_j) / sigma^2 at the worst pair
    tr_full = {t: float((offs[t] ** 2).sum(1).mean()) for t in range(T)}      # (1/C) sum_k ||mu_k - mu^(t)||^2 = tr(Sigma_between^(t))
    for trk, trf in (("tr_full", lambda t: tr_full[t]), ("tr_over_d", lambda t: tr_full[t] / M.d)):
        for sk, sv in (("lambda_min", float(ev.min())), ("lambda_max", float(ev.max())), ("mean_eig", float(ev.mean())),
                       ("lambda_min_unshrunk", float(np.linalg.eigvalsh(M.S_w).min()))):
            for cf in ("32C", "32"):
                mult = 32.0 * C if cf == "32C" else 32.0
                out[f"rhs | {trk} | sigma2={sk} | {cf}"] = float(mult * (trf(i0) + trf(j0)) / max(sv, 1e-300))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(root: Path, out_path: Path, seeds=(0,), backbones=R.PAPER_A_BACKBONES, datasets=R.PAPER_A_DATASETS, subsample: int = 50, log=print) -> dict:
    t0 = time.time(); report = {"configs": {}, "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    printed_g = {k: v[0] for k, v in R.PRINTED["A.tab:gamma-min"].items()}
    for d0, (g0, _) in R.PRINTED["A.tab:random-anchor"].items():
        printed_g[("random_vit", d0)] = g0
    for b in list(backbones) + (["random_vit"] if "random_vit" not in backbones else []):
        for d in datasets:
            if not cache_exists(b, d, root):
                continue
            (Xtr, ytr, _), _, meta = load_pair(b, d, root); N = R.DATASETS[d].n_classes
            M = Moments(Xtr, ytr, N); Mn = Moments(Xtr, ytr, N, normalize=True); Ms = Moments(Xtr, ytr, N, subsample=subsample, seed_purpose=f"fp:{b}:{d}")
            Ml = Moments(Xtr, ytr, N, rowln=True)
            class_lmax = M.class_lambda_max()
            cfg = dict(backbone=b, dataset=d, provenance=meta.get("provenance", "real"), printed_gamma=printed_g.get((b, d)),
                       printed_nu=PRINTED_NU.get((b, d)), printed_rhs=PRINTED_RHS.get((b, d)), seeds={})
            for s in seeds:
                per = {}
                for oname, order in orderings(d, s).items():
                    toc = R.task_of_class(d, order)
                    gv = gamma_variants(M, Mn, Ms, toc, Ml=Ml if oname == "ours (sha256 rng)" else None, order=order if oname == "ours (sha256 rng)" else None, dataset=d)
                    per[oname] = dict(gamma=gv, pair_hits=[])
                    pg = cfg["printed_gamma"]
                    if pg:
                        G = np.array(gv["lw_pooled (ours)"]["G"]); iu = np.triu_indices(G.shape[0], 1)
                        hits = [(int(a), int(c), float(G[a, c])) for a, c in zip(*iu) if abs(G[a, c] - pg) / pg < 0.03]
                        per[oname]["pair_hits"] = hits
                    if oname == "ours (sha256 rng)":
                        per[oname]["nu_rhs"] = nu_rhs_variants(M, toc, np.array(gv["lw_pooled (ours)"]["G"]), pg, class_lmax)
                cfg["seeds"][str(s)] = per
                log(f"[fingerprint] {b}/{d} s{s}: printed gamma {cfg['printed_gamma']}, ours {per['ours (sha256 rng)']['gamma']['lw_pooled (ours)']['gamma_min']:.3f}, "
                    f"ridge 1e-2 {per['ours (sha256 rng)']['gamma']['ridge_abs eps=0.01']['gamma_min']:.3f}, sample {per['ours (sha256 rng)']['gamma']['sample_pooled']['gamma_min']:.3f} ({time.time()-t0:.0f}s)")
            cfg["mediators"] = per["ours (sha256 rng)"]["gamma"]["_mediators"]
            report["configs"][f"{b}|{d}"] = cfg
    report["ranking"] = rank(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    out_path.write_text(render(report), encoding="utf-8")
    log(f"[fingerprint] wrote {out_path}")
    return report


def rank(report: dict) -> dict:
    """Worst-case |log(variant / printed)| across configurations with a printed value, per convention (ours ordering,
    first seed); the same for the nu and RHS cells (worst absolute error)."""
    gam, nu, rhs = {}, {}, {}
    for key, cfg in report["configs"].items():
        pg = cfg.get("printed_gamma")
        s0 = cfg["seeds"][sorted(cfg["seeds"])[0]]["ours (sha256 rng)"]
        if pg:
            for vname, v in s0["gamma"].items():
                if vname.startswith("_"):
                    continue
                gam.setdefault(vname, []).append(abs(np.log(max(v["gamma_min"], 1e-12) / pg)))
        for cell, val in s0.get("nu_rhs", {}).items():
            if cell.startswith("nu") and cfg.get("printed_nu") is not None and np.isfinite(val):
                nu.setdefault(cell, []).append(abs(val - cfg["printed_nu"]))
            if cell.startswith("rhs") and cfg.get("printed_rhs") is not None and np.isfinite(val):
                rhs.setdefault(cell, []).append(abs(np.log(max(val, 1e-12) / cfg["printed_rhs"])))
    srt = lambda dct: sorted(((k, float(max(v)), float(np.mean(v)), len(v)) for k, v in dct.items()), key=lambda t: t[1])
    return dict(gamma=srt(gam), nu=srt(nu), rhs=srt(rhs))


def render(report: dict) -> str:
    L = ["# Fingerprint report: which convention reproduces the printed gamma_min, nu and RHS", "",
         "Feature caches only. A convention is a candidate only if ONE setting fits every configuration; the tables rank conventions by their "
         "worst case across configurations. Synthetic provenance means the report exercises the tool, not the paper.", ""]
    L += ["## Per-backbone mediators (direction of each convention's effect)", "", "| config | provenance | d | n | tr S/d | lambda_max | lambda_min | LW rho | mean row norm | total lambda_max |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, cfg in report["configs"].items():
        m = cfg["mediators"]
        L.append(f"| {key} | {cfg['provenance']} | {m['d']} | {m['n']} | {m['tr_S_over_d']:.4g} | {m['lambda_max_w']:.4g} | {m['lambda_min_w']:.3g} | {m['lw_rho']:.4f} | {m['mean_row_norm']:.4g} | {m['total_lambda_max']:.4g} |")
    L += ["", "## gamma_min: printed against each convention (ours ordering, first seed)", ""]
    names = None
    for key, cfg in report["configs"].items():
        s0 = cfg["seeds"][sorted(cfg["seeds"])[0]]["ours (sha256 rng)"]["gamma"]
        names = [k for k in s0 if not k.startswith("_")] if names is None else names
        L.append(f"**{key}** printed {cfg.get('printed_gamma')}: " + "; ".join(f"{k} {s0[k]['gamma_min']:.3g}" for k in names))
    L += ["", "### Ranking (worst |log ratio| to the printed value; 0 = exact everywhere)", "", "| convention | worst | mean | n configs |", "|---|---|---|---|"]
    for k, w, m, n in report["ranking"]["gamma"]:
        L.append(f"| {k} | {w:.3f} | {m:.3f} | {n} |")
    # the ratio grid: convention / printed per configuration, so a uniform per-backbone factor is visible at a glance
    keys = [k for k, cfg in report["configs"].items() if cfg.get("printed_gamma")]
    if keys and names:
        L += ["", "### Ratio grid: convention value / printed value (1.00 = exact; a constant column within one backbone is a scale convention, a constant row is an ordering or data effect)", "",
              "| convention | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for k in names:
            cells = []
            for key in keys:
                cfg = report["configs"][key]; s0 = cfg["seeds"][sorted(cfg["seeds"])[0]]["ours (sha256 rng)"]["gamma"]
                cells.append(f"{s0[k]['gamma_min'] / cfg['printed_gamma']:.2f}" if k in s0 else "")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "## Pair-set test: does the printed gamma_min equal some pair's gamma under our estimator (within 3 percent)?", ""]
    for key, cfg in report["configs"].items():
        for s, per in cfg["seeds"].items():
            for oname, v in per.items():
                hits = v.get("pair_hits") or []
                L.append(f"- {key} seed {s} ordering {oname}: ours gamma_min {v['gamma']['lw_pooled (ours)']['gamma_min']:.3g}; pair hits {hits if hits else 'none'}")
    L += ["", "## nu: best formula conventions (worst absolute error to the printed nu)", "", "| cell | worst | mean | n |", "|---|---|---|---|"]
    for k, w, m, n in report["ranking"]["nu"][:25]:
        L.append(f"| {k} | {w:.3f} | {m:.3f} | {n} |")
    L += ["", "## RHS: best formula conventions (worst |log ratio| to the printed RHS)", "", "| cell | worst | mean | n |", "|---|---|---|---|"]
    for k, w, m, n in report["ranking"]["rhs"][:16]:
        L.append(f"| {k} | {w:.3f} | {m:.3f} | {n} |")
    return "\n".join(L) + "\n"
