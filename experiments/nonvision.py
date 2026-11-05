"""experiments.nonvision -- Paper B 20 Newsgroups experiment (G03 SPEC 9).

MERGE NOTE (W3-R1, 2026-10-22, coordinator union ruling; VA4 audit). This
module is the union of the fleet H07 implementation (the live repo's prior
experiments/nonvision.py) and Claude's 2026-10-22 replication zip. Adopted
from Claude: the ALLOW_DOWNLOAD flag (DEFAULT FALSE -- a plain import or run
keeps the OWNER STORAGE LAW: zero network, always; download paths open only
behind the explicit --allow-download flag, intended for Colab), the per-pair
bias-direction diagnostic (the paper's C^2-averaged statistic is retained as
the ungated gauge identity ``paper_averaged_*``; the pre-declared gates now
evaluate on the per-pair aggregates), ``control_rng`` negative controls
wired into the run path, and the ``--run-label`` CLI option. Kept from the
fleet: the H07 surface consumed by experiments/diagnostic_controls.py (CI3)
-- ``load_20ng`` (cache-only under the default flag), ``build_embedder``,
``make_partition``, ``_cap_per_task``, ``fit_ce_head_lbfgs``,
``build_task_stats``, ``derive_seed``, ``LAM_DEFAULT``, and the T/C/N
constants -- behavior-identical at ALLOW_DOWNLOAD=False, so the stored CI3
record stays reproducible. Rerun note: the stored
results/runs/B/nonvision/B_nonvision_smoke.json predates this merge (H07
averaged-metric variant); under this module the same data additionally
yields the per-pair statistics while the averaged identity still reads 1.0.
Gate verdicts remain properties of the FULL run.

Status: IMPLEMENTED by H07, 2026-10-21 (replaces the G09 skeleton).

Spec source (read-only): ``N1000_INIT_2026-10-21/exec/G03_log.md`` SPEC 9
(app:nonvision L2690-2771): 20 Newsgroups standard BY-DATE split with
headers/footers/quotes removed; frozen ``all-MiniLM-L6-v2`` sentence
embeddings d=384; 20 classes partitioned into T=5 disjoint-label tasks of
C=4 classes; arms WT (within-task eq:ce-within, weight+bias decay exactly
per app:experiments), J (eq:ce-full on the union), FIX (Def 7.1 at
evaluation); 5 random class orderings seeds 0-4; diagnostics on training
data only per the tab:bias-direction protocol; pre-declared falsification:
fails if cosine < 0.90 or CV > 0.20.

OWNER STORAGE LAW (binding): NO dataset downloads on this machine, ever.
``fetch_20newsgroups`` is always called with ``download_if_missing=False``.
The FULL run happens on Google Colab: ``--mode colab`` prints the exact
self-contained Colab cell (also embedded verbatim in ``exec/H07_log.md``
for J-wave notebook assembly). Local execution is the capped labeled
SMOKE only, and only if the sklearn 20NG cache already exists on disk
(probed by catching the no-cache error); otherwise local is SKIPPED.

Embedding interface: sentence-transformers MiniLM when importable AND
loadable from the local HF cache with zero network (HF_HUB_OFFLINE=1 is
forced before the import attempt); else a clearly labeled TF-IDF fallback
(TfidfVectorizer fit on train texts only). TF-IDF geometry is NOT
comparable to the paper's MiniLM rows; every artifact records the backend.

IMPLEMENTATION DECISIONS (declared, G03 OWNER-ASK #7 territory):
  D1 GROUPING (coordinator pin "canonical first-20 grouping"): the 20
     classes in sklearn canonical order, chunked consecutively into
     {0-3},{4-7},{8-11},{12-15},{16-19}. The paper says only "5 random
     class orderings, seeds 0-4"; the grouping is unstated in the source.
     Under the canonical grouping the per-seed random ordering permutes
     the task TRAINING sequence only, and this evaluation-only L2 head is
     order-invariant, so the seed spread is structurally 0 (recorded
     honestly). The alternative reading -- per-seed random permutation of
     the 20-class list, chunked consecutively (the standard CIL
     "ordering" protocol, which resamples the partition and reproduces a
     nonzero seed spread) -- is implemented as ``--grouping permuted``.
  D2 OPTIMIZER: full-batch L-BFGS (numpy/scipy only, no torch) on the
     exact regularized objectives: mean within-task CE (eq:ce-within) or
     full CE (eq:ce-full) + (lam/2)(||W||^2 + ||b||^2), lam=1e-4 applied
     to weights AND biases (canonical gauge, SPEC 9 wording). The paper
     protocol trains AdamW lr 1e-3 / batch 128 / cosine / 50 epochs; the
     objective is matched exactly and L-BFGS converges it tighter
     (zero-init; strongly convex for lam>0).
  D3 SIGMA_HAT: per-task Ledoit-Wolf shrunk pooled within-class
     covariance (analytical 2004 intensity, sklearn-compatible) computed
     on that task's training data, pooled by averaging across tasks (SPEC
     5 protocol). One Sigma_hat per run, shared by the FIX statistics
     (TaskStats / S_bar) and the diagnostic.
  D4 DIAGNOSTIC AGGREGATION: cos/rel/CV per ordered task pair (i != j)
     on the WT head, means over the 20 ordered pairs x seeds; the
     pre-declared gates (cos >= 0.90, CV <= 0.20) are evaluated on the
     aggregates. Gate verdicts are properties of the FULL run; a smoke
     records them labeled, without claiming the experiment's verdict.
  D5 CI: the paper's 95% CI method for +7.30 is unstated (OWNER-ASK #7);
     this module reports mean +- SD over seeds and computes no CI.

PAPER CONTEXT, QUOTED ONLY, NEVER ASSERTED (SPEC 9 targets tab:nonvision
L2729-2751): WT 64.20 +- 0.98; FIX 71.50 +- 0.95 (+7.30, 95% CI
[7.21, 7.39]); J 73.84 +- 1.00; gap 9.64 with FIX recovering 76%;
cosine 0.93; CV 0.12. Real-run numbers are compared side by side with
this context only; no assertion ever tests these values.

Fleet imports (guarded, each fallback labeled): H01 core.gaussian_model
(seed policy), H02 core.fix (Def 7.1 de-center, TaskStats), H03
core.lda_ncm (LW shrunk pooled covariance), H04 core.diagnostic
(bias-direction algebra).

Writes: ``results/runs/B/nonvision/B_nonvision_<run>.json`` (G08-S2-style
envelope) on a real run; INDEX.json update is OPT-IN (``--update-index``)
so a caller can stay inside a strict file budget. Smoke caps at
``--cap-per-task 200`` docs/task (stratified 50/class), train and test,
all in memory, zero network.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # allow direct-script execution
    sys.path.insert(0, str(_REPO_ROOT))

EXPERIMENT = "nonvision"
PAPER = "B"
T_TASKS = 5
C_CLASSES = 4
N_CLASSES = T_TASKS * C_CLASSES
LAM_DEFAULT = 1e-4  # weight AND bias decay, SPEC 9 / app:experiments

#: Quoted context from tab:nonvision (SPEC 9). NEVER asserted; recorded
#: verbatim and printed next to real numbers for side-by-side reading.
PAPER_CONTEXT: Dict[str, Any] = {
    "never_asserted": True,
    "source": "G03_log.md SPEC 9 (tab:nonvision L2729-2751)",
    "WT_acc_pct": 64.20, "WT_sd": 0.98,
    "FIX_acc_pct": 71.50, "FIX_sd": 0.95,
    "FIX_gain": 7.30, "FIX_gain_ci95": [7.21, 7.39],
    "J_acc_pct": 73.84, "J_sd": 1.00,
    "gap_J_minus_WT": 9.64, "fix_recovery_pct": 76.0,
    "cosine": 0.93, "cv": 0.12,
}

#: Pre-declared falsification criteria of the experiment itself (SPEC 9).
FALSIFICATION = {"cosine_min": 0.90, "cv_max": 0.20}

DRIFT_NOTES: List[str] = []

# --- H02 (core.fix) ------------------------------------------------------------
try:
    from core import fix as fixm
    DRIFT_NOTES.append("H02 core.fix imported (de_center / TaskStats).")
except ImportError:  # pragma: no cover - FALLBACK-H02
    fixm = None  # type: ignore[assignment]
    DRIFT_NOTES.append(
        "FALLBACK-H02 ACTIVE: core.fix missing; local Def 7.1 algebra used."
    )

# --- H03 (core.lda_ncm: Ledoit-Wolf) -------------------------------------------
try:
    from core.lda_ncm import shrunk_pooled_covariance as _lw_pooled
    DRIFT_NOTES.append("H03 core.lda_ncm.shrunk_pooled_covariance imported "
                       "(returns (cov, intensity)).")
except ImportError:  # pragma: no cover - FALLBACK-H03
    _lw_pooled = None  # type: ignore[assignment]
    DRIFT_NOTES.append(
        "FALLBACK-H03 ACTIVE: core.lda_ncm missing; unshrunk pooled covariance."
    )

# --- H04 (core.diagnostic: bias-direction algebra) -----------------------------
try:
    from core.diagnostic import (
        bias_residuals as _bias_residuals,
        cv_cross_pair as _cv_cross_pair,
        delta_w_hat as _delta_w_hat,
        delta_w_pred as _delta_w_pred,
        relative_magnitude as _relative_magnitude,
        sigma_inner_cosine as _sigma_inner_cosine,
        symmetric_inverse as _symmetric_inverse,
        diagnostic_block as _diagnostic_block,
        closed_form_alignment as _closed_form_alignment,
    )
    DRIFT_NOTES.append("H04 core.diagnostic bias-direction API imported.")
except ImportError:  # pragma: no cover - FALLBACK-H04
    _bias_residuals = _cv_cross_pair = _delta_w_hat = _delta_w_pred = None  # type: ignore[assignment]
    _relative_magnitude = _sigma_inner_cosine = _symmetric_inverse = None  # type: ignore[assignment]
    _diagnostic_block = _closed_form_alignment = None  # type: ignore[assignment]
    DRIFT_NOTES.append(
        "FALLBACK-H04 ACTIVE: core.diagnostic missing; local algebra used."
    )

# --- H01 (core.gaussian_model: conftest seed policy re-export) ------------------
try:
    from core.gaussian_model import derive_seed as _derive_seed
    DRIFT_NOTES.append("H01 seed policy (derive_seed) imported.")
except ImportError:  # pragma: no cover - FALLBACK-H01
    _derive_seed = None  # type: ignore[assignment]
    DRIFT_NOTES.append(
        "FALLBACK-H01 ACTIVE: local SHA-256 purpose-seed copy used "
        "(same formula as the conftest policy)."
    )


def derive_seed(purpose: str, salt: int = 0) -> int:
    """Conftest SEED=0 policy: sha256('0|purpose|salt'), first 16 hex."""
    if _derive_seed is not None:
        return int(_derive_seed(purpose, salt))
    d = hashlib.sha256(f"0|{purpose}|{salt}".encode()).hexdigest()
    return int(d[:16], 16)


def new_rng(purpose: str, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng(derive_seed(purpose, salt))


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# Fallback algebra (only exercised when the fleet imports above fail)
# =============================================================================


def _fb_symmetric_inverse(a: np.ndarray) -> np.ndarray:  # FALLBACK-H04
    inv = np.linalg.solve(a, np.eye(a.shape[0]))
    return 0.5 * (inv + inv.T)


def _fb_sigma_inv(sigma: np.ndarray) -> np.ndarray:
    if _symmetric_inverse is not None:
        return _symmetric_inverse(np.asarray(sigma, float))
    return _fb_symmetric_inverse(np.asarray(sigma, float))  # pragma: no cover


def _fb_lw_pooled(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Unshrunk pooled within-class covariance, labeled fallback (H03 path
    returns the LW-shrunk version with the analytical intensity)."""
    K = int(y.max()) + 1
    means = np.stack([X[y == k].mean(axis=0) for k in range(K)])
    Xc = X - means[y]
    return Xc.T @ Xc / max(len(y) - K, 1), float("nan")


class _FBTaskStats:  # pragma: no cover - FALLBACK-H02 container
    def __init__(self, means, Sigma, task_of_class):
        si = _fb_sigma_inv(Sigma)
        self.means = np.asarray(means, float)
        self.Sigma = np.asarray(Sigma, float)
        self.task_of_class = np.asarray(task_of_class, np.int64)
        T = int(self.task_of_class.max()) + 1
        self.mu_bar = np.vstack(
            [self.means[self.task_of_class == t].mean(axis=0) for t in range(T)])
        q = np.einsum("ri,ij,rj->r", self.means, si, self.means)
        self.S_bar = np.array([q[self.task_of_class == t].mean() for t in range(T)])
        self.n_classes, self.d, self.n_tasks = (
            self.means.shape[0], self.means.shape[1], T)


def _fb_de_center(W: np.ndarray, b: np.ndarray, stats) -> Tuple[np.ndarray, np.ndarray]:
    """Def 7.1: w += Sigma^{-1} mu_bar^{t(r)}, b -= 0.5 S_bar^{t(r)}."""
    if fixm is not None:
        head = fixm.de_center(fixm.LinearHead(np.asarray(W, float),
                                              np.asarray(b, float)), stats)
        return head.W, head.b
    si = _fb_sigma_inv(stats.Sigma)  # pragma: no cover - FALLBACK-H02
    Wf = W + (stats.mu_bar @ si)[stats.task_of_class]
    bf = b - 0.5 * stats.S_bar[stats.task_of_class]
    return Wf, bf


def build_task_stats(X: np.ndarray, y: np.ndarray, task_of_class: np.ndarray,
                     per_task_blocks: Dict[int, Tuple[np.ndarray, np.ndarray]]
                     ) -> Tuple[Any, np.ndarray, float]:
    """TaskStats (H02) with the SPEC-5 Sigma_hat: per-task LW shrunk pooled
    within-class covariance, averaged across tasks (D3). Returns
    (stats, sigma_hat, mean_lw_intensity)."""
    sigmas, intens = [], []
    for t in sorted(per_task_blocks):
        Xt, yt = per_task_blocks[t]
        if _lw_pooled is not None:
            cov, rho = _lw_pooled(Xt, yt)
        else:  # pragma: no cover - FALLBACK-H03
            cov, rho = _fb_lw_pooled(Xt, yt)
        sigmas.append(cov)
        intens.append(float(rho) if np.isfinite(rho) else float("nan"))
    sigma_hat = np.mean(np.stack(sigmas), axis=0)
    means = np.stack([X[y == k].mean(axis=0) for k in range(int(y.max()) + 1)])
    if fixm is not None:
        stats = fixm.TaskStats.from_class_stats(means, sigma_hat, task_of_class)
    else:  # pragma: no cover - FALLBACK-H02
        stats = _FBTaskStats(means, sigma_hat, task_of_class)
    return stats, sigma_hat, float(np.mean(intens))


# =============================================================================
# Stage 1 -- data loading (zero network; cache-only)
# =============================================================================


class NoCacheError(RuntimeError):
    """Raised when the sklearn 20NG cache is absent (never a download)."""


ALLOW_DOWNLOAD = False  # set True only on Colab (--allow-download); never on the owner machine


def load_20ng(subset: str = "train", data_home: Optional[str] = None):
    """fetch_20newsgroups BY-DATE, headers/footers/quotes removed,
    download_if_missing=False (STORAGE LAW: zero network on the owner machine).
    On Colab, ``--allow-download`` flips ALLOW_DOWNLOAD and the fetch may download."""
    from sklearn.datasets import fetch_20newsgroups  # local import
    try:
        return fetch_20newsgroups(
            subset=subset, data_home=data_home,
            download_if_missing=ALLOW_DOWNLOAD,
            remove=("headers", "footers", "quotes"),
        )
    except Exception as exc:  # sklearn raises OSError/IOError variants
        raise NoCacheError(
            f"20NG {subset} cache unavailable (download forbidden by the "
            f"owner storage law; original error: {type(exc).__name__}: {exc})"
        ) from exc


def cache_present(data_home: Optional[str] = None) -> bool:
    try:
        load_20ng("train", data_home)
        return True
    except NoCacheError:
        return False


# =============================================================================
# Stage 2 -- embedding interface (MiniLM offline-first; labeled TF-IDF fallback)
# =============================================================================


class MinilmEmbedder:
    """Frozen all-MiniLM-L6-v2 (d=384), offline-only load.

    HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are forced to '1' BEFORE the
    sentence_transformers import so a missing local cache can never turn
    into a network download; failure falls back to TF-IDF with a drift
    note. all-MiniLM-L6-v2's own pipeline L2-normalizes its outputs.
    """

    name = "sentence-transformers/all-MiniLM-L6-v2"
    backend = "minilm-l6-v2"
    d = 384

    def __init__(self):
        import os
        if not ALLOW_DOWNLOAD:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer  # deferred
        self._model = SentenceTransformer(self.name)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(
            list(texts), batch_size=64, convert_to_numpy=True,
            show_progress_bar=False), dtype=float)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit_transform(texts)


class TfidfEmbedder:
    """CLEARLY LABELED FALLBACK backend (sklearn TfidfVectorizer, fit on
    train texts only). Geometry is NOT comparable to MiniLM rows."""

    name = "tfidf-fallback (sublinear_tf, l2-normalized rows)"
    backend = "tfidf-fallback"
    d: Optional[int] = None

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.9)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        X = self._vec.fit_transform(list(texts))
        self.d = X.shape[1]
        return np.asarray(X.todense(), dtype=float)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        X = self._vec.transform(list(texts))
        return np.asarray(X.todense(), dtype=float)


def build_embedder(prefer: str = "minilm") -> Tuple[Any, str]:
    """(embedder, backend_label). Tries MiniLM offline; any failure falls
    back to the labeled TF-IDF backend."""
    if prefer == "minilm":
        try:
            return MinilmEmbedder(), "minilm-l6-v2 (offline cache, d=384)"
        except Exception as exc:
            DRIFT_NOTES.append(
                "FALLBACK-EMBEDDER ACTIVE: MiniLM offline load failed "
                f"({type(exc).__name__}: {str(exc)[:120]}); TF-IDF fallback "
                "used -- numbers NOT comparable to paper MiniLM rows."
            )
    return TfidfEmbedder(), "tfidf-fallback (NOT comparable to paper rows)"


# =============================================================================
# Stage 3 -- 5x4 task partition (D1)
# =============================================================================


def make_partition(grouping: str = "canonical", seed: int = 0) -> Dict[str, Any]:
    """T=5 disjoint-label tasks x C=4 classes from the canonical 20 classes.

    canonical (coordinator pin, D1): consecutive chunks of the canonical
    class list 0..19 -> {0-3},{4-7},{8-11},{12-15},{16-19}; the seed
    permutes the task TRAINING order only (evaluation is order-invariant
    for this head, so seed spread is structurally 0 under this grouping).
    permuted: per-seed random permutation of the 20-class list, chunked
    consecutively (standard CIL ordering protocol; resamples the partition).
    Chunk rows are sorted ascending (deterministic within-task labels).
    """
    classes = np.arange(N_CLASSES)
    if grouping == "canonical":
        chunks = classes.reshape(T_TASKS, C_CLASSES)
        order = list(np.random.default_rng(
            derive_seed(f"h07:order:{grouping}", seed)).permutation(T_TASKS))
        note = ("CANONICAL_PARTITION: consecutive 4-class chunks of the "
                "canonical class list (coordinator pin; grouping unstated "
                "in source, G03 OWNER-ASK #7). Seed permutes task training "
                "order only; evaluation is order-invariant, so the seed "
                "spread is structurally 0 under this grouping.")
    elif grouping == "permuted":
        perm = np.random.default_rng(
            derive_seed("h07:perm", seed)).permutation(N_CLASSES)
        chunks = np.sort(perm.reshape(T_TASKS, C_CLASSES), axis=1)
        order = list(range(T_TASKS))
        note = ("PERMUTED_PARTITION: per-seed random permutation of the "
                "canonical class list, consecutive 4-class chunks (standard "
                "CIL ordering reading; resamples the partition).")
    else:
        raise ValueError(f"unknown grouping {grouping!r}")
    task_of_class = np.zeros(N_CLASSES, dtype=np.int64)
    for t, chunk in enumerate(chunks):
        task_of_class[chunk] = t
    return dict(chunks=[list(map(int, c)) for c in chunks],
                task_of_class=task_of_class,
                train_order=[int(t) for t in order],
                grouping=grouping, note=note)


# =============================================================================
# Stage 4 -- CIL training: L2 CE head via full-batch L-BFGS (D2, no torch)
# =============================================================================


def fit_ce_head_lbfgs(X: np.ndarray, y: np.ndarray, n_classes: int,
                      lam: float, maxiter: int = 1000, gtol: float = 1e-9,
                      ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Minimize mean CE + (lam/2)(||W||^2 + ||b||^2) over one class block.

    The objective is eq:ce-within (task block) or eq:ce-full (all classes)
    with the app:experiments weight+bias decay; full-batch L-BFGS, zero
    init, analytic gradient (D2). Convex; strongly convex for lam>0.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, np.int64)
    n, d = X.shape
    C = int(n_classes)
    idx = np.arange(n)

    def obj(theta: np.ndarray):
        W = theta[: C * d].reshape(C, d)
        b = theta[C * d:]
        S = X @ W.T + b
        S = S - S.max(axis=1, keepdims=True)
        lse = np.log(np.exp(S).sum(axis=1))
        loss = float(
            (lse - S[idx, y]).mean()
            + 0.5 * lam * (np.sum(W * W) + np.sum(b * b))
        )
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        G = P.copy()
        G[idx, y] -= 1.0
        gW = (X.T @ G).T / n + lam * W
        gb = G.mean(axis=0) + lam * b
        return loss, np.concatenate([gW.ravel(), gb])

    res = scipy.optimize.minimize(
        obj, np.zeros(C * d + C), jac=True, method="L-BFGS-B",
        options=dict(maxiter=maxiter, ftol=1e-14, gtol=gtol),
    )
    W = res.x[: C * d].reshape(C, d)
    b = res.x[C * d:]
    info = dict(nit=int(res.nit), max_grad=float(np.abs(res.jac).max()),
                fun=float(res.fun), success=bool(res.success))
    return W, b, info


# =============================================================================
# Stage 5 -- arms WT / J / FIX + Stage 6 -- tab:bias-direction diagnostic
# =============================================================================


@dataclasses.dataclass
class DiagHead:
    """H04 diagnostic head protocol: .W, .mu_hat, .task_of_class."""

    W: np.ndarray
    b: np.ndarray
    mu_hat: np.ndarray
    task_of_class: np.ndarray


def _accuracy(W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(X @ W.T + b, axis=1) == y))


def diagnostic_on_head(W: np.ndarray, mu_hat: np.ndarray,
                       task_of_class: np.ndarray, sigma_hat: np.ndarray,
                       control_rng: Optional[np.random.Generator] = None,
                       ) -> Dict[str, Any]:
    """tab:bias-direction protocol on the trained WT head, training stats only.

    TWO STATISTICS ARE REPORTED (Claude, 2026-10-22, Q7):

    * ``paper_averaged_cos`` / ``paper_averaged_rel``: the paper's metric (iii),
      cosine between the C^2-AVERAGED residual and the prediction. Under Lemma
      3.2 (within-task weights sum to zero at any regularized minimizer) the
      averaged residual EQUALS the prediction for any data, so these read 1.0 by
      identity whenever ``gauge_residual`` is ~0. They are kept for comparison
      with the printed table only and are NOT gated.
    * ``cos_mean`` / ``cos_min`` / ``rel_mean`` / ``cv_mean``: the per-pair
      statistics (cosine and relative magnitude of EACH pair residual against
      the prediction, averaged over the C^2 pairs and the T(T-1) ordered task
      pairs). This is the test of Theorem 4.1 + Corollary 4.2. The pre-declared
      gates (cos >= 0.90, CV <= 0.20) are evaluated on THESE.
    * ``thm33_alignment_mean/min``: Theorem 3.3 per class, cos_Sigma(w_k,
      Sigma^{-1}(mu_k - mu_bar_t)).
    * Controls (when ``control_rng`` is given): permuted-pairing and random-head
      negative controls per block; both must collapse the per-pair cosine.
    """
    si = _fb_sigma_inv(sigma_hat)
    head = DiagHead(W=np.asarray(W, float), b=np.zeros(W.shape[0]),
                    mu_hat=np.asarray(mu_hat, float),
                    task_of_class=np.asarray(task_of_class, np.int64))
    T = int(task_of_class.max()) + 1
    per_pair = []
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            if _diagnostic_block is not None:
                blk = _diagnostic_block(head, i, j, si, control_rng)
            else:  # pragma: no cover - FALLBACK-H04: per-pair algebra inline
                ki = np.flatnonzero(task_of_class == i)
                kj = np.flatnonzero(task_of_class == j)
                res = np.stack([np.stack([(W[k] - W[l]) - si @ (mu_hat[k] - mu_hat[l])
                                          for l in kj]) for k in ki])
                cents = np.vstack([mu_hat[task_of_class == t].mean(0) for t in range(T)])
                dp = si @ (cents[j] - cents[i])
                dh = res.mean(axis=(0, 1))
                flat = res.reshape(-1, res.shape[-1])
                mags = np.sqrt(np.maximum((flat @ si * flat).sum(1), 0.0))
                npred = np.sqrt(dp @ si @ dp)
                pc = (flat @ si @ dp) / np.maximum(mags * npred, 1e-300)
                blk = dict(paper_averaged_cos=float((dh @ si @ dp) / np.sqrt((dh @ si @ dh) * (dp @ si @ dp))),
                           paper_averaged_rel=float(np.sqrt((dh @ si @ dh) / (dp @ si @ dp))),
                           gauge_residual_i=float("nan"), gauge_residual_j=float("nan"),
                           pair_cos_mean=float(pc.mean()), pair_cos_min=float(pc.min()),
                           pair_rel_mean=float((mags / npred).mean()), pair_rel_min=float((mags / npred).min()),
                           pair_rel_max=float((mags / npred).max()),
                           cv=float(mags.std() / mags.mean()) if mags.mean() > 0 else float("nan"))
            row = dict(i=i, j=j)
            for k, v in blk.items():
                row[k] = (v if isinstance(v, dict) else (None if (v is None or np.isnan(v)) else round(float(v), 6)))
            per_pair.append(row)

    def _agg(key):
        vals = [p[key] for p in per_pair if p.get(key) is not None]
        return vals
    cos_all, rel_all, cv_all = _agg("pair_cos_mean"), _agg("pair_rel_mean"), _agg("cv")
    out = dict(
        per_pair=per_pair, n_pairs=len(per_pair),
        statistic="per-pair (Theorem 4.1 / Cor 4.2); paper_averaged_* is the gauge identity, not gated",
        cos_mean=float(np.mean(cos_all)), cos_min=float(np.min(_agg("pair_cos_min"))),
        rel_mean=float(np.mean(rel_all)),
        cv_mean=float(np.mean(cv_all)), cv_max=float(np.max(cv_all)),
        paper_averaged_cos_mean=float(np.mean(_agg("paper_averaged_cos"))),
        paper_averaged_rel_mean=float(np.mean(_agg("paper_averaged_rel"))),
        gauge_residual_max=float(np.max(_agg("gauge_residual_i") + _agg("gauge_residual_j") or [float("nan")])),
    )
    if _closed_form_alignment is not None:
        al = _closed_form_alignment(head, si)
        out["thm33_alignment_mean"] = float(np.mean(al)); out["thm33_alignment_min"] = float(np.min(al))
    if control_rng is not None and per_pair and isinstance(per_pair[0].get("control_random_head"), dict):
        out["control_random_head_pair_cos_mean"] = float(np.mean([p["control_random_head"]["pair_cos_mean"] for p in per_pair]))
        out["control_random_head_averaged_cos_mean"] = float(np.mean([p["control_random_head"]["averaged_cos"] for p in per_pair]))
        out["control_permuted_pair_cos_mean"] = float(np.mean([p["control_permuted"]["pair_cos_mean"] for p in per_pair]))
    return out


# =============================================================================
# Stage 7 -- the experiment driver
# =============================================================================


def _cap_per_task(texts: Sequence[str], targets: np.ndarray,
                  chunks: List[List[int]], cap_per_task: int,
                  purpose: str) -> Tuple[List[str], np.ndarray]:
    """SMOKE CAP: at most cap_per_task docs per task (stratified equally
    across the task's C classes), purpose-pinned rng, in memory."""
    idx_keep = []
    per_class = max(cap_per_task // C_CLASSES, 1)
    for chunk in chunks:
        for cls in chunk:
            idx_cls = np.flatnonzero(targets == cls)
            rng = new_rng(f"{purpose}:c{cls}")
            take = per_class if len(idx_cls) > per_class else len(idx_cls)
            idx_keep.append(rng.choice(idx_cls, size=take, replace=False))
    idx = np.sort(np.concatenate(idx_keep))
    return [texts[i] for i in idx], targets[idx]


def run_experiment(train_texts: Sequence[str], train_y: np.ndarray,
                   test_texts: Sequence[str], test_y: np.ndarray,
                   embedder: Any, backend_label: str,
                   seeds: Sequence[int], grouping: str, lam: float,
                   cap_per_task: Optional[int], purpose_tag: str,
                   ) -> Dict[str, Any]:
    """Full pipeline on one embedding backend: WT / J / FIX + diagnostic,
    per seed, then aggregation and (recorded, full-run-only) gates."""
    t0 = time.time()
    train_texts, test_texts = list(train_texts), list(test_texts)
    train_y, test_y = np.asarray(train_y), np.asarray(test_y)
    # uncapped runs embed once and reuse across seeds; capped (smoke) runs
    # cap FIRST so only the capped subsets are ever embedded.
    Xtr_full = Xte_full = None
    if cap_per_task is None:
        Xtr_full = np.asarray(embedder.fit_transform(train_texts), float)
        Xte_full = np.asarray(embedder.transform(test_texts), float)
    d = Xtr_full.shape[1] if Xtr_full is not None else None

    per_seed = []
    for seed in seeds:
        part = make_partition(grouping, seed)
        toc = part["task_of_class"]
        if cap_per_task:
            tr_s, ytr_s = _cap_per_task(train_texts, train_y, part["chunks"],
                                        cap_per_task, f"{purpose_tag}:cap:s{seed}")
            te_s, yte_s = _cap_per_task(test_texts, test_y, part["chunks"],
                                        cap_per_task, f"{purpose_tag}:capte:s{seed}")
            Xtr_s = np.asarray(embedder.fit_transform(tr_s), float)
            Xte_s = np.asarray(embedder.transform(te_s), float)
        else:
            Xtr_s, Xte_s = Xtr_full, Xte_full
            ytr_s, yte_s = train_y, test_y
        d = Xtr_s.shape[1]

        # --- WT arm: sequential within-task CE fits in training order -----
        W = np.zeros((N_CLASSES, d))
        b = np.zeros(N_CLASSES)
        blocks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        max_grad = 0.0
        for t in part["train_order"]:
            chunk = np.array(part["chunks"][t])  # sorted ascending (D1)
            m = np.isin(ytr_s, chunk)
            Xb, yb = Xtr_s[m], np.searchsorted(chunk, ytr_s[m])
            Wt, bt, info = fit_ce_head_lbfgs(Xb, yb, C_CLASSES, lam)
            W[chunk], b[chunk] = Wt, bt
            blocks[t] = (Xb, ytr_s[m])  # global labels for the stats
            max_grad = max(max_grad, info["max_grad"])
        acc_wt = _accuracy(W, b, Xte_s, yte_s)

        # --- J arm: eq:ce-full on the union --------------------------------
        Wj, bj, info_j = fit_ce_head_lbfgs(Xtr_s, ytr_s, N_CLASSES, lam)
        acc_j = _accuracy(Wj, bj, Xte_s, yte_s)

        # --- task statistics + FIX arm (Def 7.1, evaluation only) ---------
        Xtr_all = np.vstack([blocks[t][0] for t in sorted(blocks)])
        ytr_all = np.concatenate([blocks[t][1] for t in sorted(blocks)])
        stats, sigma_hat, lw_int = build_task_stats(
            Xtr_all, ytr_all, toc, blocks)
        W_fix, b_fix = _fb_de_center(W, b, stats)
        acc_fix = _accuracy(W_fix, b_fix, Xte_s, yte_s)

        # --- diagnostic on the WT head (training stats only) --------------
        mu_hat = np.stack([Xtr_all[ytr_all == k].mean(axis=0)
                           for k in range(N_CLASSES)])
        diag = diagnostic_on_head(W, mu_hat, toc, sigma_hat,
                                  control_rng=new_rng(f"{purpose_tag}:controls:s{seed}"))

        gap = acc_j - acc_wt
        rec = ((acc_fix - acc_wt) / gap) if abs(gap) > 1e-12 else float("nan")
        per_seed.append(dict(
            seed=seed, acc_wt=round(acc_wt, 6), acc_fix=round(acc_fix, 6),
            acc_j=round(acc_j, 6), fix_gain=round(acc_fix - acc_wt, 6),
            gap_j_wt=round(gap, 6),
            fix_recovery=(None if np.isnan(rec) else round(rec, 6)),
            diagnostic=diag, lw_intensity_mean=round(lw_int, 6),
            n_train=int(len(ytr_s)), n_test=int(len(yte_s)),
            fit_max_grad_wt=round(max_grad, 12),
            fit_j_max_grad=round(info_j["max_grad"], 12),
            partition=dict(chunks=part["chunks"], order=part["train_order"],
                           note=part["note"]),
        ))

    def _ms(key: str, per_seed_key: Optional[str] = None) -> Dict[str, Any]:
        src = per_seed_key or key
        vals = [s[src] for s in per_seed]
        if any(v is None for v in vals):
            return dict(mean=None, sd=None, per_seed=vals)
        fv = [float(v) for v in vals]
        return dict(mean=round(float(np.mean(fv)), 6),
                    sd=round(float(np.std(fv, ddof=1)) if len(fv) > 1 else 0.0, 6),
                    per_seed=[round(v, 6) for v in fv])

    agg = {k: _ms(k) for k in ("acc_wt", "acc_fix", "acc_j", "fix_gain",
                               "gap_j_wt", "fix_recovery")}
    for key, src in (("cos_mean", "cos_mean"), ("rel_mean", "rel_mean"),
                     ("cv_mean", "cv_mean")):
        vals = [s["diagnostic"][src] for s in per_seed]
        agg[key] = dict(mean=round(float(np.mean(vals)), 6),
                        sd=round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 6),
                        per_seed=[round(v, 6) for v in vals])
    gates = dict(
        cos_ge_090=bool(agg["cos_mean"]["mean"] >= FALSIFICATION["cosine_min"]),
        cv_le_020=bool(agg["cv_mean"]["mean"] <= FALSIFICATION["cv_max"]),
        criteria="pre-declared falsification (SPEC 9): fails if cosine < "
                 "0.90 or CV > 0.20 -- verdicts attach to the FULL run; a "
                 "smoke records them without claiming the verdict",
    )
    return dict(
        backend=backend_label, d=int(d), seeds=list(seeds), grouping=grouping,
        lam=lam, cap_per_task=cap_per_task,
        per_seed=per_seed, aggregate=agg, gates=gates,
        paper_context=dict(PAPER_CONTEXT),
        runtime_s=round(time.time() - t0, 1),
    )


# =============================================================================
# Stage 8 -- results-store JSON writer (G08-S2-style envelope)
# =============================================================================


def _file_sha(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def write_run_json(record: Dict[str, Any], out_root: Path, run_id: str,
                   provenance: str, started_utc: str, cfg: Dict[str, Any],
                   update_index: bool = False) -> Path:
    run_dir = out_root / "runs" / PAPER / EXPERIMENT
    run_dir.mkdir(parents=True, exist_ok=True)
    gates = record.get("gates", {})
    payload = dict(
        run_id=run_id, paper=PAPER, experiment=EXPERIMENT,
        provenance=provenance, created_utc=_utcnow(), executed_utc=started_utc,
        schema="G08-S2-proposal envelope (H05 precedent); SPEC 9 experiment",
        config=cfg, seeds=cfg.get("seeds", []),
        metrics=record,
        paper_context=dict(PAPER_CONTEXT, never_asserted=True),
        env=dict(python=platform.python_version(), numpy=np.__version__,
                 scipy=__import__("scipy").__version__,
                 platform=platform.platform()),
        code_version=_file_sha(Path(__file__)),
        inputs={name: _file_sha(_REPO_ROOT / "core" / f"{name}.py")
                for name in ("fix", "lda_ncm", "diagnostic")},
        drift_notes=DRIFT_NOTES,
    )
    path = run_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    if update_index:
        index_path = out_root / "INDEX.json"
        index: Dict[str, Any] = {}
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                index = {}
        runs = [r for r in index.setdefault("runs", [])
                if r.get("run_id") != run_id]
        runs.append(dict(
            run_id=run_id, path=str(path.relative_to(out_root)),
            provenance=provenance, paper=PAPER, experiment=EXPERIMENT,
            suite_status=("PASS" if (gates.get("cos_ge_090")
                                     and gates.get("cv_le_020")) else "RECORDED"),
            created_utc=payload["created_utc"]))
        index["runs"] = runs
        index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return path


def print_summary(record: Dict[str, Any], run_label: str) -> None:
    agg = record["aggregate"]
    print(f"[h07] {run_label} summary (backend {record['backend']}, "
          f"d={record['d']}, grouping={record['grouping']}, "
          f"cap_per_task={record['cap_per_task']}, seeds={record['seeds']})")
    ctx = PAPER_CONTEXT

    def line(name, mean, ctx_val, pct=False):
        ours = "n/a" if mean is None else (
            f"{mean * 100:.2f}" if pct else f"{mean:.4f}")
        print(f"  {name:<14} ours {ours:<12} paper-context {ctx_val} "
              "(quoted, never asserted)")

    line("WT acc", agg["acc_wt"]["mean"], "64.20 +- 0.98", pct=True)
    line("FIX acc", agg["acc_fix"]["mean"], "71.50 +- 0.95", pct=True)
    line("J acc", agg["acc_j"]["mean"], "73.84 +- 1.00", pct=True)
    line("FIX gain", agg["fix_gain"]["mean"], "7.30 [CI 7.21, 7.39]", pct=True)
    line("gap J-WT", agg["gap_j_wt"]["mean"], "9.64", pct=True)
    line("FIX recovery", agg["fix_recovery"]["mean"], "76%", pct=True)
    line("cos per-pair", agg["cos_mean"]["mean"], "0.93 (paper metric is the averaged one)")
    line("CV cross-pair", agg["cv_mean"]["mean"], "0.12")
    d0 = record["per_seed"][0]["diagnostic"]
    print(f"  paper-averaged cos/rel (gauge identity, not gated): "
          f"{d0.get('paper_averaged_cos_mean', float('nan')):.6f} / {d0.get('paper_averaged_rel_mean', float('nan')):.6f}; "
          f"gauge residual max {d0.get('gauge_residual_max', float('nan')):.2e}")
    if "thm33_alignment_mean" in d0:
        print(f"  Thm 3.3 per-class alignment: mean {d0['thm33_alignment_mean']:.4f} min {d0['thm33_alignment_min']:.4f}")
    if "control_random_head_pair_cos_mean" in d0:
        print(f"  controls: random-head per-pair cos {d0['control_random_head_pair_cos_mean']:.3f} "
              f"(averaged stays {d0['control_random_head_averaged_cos_mean']:.3f}); "
              f"permuted-pairing per-pair cos {d0['control_permuted_pair_cos_mean']:.3f}")
    g = record["gates"]
    print(f"[h07] pre-declared gates (verdicts attach to the FULL run): "
          f"cos>=0.90 {g['cos_ge_090']} | CV<=0.20 {g['cv_le_020']}")


# =============================================================================
# Colab cell (J-wave assembly) -- exact contents emitted by --mode colab
# =============================================================================

COLAB_CELL = r'''# ============================================================================
# H07 COLAB CELL -- Paper B nonvision: 20 Newsgroups FULL run (G03 SPEC 9)
# SUPERSEDED (2026-10-22): colab/05_nonvision_paperB.ipynb runs this module itself
# (python replicate.py nonvision --allow-download), which reports the per-pair
# bias-direction statistics beside the paper's averaged one.  The standalone cell
# below is kept for reference only and still mirrors the older averaged metric.
# Mirrors repo/experiments/nonvision.py (H07, 2026-10-21). Keep in sync.
# On Colab, downloads are fine (the owner storage law forbids them only on
# the owner machine). Output: /content/B_nonvision_full.json -> download
# into repo/results/runs/B/nonvision/B_nonvision_full.json and register it
# in repo/results/INDEX.json (J-wave task).
# Arms: WT (within-task CE), J (full CE on the union), FIX (Def 7.1 at
# evaluation). Pre-declared falsification (SPEC 9): fail if cosine < 0.90
# or CV > 0.20. Paper numbers below are quoted context, never asserted.
# ============================================================================
%pip install -q sentence-transformers scikit-learn
import datetime, hashlib, json, platform, time
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

SEEDS = [0, 1, 2, 3, 4]   # 5 random class orderings
GROUPING = "canonical"    # "canonical" (coordinator pin) | "permuted"
LAM = 1e-4                # weight AND bias decay (SPEC 9)
T, C, N = 5, 4, 20        # tasks x classes/task x total classes
PAPER_CONTEXT = {         # tab:nonvision, QUOTED, never asserted
    "never_asserted": True,
    "WT": (64.20, 0.98), "FIX": (71.50, 0.95), "J": (73.84, 1.00),
    "fix_gain": (7.30, [7.21, 7.39]), "gap": 9.64, "recovery_pct": 76.0,
    "cos": 0.93, "cv": 0.12,
}

def derive_seed(purpose, salt=0):  # conftest SEED=0 policy
    return int(hashlib.sha256(f"0|{purpose}|{salt}".encode()).hexdigest()[:16], 16)

# ---- data: fresh by-date fetch (allowed here), headers/footers/quotes off --
print("[cell] fetching 20NG by-date (train+test) ...", flush=True)
tr = fetch_20newsgroups(subset="train", download_if_missing=True,
                        remove=("headers", "footers", "quotes"))
te = fetch_20newsgroups(subset="test", download_if_missing=True,
                        remove=("headers", "footers", "quotes"))
train_texts, train_y = list(tr.data), np.asarray(tr.target)
test_texts, test_y = list(te.data), np.asarray(te.target)
print(f"[cell] train {len(train_texts)} docs, test {len(test_texts)} docs")

# ---- embeddings: frozen all-MiniLM-L6-v2, d=384 (labeled TF-IDF fallback) --
t0 = time.time()
try:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb_name = "sentence-transformers/all-MiniLM-L6-v2"
    def embed(texts):
        return np.asarray(st.encode(texts, batch_size=128,
                                    convert_to_numpy=True,
                                    show_progress_bar=True), dtype=float)
except Exception as exc:  # CLEARLY LABELED FALLBACK
    print(f"[cell] MiniLM unavailable ({type(exc).__name__}); TF-IDF FALLBACK "
          "active -- numbers NOT comparable to the paper's MiniLM rows.")
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.9)
    _fit = [False]
    def embed(texts):
        if not _fit[0]:
            X, _fit[0] = vec.fit_transform(texts), True
        else:
            X = vec.transform(texts)
        return np.asarray(X.todense(), dtype=float)
    emb_name = "tfidf-fallback (NOT comparable to paper rows)"
Xtr, Xte = embed(train_texts), embed(test_texts)
print(f"[cell] embedded: train {Xtr.shape}, test {Xte.shape} "
      f"({time.time()-t0:.1f}s, backend {emb_name})", flush=True)

def make_partition(grouping, seed):
    classes = np.arange(N)
    if grouping == "canonical":
        chunks = classes.reshape(T, C)
        order = list(np.random.default_rng(
            derive_seed(f"h07:order:{grouping}", seed)).permutation(T))
        note = ("CANONICAL_PARTITION: consecutive 4-class chunks of the "
                "canonical class list (coordinator pin; grouping unstated in "
                "source, G03 OWNER-ASK #7); seed permutes training order only "
                "-> seed spread structurally 0.")
    else:
        perm = np.random.default_rng(derive_seed("h07:perm", seed)).permutation(N)
        chunks, order = np.sort(perm.reshape(T, C), axis=1), list(range(T))
        note = "PERMUTED_PARTITION: per-seed permutation, consecutive chunks."
    toc = np.zeros(N, dtype=np.int64)
    for t, chunk in enumerate(chunks):
        toc[chunk] = t
    return chunks, toc, order, note

def fit_ce_head_lbfgs(X, y, n_classes, lam, maxiter=1000, gtol=1e-9):
    # mean CE + (lam/2)(||W||^2 + ||b||^2); full-batch L-BFGS, zero init
    n, d = X.shape
    idx = np.arange(n)
    def obj(th):
        W, b = th[: n_classes * d].reshape(n_classes, d), th[n_classes * d:]
        S = X @ W.T + b
        S -= S.max(1, keepdims=True)
        loss = (np.log(np.exp(S).sum(1)) - S[idx, y]).mean() \
            + .5 * lam * (np.sum(W * W) + np.sum(b * b))
        P = np.exp(S)
        P /= P.sum(1, keepdims=True)
        G = P.copy()
        G[idx, y] -= 1.0
        gW = (X.T @ G).T / n + lam * W
        gb = G.mean(0) + lam * b
        return loss, np.concatenate([gW.ravel(), gb])
    res = minimize(obj, np.zeros(n_classes * d + n_classes), jac=True,
                   method="L-BFGS-B",
                   options=dict(maxiter=maxiter, ftol=1e-14, gtol=gtol))
    return (res.x[: n_classes * d].reshape(n_classes, d), res.x[n_classes * d:],
            float(np.abs(res.jac).max()))

def lw_shrunk_pooled(X, y):
    # Ledoit-Wolf analytical shrinkage on pooled within-class residuals
    Z = np.vstack([X[y == k] - X[y == k].mean(0) for k in np.unique(y)
                   if (y == k).sum() >= 2])
    n, d = Z.shape
    S = Z.T @ Z / n
    mu = np.trace(S) / d
    delta = (S ** 2).sum() - d * mu ** 2
    num = ((np.sum(Z ** 2, 1) ** 2).sum() - n * (S ** 2).sum()) / (n * n)
    rho = min(num, delta) / delta if (delta > 0 and num > 0) else 0.0
    return (1 - rho) * S + rho * mu * np.eye(d), rho

def sym_inv(A):
    inv = np.linalg.solve(A, np.eye(A.shape[0]))
    return .5 * (inv + inv.T)

def accuracy(W, b, X, y):
    return float(np.mean(np.argmax(X @ W.T + b, 1) == y))

per_seed = []
for seed in SEEDS:
    chunks, toc, order, note = make_partition(GROUPING, seed)
    d = Xtr.shape[1]
    W, b = np.zeros((N, d)), np.zeros(N)
    sigmas, rhos, blocks = [], [], {}
    for t in order:
        chunk = np.array(chunks[t])  # sorted ascending
        m = np.isin(train_y, chunk)
        Wt, bt, gmax = fit_ce_head_lbfgs(Xtr[m], np.searchsorted(chunk, train_y[m]),
                                         C, LAM)
        W[chunk], b[chunk] = Wt, bt
        blocks[t] = (Xtr[m], train_y[m])
        Sg, rho = lw_shrunk_pooled(Xtr[m], train_y[m])
        sigmas.append(Sg)
        rhos.append(rho)
    acc_wt = accuracy(W, b, Xte, test_y)
    Wj, bj, _ = fit_ce_head_lbfgs(Xtr, train_y, N, LAM)   # J arm (union)
    acc_j = accuracy(Wj, bj, Xte, test_y)
    # FIX arm: Def 7.1 with training-side stats, SPEC-5 pooled LW Sigma
    Sigma = np.mean(np.stack(sigmas), 0)
    si = sym_inv(Sigma)
    Xa = np.vstack([blocks[t][0] for t in sorted(blocks)])
    ya = np.concatenate([blocks[t][1] for t in sorted(blocks)])
    mu_hat = np.stack([Xa[ya == k].mean(0) for k in range(N)])
    mu_bar = np.vstack([mu_hat[toc == t].mean(0) for t in range(T)])
    q = np.einsum("ri,ij,rj->r", mu_hat, si, mu_hat)
    S_bar = np.array([q[toc == t].mean() for t in range(T)])
    W_fix = W + (mu_bar @ si)[toc]         # Def 7.1, weight correction
    b_fix = b - .5 * S_bar[toc]            # Def 7.1, threshold correction
    acc_fix = accuracy(W_fix, b_fix, Xte, test_y)
    # diagnostic (tab:bias-direction, training stats only, WT head)
    cos_l, rel_l, cv_l = [], [], []
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            ki, kj = np.flatnonzero(toc == i), np.flatnonzero(toc == j)
            res = np.stack([np.stack([(W[k] - W[l]) - si @ (mu_hat[k] - mu_hat[l])
                                      for l in kj]) for k in ki])
            dh = res.mean((0, 1))
            dp = si @ (mu_bar[j] - mu_bar[i])
            n2h, n2p = dh @ si @ dh, dp @ si @ dp
            if n2h > 0 and n2p > 0:
                cos_l.append((dh @ si @ dp) / np.sqrt(n2h * n2p))
                rel_l.append(np.sqrt(n2h / n2p))
            flat = res.reshape(-1, d)
            mags = np.sqrt(np.maximum((flat @ si * flat).sum(1), 0.0))
            if mags.mean() > 0:
                cv_l.append(mags.std() / mags.mean())
    gap = acc_j - acc_wt
    per_seed.append(dict(
        seed=seed, acc_wt=acc_wt, acc_fix=acc_fix, acc_j=acc_j,
        fix_gain=acc_fix - acc_wt, gap_j_wt=gap,
        fix_recovery=(acc_fix - acc_wt) / gap if abs(gap) > 1e-12 else None,
        cos_mean=float(np.mean(cos_l)), rel_mean=float(np.mean(rel_l)),
        cv_mean=float(np.mean(cv_l)), lw_rho_mean=float(np.mean(rhos)),
        partition=dict(chunks=[list(map(int, c)) for c in chunks],
                       order=[int(t) for t in order], note=note)))
    print(f"[cell] seed {seed}: WT {acc_wt:.4f} FIX {acc_fix:.4f} "
          f"J {acc_j:.4f} cos {np.mean(cos_l):.4f} CV {np.mean(cv_l):.4f}",
          flush=True)

def ms(key):
    v = [s[key] for s in per_seed]
    if any(x is None for x in v):
        return dict(mean=None, sd=None, per_seed=v)
    return dict(mean=float(np.mean(v)),
                sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, per_seed=v)

agg = {k: ms(k) for k in ("acc_wt", "acc_fix", "acc_j", "fix_gain",
                          "gap_j_wt", "fix_recovery", "cos_mean",
                          "rel_mean", "cv_mean")}
gates = dict(cos_ge_090=agg["cos_mean"]["mean"] >= 0.90,
             cv_le_020=agg["cv_mean"]["mean"] <= 0.20)
record = dict(
    run_id="B_nonvision_full", paper="B", experiment="nonvision",
    provenance="colab-full",
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    backend=emb_name, d=int(Xtr.shape[1]), seeds=SEEDS, grouping=GROUPING,
    lam=LAM, cap_per_task=None, per_seed=per_seed, aggregate=agg, gates=gates,
    paper_context=PAPER_CONTEXT,
    decisions=("canonical grouping pin (G03 OWNER-ASK #7); full-batch "
               "L-BFGS on the exact eq:ce objective with weight+bias decay "
               "lam=1e-4 (paper: AdamW 50 epochs -- objective matched, "
               "optimizer substituted); SPEC-5 pooled LW Sigma_hat; mean "
               "over 20 ordered task pairs x 5 seeds; no CI (method "
               "unstated in source)"),
    env=dict(python=platform.python_version(), numpy=np.__version__,
             platform=platform.platform(), machine=platform.machine()),
)
out = "/content/B_nonvision_full.json"
with open(out, "w") as fh:
    json.dump(record, fh, indent=1, default=str)
print("\n[cell] FULL RUN SUMMARY (ours vs paper context, quoted only)")
print(f"  WT  {agg['acc_wt']['mean']*100:.2f} +- {agg['acc_wt']['sd']*100:.2f}   "
      f"(context 64.20 +- 0.98)")
print(f"  FIX {agg['acc_fix']['mean']*100:.2f} +- {agg['acc_fix']['sd']*100:.2f}   "
      f"(context 71.50 +- 0.95)")
print(f"  J   {agg['acc_j']['mean']*100:.2f} +- {agg['acc_j']['sd']*100:.2f}   "
      f"(context 73.84 +- 1.00)")
print(f"  gain {agg['fix_gain']['mean']*100:.2f} | recovery "
      f"{agg['fix_recovery']['mean']*100:.1f}% | cos "
      f"{agg['cos_mean']['mean']:.3f} | CV {agg['cv_mean']['mean']:.3f}")
print(f"  gates: cos>=0.90 {gates['cos_ge_090']} | CV<=0.20 {gates['cv_le_020']}")
print(f"[cell] saved {out} -> download into "
      "repo/results/runs/B/nonvision/B_nonvision_full.json "
      "(then register in results/INDEX.json)")
'''.strip()


# =============================================================================
# CLI
# =============================================================================


def _selfcheck() -> int:
    """Fast, data-free identity checks (zero network, no 20NG load)."""
    # partition invariants
    for grouping in ("canonical", "permuted"):
        part = make_partition(grouping, 3)
        cover = sorted(c for ch in part["chunks"] for c in ch)
        assert cover == list(range(N_CLASSES)), cover
        assert len(part["chunks"]) == T_TASKS
        assert all(len(c) == C_CLASSES for c in part["chunks"])
        assert all(c == sorted(c) for c in part["chunks"])
        assert sorted(part["train_order"]) == list(range(T_TASKS))
    print("PASS partition: 5x4 disjoint cover of the canonical 20 classes, "
          "sorted chunks, valid orders (canonical and permuted)")
    # Def 7.1 algebra on a synthetic Gaussian block (population)
    rng = np.random.default_rng(derive_seed("h07:selfcheck"))
    means = rng.standard_normal((6, 5))
    A = rng.standard_normal((5, 5))
    cov = A @ A.T + 0.5 * np.eye(5)
    toc = np.repeat(np.arange(2), 3)
    if fixm is not None:
        stats = fixm.TaskStats.from_class_stats(means, cov, toc)
        cf = fixm.cf_optimal_head(stats)
        fixed = fixm.de_center(cf, stats)
        bay = fixm.bayes_head(stats)
        dw = float(np.abs(fixed.W - bay.W).max())
        db = float(np.abs(fixed.b - bay.b).max())
        assert dw < 1e-10 and db < 1e-10, (dw, db)
        print(f"PASS de_center == Bayes exactly via H02 (|dW| {dw:.2e}, "
              f"|db| {db:.2e})")
    else:  # pragma: no cover - FALLBACK-H02
        print("NOTE FALLBACK-H02: core.fix absent; Bayes identity check "
              "limited to the local algebra path")
    # WT-style sequential fits vs the within-task closed form + Thm 4.1
    # alignment (n=2000/class, d=5). NOTE: per-class weight cosine at
    # lam=0 is ill-posed (within-task gauge freedom), and comparing a FULL
    # softmax MLE against the within-task CF head measures exactly the
    # Theorem 4.1 structural bias (verified vs sklearn to 8 decimals
    # during H07 bring-up: MLE==ours, MLE vs CF-head disc-cos ~0.88 is
    # the theorem, not a defect). Discriminators are gauge-invariant.
    n_pc = 2000
    L = np.linalg.cholesky(cov)
    X = np.vstack([means[k] + rng.standard_normal((n_pc, 5)) @ L.T
                   for k in range(6)])
    y = np.repeat(np.arange(6), n_pc)
    stats2 = fixm.TaskStats.from_data(X, y, toc)
    cf2 = fixm.cf_optimal_head(stats2)
    W_as = np.zeros((6, 5))
    b_as = np.zeros(6)
    for t in range(2):
        idx = np.flatnonzero(toc == t)
        m = np.isin(y, idx)
        Wt, bt, _ = fit_ce_head_lbfgs(X[m], np.searchsorted(idx, y[m]),
                                      len(idx), LAM_DEFAULT)
        W_as[idx], b_as[idx] = Wt, bt
        for a in range(len(idx)):
            for b_i in range(a + 1, len(idx)):
                d1 = Wt[a] - Wt[b_i]
                d2 = cf2.W[idx[a]] - cf2.W[idx[b_i]]
                c = float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2)))
                assert c > 0.999, (t, a, b_i, c)
    print("PASS sequential WT fits == within-task closed form on Gaussian "
          "block (within-task discriminator cos > 0.999, lam=1e-4)")
    # Theorem 4.1 on the assembled WT head: residual block aligns with
    # Sigma^{-1}(mu_bar_j - mu_bar_i)
    si = _fb_sigma_inv(stats2.Sigma)
    res = np.stack([np.stack([(W_as[k] - W_as[l]) - si @ (stats2.means[k]
                                                          - stats2.means[l])
                              for l in range(6) if toc[l] != toc[k]])
                    for k in range(6)])
    cents = np.vstack([stats2.means[toc == t].mean(0) for t in range(2)])
    worst = 1.0
    for k in range(6):
        for jj, l in enumerate([q for q in range(6) if toc[q] != toc[k]]):
            pred = si @ (cents[toc[l]] - cents[toc[k]])
            B = res[k, jj]
            worst = min(worst, float(
                B @ si @ pred / np.sqrt((B @ si @ B) * (pred @ si @ pred))))
    assert worst > 0.99, worst
    print(f"PASS Theorem 4.1 alignment on assembled WT head: min Sigma-inner "
          f"cos(residual, predicted) = {worst:.4f} (> 0.99)")
    # TF-IDF fallback path executes on a toy corpus (no download involved)
    emb = TfidfEmbedder()
    Xt = emb.fit_transform(["alpha beta gamma", "beta gamma delta",
                            "epsilon zeta eta", "zeta eta theta"])
    assert Xt.shape[0] == 4 and np.isfinite(Xt).all()
    print(f"PASS TF-IDF fallback path executes (toy corpus, d={emb.d})")
    # Colab cell sanity
    for marker in ("B_nonvision_full", "all-MiniLM-L6-v2", "fetch_20newsgroups",
                   "G03 SPEC 9", "never asserted", "L-BFGS"):
        assert marker in COLAB_CELL, marker
    print(f"PASS Colab cell present ({len(COLAB_CELL.splitlines())} lines, "
          "key markers verified)")
    print(f"drift notes ({len(DRIFT_NOTES)}):")
    for note in DRIFT_NOTES:
        print(f"  - {note}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", default="smoke",
                   choices=["smoke", "full", "colab", "selfcheck"])
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--grouping", default="canonical",
                   choices=["canonical", "permuted"])
    p.add_argument("--lam", type=float, default=LAM_DEFAULT)
    p.add_argument("--cap-per-task", type=int, default=200,
                   help="smoke cap: max docs per task (stratified 50/class)")
    p.add_argument("--data-home", default=None)
    p.add_argument("--out-root", default=str(_REPO_ROOT / "results"))
    p.add_argument("--update-index", action="store_true",
                   help="also update results/INDEX.json (opt-in: callers on "
                        "a strict file budget leave this off)")
    p.add_argument("--allow-local-full", action="store_true",
                   help="full mode is for Colab-side execution; locally the "
                        "owner storage law routes the FULL run to Colab")
    p.add_argument("--allow-download", action="store_true",
                   help="COLAB ONLY: permit the 20NG fetch and the MiniLM hub "
                        "load to download. Never pass this on the owner machine.")
    p.add_argument("--run-label", default=None,
                   help="override the run id suffix (default: smoke|full)")
    args = p.parse_args(argv)
    global ALLOW_DOWNLOAD
    if args.allow_download:
        ALLOW_DOWNLOAD = True
        DRIFT_NOTES.append("ALLOW_DOWNLOAD=True (Colab mode): 20NG fetch and MiniLM hub load may download.")

    seeds = [int(s) for s in str(args.seeds).split(",") if s != ""]

    if args.mode == "selfcheck":
        return _selfcheck()

    if args.mode == "colab":
        print(COLAB_CELL)
        print("\n[h07] emit the cell above verbatim into the J-wave notebook; "
              "its JSON lands at results/runs/B/nonvision/"
              "B_nonvision_full.json after download.", file=sys.stderr)
        return 0

    # data-requiring modes: probe the cache first (never download)
    if not cache_present(args.data_home):
        print("[h07] SKIP local execution: sklearn 20NG cache absent "
              "(download_if_missing=False raised); downloads are forbidden "
              "by the owner storage law. The FULL run happens on Colab "
              "(--mode colab emits the cell). Nothing was written.",
              file=sys.stderr)
        return 0

    if args.mode == "full" and not args.allow_local_full:
        print("[h07] REFUSED: the FULL run happens on Google Colab per the "
              "owner storage law. Use --mode colab (emits the cell), or pass "
              "--allow-local-full deliberately on a machine whose cache "
              "legitimately covers the full data.", file=sys.stderr)
        return 2

    started = _utcnow()
    cap = args.cap_per_task if args.mode == "smoke" else None
    tr = load_20ng("train", args.data_home)
    te = load_20ng("test", args.data_home)
    embedder, backend = build_embedder("minilm")
    record = run_experiment(
        list(tr.data), np.asarray(tr.target), list(te.data),
        np.asarray(te.target), embedder, backend, seeds, args.grouping,
        args.lam, cap, purpose_tag=f"h07:{args.mode}")
    run_id = f"B_{EXPERIMENT}_{args.run_label or ('smoke' if args.mode == 'smoke' else 'full')}"
    provenance = ("smoke-local-cached" if args.mode == "smoke"
                  else ("colab-full" if ALLOW_DOWNLOAD else "local-cached-full"))
    cfg = dict(seeds=seeds, grouping=args.grouping, lam=args.lam,
               cap_per_task=cap, dataset="20newsgroups-by-date",
               remove=("headers", "footers", "quotes"), d_expected=384)
    path = write_run_json(record, Path(args.out_root), run_id, provenance,
                          started, cfg, update_index=args.update_index)
    print_summary(record, run_id)
    print(f"[h07] {args.mode} record -> {path} "
          f"(index update {'ON' if args.update_index else 'OFF'})")
    print(f"[h07] drift notes: {DRIFT_NOTES}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
