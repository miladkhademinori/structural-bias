"""core.lda_ncm -- LDA / NCM heads and ordering-aware evaluation on cached features (Paper B).

IMPLEMENTED by H03, 2026-10-21 (replaces the G09 skeleton placeholder).
Spec sources:
  - N1000_INIT_2026-10-21/exec/G03_log.md SPEC 5 (LDA/NCM columns of
    tab:lda-ncm-comparison) and SPEC 8 (benchmark protocol: task stages,
    seeds 0-4 orderings, final average accuracy convention).
  - exec/G08_log.md S2 (results-store JSON schema, provenance values) and
    S4 (b_lda_ncm input contract: features npz per backbone/dataset/seed,
    one-decimal mean +- sd over seeds, MISSING never estimated).

Semantics (G03 SPEC 5, all closed-form, evaluation-only, no training):
  - LDA (SLDA-style): score_r(x) = mu_r' Sigma^-1 x - (1/2) mu_r' Sigma^-1 mu_r
    with mu_r the class means from TRAINING data only (uniform priors: no
    prior term) and Sigma the Ledoit-Wolf shrunk pooled within-class
    covariance. Population equivalence: the de-centered head of Paper B
    equals this shared-covariance LDA discriminant exactly.
  - NCM: same discriminant with Sigma = I (nearest class mean over
    Euclidean cached features).
  - Sigma_hat rule (source L2050-2062): computed once per task on that
    task's training data, then pooled across tasks by sample-weighted
    averaging (balanced tasks => plain average).
  - Metric: final average accuracy on the CUMULATIVE test set at sequence
    end (all tasks arrived), percent, one decimal, mean +- sd over seeds.

Ledoit-Wolf implementation: analytical closed form of Ledoit & Wolf (2004),
"A well-conditioned estimator for large-dimensional covariance matrices",
J. Multivariate Anal. 88, 365-411 -- identical to
sklearn.covariance.LedoitWolf (biased empirical covariance, target mu*I,
intensity beta/delta). NOT cross-validated. The self-check asserts
bit-level agreement with sklearn (< 1e-12 relative) when sklearn is
installed; the formula is therefore "sklearn-compatible analytical".
Single deviation: when the numerator/denominator are non-positive
(degenerate constant data) we return intensity 0 instead of sklearn's
signed ratio; this only differs on measure-zero pathological inputs.

MISSING policy (G08 S4 / N24): absent or unreadable features for a seed are
recorded as null and reported as MISSING at aggregation time. Never
imputed, never estimated, never silently dropped.

Bridge exports `task_lw_covariance` / `pooled_covariance` /
`lda_final_accuracy` / `ncm_final_accuracy` give experiments/
lda_ncm_eval.py one shared source for the same numbers (it prefers this
module and labels its fallback path when these are absent).

Laws honoured: writes only this module + its exec log (H03); no OneDrive
paths; dependencies numpy/scipy (+ sklearn for the self-check cross-check
only -- the estimator itself is pure numpy/scipy).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.linalg

__all__ = [
    "LinearHead",
    "StreamingGaussianHead",
    "LDA_METHOD",
    "NCM_METHOD",
    "PROVENANCE_VALUES",
    "MISSING",
    "ledoit_wolf_shrinkage",
    "lw_pooled_covariance",
    "head_from_parameters",
    "fit_lda_head",
    "fit_ncm_head",
    "evaluate_head",
    "evaluate_cil_sequence",
    "task_lw_covariance",
    "pooled_covariance",
    "lda_final_accuracy",
    "ncm_final_accuracy",
    "format_pm",
    "FeaturesFile",
    "load_features_npz",
    "save_features_npz",
    "self_check",
    "main",
]

LDA_METHOD = "lda"
NCM_METHOD = "ncm"
#: Exactly the three provenance values allowed by G08 S2 rule (b).
PROVENANCE_VALUES = ("synthetic", "colab", "owner-artifact")
#: The MISSING cell string (G08 S4); never a number, never an estimate.
MISSING = "MISSING"


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrunk covariance (analytical, sklearn-compatible)
# ---------------------------------------------------------------------------


def ledoit_wolf_shrinkage(x: np.ndarray, *, assume_centered: bool = False) -> float:
    """Analytical Ledoit-Wolf shrinkage intensity of Ledoit & Wolf (2004).

    Matches sklearn.covariance.LedoitWolf.shrinkage_ exactly (verified in
    self_check to < 1e-12 on random and degenerate inputs). Uses the biased
    empirical covariance S = X'X/n and the scaled identity target mu*I with
    mu = tr(S)/d. sklearn works with the d-scaled misfit delta/d; the same
    ratio results from fully unscaled quantities:

        delta = ||S - mu*I||_F^2 = tr(S @ S) - d * mu^2      (unscaled)
        num   = (sum_k ||x_k||^4 - n * tr(S @ S)) / n^2      (unscaled beta)
        shrinkage = min(num, delta) / delta

    sum_k ||x_k||^4 is the fourth power of ROW NORMS (cross terms inside
    the norm), not the sum of elementwise fourth powers -- the distinction
    is load-bearing.

    Parameters
    ----------
    x : (n, d) array of samples.
    assume_centered : if False (default) rows are centered by the feature
        mean first, mirroring the sklearn flag.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (n, d), got shape {x.shape}")
    n, d = x.shape
    if d == 1:  # sklearn short-circuits: intensity irrelevant for 1 feature
        return 0.0
    xc = x if assume_centered else x - x.mean(axis=0)
    s = (xc.T @ xc) / n
    mu = float(np.trace(s)) / d
    tr_s2 = float((s**2).sum())
    # sum_k ||x_k||^4 (row-norm fourth powers, WITH cross terms) -- NOT
    # the elementwise fourth power; this distinction is load-bearing.
    sum_row4 = float((np.sum(xc**2, axis=1) ** 2).sum())
    delta = tr_s2 - d * mu**2  # = ||S - mu*I||_F^2 since tr(S) = d*mu
    num = (sum_row4 - n * tr_s2) / (n * n)
    if delta <= 0.0 or num <= 0.0:
        return 0.0
    return float(min(num, delta) / delta)


def shrunk_pooled_covariance(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    shrinkage: str | float | None = "ledoit-wolf",
) -> tuple[np.ndarray, float]:
    """Pooled within-class covariance of one data block, LW-shrunk.

    Centering is per class (within-class), concatenating every class with
    >= 2 samples; classes with a single sample contribute to means but not
    to the covariance pool. Returns (covariance, intensity).
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(
            f"features {x.shape} and labels {y.shape} disagree on n samples"
        )
    blocks = []
    for cls in np.unique(y):
        rows = x[y == cls]
        if rows.shape[0] >= 2:
            blocks.append(rows - rows.mean(axis=0))
    if not blocks:
        raise ValueError("no class has >= 2 samples; within-class covariance undefined")
    centered = np.vstack(blocks)
    n = centered.shape[0]
    emp = (centered.T @ centered) / n  # biased, matches sklearn
    d = x.shape[1]
    if shrinkage is None or shrinkage in ("none", "unshrunk"):
        intensity = 0.0
    elif isinstance(shrinkage, str) and shrinkage.lower() in ("ledoit-wolf", "lw"):
        intensity = ledoit_wolf_shrinkage(centered, assume_centered=True)
    elif isinstance(shrinkage, (int, float)):
        intensity = float(shrinkage)
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"shrinkage intensity must be in [0, 1], got {intensity}")
    else:
        raise ValueError(f"unknown shrinkage setting: {shrinkage!r}")
    if intensity > 0.0:
        mu = float(np.trace(emp)) / d
        cov = (1.0 - intensity) * emp + intensity * mu * np.eye(d)
    else:
        cov = emp
    return cov, intensity


def lw_pooled_covariance(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    shrinkage: str | float | None = "ledoit-wolf",
) -> np.ndarray:
    """Ledoit-Wolf shrunk pooled within-class covariance (single block).

    Exported for reuse by the A-side modules (margin/diagnostic: gamma_min
    and bias-direction statistics need the same Sigma_hat from cached
    features). Sigma_hat = (1 - rho) S_pool + rho * mu * I with the
    analytical LW intensity rho (Ledoit & Wolf 2004, sklearn-compatible,
    NOT cross-validated) and mu = tr(S_pool)/d.
    """
    return shrunk_pooled_covariance(features, labels, shrinkage=shrinkage)[0]


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LinearHead:
    """Linear Gaussian head in discriminant form scores(x) = x @ W' + b.

    For LDA: W = Mu Sigma^-1, b_r = -(1/2) mu_r' Sigma^-1 mu_r
    (uniform priors; G03 SPEC 5). For NCM: Sigma = I, i.e. W = Mu,
    b_r = -(1/2) ||mu_r||^2. `covariance` keeps the Sigma used (None for
    identity/NCM) so diagnostics can reuse it.
    """

    classes: np.ndarray  # (k,) sorted int class ids
    weights: np.ndarray  # (k, d)
    bias: np.ndarray  # (k,)
    kind: str = "custom"  # "lda" | "ncm" | "custom"
    covariance: np.ndarray | None = None  # (d, d) if not identity

    def scores(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        return x @ self.weights.T + self.bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.scores(features), axis=1)]

    @property
    def n_classes(self) -> int:
        return int(self.classes.shape[0])


def head_from_parameters(
    means: np.ndarray,
    covariance: np.ndarray | None = None,
    classes: np.ndarray | None = None,
) -> LinearHead:
    """Head from explicit parameters (Bayes ground truth; A-side reuse).

    With the TRUE class means and TRUE shared covariance this is exactly
    the Bayes-optimal linear discriminant under uniform priors (canonical
    form w_r = Sigma^-1 mu_r, b_r = -(1/2)||mu_r||^2_{Sigma^-1}; the global
    centroid shift cancels in argmax). covariance=None means identity (NCM
    form).
    """
    mu = np.asarray(means, dtype=np.float64)
    if mu.ndim != 2:
        raise ValueError(f"means must be (k, d), got {mu.shape}")
    k, d = mu.shape
    if classes is None:
        classes = np.arange(k)
    classes = np.asarray(classes)
    if classes.shape[0] != k:
        raise ValueError("classes length must match number of mean rows")
    if covariance is None:
        weights, bias = mu, -0.5 * np.sum(mu**2, axis=1)
    else:
        cov = np.asarray(covariance, dtype=np.float64)
        chol = scipy.linalg.cho_factor(cov, lower=True)
        weights = scipy.linalg.cho_solve(chol, mu.T).T  # Mu @ cov^-1
        bias = -0.5 * np.sum(mu * weights, axis=1)
    return LinearHead(classes=classes, weights=weights, bias=bias,
                      kind="custom", covariance=covariance)


class StreamingGaussianHead:
    """Incremental class-statistics maintainer (SLDA-style, no training).

    One `update` per arriving task stage with that task's TRAINING
    features/labels. Maintains per-class running sums/counts (new classes
    added, existing classes updated -- shared-label DomainNet stress rows
    therefore work) and the per-task LW covariance pool averaged across
    tasks with weights = per-task within-class sample counts (balanced
    tasks => plain average, per source L2050-2062).

    method="lda": Sigma = LW shrunk pooled covariance.
    method="ncm": Sigma = I (identity covariance specialization).
    shrinkage: "ledoit-wolf" (default) | "none"/None | float intensity,
    applied per task block before pooling.
    """

    def __init__(
        self,
        *,
        method: str = LDA_METHOD,
        shrinkage: str | float | None = "ledoit-wolf",
    ) -> None:
        if method not in (LDA_METHOD, NCM_METHOD):
            raise ValueError(f"method must be '{LDA_METHOD}' or '{NCM_METHOD}'")
        self.method = method
        self.shrinkage = shrinkage
        self._d: int | None = None
        self._sums: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}
        self._cov_acc: np.ndarray | None = None  # weight * cov accumulator
        self._cov_weight = 0.0
        self.last_intensity: float | None = None  # last task block intensity

    # -- statistics ------------------------------------------------------

    def update(self, features: np.ndarray, labels: np.ndarray) -> "StreamingGaussianHead":
        """Absorb one task stage (training features/labels)."""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels)
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(
                f"features {x.shape} and labels {y.shape} disagree on n samples"
            )
        if self._d is None:
            self._d = x.shape[1]
        elif x.shape[1] != self._d:
            raise ValueError(f"feature dim changed: {self._d} -> {x.shape[1]}")

        block_means: dict[int, np.ndarray] = {}
        for cls in np.unique(y):
            cls = int(cls)
            rows = x[y == cls]
            m = rows.mean(axis=0)
            block_means[cls] = m
            self._sums[cls] = m * rows.shape[0] + self._sums.get(cls, 0.0)
            self._counts[cls] = int(rows.shape[0]) + self._counts.get(cls, 0)

        if self.method == NCM_METHOD:
            return self  # identity covariance: nothing to pool

        cov, intensity = shrunk_pooled_covariance(x, y, shrinkage=self.shrinkage)
        self.last_intensity = intensity
        blocks = [x[y == c] - block_means[c] for c in block_means if (y == c).sum() >= 2]
        weight = float(sum(b.shape[0] for b in blocks))
        if self._cov_acc is None:
            self._cov_acc = np.zeros_like(cov)
        self._cov_acc += weight * cov
        self._cov_weight += weight
        return self

    # -- head ------------------------------------------------------------

    def means(self) -> tuple[np.ndarray, np.ndarray]:
        """(sorted classes, means matrix) from statistics seen so far."""
        if not self._counts:
            raise ValueError("no data observed; head undefined")
        classes = np.array(sorted(self._counts), dtype=np.int64)
        means = np.vstack(
            [self._sums[int(c)] / self._counts[int(c)] for c in classes]
        )
        return classes, means

    def covariance(self) -> np.ndarray | None:
        """Current pooled Sigma (None means identity for NCM)."""
        if self.method == NCM_METHOD:
            return None
        if self._cov_weight <= 0.0:
            return None  # no class had >= 2 samples yet: identity fallback
        return self._cov_acc / self._cov_weight

    def n_classes(self) -> int:
        """Number of distinct classes observed so far."""
        return len(self._counts)

    def head(self) -> LinearHead:
        classes, means = self.means()
        cov = self.covariance()
        if cov is None:
            weights, bias = means, -0.5 * np.sum(means**2, axis=1)
        else:
            chol = scipy.linalg.cho_factor(cov, lower=True)
            weights = scipy.linalg.cho_solve(chol, means.T).T
            bias = -0.5 * np.sum(means * weights, axis=1)
        return LinearHead(classes=classes, weights=weights, bias=bias,
                          kind=self.method, covariance=cov)


def fit_lda_head(
    features: np.ndarray,
    labels: np.ndarray,
    shrinkage: str | float | None = "ledoit-wolf",
) -> LinearHead:
    """Fit the SLDA-style LDA head from one feature block (single task).

    Analytical Ledoit-Wolf shrunk pooled within-class covariance (Ledoit &
    Wolf 2004 closed form == sklearn.covariance.LedoitWolf; NOT
    cross-validated). Discriminant: mu_r' Sigma^-1 x - (1/2) mu_r'
    Sigma^-1 mu_r, uniform priors, means from training data only.
    """
    return StreamingGaussianHead(method=LDA_METHOD, shrinkage=shrinkage).update(
        features, labels
    ).head()


def fit_ncm_head(features: np.ndarray, labels: np.ndarray) -> LinearHead:
    """Fit the NCM head: class means + identity covariance (Sigma = I)."""
    return StreamingGaussianHead(method=NCM_METHOD).update(features, labels).head()


def evaluate_head(head: LinearHead, features: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy of `head` on (features, labels), as a fraction in [0, 1].

    Labels the head has never seen simply count as errors (they can never
    be argmax-predicted). Multiply by 100 for the paper's percent metric.
    """
    y = np.asarray(labels)
    pred = head.predict(features)
    if pred.shape != y.shape:
        raise ValueError(f"prediction shape {pred.shape} vs labels {y.shape}")
    return float(np.mean(pred == y))


# ---------------------------------------------------------------------------
# Ordering-aware CIL evaluation harness
# ---------------------------------------------------------------------------


def evaluate_cil_sequence(
    train_stages: Sequence[tuple[np.ndarray, np.ndarray]],
    test_stages: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    method: str = LDA_METHOD,
    shrinkage: str | float | None = "ledoit-wolf",
) -> dict[str, Any]:
    """Run the class-incremental protocol over cached per-task features.

    Protocol (G03 SPEC 8 / G08 S4): tasks arrive in the given order; after
    each stage t the head is updated with task t's TRAINING data (new
    classes join incrementally) and evaluated on the CUMULATIVE test set of
    tasks 1..t. The benchmark metric -- final average accuracy A-bar -- is
    the accuracy after the LAST task on the full cumulative test set.

    Returns a dict with per-stage cumulative accuracies (fractions),
    `final_accuracy` (fraction), counts, and method metadata. Evaluation
    only: no gradients, no epochs, cached features only.
    """
    if len(train_stages) != len(test_stages):
        raise ValueError(
            f"train/test stage count mismatch: {len(train_stages)} vs {len(test_stages)}"
        )
    if not train_stages:
        raise ValueError("empty task sequence")
    stream = StreamingGaussianHead(method=method, shrinkage=shrinkage)
    per_stage: list[float] = []
    test_acc: list[np.ndarray] = []
    test_labels_acc: list[np.ndarray] = []
    n_classes = 0
    for stage, ((xtr, ytr), (xte, yte)) in enumerate(zip(train_stages, test_stages)):
        stream.update(xtr, ytr)
        test_acc.append(np.asarray(xte, dtype=np.float64))
        test_labels_acc.append(np.asarray(yte))
        n_classes = stream.n_classes()
        head = stream.head()
        per_stage.append(evaluate_head(head, np.vstack(test_acc), np.concatenate(test_labels_acc)))
    return {
        "method": method,
        "shrinkage": str(shrinkage),
        "n_tasks": len(train_stages),
        "n_classes": n_classes,
        "per_stage_accuracy": per_stage,  # cumulative test after each stage
        "final_accuracy": per_stage[-1],  # A-bar convention (sequence end)
    }


def format_pm(mean_pct: float | None, sd_pct: float | None) -> str:
    """One-decimal mean +- sd cell (G08 S4: `84.1 \\pm 0.4`); MISSING if absent."""
    if mean_pct is None or sd_pct is None:
        return MISSING
    return f"{mean_pct:.1f} \u00b1 {sd_pct:.1f}"


# ---------------------------------------------------------------------------
# Bridge exports for experiments/lda_ncm_eval.py (same numbers, one source)
# ---------------------------------------------------------------------------


def task_lw_covariance(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray | None = None,
) -> np.ndarray:
    """Ledoit-Wolf shrunk WITHIN-class covariance of ONE task's train block.

    Bridge export (experiments/lda_ncm_eval.py prefers core when present).
    Numerically identical to sklearn
    `LedoitWolf(assume_centered=True).fit(task_centered_rows).covariance_`
    (analytical 2004 closed form; asserted against sklearn in self_check).
    `classes` is accepted for signature parity and ignored -- the labels in
    `features`/`labels` are the authority.
    """
    return lw_pooled_covariance(features, labels)


def pooled_covariance(covariances: Sequence[np.ndarray]) -> np.ndarray:
    """Equal-weight average of per-task LW covariances (G03 SPEC 5 pooling:
    balanced tasks, so sample-weighted and equal-weight coincide)."""
    arr = np.asarray(covariances, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] == 0:
        raise ValueError(
            f"pooled_covariance expects a non-empty stack of (d, d) matrices, "
            f"got shape {arr.shape}"
        )
    return arr.mean(axis=0)


def _final_accuracy_from_stream(
    train_tasks: Sequence[tuple[np.ndarray, np.ndarray]],
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    method: str,
) -> float:
    stream = StreamingGaussianHead(method=method)
    for x_t, y_t in train_tasks:
        stream.update(x_t, y_t)
    head = stream.head()
    unseen = set(np.unique(np.asarray(test_y)).tolist()) - set(
        head.classes.tolist()
    )
    if unseen:
        raise ValueError(
            f"test set contains classes with no training data: {sorted(unseen)}"
        )
    return 100.0 * evaluate_head(head, test_x, test_y)


def lda_final_accuracy(
    train_tasks: Sequence[tuple[np.ndarray, np.ndarray]],
    classes: np.ndarray | None,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> float:
    """LDA final average accuracy A-bar in PERCENT on the cumulative test set.

    Bridge export matching the experiments/lda_ncm_eval.py contract: means
    pool every training row of each class; Sigma = sample-weighted pooled
    per-task LW covariance (equal-weight on balanced tasks). `classes` is
    accepted for signature parity and ignored (data is the authority).
    """
    return _final_accuracy_from_stream(
        train_tasks, test_x, test_y, method=LDA_METHOD
    )


def ncm_final_accuracy(
    train_tasks: Sequence[tuple[np.ndarray, np.ndarray]],
    classes: np.ndarray | None,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> float:
    """NCM final average accuracy in PERCENT (Sigma = I; bridge export)."""
    return _final_accuracy_from_stream(
        train_tasks, test_x, test_y, method=NCM_METHOD
    )


# ---------------------------------------------------------------------------
# Features npz I/O (b_lda_ncm input contract, G08 S4)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FeaturesFile:
    """One seed's cached features for a (backbone, dataset) row."""

    path: Path
    backbone: str
    dataset: str
    seed: int
    provenance: str
    provenance_source: str  # "npz" | "meta" | "cli"
    executed_utc: str | None
    meta: dict[str, Any]
    train_stages: list[tuple[np.ndarray, np.ndarray]]
    test_stages: list[tuple[np.ndarray, np.ndarray]]


_TRAIN_KEY = re.compile(r"^train_x_(\d+)$")


def _str_from_npz(z: Any, key: str) -> str | None:
    if key not in z:
        return None
    val = z[key]
    return val.item() if val.shape == () else str(val)


def load_features_npz(
    path: Path | str,
    *,
    backbone: str | None = None,
    dataset: str | None = None,
    seed: int | None = None,
    provenance_fallback: str | None = None,
) -> FeaturesFile:
    """Load a cached-features npz (G08 S4 layout).

    Required keys: `train_x_<i>`, `train_y_<i>` for i = 0..T-1 (task
    arrival order, global integer labels). Test data: either per-stage
    `test_x_<i>`/`test_y_<i>` or one cumulative `test_x`/`test_y` (split by
    label ownership inferred from the training labels). Source marker:
    string key `provenance`, or `provenance` inside the JSON string key
    `meta`, or the caller's `provenance_fallback` (recorded as
    provenance_source). Marker value must be one of PROVENANCE_VALUES
    (G08 S2 rule b); an absent or invalid marker raises ValueError --
    provenance is never guessed.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        idx = sorted(int(m.group(1)) for k in z.files if (m := _TRAIN_KEY.match(k)))
        if not idx:
            raise ValueError(f"{path.name}: no train_x_<i> arrays")
        if idx != list(range(len(idx))):
            raise ValueError(f"{path.name}: non-contiguous task indices {idx}")
        train_stages = [
            (np.asarray(z[f"train_x_{i}"], dtype=np.float64),
             np.asarray(z[f"train_y_{i}"]))
            for i in idx
        ]
        test_stages: list[tuple[np.ndarray, np.ndarray]] | None = None
        if "test_x_0" in z.files:
            test_stages = [
                (np.asarray(z[f"test_x_{i}"], dtype=np.float64),
                 np.asarray(z[f"test_y_{i}"]))
                for i in idx
                if f"test_x_{i}" in z.files
            ]
            if len(test_stages) != len(idx):
                raise ValueError(f"{path.name}: partial per-stage test arrays")
        elif "test_x" in z.files:
            xte = np.asarray(z["test_x"], dtype=np.float64)
            yte = np.asarray(z["test_y"])
            owner: dict[int, int] = {}
            for i, (_, ytr) in enumerate(train_stages):
                for lab in np.unique(ytr):
                    owner.setdefault(int(lab), i)
            parts_x: list[np.ndarray] = []
            parts_y: list[np.ndarray] = []
            stage_rows: list[list[int]] = [[] for _ in idx]
            for row, lab in enumerate(yte):
                lab_i = int(lab)
                if lab_i not in owner:
                    raise ValueError(f"{path.name}: test label {lab_i} owned by no task")
                stage_rows[owner[lab_i]].append(row)
            for i in idx:
                rows = np.asarray(stage_rows[i], dtype=int)
                parts_x.append(xte[rows])
                parts_y.append(yte[rows])
            test_stages = list(zip(parts_x, parts_y))
        else:
            raise ValueError(f"{path.name}: no test arrays (test_x_<i> or test_x)")

        meta: dict[str, Any] = {}
        raw_meta = _str_from_npz(z, "meta")
        if raw_meta:
            meta = json.loads(raw_meta)
        meta_backbone = meta.get("backbone", _str_from_npz(z, "backbone"))
        meta_dataset = meta.get("dataset", _str_from_npz(z, "dataset"))
        meta_seed = meta.get("seed", _str_from_npz(z, "seed"))
        executed = meta.get("executed_utc", _str_from_npz(z, "executed_utc"))
        prov_npz = _str_from_npz(z, "provenance")
        prov_meta = meta.get("provenance")

    backbone = backbone or meta_backbone
    dataset = dataset or meta_dataset
    seed = seed if seed is not None else (int(meta_seed) if meta_seed is not None else -1)
    provenance, source = None, None
    for cand, src in ((prov_npz, "npz"), (prov_meta, "meta"),
                      (provenance_fallback, "cli")):
        if cand is not None:
            provenance, source = str(cand), src
            break
    if provenance not in PROVENANCE_VALUES:
        raise ValueError(
            f"{path.name}: provenance marker absent/invalid ({provenance!r}); "
            f"expected one of {PROVENANCE_VALUES}; pass --provenance explicitly"
        )
    return FeaturesFile(
        path=path,
        backbone=str(backbone) if backbone is not None else "",
        dataset=str(dataset) if dataset is not None else "",
        seed=int(seed),
        provenance=provenance,
        provenance_source=source or "",
        executed_utc=executed,
        meta=meta,
        train_stages=train_stages,
        test_stages=test_stages,
    )


def save_features_npz(
    path: Path | str,
    *,
    train_stages: Sequence[tuple[np.ndarray, np.ndarray]],
    test_stages: Sequence[tuple[np.ndarray, np.ndarray]],
    backbone: str,
    dataset: str,
    seed: int,
    provenance: str,
    executed_utc: str | None = None,
    extra_meta: dict[str, Any] | None = None,
    per_stage_test: bool = False,
) -> Path:
    """Write a features npz conforming to load_features_npz (inverse op).

    Used by the self-check / tests to prove the CLI round-trip; the real
    store is produced by the colab feature cache (provenance "colab").
    """
    if provenance not in PROVENANCE_VALUES:
        raise ValueError(f"provenance must be one of {PROVENANCE_VALUES}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    for i, (x, y) in enumerate(train_stages):
        arrays[f"train_x_{i}"] = np.asarray(x, dtype=np.float64)
        arrays[f"train_y_{i}"] = np.asarray(y)
    if per_stage_test:
        for i, (x, y) in enumerate(test_stages):
            arrays[f"test_x_{i}"] = np.asarray(x, dtype=np.float64)
            arrays[f"test_y_{i}"] = np.asarray(y)
    else:
        arrays["test_x"] = np.vstack([np.asarray(x) for x, _ in test_stages])
        arrays["test_y"] = np.concatenate([np.asarray(y) for _, y in test_stages])
    meta = {
        "backbone": backbone,
        "dataset": dataset,
        "seed": int(seed),
        "provenance": provenance,
        "T": len(train_stages),
        "n_classes": int(sum(len(np.unique(y)) for _, y in train_stages)),
        "executed_utc": executed_utc,
        **(extra_meta or {}),
    }
    arrays["meta"] = json.dumps(meta)
    np.savez(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# Results-store emission (G08 S2 format)
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("_")


def aggregate_seeds(
    finals: list[float | None],
) -> tuple[float | None, float | None, int, int]:
    """mean/sd (PERCENT) over non-null seed finals; ddof=1 sd for n>=2.

    Returns (mean, sd, n_used, n_missing). Never imputes: nulls are only
    counted. sd of a single seed is 0.0 (undefined sample sd, convention:
    report the value itself with 0 spread).
    """
    vals = [v for v in finals if v is not None]
    n_missing = len(finals) - len(vals)
    if not vals:
        return None, None, 0, n_missing
    pct = np.asarray(vals, dtype=np.float64) * 100.0
    mean = float(pct.mean())
    sd = float(pct.std(ddof=1)) if pct.size >= 2 else 0.0
    return mean, sd, len(vals), n_missing


def run_one_features_file(
    feat: FeaturesFile, methods: Iterable[str] = (LDA_METHOD, NCM_METHOD)
) -> dict[str, Any]:
    """Per-seed final accuracies (fractions) for each requested method."""
    out: dict[str, Any] = {}
    for m in methods:
        rep = evaluate_cil_sequence(
            feat.train_stages, feat.test_stages, method=m
        )
        out[m] = rep
    return out


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _import_conftest() -> Any:
    tests_dir = Path(__file__).resolve().parent.parent / "tests"
    sys.path.insert(0, str(tests_dir))
    import conftest  # noqa: E402  (G10 test framework, shared factories)

    return conftest


def self_check(verbose: bool = True) -> int:
    """Self-check main: LW-vs-sklearn + synthetic CIL LDA/NCM/Bayes.

    Tier: VERIFICATION on synthetic ground truth via the G10 conftest
    fixtures (GaussianTask). Prints LDA vs NCM vs Bayes final accuracies.
    Returns number of failures (0 = pass). No paper number is asserted
    (NO REAL-BACKBONE RULE; synthetic-proxy only).
    """
    failures = 0

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("== core.lda_ncm self-check (H03) ==")

    # --- 1. analytical LW vs sklearn.covariance.LedoitWolf -------------
    ct = _import_conftest()
    try:
        from sklearn.covariance import LedoitWolf

        rng = ct.new_rng(purpose="h03-lw-crosscheck")
        worst_s, worst_c = 0.0, 0.0
        for n, d in ((500, 8), (60, 20), (2000, 50), (5000, 200), (30, 5)):
            a = rng.standard_normal((d, d))
            x = rng.standard_normal((n, d)) @ a.T
            sk = LedoitWolf().fit(x)
            ana_rho = ledoit_wolf_shrinkage(x)
            emp = (x - x.mean(0)).T @ (x - x.mean(0)) / n
            mu = float(np.trace(emp)) / d
            ana_cov = (1 - ana_rho) * emp + ana_rho * mu * np.eye(d)
            worst_s = max(worst_s, abs(sk.shrinkage_ - ana_rho))
            worst_c = max(worst_c, float(np.abs(sk.covariance_ - ana_cov).max()))
        ok = worst_s < 1e-12 and worst_c < 1e-10
        failures += 0 if ok else 1
        log(f"[{'PASS' if ok else 'FAIL'}] analytical LW == sklearn LedoitWolf: "
            f"max |dshrink|={worst_s:.2e}, max |dcov|={worst_c:.2e} (5 configs)")
    except ImportError:  # pragma: no cover
        log("[SKIP] sklearn absent; LW cross-check not run (formula is "
            "the Ledoit-Wolf 2004 closed form)")

    # --- 2. synthetic CIL sequence from the conftest GaussianTask ------
    task = ct.make_gaussian_task(
        d=8, n_classes=9, separation=3.0, n_per_class=700,
        purpose="h03-lda-ncm-selfcheck",
    )
    n_train, n_test = 500, 200
    classes_per_task = 3
    train_stages, test_stages = [], []
    for t0 in range(0, 9, classes_per_task):
        xtr, xte, ytr, yte = [], [], [], []
        for c in range(t0, t0 + classes_per_task):
            rows = task.X[task.y == c]
            xtr.append(rows[:n_train])
            xte.append(rows[n_train:n_train + n_test])
            ytr += [c] * n_train
            yte += [c] * n_test
        train_stages.append((np.vstack(xtr), np.asarray(ytr)))
        test_stages.append((np.vstack(xte), np.asarray(yte)))

    lda_rep = evaluate_cil_sequence(train_stages, test_stages, method=LDA_METHOD)
    ncm_rep = evaluate_cil_sequence(train_stages, test_stages, method=NCM_METHOD)
    # Bayes head from TRUE parameters (closed-form ground truth).
    bayes = head_from_parameters(task.means, task.cov, np.arange(9))
    x_all = np.vstack([x for x, _ in test_stages])
    y_all = np.concatenate([y for _, y in test_stages])
    bayes_acc = evaluate_head(bayes, x_all, y_all)

    log(f"CIL sequence: T=3 tasks x 3 classes, d=8, sep=3.0, "
        f"n_train/class={n_train}, n_test/class={n_test} (GaussianTask, "
        f"purpose='h03-lda-ncm-selfcheck', seed={task.seed})")
    log(f"  stage accuracies (cumulative test): "
        f"LDA {[f'{a * 100:.2f}' for a in lda_rep['per_stage_accuracy']]} | "
        f"NCM {[f'{a * 100:.2f}' for a in ncm_rep['per_stage_accuracy']]}")
    log(f"  FINAL average accuracy (A-bar, percent): "
        f"LDA {lda_rep['final_accuracy'] * 100:.2f} | "
        f"NCM {ncm_rep['final_accuracy'] * 100:.2f} | "
        f"Bayes {bayes_acc * 100:.2f}")

    gap_bayes = (bayes_acc - lda_rep["final_accuracy"]) * 100.0
    ok_close = gap_bayes <= 2.5  # estimation slack at 500 train/class, d=8
    ok_order = lda_rep["final_accuracy"] >= ncm_rep["final_accuracy"]
    failures += 0 if ok_close else 1
    failures += 0 if ok_order else 1
    log(f"[{'PASS' if ok_close else 'FAIL'}] LDA within 2.5 pct-pts of Bayes "
        f"(gap={gap_bayes:.2f}) -- population equivalence, SLDA-style")
    log(f"[{'PASS' if ok_order else 'FAIL'}] LDA >= NCM on anisotropic "
        f"covariance (LDA-NCM={100 * (lda_rep['final_accuracy'] - ncm_rep['final_accuracy']):.2f})")

    # Bridge parity: core functions must reproduce the harness numbers.
    bridge_lda = lda_final_accuracy(train_stages, None, x_all, y_all)
    bridge_ncm = ncm_final_accuracy(train_stages, None, x_all, y_all)
    ok_bridge = (
        abs(bridge_lda - 100.0 * lda_rep["final_accuracy"]) < 1e-9
        and abs(bridge_ncm - 100.0 * ncm_rep["final_accuracy"]) < 1e-9
    )
    failures += 0 if ok_bridge else 1
    log(f"[{'PASS' if ok_bridge else 'FAIL'}] bridge exports match harness "
        f"(lda {bridge_lda:.2f}, ncm {bridge_ncm:.2f}, percent)")

    # --- 3. CLI + features-npz round-trip (temp store; synthetic only) --
    tmp = Path(tempfile.mkdtemp(prefix="h03-lda-ncm-selfcheck-"))
    try:
        fdir = tmp / "features" / "synthetic-gaussian" / "selfcheck-T3x3"
        for s in (0, 1):  # two valid seeds -> mean/sd aggregation path
            save_features_npz(
                fdir / f"seed{s}.npz", train_stages=train_stages,
                test_stages=test_stages, backbone="synthetic-gaussian",
                dataset="selfcheck-T3x3", seed=s, provenance="synthetic",
            )
        (fdir / "seed2.npz").write_bytes(b"not an npz")  # unusable -> MISSING
        out_dir = tmp / "results"
        rc = main([
            "--features-dir", str(tmp / "features"), "--out", str(out_dir),
            "--index", str(out_dir / "INDEX.json"),  # keep the temp store sealed
        ])
        ok_cli = rc == 0
        run_path = out_dir / "B_lda_ncm_synthetic-gaussian_selfcheck-T3x3.json"
        rec = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
        ok_cli = ok_cli and rec.get("provenance") == "synthetic"
        ok_cli = ok_cli and rec.get("config", {}).get("T") == len(train_stages)
        ok_cli = ok_cli and bool(rec.get("metrics", {}).get("missing_seeds"))
        ok_cli = ok_cli and rec.get("status") == "partial"
        cell = re.fullmatch(r"\d+\.\d+ \u00b1 \d+\.\d+",
                            rec.get("summary", {}).get("lda", ""))
        ok_cli = ok_cli and cell is not None
        # stored per-seed values must reproduce the harness final accuracy
        seeds_ok = all(
            abs(rec["metrics"]["per_seed"][s]["lda"]
                - 100.0 * lda_rep["final_accuracy"]) < 5e-4
            for s in ("0", "1")
        )
        ok_cli = ok_cli and seeds_ok and (out_dir / "INDEX.json").exists()
        failures += 0 if ok_cli else 1
        log(f"[{'PASS' if ok_cli else 'FAIL'}] CLI round-trip: 2 synthetic "
            f"seeds + 1 corrupt (MISSING recorded, never estimated); run JSON "
            f"+ INDEX.json written under a temp store")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    log(f"== self-check {'PASS' if failures == 0 else 'FAIL'} "
        f"({failures} failures) ==")
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _repo_default(rel: str) -> Path:
    return Path(__file__).resolve().parent.parent / rel


def _walk_features_dir(features_dir: Path) -> list[tuple[str, str, list[Path]]]:
    """(backbone, dataset, [seed files]) rows under <dir>/<backbone>/<dataset>/seed<k>.npz."""
    rows: list[tuple[str, str, list[Path]]] = []
    if not features_dir.is_dir():
        return rows
    for b_dir in sorted(p for p in features_dir.iterdir() if p.is_dir()):
        for d_dir in sorted(p for p in b_dir.iterdir() if p.is_dir()):
            seeds = [
                p for p in d_dir.glob("seed*.npz")
                if re.match(r"^seed\d+\.npz$", p.name)
            ]
            if seeds:
                rows.append((b_dir.name, d_dir.name, sorted(seeds)))
    return rows


def evaluate_features_row(
    backbone: str,
    dataset: str,
    seed_files: Sequence[Path],
    *,
    methods: Sequence[str] = (LDA_METHOD, NCM_METHOD),
    provenance_fallback: str | None = None,
) -> dict[str, Any] | None:
    """Evaluate one (backbone, dataset) row across seeds; None = all-MISSING."""
    per_seed: dict[int, dict[str, Any]] = {}
    provenance: str | None = None
    executed_utc: str | None = None
    n_tasks: int | None = None
    for f in seed_files:
        m = re.match(r"^seed(\d+)\.npz$", f.name)
        seed = int(m.group(1))  # type: ignore[union-attr]
        try:
            feat = load_features_npz(
                f, backbone=backbone, dataset=dataset, seed=seed,
                provenance_fallback=provenance_fallback,
            )
            per_seed[seed] = run_one_features_file(feat, methods)
            if provenance is None:
                provenance = feat.provenance
                executed_utc = feat.executed_utc
            if n_tasks is None:
                n_tasks = len(feat.train_stages)
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
            print(f"  WARN seed{seed}: {f.name} unusable ({exc}); recorded MISSING")
            per_seed[seed] = None
    if provenance is None:
        # only reached if every file failed BEFORE yielding a marker
        return None
    seeds = sorted(per_seed)
    metrics: dict[str, Any] = {"per_seed": {}, "missing_seeds": []}
    summary: dict[str, str] = {}
    for method in methods:
        finals = [
            per_seed[s][method]["final_accuracy"] if per_seed[s] else None
            for s in seeds
        ]
        mean, sd, n_used, n_missing = aggregate_seeds(finals)
        metrics[f"{method}_final_acc_pct_mean"] = mean
        metrics[f"{method}_final_acc_pct_sd"] = sd
        summary[method] = format_pm(mean, sd)
        for s, fin in zip(seeds, finals):
            metrics["per_seed"].setdefault(str(s), {})[method] = (
                None if fin is None else round(fin * 100.0, 4)
            )
        if n_missing:
            metrics["missing_seeds"].append(f"{method}:{n_missing}")
    run_id = f"B_lda_ncm_{_slug(backbone)}_{_slug(dataset)}"
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "paper": "B",
        "experiment": "lda_ncm",
        "provenance": provenance,  # from the features source marker (G08 S2b)
        "created_utc": _utc_now(),
        "executed_utc": executed_utc,
        "config": {
            "backbone": backbone,
            "dataset": dataset,
            "T": n_tasks,  # task stages in the CIL sequence (first usable seed)
            "seeds": seeds,
            "methods": list(methods),
            "shrinkage": "ledoit-wolf",
            "evaluation_only": True,
        },
        "seeds": seeds,
        "metrics": metrics,
        "summary": summary,
        "env": _env_block(),
        "code_version": {
            "core/lda_ncm.py": _sha256_file(Path(__file__)),
            "git": None,
            "implemented_by": "H03 (2026-10-21)",
        },
        "inputs": {f.name: _sha256_file(f) for f in seed_files},
        "derived_tables": ["tab:lda-ncm-comparison"],
        "status": "done" if not metrics["missing_seeds"] else "partial",
    }


def _env_block() -> dict[str, Any]:
    try:
        import sklearn

        sk = sklearn.__version__
    except ImportError:  # pragma: no cover
        sk = None
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sk,
    }


def _update_index(index_path: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Create/merge results INDEX.json (G08 S2 rule d: run_id -> path)."""
    index: dict[str, Any] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index = {}
    runs = index.setdefault("runs", {})
    for run_id, rec in entries.items():
        runs[run_id] = rec
    index["updated_utc"] = _utc_now()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, sort_keys=True, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: walk a features directory, emit the comparison JSON per row.

    Usage:
        python -m core.lda_ncm --features-dir <dir> --out <dir>
            [--index <path>] [--methods lda,ncm] [--provenance <val>]
        python -m core.lda_ncm --self-check   (or no args)

    Walks <features-dir>/<backbone>/<dataset>/seed<k>.npz (G08 S4 layout),
    runs the ordering-aware CIL harness per seed, aggregates mean +- sd in
    percent at one decimal over seeds, writes one G08-S2 run JSON per row
    into --out, merges results INDEX.json, and prints the comparison table
    (MISSING for absent features; never estimated).
    """
    parser = argparse.ArgumentParser(
        prog="core.lda_ncm",
        description="LDA/NCM comparison on cached features (Paper B, tab:lda-ncm-comparison)",
    )
    parser.add_argument("--features-dir", type=Path,
                        default=_repo_default("data/B/features"),
                        help="root of <backbone>/<dataset>/seed<k>.npz")
    parser.add_argument("--out", type=Path,
                        default=_repo_default("results/runs/B/lda_ncm"),
                        help="output directory for run JSONs")
    parser.add_argument("--index", type=Path,
                        default=_repo_default("results/INDEX.json"),
                        help="INDEX.json path (G08 S2 rule d; default: "
                             "repo results/INDEX.json)")
    parser.add_argument("--methods", default="lda,ncm",
                        help="comma-separated subset of: lda,ncm")
    parser.add_argument("--provenance", default=None,
                        help="fallback provenance when the npz marker is absent "
                             f"(one of {PROVENANCE_VALUES})")
    parser.add_argument("--self-check", action="store_true",
                        help="run the self-check only")
    if argv is None:
        argv = list(sys.argv[1:])
    args = parser.parse_args(argv)

    if args.self_check or not argv:  # --self-check, or zero CLI args
        return 1 if self_check() else 0

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    for m in methods:
        if m not in (LDA_METHOD, NCM_METHOD):
            parser.error(f"unknown method {m!r}")

    rows = _walk_features_dir(args.features_dir)
    if not rows:
        print(f"No seed*.npz rows found under {args.features_dir}; nothing to do. "
              f"(Run --self-check for the synthetic verification.)")
        return 1

    print(f"core.lda_ncm CLI: features={args.features_dir} out={args.out} "
          f"methods={','.join(methods)}")
    header = f"{'backbone':<28} {'dataset':<24} " + " ".join(
        f"{m.upper():>13}" for m in methods
    )
    print(header)
    print("-" * len(header))

    index_entries: dict[str, dict[str, Any]] = {}
    produced = 0
    for backbone, dataset, seed_files in rows:
        record = evaluate_features_row(
            backbone, dataset, seed_files,
            methods=methods, provenance_fallback=args.provenance,
        )
        cells = (
            " ".join(f"{record['summary'][m]:>13}" for m in methods)
            if record
            else " ".join(f"{MISSING:>13}" for _ in methods)
        )
        print(f"{backbone:<28} {dataset:<24} {cells}")
        if record is None:
            print(f"  WARN row {backbone}/{dataset}: no usable seed file "
                  f"(all MISSING); no run JSON written")
            continue
        out_path = args.out / f"{record['run_id']}.json"
        args.out.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, sort_keys=True, indent=2),
                            encoding="utf-8")
        produced += 1
        index_entries[record["run_id"]] = {
            "path": str(out_path),
            "derived_tables": record["derived_tables"],
            "provenance": record["provenance"],
            "updated_utc": _utc_now(),
        }
        print(f"  -> {out_path}")

    index_path = args.index or (args.out / "INDEX.json")
    _update_index(index_path, index_entries)
    print(f"rows={len(rows)} json_written={produced} "
          f"index={index_path} ({len(index_entries)} entries updated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
