"""tests.conftest -- shared test framework: fixtures, factories, tolerances.

Status: IMPLEMENTED by G10 (test-framework wave) 2026-10-21, replacing the
G09 skeleton placeholder. Spec source: G10 task order (coordinator). Later
H/I-wave implementers of test_core/test_theory_A/test_theory_B/test_tables
MUST import their fixtures/helpers from this module and follow the policy
below; do not fork private copies of seeding or tolerance logic.

TEST POLICY (binding for every test in repo/tests/)
===================================================

Tiers. Every test MUST declare its tier in its docstring.

1. UNIT (fast, synthetic)
   - Runtime budget < 1 s per test; no network, no external models, no repo
     results files. Fully deterministic synthetic data from the factories
     in this module.
   - Asserts mechanical invariants: imports, shapes, seeds, round-trips.

2. VERIFICATION (theorem-level, tolerances)
   - Checks a mathematical claim of the paper against a synthetic
     construction with closed-form ground truth (e.g. Gaussian class
     separation with known means/covariance, exactly computable Bayes risk).
   - Every assertion MUST go through the tolerance helpers below and MUST
     name which epsilon it uses and why (float64 round-off, finite-sample
     fluctuation, or eigen/cholesky solver error). Bare
     `assert abs(x - y) < 1e-3` without a documented bound is a review
     failure.

3. INTEGRATION (results-store round-trip)
   - Exercises repo modules end-to-end and persists artifacts through the
     results-store fixture (an auto-cleaned temp directory). A module must
     write a results file, read it back, and recover identical values.
   - Still synthetic data only.

SEED POLICY
-----------
- Global SEED = 0 for the whole suite (owner policy). Never touch the
  np.random.* global state; all randomness flows through new_rng() /
  make_gaussian_task().
- Child seeds derive from (SEED, purpose-string) via SHA-256. Python's
  builtin hash() is process-salted and therefore FORBIDDEN for seed
  derivation.
- Two calls with the same purpose string MUST produce bit-identical arrays
  (asserted in test_smoke.py).
- Streams are pinned to the recorded numpy version (2.5.2, Generator/
  PCG64 + ziggurat normals); a numpy major bump may require re-recording.

NO REAL-BACKBONE RULE (synthetic-only assertions, labeled)
---------------------------------------------------------
- No test may assert a paper number that requires running a real backbone
  (ImageNet-scale CNN/ViT/CLIP etc.). Those numbers are facts of nature
  from costly runs, not code invariants: they live in repo/results/ store
  files and paper tables, NEVER in test assertions.
- A test MAY touch a constant that also appears in the paper ONLY as a
  SYNTHETIC PROXY: its docstring must carry the label "synthetic-proxy"
  and it may assert only that the synthetic construction reproduces the
  theorem's closed form, never the paper's measured value.
- Violations are review failures. Integration tests verify round-trip
  fidelity of stored numbers, not the numbers themselves.

RUNNER NOTE
-----------
pytest is NOT installed in this environment (probed 2026-10-21,
Python 3.12.10; pip installs forbidden), so the suite runs on stdlib
unittest::

    python repo/tests/test_smoke.py             # direct, verbose
    python -m unittest discover -s repo/tests   # discovery

This file is importable as a plain module (`import conftest`); the tests
add their own directory to sys.path. If pytest is ever installed, the
guarded fixtures at the bottom activate automatically and this file also
serves as a classic pytest conftest.py without any edits.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Seed policy
# ---------------------------------------------------------------------------

#: Global deterministic seed for the entire suite. Owner-set policy: SEED=0.
SEED: int = 0


def derive_seed(purpose: str, salt: int = 0) -> int:
    """Deterministically derive a child seed from the global SEED.

    SHA-256 based: stable across processes and platforms. Python's builtin
    hash() is salted per process and MUST NOT be used for this.
    """
    digest = hashlib.sha256(f"{SEED}|{purpose}|{salt}".encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**63)


def new_rng(
    seed: int | None = None, *, purpose: str | None = None, salt: int = 0
) -> np.random.Generator:
    """Fresh Generator. Pass `seed` explicitly, or a `purpose` label to derive one."""
    if seed is None:
        if purpose is None:
            raise ValueError("new_rng requires either seed= or purpose=")
        seed = derive_seed(purpose, salt)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Tolerance helpers (documented epsilons)
# ---------------------------------------------------------------------------

#: Bitwise identity. Determinism checks only.
EPS_EXACT = 0.0
#: Pure float64 round-trip error (serialize/deserialize, arithmetic identity).
EPS_TIGHT = 1e-12
#: Linear-algebra round-trip error (cholesky/eigh/qr/solve) in double
#: precision for condition numbers up to ~1e6.
EPS_LINALG = 1e-9
#: Z-score for the finite-sample tolerance helpers. 6 sigma keeps the
#: false-failure rate < 1e-9 per assertion under Gaussianity.
Z_STAT = 6.0


def allclose(
    actual: Any,
    desired: Any,
    *,
    rtol: float = EPS_LINALG,
    atol: float = EPS_TIGHT,
    label: str = "",
) -> bool:
    """np.allclose with diagnostics; raises a labeled AssertionError on failure.

    Always pass an explicit `label` in tests so failures are self-describing.
    """
    a = np.asarray(actual, dtype=float)
    d = np.asarray(desired, dtype=float)
    if a.shape != d.shape:
        raise AssertionError(f"[{label}] shape mismatch: {a.shape} vs {d.shape}")
    if not np.allclose(a, d, rtol=rtol, atol=atol):
        err = np.abs(a - d)
        idx = np.unravel_index(int(np.argmax(err)), err.shape)
        raise AssertionError(
            f"[{label}] not close (rtol={rtol:g}, atol={atol:g}): "
            f"max|diff|={err.max():.3e} at {idx}, "
            f"actual={a[idx]!r}, desired={d[idx]!r}"
        )
    return True


def arrays_identical(a: np.ndarray, b: np.ndarray, *, label: str) -> bool:
    """Bitwise determinism check (EPS_EXACT tier)."""
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise AssertionError(f"[{label}] not ndarrays: {type(a)}, {type(b)}")
    if a.dtype != b.dtype or a.shape != b.shape:
        raise AssertionError(
            f"[{label}] dtype/shape mismatch {a.dtype}{a.shape} vs {b.dtype}{b.shape}"
        )
    if not np.array_equal(a, b):
        raise AssertionError(
            f"[{label}] arrays differ: max|diff|={np.abs(a - b).max():.3e}"
        )
    return True


def mean_tol(sigma: float, n: int, *, z: float = Z_STAT) -> float:
    """Half-width for empirical-mean agreement: z * sigma / sqrt(n).

    sigma: per-coordinate population std (use the worst coordinate).
    """
    return float(z * sigma / np.sqrt(n))


def cov_tol(sigma2: float, n: int, *, z: float = Z_STAT) -> float:
    """Half-width for sample-covariance ENTRY agreement, Gaussian data.

    Bound: z * sigma2 * sqrt(2/(n-1)) with sigma2 >= cov_ii * cov_jj
    (diagonal: variance of sample variance; off-diagonal: same order via
    the Gaussian fourth-moment identity). Conservative, not exact.
    """
    return float(z * sigma2 * np.sqrt(2.0 / max(n - 1, 1)))


# ---------------------------------------------------------------------------
# Synthetic Gaussian task factory (shared covariance)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GaussianTask:
    """A synthetic classification task with known closed-form ground truth.

    X: (n_classes*n_per_class, d) float64 samples, class-block ordered.
    y: int64 labels in [0, n_classes).
    means: (n_classes, d) true class means.
    cov: (d, d) TRUE shared covariance, identical for every class.
    Class-mean directions are orthonormal when n_classes <= d, hence
    ||mean_i - mean_j||_2 == separation * sqrt(2) exactly.
    """

    X: np.ndarray
    y: np.ndarray
    means: np.ndarray
    cov: np.ndarray
    d: int
    n_classes: int
    n_per_class: int
    separation: float
    seed: int
    purpose: str


def make_gaussian_task(
    *,
    d: int = 4,
    n_classes: int = 3,
    separation: float = 3.0,
    n_per_class: int = 1000,
    seed: int | None = None,
    purpose: str = "gaussian-task",
) -> GaussianTask:
    """Synthetic Gaussian task: shared covariance; configurable d, classes,
    separation, per-class sample count.

    Determinism: same seed (explicit or purpose-derived) => bit-identical
    arrays. Ground truth (means, cov) is returned alongside the samples so
    VERIFICATION-tier tests can compare empirical estimates against exact
    parameters.
    """
    if seed is None:
        seed = derive_seed(purpose)
    rng = np.random.default_rng(seed)

    # Shared covariance: Wishart-style SPD; scaling keeps diag ~ O(1).
    A = rng.standard_normal((d, d))
    cov = A @ A.T / d + 0.05 * np.eye(d)
    L = np.linalg.cholesky(cov)

    # Class-mean directions: orthonormal (QR) when n_classes <= d.
    if n_classes <= d:
        Q, _ = np.linalg.qr(rng.standard_normal((d, n_classes)))
        dirs = np.ascontiguousarray(Q.T)  # rows orthonormal
    else:
        raw = rng.standard_normal((n_classes, d))
        dirs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    means = separation * dirs

    chunks = []
    for k in range(n_classes):
        Z = rng.standard_normal((n_per_class, d))
        chunks.append(means[k] + Z @ L.T)
    X = np.ascontiguousarray(np.vstack(chunks))
    y = np.ascontiguousarray(np.repeat(np.arange(n_classes), n_per_class))
    return GaussianTask(
        X=X,
        y=y,
        means=means,
        cov=cov,
        d=d,
        n_classes=n_classes,
        n_per_class=n_per_class,
        separation=separation,
        seed=seed,
        purpose=purpose,
    )


# ---------------------------------------------------------------------------
# Results store (tmp; INTEGRATION tier)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def results_store(prefix: str = "n26-results") -> Iterator[Path]:
    """Temp directory yielded as a Path, always removed on exit.

    INTEGRATION tests write artifacts here (JSON via store_write_json /
    store_read_json) to prove round-trip fidelity without touching repo/results.
    """
    path = Path(tempfile.mkdtemp(prefix=f"{prefix}-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def store_write_json(store_dir: Path, name: str, obj: Any) -> Path:
    """Write obj to <store_dir>/<name>.json (deterministic: sort_keys)."""
    p = store_dir / f"{name}.json"
    p.write_text(json.dumps(obj, sort_keys=True, indent=2), encoding="utf-8")
    return p


def store_read_json(store_dir: Path, name: str) -> Any:
    """Read back a JSON written by store_write_json."""
    return json.loads((store_dir / f"{name}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Optional pytest bridge (inactive while pytest is not installed)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised only if pytest is ever installed
    import pytest  # type: ignore
except ImportError:  # probed absent 2026-10-21; suite runs on stdlib unittest
    pytest = None  # type: ignore[assignment]

if pytest is not None:  # pragma: no cover
    @pytest.fixture(scope="session")
    def global_seed() -> int:
        """Global SEED=0 policy seed."""
        return SEED

    @pytest.fixture()
    def gaussian_task() -> GaussianTask:
        """Default synthetic Gaussian task (deterministic, purpose-derived)."""
        return make_gaussian_task(purpose="pytest-default")

    @pytest.fixture()
    def results_store_dir(tmp_path):  # type: ignore[no-untyped-def]
        """pytest-native results store under tmp_path."""
        p = tmp_path / "results"
        p.mkdir()
        return p
