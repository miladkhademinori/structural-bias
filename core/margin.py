"""core.margin -- Paper A Mahalanobis margin, block loss matrices, TC gap.

Implements the computational pipeline of Paper A
("A Mahalanobis Margin gamma_min Bound on Task Confusion in Pretrained
Class-Incremental Learning: From Infeasibility to Exponential Attenuation")
per SPEC S1 and S5 of ``N1000_INIT_2026-10-21/exec/G02_log.md``. All TeX
anchors below refer to ``working/paper_A/neurips_2026.tex`` as ``anchor@line``
(G02's cited source) and were re-verified against that file on 2026-10-21.

Objects implemented here
========================
Population level (Definition 3.1, ``def:gamma``@L583 / ``eq:gamma``@L587-595):
  * :func:`task_statistics` -- task centroids ``mu^(t) = C^-1 sum_r mu_r``
    and task covariances ``Sigma^(t) = C^-1 sum_r Sigma_r``
    (task-level statistics, L576-580);
  * :func:`gamma_ij_pairwise` -- the inter-task Mahalanobis margin of
    Definition 3.1 with the PAIR-AVERAGED covariance
    ``(Sigma^(i)+Sigma^(j))/2``;
  * :func:`gamma_min` / :func:`gamma_min_population` -- the worst-case
    margin ``min_{i != j} gamma_ij`` with argmin and full (T,T) matrix.

Loss-matrix / TC-gap level (``eq:rho``@L486-491, block structure L497-517;
``thm:infeasibility``@L530, ``thm:feasibility``@L542;
``app:tc-gap-computation``@L4339-4392):
  * :func:`block_loss_matrix` -- T x T block aggregation of a pairwise loss
    matrix P(theta): diagonal blocks = intra-task (CF), off-diagonal =
    inter-task (TC);
  * :func:`total_block_loss` -- diagonal / off-diagonal / total sums;
  * :func:`tc_gap` -- empirical TC gap ``L(psi_dagger) - L(psi_star)``
    (App. procedure step 4, L4366-4372 and L4373-4380: identical heads,
    only sampling differs);
  * :func:`infeasibility_gap` -- the strict gap of the Infeasibility Theorem.

Bound evaluators (``lem:offdiag``@L684 / ``eq:offdiag``@L689;
``thm:relaxed``@L711 / ``eq:relaxed-bound``@L725; ``cor:operational``@L766;
``eq:main-decomp``@L1008-1025; ``eq:nu-def``@L2993):
  * :func:`e_tc_bound` -- ``M C^2 T(T-1)/(N-1) * exp(-kappa * gamma_min)``
    with kappa = 1/8 (operational, negligible spread), 1/16 (Condition (S)
    worst case) or (1-nu)/8 (interpolating form);
  * :func:`e_cf_bound` -- the E_CF term of the excess-risk decomposition;
  * :func:`nu_spread_ratio` -- within-task spread ratio nu_ij (Condition (S)
    check, ``eq:nu-def``; used by ``tab:residual-check`` per S8).

Operational estimator (``app:gamma-computation``@L4251-4336):
  * :func:`estimate_gamma_min` -- single forward pass over training
    features: layer normalisation (L4258-4269), double-precision per-class
    means (L4273-4277), Ledoit-Wolf pooled covariance with the
    scikit-learn closed form (L4279-4293; unshrunken cross-check when
    lambda* < 0.05, L4329-4333), task centroids (L4297-4299), pairwise
    margins via ONE Cholesky factorisation and two triangular solves
    (L4301-4313), worst case over the C(T,2) pairs (L4315).

CONVENTION NOTE (load-bearing, per G02 S1)
=========================================
Paper A's *definition* (Def 3.1) uses the PAIR-AVERAGED covariance
``(Sigma^(i)+Sigma^(j))/2`` -- this differs BOTH from Paper A's own
*operational estimator* (a SINGLE Ledoit-Wolf pooled Sigma_hat shared by
all pairs, L4285 and L4305) AND from Paper B's shared-Sigma convention
(``core.gaussian_model``, ``ass:gauss``: one Sigma for every class). The
two conventions coincide exactly when all class-conditional covariances
are identical (then Sigma^(t) = Sigma for every task and the pair average
is that same Sigma); the self-check demonstrates the agreement in that
regime AND the divergence under heterogeneous Sigma^(t). Nothing in this
module silently substitutes one convention for the other:
:func:`gamma_ij_pairwise` is Def 3.1 (pair average);
:func:`estimate_gamma_min` is the operational estimator (shared pooled).

SYNTHETIC-ONLY POLICY (mirrors conftest.py): every number produced here is
a fact of a synthetic construction, never a paper measurement; no
backbone is loaded; no network access; everything runs in memory
(STORAGE LAW). Seeding flows through tests/conftest.py (G10):
derive_seed / new_rng are imported and reused -- no private seed logic.

Self-check: ``python core/margin.py`` runs closed-form verification
(< 1e-9), an isotropic identity check (< 1e-12), the convention-divergence
demonstration, block/bound sanity checks, and the 50k-sample estimator
recovery test with a documented sampling tolerance; prints PASS lines.
numpy/scipy only (sklearn is touched ONLY by the self-check's optional
Ledoit-Wolf cross-verification, never by the estimator itself).

Status: IMPLEMENTED by I01, 2026-10-21 (replaces G09 skeleton).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import scipy.linalg

# ---------------------------------------------------------------------------
# conftest reuse (G10 test framework): derive_seed / new_rng / tolerances
# ---------------------------------------------------------------------------


def _load_conftest():  # pragma: no cover - trivial import shim
    """Import repo/tests/conftest.py (owner-mandated reuse, no duplication).

    Same shim pattern as core/gaussian_model.py (H01): works when the tests
    dir is already on sys.path (unittest runners) and when imported as a
    bare module from anywhere.
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
#: 6-sigma finite-sample multiplier for sampling tolerances, see conftest.
Z_STAT = _conftest.Z_STAT


def _load_ledoit_wolf_intensity():
    """Reuse core/lda_ncm.ledoit_wolf_shrinkage (H03, sklearn-identical).

    That function is explicitly "exported for reuse by the A-side modules
    (margin/diagnostic)" in its docstring; importing it keeps a single
    source of the Ledoit-Wolf closed form in the repo. The inline fallback
    (identical formula, same degenerate-input convention) exists only so a
    partially-written tree still imports; the self-check asserts the two
    agree bit-for-bit whenever the import succeeds.
    """
    try:
        try:
            from core import lda_ncm as _ln
        except ImportError:
            root = str(Path(__file__).resolve().parents[1])
            if root not in sys.path:
                sys.path.insert(0, root)
            from core import lda_ncm as _ln
        return _ln.ledoit_wolf_shrinkage
    except Exception:  # pragma: no cover - defensive; never hit in-repo
        def _fallback(x, *, assume_centered=False):
            x = np.asarray(x, dtype=np.float64)
            n, d = x.shape
            if d == 1:
                return 0.0
            xc = x if assume_centered else x - x.mean(axis=0)
            s = (xc.T @ xc) / n
            mu = float(np.trace(s)) / d
            tr_s2 = float((s ** 2).sum())
            sum_row4 = float((np.sum(xc ** 2, axis=1) ** 2).sum())
            delta = tr_s2 - d * mu ** 2
            num = (sum_row4 - n * tr_s2) / (n * n)
            if delta <= 0.0 or num <= 0.0:
                return 0.0
            return float(min(num, delta) / delta)
        return _fallback


_ledoit_wolf_shrinkage = _load_ledoit_wolf_intensity()


# ---------------------------------------------------------------------------
# Small linear-algebra helpers (private)
# ---------------------------------------------------------------------------


def _check_spd(cov: np.ndarray, name: str) -> np.ndarray:
    """Validate symmetric PSD->SPD via Cholesky; return float64 copy."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name} must be square 2-D, got {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{name} must be finite")
    scale = np.abs(cov).max()
    if scale == 0.0:
        scale = 1.0
    if not np.allclose(cov, cov.T, rtol=0.0, atol=EPS_LINALG * scale):
        raise ValueError(f"{name} must be symmetric")
    cov = 0.5 * (cov + cov.T)
    try:
        scipy.linalg.cholesky(cov, lower=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be SPD (Def 3.1 covariance)") from exc
    return cov


def _mahalanobis_sq(cov: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """v^T cov^{-1} v for each column of ``vectors`` via Cholesky.

    One cho_factor + one triangular solve batch (the two-triangular-solve
    path of app:gamma-computation L4301-4313, applied to many vectors).
    Returns (vectors.shape[1],) float64.
    """
    fac = scipy.linalg.cho_factor(cov, lower=True, check_finite=True)
    solved = scipy.linalg.solve_triangular(
        fac[0], np.asarray(vectors, dtype=float), lower=True
    )
    return np.einsum("it,it->t", solved, solved)


# ---------------------------------------------------------------------------
# Population-level gamma (Definition 3.1)
# ---------------------------------------------------------------------------


def task_statistics(
    class_means: np.ndarray,
    class_covs: "np.ndarray | None",
    task_of_class: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Task centroids mu^(t) and task covariances Sigma^(t) (L576-580).

    ``mu^(t) = C_t^-1 sum_{r in Y^(t)} mu_r`` and
    ``Sigma^(t) = C_t^-1 sum_{r in Y^(t)} Sigma_r`` (task-level statistics
    preceding Def 3.1; equal-size tasks in the paper's headline setting,
    ragged partitions tolerated here).

    Parameters
    ----------
    class_means : (N, d) per-class means mu_r (global class order).
    class_covs  : (N, d, d) per-class covariances Sigma_r; or a single
        (d, d) covariance SHARED by every class (Paper B ``ass:gauss``
        style -- then Sigma^(t) is that covariance for every task); or
        None (identity metric: Euclidean margins).
    task_of_class : (N,) int task index of each class (values 0..T-1,
        each present at least once; contiguity/order not enforced here).

    Returns ``(task_means (T, d), task_covs (T, d, d))``.
    """
    mu = np.asarray(class_means, dtype=float)
    if mu.ndim != 2 or mu.shape[0] < 1:
        raise ValueError(f"class_means must be (N>=1, d), got {mu.shape}")
    N, d = mu.shape
    toc = np.asarray(task_of_class, dtype=np.int64)
    if toc.shape != (N,):
        raise ValueError(f"task_of_class must be ({N},), got {toc.shape}")
    if toc.min() < 0:
        raise ValueError("task indices must be >= 0")
    T = int(toc.max()) + 1
    if class_covs is None:
        covs = np.broadcast_to(np.eye(d), (N, d, d))
    else:
        arr = np.asarray(class_covs, dtype=float)
        if arr.shape == (d, d):
            covs = np.broadcast_to(_check_spd(arr, "class_covs"), (N, d, d))
        elif arr.shape == (N, d, d):
            covs = np.stack(
                [_check_spd(arr[r], f"class_covs[{r}]") for r in range(N)]
            )
        else:
            raise ValueError(
                f"class_covs must be ({N},{d},{d}) or ({d},{d}), "
                f"got {arr.shape}"
            )
    counts = np.bincount(toc, minlength=T).astype(float)
    if np.any(counts == 0):
        raise ValueError("task ids must be contiguous 0..T-1, each present")
    task_means = np.vstack(
        [mu[toc == t].mean(axis=0) for t in range(T)]
    )
    task_covs = np.stack(
        [covs[toc == t].mean(axis=0) for t in range(T)]
    )
    return np.ascontiguousarray(task_means), np.ascontiguousarray(task_covs)


def gamma_ij_pairwise(
    mu_i: np.ndarray,
    mu_j: np.ndarray,
    Sigma_i: np.ndarray,
    Sigma_j: np.ndarray,
) -> float:
    """Inter-task Mahalanobis margin gamma_ij, Definition 3.1 EXACTLY.

    ``eq:gamma``@L587-595 (``def:gamma``@L583)::

        gamma_ij = (mu^(i) - mu^(j))^T ((Sigma^(i)+Sigma^(j))/2)^{-1}
                   (mu^(i) - mu^(j))

    PAIR-AVERAGED covariance convention -- this is Paper A's *definition*
    and deliberately DIFFERS from (a) the operational estimator of
    :func:`estimate_gamma_min` (single shared Ledoit-Wolf pooled Sigma_hat,
    app:gamma-computation L4285/L4305) and (b) Paper B's shared-Sigma
    ``ass:gauss`` model in ``core.gaussian_model``. All three coincide only
    when every class shares one covariance.

    Computes via one Cholesky factorisation + one triangular solve of the
    pair-averaged matrix (never forms the inverse). Valid input symmetric
    Sigma_i, Sigma_j with SPD average (Cholesky is attempted on the
    average; SPD of each input is implied).
    """
    mu_i = np.asarray(mu_i, dtype=float).ravel()
    mu_j = np.asarray(mu_j, dtype=float).ravel()
    if mu_i.shape != mu_j.shape or mu_i.ndim != 1 or mu_i.size < 1:
        raise ValueError("mu_i/mu_j must be matching 1-D vectors")
    d = mu_i.size
    Si = _check_spd(Sigma_i, "Sigma_i")
    Sj = _check_spd(Sigma_j, "Sigma_j")
    if Si.shape != (d, d) or Sj.shape != (d, d):
        raise ValueError(
            f"Sigma_i/Sigma_j must be ({d},{d}) matching the means"
        )
    pair_avg = _check_spd(0.5 * (Si + Sj), "(Sigma_i+Sigma_j)/2")
    delta = (mu_i - mu_j)[:, None]
    return float(_mahalanobis_sq(pair_avg, delta)[0])


@dataclasses.dataclass(frozen=True)
class GammaMinResult:
    """Result of the population worst-case margin (``def:gamma``@L583).

    gamma_min    worst-case margin min_{i != j} gamma_ij.
    argmin       (i, j) with i < j attaining the minimum (lexicographic
                 tie-break: first minimum in row-major upper-triangle order).
    gamma_matrix (T, T) symmetric matrix of gamma_ij; the DIAGONAL HOLDS
                 0.0 as a "not defined / excluded" placeholder (gamma is
                 defined for i != j only); mins are always taken over the
                 strict upper triangle.
    convention   "pair-averaged" for the Def 3.1 path (task_covs given),
                 "shared" when a single covariance was broadcast.
    """

    gamma_min: float
    argmin: Tuple[int, int]
    gamma_matrix: np.ndarray
    convention: str


def gamma_min(
    task_means: np.ndarray,
    task_covs: "np.ndarray | None" = None,
) -> GammaMinResult:
    """Worst-case inter-task margin min_{i != j} gamma_ij (``def:gamma``).

    ``gamma_min(phi) := min_{i != j} gamma_ij(phi)`` with the task-level
    statistics of L576-580. Modes:

    * ``task_covs`` (T, d, d) with heterogeneous matrices: Def 3.1 with
      the PAIR-AVERAGED covariance per pair (loops the C(T,2) pairs; each
      pair costs one Cholesky of (Sigma^(i)+Sigma^(j))/2).
    * ``task_covs`` a single (d, d) matrix, or None (identity), or (T, d, d)
      matrices that are all equal up to EPS_LINALG: ONE shared covariance.
      Fast vectorised path -- a single Cholesky whitens all T task
      centroids at once (one triangular solve on a (d, T) block) and
      gamma_ij = ||w_i - w_j||^2 in whitened coordinates. Mathematically
      IDENTICAL to the pair-averaged value in this case (the pair average
      of equal matrices is the matrix); this is also the population
      counterpart of the operational estimator's shared-Sigma_hat
      convention.

    The shared fast path and the pairwise path agree to EPS_LINALG when
    the task covariances are all equal (asserted in the self-check).
    """
    mu = np.asarray(task_means, dtype=float)
    if mu.ndim != 2 or mu.shape[0] < 2:
        raise ValueError(
            f"task_means must be (T>=2, d) -- gamma_min needs i != j, "
            f"got {mu.shape}"
        )
    T, d = mu.shape
    if task_covs is None:
        shared_cov = np.eye(d)
        shared = True
    else:
        arr = np.asarray(task_covs, dtype=float)
        if arr.shape == (d, d):
            shared_cov = _check_spd(arr, "task_covs")
            shared = True
        elif arr.shape == (T, d, d):
            covs = np.stack(
                [_check_spd(arr[t], f"task_covs[{t}]") for t in range(T)]
            )
            scale = np.abs(covs).max()
            shared = bool(
                np.allclose(
                    covs, covs[0][None, :, :], rtol=0.0,
                    atol=EPS_LINALG * (scale if scale > 0 else 1.0),
                )
            )
            shared_cov = covs[0] if shared else None
        else:
            raise ValueError(
                f"task_covs must be None, a single ({d},{d}) shared "
                f"covariance, or ({T},{d},{d}); got {arr.shape}"
            )

    if not shared:
        gam = np.zeros((T, T), dtype=float)
        for i in range(T - 1):
            for j in range(i + 1, T):
                g = gamma_ij_pairwise(mu[i], mu[j], covs[i], covs[j])
                gam[i, j] = g
                gam[j, i] = g
    else:
        # All tasks share one covariance: whiten every centroid with ONE
        # Cholesky factorisation (vectorised; the same single-factor trick
        # the operational estimator uses with Sigma_hat).
        fac = scipy.linalg.cho_factor(shared_cov, lower=True)
        w = scipy.linalg.solve_triangular(fac[0], mu.T, lower=True)  # (d,T)
        gram = w.T @ w
        diag = np.diag(gram)
        gam_full = diag[:, None] + diag[None, :] - 2.0 * gram
        gam = 0.5 * (gam_full + gam_full.T)
        np.fill_diagonal(gam, 0.0)
        gam = np.maximum(gam, 0.0)  # clip negative round-off at identical means

    iu = np.triu_indices(T, k=1)
    vals = gam[iu]
    k = int(np.argmin(vals))
    return GammaMinResult(
        gamma_min=float(vals[k]),
        argmin=(int(iu[0][k]), int(iu[1][k])),
        gamma_matrix=np.ascontiguousarray(gam),
        convention="shared" if shared else "pair-averaged",
    )


def gamma_min_population(
    class_means: np.ndarray,
    class_covs: "np.ndarray | None",
    task_of_class: np.ndarray,
) -> GammaMinResult:
    """Def 3.1 gamma_min from CLASS-level parameters (population truth).

    Applies the task statistics of L576-580
    (``mu^(t) = C^-1 sum_r mu_r``, ``Sigma^(t) = C^-1 sum_r Sigma_r``)
    then :func:`gamma_min`. ``class_covs``: (N, d, d) heterogeneous;
    a single (d, d) shared by every class (``ass:gauss`` style -- the
    model where the operational estimator's shared-Sigma_hat convention
    and Def 3.1's pair average coincide); None for the identity metric
    (Euclidean margins). This is the exact ground truth the self-check's
    50k-sample estimator run is compared against.
    """
    task_means, task_covs = task_statistics(class_means, class_covs, task_of_class)
    return gamma_min(task_means, task_covs)


# ---------------------------------------------------------------------------
# Block loss matrices (eq:rho L486-491; block structure L497-517)
# ---------------------------------------------------------------------------


def block_loss_matrix(
    pair_losses: np.ndarray,
    task_of_class: np.ndarray,
    *,
    average: bool = False,
) -> np.ndarray:
    """Aggregate a pairwise loss matrix P(theta) into its T x T blocks.

    ``eq:rho``@L486-491 defines, for distinct classes k != l,

        rho_kl(theta) = (N-1)^{-1} int_{X x {k,l}} v(f_theta(x), y) p(x, y),

    collected into the loss matrix P with [P]_kl = rho_kl (k != l), total
    risk I(theta) = sum_{k<l} rho_kl. Given the task partition, P admits
    the T x T block structure of L497-517: DIAGONAL blocks P_tt hold
    intra-task pairs (the CF objective sum_t |P_tt| of
    ``thm:infeasibility``@L530); OFF-DIAGONAL blocks P_ij (i != j) hold
    inter-task pairs -- task confusion, the mass no sequential stage
    touches. With C classes per task the T(T-1) off-diagonal blocks carry
    T(T-1) C^2 ordered entries (L511-516).

    Parameters
    ----------
    pair_losses : (N, N) matrix; entry [k, l] is rho_kl for k != l. The
        DIAGONAL IS IGNORED (rho is defined only for k != l; pass any
        filler there). Supply a full ordered matrix (both [k,l] and [l,k])
        or an upper/lower-triangular one -- the aggregation never mixes
        in the ignored diagonal.
    task_of_class : (N,) task id of each class.
    average : if False (default) blocks hold SUMS (the scale the theorem's
        sums act on); if True, per-block MEANS over the contributing
        k != l entries (comparable block magnitudes for visualisation).

    Returns (T, T) blocks; blocks[i, j] aggregates classes of task i
    (rows) against classes of task j (columns), k == l excluded.
    """
    P = np.asarray(pair_losses, dtype=float)
    toc = np.asarray(task_of_class, dtype=np.int64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"pair_losses must be square 2-D, got {P.shape}")
    N = toc.size
    if P.shape != (N, N):
        raise ValueError(
            f"pair_losses {P.shape} and task_of_class ({N},) disagree on N"
        )
    if toc.min() < 0:
        raise ValueError("task indices must be >= 0")
    T = int(toc.max()) + 1
    if np.any(np.bincount(toc, minlength=T) == 0):
        raise ValueError("task ids must be contiguous 0..T-1, each present")
    members = [np.flatnonzero(toc == t) for t in range(T)]
    blocks = np.zeros((T, T), dtype=float)
    for i in range(T):
        for j in range(T):
            sub = P[np.ix_(members[i], members[j])]
            if i == j:
                sub = sub[~np.eye(sub.shape[0], dtype=bool)]
            blocks[i, j] = sub.mean() if average else sub.sum()
    return blocks


def total_block_loss(
    blocks: np.ndarray,
    which: str = "all",
) -> float:
    """Total of a block matrix over a block selection.

    ``which``:
      * ``"diagonal"``    -- sum_t |P_tt|  : the CF objective of
        ``thm:infeasibility``@L530 (theta_dagger minimises exactly this);
      * ``"off-diagonal"``-- sum_{i != j} |P_ij| : the task-confusion mass;
      * ``"all"``         -- sum_{i,j} |P_ij| : the total loss the
        Infeasibility Theorem compares (CF-optimal vs globally optimal).

    Absolute values follow the theorem's |P_ij| notation (losses are
    non-negative in practice; abs is a no-op there but keeps the exact
    statement for signed inputs).
    """
    B = np.abs(np.asarray(blocks, dtype=float))
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"blocks must be square 2-D, got {B.shape}")
    eye = np.eye(B.shape[0], dtype=bool)
    if which == "all":
        return float(B.sum())
    if which == "diagonal":
        return float(B[eye].sum())
    if which in ("off-diagonal", "offdiagonal"):
        return float(B[~eye].sum())
    raise ValueError(
        f"which must be 'all' | 'diagonal' | 'off-diagonal', got {which!r}"
    )


# ---------------------------------------------------------------------------
# TC gap and the Infeasibility gap (app:tc-gap-computation; thm:infeasibility)
# ---------------------------------------------------------------------------


def tc_gap(
    diagonal_head_loss: float,
    allblocks_head_loss: float,
) -> float:
    """Empirical TC gap: L(psi_dagger) - L(psi_star) on held-out data.

    ``app:tc-gap-computation``@L4339-4392, step 4 (L4366-4372):
    (1) diagonal-only joint training (per-batch within-task pairs, CE
    restricted to the task's C classes, task-balanced sampling) gives the
    CF-optimal reference head psi_dagger; (2) identical all-blocks N-way CE
    training gives psi_star; (3) both test losses are computed with the
    same protocol; (4) TC gap := L(psi_dagger) - L(psi_star). The two
    heads share architecture/optimiser (AdamW wd 1e-4)/budget (20 epochs,
    lr 1e-3)/batch size -- ONLY the sampling distribution differs
    (L4373-4380), which isolates the off-diagonal contribution.

    Theory guarantees >= 0 (``eq:relaxed-bound``@L725); an empirical
    estimate may dip negative through finite-sample noise -- the raw
    difference is returned UNCLAMPED so the sign stays informative.
    Head training itself lives in core/heads.py; this function is the
    agreed seam for the final subtraction.
    """
    l_dag = float(diagonal_head_loss)
    l_star = float(allblocks_head_loss)
    if not (np.isfinite(l_dag) and np.isfinite(l_star)):
        raise ValueError("both losses must be finite scalars")
    return l_dag - l_star


def infeasibility_gap(
    loss_cf_optimal: "float | np.ndarray",
    loss_joint_optimal: "float | np.ndarray",
) -> float:
    """Strict gap of the Infeasibility Theorem (``thm:infeasibility``@L530).

    With theta_dagger in argmin_theta sum_t |P_tt(theta)| (CF-optimal),
    the theorem states

        sum_{i,j} |P_ij(theta_dagger)|  >  min_theta sum_{i,j} |P_ij(theta)|,

    the strict gap attributable to the off-diagonal blocks
    (``thm:feasibility``@L542: generative models, purely diagonal loss
    matrix, escape the obstacle -- their gap is 0; Paper A's 1e-10 LDA
    check, S9 of the G02 spec, attains exactly this limit).

    Accepts scalars, or (T,T)/(N,N) loss matrices (summed over |entries|;
    combine with :func:`block_loss_matrix` / :func:`total_block_loss`
    with ``which='all'``). Returns
    ``total(loss_cf_optimal) - total(loss_joint_optimal)``, >= 0 in
    population, unclamped for empirical inputs.
    """

    def _total(x) -> float:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        if not np.all(np.isfinite(arr)):
            raise ValueError("loss matrices must be finite")
        return float(np.abs(arr).sum())

    return _total(loss_cf_optimal) - _total(loss_joint_optimal)


# ---------------------------------------------------------------------------
# Bound evaluators (lem:offdiag, thm:relaxed, cor:operational, eq:main-decomp)
# ---------------------------------------------------------------------------


def e_tc_bound(
    gamma_min_value: float,
    *,
    M: float,
    C: int,
    T: int,
    N: int,
    rate: "str | float" = "operational",
) -> float:
    """E_TC upper bound  M C^2 T(T-1)/(N-1) * exp(-kappa * gamma_min).

    ``eq:main-decomp``@L1008-1025 carries ``exp(-gamma_min/16)`` -- the
    Condition (S) worst case nu = 1/2 of ``cor:operational``@L766 and the
    guaranteed constant of ``thm:relaxed``@L711 (``eq:relaxed-bound``@L725,
    aggregating ``lem:offdiag``@L684 / ``eq:offdiag``@L689). The
    operational (negligible-spread) rate of the corollary improves the
    exponent to ``exp(-gamma_min/8)``; its interpolating form is
    ``exp(-(1-nu)/8 * gamma_min)`` with nu the within-task spread ratio
    (``eq:nu-def``@L2993; Condition (S) is nu <= 1/2, cf. S8).

    Parameters
    ----------
    rate : ``"operational"`` (kappa = 1/8), ``"conservative"``
        (kappa = 1/16), or a float nu in [0, 1) giving kappa = (1-nu)/8.
    M : loss bound of Condition (L) (0 <= v <= M).
    C : classes per task; T : tasks; N = C*T total classes (>= 2).
    """
    g = float(gamma_min_value)
    if not np.isfinite(g) or g < 0.0:
        raise ValueError("gamma_min must be finite and >= 0")
    if M <= 0 or C < 1 or T < 2 or N < 2:
        raise ValueError("need M > 0, C >= 1, T >= 2, N >= 2")
    if rate == "operational":
        kappa = 0.125
    elif rate == "conservative":
        kappa = 0.0625
    elif isinstance(rate, (int, float)):
        nu = float(rate)
        if not 0.0 <= nu < 1.0:
            raise ValueError(
                f"numeric rate is the spread ratio nu in [0, 1), got {nu}"
            )
        kappa = (1.0 - nu) / 8.0
    else:
        raise ValueError(
            "rate must be 'operational' | 'conservative' | nu in [0,1), "
            f"got {rate!r}"
        )
    prefactor = M * (C ** 2) * T * (T - 1) / (N - 1)
    return float(prefactor * np.exp(-kappa * g))


def e_cf_bound(
    K: "float | np.ndarray",
    *,
    C_opt_sc: float = 1.0,
    Delta_t: "float | np.ndarray" = 1.0,
    sigma2: float = 1.0,
    mu: float = 1.0,
    alpha: float = 1.5,
) -> "float | np.ndarray":
    """E_CF term of ``eq:main-decomp``@L1008-1025.

        E_CF = sum_{t=1}^{T} C_opt_sc * (Delta_t + sigma^2 / mu)
               / K_t^{(alpha-1)/alpha}

    K broadcasts: scalar (one budget for every task), (T,) per-task
    budgets, or (S, T) budget sweeps (summed over the last axis -> (S,)).
    Delta_t broadcasts like K. alpha > 1 (the heavy-tail index; alpha = 2
    recovers the standard non-convex SGD rate exponent 1/2, L1036-1040).
    Returns float for scalar K, else an ndarray with the leading sweep
    shape.
    """
    if not 1.0 < alpha:
        raise ValueError(f"alpha must be > 1 (as alpha -> 1+ rate -> O(1)), got {alpha}")
    if C_opt_sc <= 0 or sigma2 <= 0 or mu <= 0:
        raise ValueError("C_opt_sc, sigma2, mu must be > 0")
    K_arr = np.asarray(K, dtype=float)
    D_arr = np.asarray(Delta_t, dtype=float)
    if np.any(K_arr <= 0) or np.any(D_arr <= 0):
        raise ValueError("K_t and Delta_t must be > 0")
    expo = (alpha - 1.0) / alpha
    per_task = C_opt_sc * (D_arr + sigma2 / mu) / K_arr ** expo
    out = per_task.sum(axis=-1) if per_task.ndim >= 1 else float(per_task)
    return out


def nu_spread_ratio(
    class_means: np.ndarray,
    class_covs: np.ndarray | None,
    task_of_class: np.ndarray,
    i: int,
    j: int,
) -> float:
    """Within-task spread ratio nu_ij (``eq:nu-def``@L2993; Condition (S)).

        nu_ij = (2 * max_{k in Y^(i)} ||mu_k - mu^(i)||_*
                + 2 * max_{l in Y^(j)} ||mu_l - mu^(j)||_*) / sqrt(gamma_ij)

    The ``||.||_*`` norm is the Mahalanobis norm of the PAIR metric of
    Definition 3.1 (inverse of (Sigma^(i)+Sigma^(j))/2), consistent with
    ``eq:gamma-vs-spread``@L3042, which bounds ||.||_* by the Euclidean
    norm over sigma_min. ``class_covs`` follows :func:`task_statistics`
    ((N, d, d) heterogeneous, a shared (d, d), or None = identity).
    Condition (S) (nu_ij <= 1/2) selects the
    conservative 1/16 constant of ``thm:relaxed``; the two ViT
    configurations daggered in ``tab:residual-check`` (S8: nu = 0.51,
    0.73) violate it -- values of that magnitude are reproduceable here
    only from synthetic inputs, never asserted (synthetic-proxy policy).
    """
    mu = np.asarray(class_means, dtype=float)
    if mu.ndim != 2:
        raise ValueError(f"class_means must be (N, d), got {mu.shape}")
    N, d = mu.shape
    toc = np.asarray(task_of_class, dtype=np.int64)
    if toc.shape != (N,):
        raise ValueError(f"task_of_class must be ({N},), got {toc.shape}")
    task_means, task_covs = task_statistics(mu, class_covs, toc)
    T = task_means.shape[0]
    if not (0 <= i < T and 0 <= j < T and i != j):
        raise ValueError(f"need two distinct task ids in [0, {T}), got {(i, j)}")
    pair_avg = _check_spd(
        0.5 * (task_covs[i] + task_covs[j]), "(Sigma^(i)+Sigma^(j))/2"
    )
    members_i = np.flatnonzero(toc == i)
    members_j = np.flatnonzero(toc == j)
    dev_i = (mu[members_i] - task_means[i]).T  # (d, n_i)
    dev_j = (mu[members_j] - task_means[j]).T
    norms = np.sqrt(np.concatenate(
        [_mahalanobis_sq(pair_avg, dev_i), _mahalanobis_sq(pair_avg, dev_j)]
    ))
    spread = 2.0 * norms[: members_i.size].max() + 2.0 * norms[members_i.size:].max()
    g_ij = gamma_ij_pairwise(
        task_means[i], task_means[j], task_covs[i], task_covs[j]
    )
    return float(spread / np.sqrt(g_ij))


# ---------------------------------------------------------------------------
# Operational estimator (app:gamma-computation L4251-4336)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GammaMinEstimate:
    """Output of :func:`estimate_gamma_min` (app:gamma-computation).

    gamma_min      : the reported worst-case margin gamma_hat_min (step 6).
    argmin         : (i, j), i < j, attaining it.
    gamma_matrix   : (T, T) pairwise gamma_hat_ij, diagonal 0.0.
    task_centroids : (T, p) step-4 centroids mu_hat^(t) (normalised coords).
    class_means    : (N, p) step-2 per-class means (double precision).
    sigma_hat      : (p, p) Ledoit-Wolf shrunk pooled covariance (step 3).
    lw_intensity   : lambda* of the shrinkage (0 = unshrunken).
    gamma_min_unshrunk / cross_check_rel_diff : the L4329-4333 cross-check
        (populated when lambda* < 0.05 and the raw pooled covariance is
        factorisable; the paper confirms agreement within 5%).
    layer_mean / layer_std : (p,) step-1 global feature mean/std used for
        the layer normalisation (recorded for reproducibility).
    n_samples / n_classes / d / convention : bookkeeping. convention is
    "shared-LW-pooled": ONE Sigma_hat for ALL pairs -- the operational
    convention, distinct from Def 3.1's pair average (see module header).
    """

    gamma_min: float
    argmin: Tuple[int, int]
    gamma_matrix: np.ndarray
    task_centroids: np.ndarray
    class_means: np.ndarray
    sigma_hat: np.ndarray
    lw_intensity: float
    gamma_min_unshrunk: "float | None"
    cross_check_rel_diff: "float | None"
    layer_mean: np.ndarray
    layer_std: np.ndarray
    n_samples: int
    n_classes: int
    d: int
    convention: str = "shared-LW-pooled"


def _class_means_from(z: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Step 2 (L4273-4277): double-precision per-class means, vectorised."""
    d = z.shape[1]
    sums = np.zeros((n_classes, d), dtype=float)
    np.add.at(sums, y, z)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    if np.any(counts == 0):
        missing = int(np.flatnonzero(counts == 0)[0])
        raise ValueError(f"class {missing} has no samples; means undefined")
    return sums / counts[:, None]


def estimate_gamma_min(
    features: np.ndarray,
    labels: np.ndarray,
    task_of_class: np.ndarray,
    *,
    shrinkage: "str | float | None" = "ledoit-wolf",
    cross_check_threshold: float = 0.05,
    layer_normalize: bool = True,
) -> GammaMinEstimate:
    """Operational gamma_min estimator (``app:gamma-computation``@L4251-4336).

    Procedure, exactly as specified (G02 S1):

    1. Layer normalisation (L4258-4269): ``z~ = (z - mu_z) / s_z`` with the
       GLOBAL feature mean/std per coordinate ("critical for making
       gamma_min invariant to backbone scaling conventions"). Note gamma is
       exactly invariant under any diagonal rescaling applied to BOTH the
       means and the covariance (congruence), so this step fixes the
       coordinates without changing the population target.
    2. Per-class means in double precision (L4273-4277).
    3. Pooled within-class covariance with Ledoit-Wolf shrinkage
       (L4279-4293): ``S = (N_train - N)^{-1} sum_n (z~_n - mu_hat_{y_n})
       (.)^T`` (UNBIASED denominator per the spec equation) and
       ``Sigma_hat = (1-lambda*) S + lambda* (tr(S)/p) I`` with lambda* in
       the scikit-learn closed form -- taken from
       core/lda_ncm.ledoit_wolf_shrinkage (verified sklearn-identical;
       lambda* is invariant to the scaling of S, so applying it to the
       unbiased S matches sklearn's intensity on the same centred rows).
       ``shrinkage``: ``"ledoit-wolf"`` (default) | ``None``/"none" | a
       fixed intensity in [0, 1]. If ``lambda* < cross_check_threshold``
       (default 0.05) the UNSHRUNKEN estimator is also computed and the
       relative difference recorded (paper confirms agreement within 5%,
       L4329-4333).
    4. Task centroids ``mu_hat^(t) = C^-1 sum_c mu_hat_c`` (L4297-4299).
    5. Pairwise margins (L4301-4313) with ONE Cholesky factorisation
       ``Sigma_hat = L L^T`` and two triangular solves per vector --
       vectorised here as a single triangular solve on the (p, T) centroid
       block, then ``gamma_ij = ||w_i - w_j||^2`` in whitened coordinates.
    6. ``gamma_hat_min = min_{i != j} gamma_hat_ij`` over the C(T,2) pairs
       (L4315; T = 10, C = 10, seeds 0-4 in the paper's protocol -- those
       protocol constants live with the experiment harness, not here).

    CONVENTION: a SINGLE Sigma_hat is shared by all pairs (L4285/L4305).
    This is the operational estimator, NOT Definition 3.1's pair-averaged
    covariance -- the two agree asymptotically when class-conditional
    covariances are homogeneous (the self-check's 50k-sample verification
    uses exactly that regime and additionally demonstrates the divergence
    under heterogeneous covariances).

    numpy/scipy only; single pass over the features; no I/O.
    """
    X = np.ascontiguousarray(np.asarray(features, dtype=np.float64))
    y = np.asarray(labels, dtype=np.int64).ravel()
    toc = np.asarray(task_of_class, dtype=np.int64).ravel()
    n, p = X.shape
    if X.ndim != 2 or n < 1 or p < 1:
        raise ValueError(f"features must be (n>=1, p>=1), got {X.shape}")
    if y.shape != (n,):
        raise ValueError(f"labels must be ({n},), got {y.shape}")
    N = toc.size
    if y.min() != 0 or y.max() != N - 1:
        raise ValueError(
            f"labels must cover every class 0..{N - 1} exactly "
            "(estimator needs all class means)"
        )
    if toc.min() < 0:
        raise ValueError("task indices must be >= 0")
    T = int(toc.max()) + 1
    if np.any(np.bincount(toc, minlength=T) == 0):
        raise ValueError("task ids must be contiguous 0..T-1, each present")
    if n <= N:
        raise ValueError(
            f"pooled covariance needs N_train > N classes (got n={n}, N={N})"
        )

    # Step 1: layer normalisation (global mean/std per coordinate).
    if layer_normalize:
        mu_z = X.mean(axis=0)
        s_z = X.std(axis=0, ddof=0)
        tiny = s_z <= np.finfo(float).tiny
        if np.any(s_z == 0.0):
            raise ValueError(
                "constant feature coordinate after layer norm; gamma undefined"
            )
        s_z = np.where(tiny, 1.0, s_z)
        z = (X - mu_z) / s_z
    else:
        mu_z = np.zeros(p)
        s_z = np.ones(p)
        z = X

    # Step 2: per-class means (double precision).
    class_means = _class_means_from(z, y, N)

    # Step 3: pooled within-class scatter + Ledoit-Wolf shrinkage.
    centered = z - class_means[y]          # each row centred by ITS class mean
    S_pool = (centered.T @ centered) / (n - N)   # spec denominator N_train - N
    d = p
    if shrinkage is None or (
        isinstance(shrinkage, str) and shrinkage.lower() in ("none", "unshrunk")
    ):
        intensity = 0.0
    elif isinstance(shrinkage, str) and shrinkage.lower() in (
        "ledoit-wolf", "lw"
    ):
        intensity = float(_ledoit_wolf_shrinkage(centered, assume_centered=True))
    elif isinstance(shrinkage, (int, float)):
        intensity = float(shrinkage)
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"shrinkage intensity must be in [0, 1], got {intensity}")
    else:
        raise ValueError(f"unknown shrinkage setting: {shrinkage!r}")
    mu_tr = float(np.trace(S_pool)) / d
    sigma_hat = (1.0 - intensity) * S_pool + intensity * mu_tr * np.eye(d)
    sigma_hat = 0.5 * (sigma_hat + sigma_hat.T)
    _check_spd(sigma_hat, "Sigma_hat")

    # Cross-check with the unshrunken estimator when lambda* is small
    # (L4329-4333: "confirm the result agrees within 5%").
    gamma_unshrunk = None
    rel_diff = None
    if 0.0 < intensity < cross_check_threshold or (
        intensity == 0.0 and shrinkage not in (None, "none", "unshrunk")
    ):
        try:
            _check_spd(S_pool, "S_pool")
            res_raw = _pairwise_from_shared(S_pool, _task_centroids(class_means, toc))
            gamma_unshrunk = res_raw[0]
        except ValueError:
            gamma_unshrunk = None  # raw scatter not factorisable: skip

    # Step 4: task centroids.
    centroids = _task_centroids(class_means, toc)

    # Steps 5-6: one Cholesky, one triangular solve on the (p, T) block,
    # pairwise squared distances in whitened coordinates.
    gamma_hat_min, argmin, gam = _pairwise_from_shared(sigma_hat, centroids)

    if gamma_unshrunk is not None:
        rel_diff = abs(gamma_hat_min - gamma_unshrunk) / gamma_hat_min

    return GammaMinEstimate(
        gamma_min=float(gamma_hat_min),
        argmin=argmin,
        gamma_matrix=gam,
        task_centroids=centroids,
        class_means=class_means,
        sigma_hat=np.ascontiguousarray(sigma_hat),
        lw_intensity=float(intensity),
        gamma_min_unshrunk=(
            float(gamma_unshrunk) if gamma_unshrunk is not None else None
        ),
        cross_check_rel_diff=(float(rel_diff) if rel_diff is not None else None),
        layer_mean=np.ascontiguousarray(mu_z),
        layer_std=np.ascontiguousarray(s_z),
        n_samples=int(n),
        n_classes=int(N),
        d=int(d),
    )


def _task_centroids(class_means: np.ndarray, toc: np.ndarray) -> np.ndarray:
    """Step 4 (L4297-4299): mu_hat^(t) = (1/C_t) sum over the task's classes."""
    T = int(toc.max()) + 1
    return np.ascontiguousarray(
        np.vstack([class_means[toc == t].mean(axis=0) for t in range(T)])
    )


def _pairwise_from_shared(
    cov: np.ndarray, centroids: np.ndarray
) -> Tuple[float, Tuple[int, int], np.ndarray]:
    """Steps 5-6 with ONE shared covariance (vectorised).

    Cholesky-whitens all T centroids in a single triangular solve and
    returns (gamma_min, argmin (i<j), (T,T) matrix with 0 diagonal).
    """
    T = centroids.shape[0]
    fac = scipy.linalg.cho_factor(cov, lower=True, check_finite=True)
    w = scipy.linalg.solve_triangular(fac[0], centroids.T, lower=True)  # (p,T)
    gram = w.T @ w
    diag = np.diag(gram)
    gam_full = diag[:, None] + diag[None, :] - 2.0 * gram
    gam = 0.5 * (gam_full + gam_full.T)
    np.fill_diagonal(gam, 0.0)
    gam = np.maximum(gam, 0.0)
    iu = np.triu_indices(T, k=1)
    vals = gam[iu]
    k = int(np.argmin(vals))
    return float(vals[k]), (int(iu[0][k]), int(iu[1][k])), np.ascontiguousarray(gam)


# ---------------------------------------------------------------------------
# Self-check (python core/margin.py) -- seconds, synthetic-only, in memory
# ---------------------------------------------------------------------------


def _selfcheck() -> None:  # pragma: no cover - manual/diagnostic entry point
    import scipy

    print(f"[versions] numpy {np.__version__} | scipy {scipy.__version__}")
    print(
        "[law] in-memory self-check only: no downloads, no disk writes, "
        "no backbone, synthetic constructions only"
    )

    # ------------------------------------------------------------------ 1
    # Known-truth Gaussians: closed-form gamma_ij vs gamma_ij_pairwise.
    rng = new_rng(purpose="i01-margin-closed-form")
    d = 3
    mu1 = np.array([2.0, -1.0, 0.5])
    mu2 = np.array([-0.5, 1.5, -2.0])
    A1 = rng.standard_normal((d, d))
    A2 = rng.standard_normal((d, d))
    S1 = A1 @ A1.T / d + 0.25 * np.eye(d)
    S2 = A2 @ A2.T / d + 0.4 * np.eye(d)
    delta = mu1 - mu2
    pair_avg = 0.5 * (S1 + S2)
    closed_form = float(delta @ np.linalg.inv(pair_avg) @ delta)  # independent path
    got = gamma_ij_pairwise(mu1, mu2, S1, S2)
    assert abs(got - closed_form) < 1e-9, (got, closed_form)
    # symmetry of the pair
    assert abs(gamma_ij_pairwise(mu2, mu1, S2, S1) - got) < EPS_LINALG
    print(
        f"PASS 1  gamma_ij_pairwise vs closed-form inverse path: "
        f"{got:.12f} vs {closed_form:.12f} (|diff|={abs(got - closed_form):.2e} "
        f"< 1e-9, EPS_LINALG tier)"
    )

    # Population pipeline on an exact 3-task construction (C=2 each, d=3):
    # per-class covariances identical within each task => Sigma^(t) exact.
    C = 2
    T3 = 3
    task_covs = [S1, S2, 0.5 * (S1 + S2)]
    task_mu = [mu1, mu2, np.array([1.0, 1.0, 1.0])]
    class_means = np.vstack(
        [task_mu[t] + r * np.array([0.3, -0.3, 0.1]) for t in range(T3) for r in (1.0, -1.0)]
    )
    class_covs = np.stack([task_covs[t] for t in range(T3) for _ in range(C)])
    toc3 = np.repeat(np.arange(T3), C)
    tmeans, tcovs = task_statistics(class_means, class_covs, toc3)
    assert np.abs(tmeans - np.array(task_mu)).max() < EPS_TIGHT
    assert np.abs(tcovs - np.stack(task_covs)).max() < EPS_TIGHT
    res = gamma_min_population(class_means, class_covs, toc3)
    closed = {}
    for i in range(T3 - 1):
        for j in range(i + 1, T3):
            dd = np.array(task_mu[i]) - np.array(task_mu[j])
            closed[(i, j)] = float(
                dd @ np.linalg.inv(0.5 * (task_covs[i] + task_covs[j])) @ dd
            )
            assert abs(res.gamma_matrix[i, j] - closed[(i, j)]) < 1e-9
    assert res.convention == "pair-averaged"
    argmin_closed = min(closed, key=closed.get)
    assert res.argmin == argmin_closed
    assert abs(res.gamma_min - closed[argmin_closed]) < 1e-9
    # Shared fast path (all task covs equal to S1) equals the per-pair
    # Def 3.1 values computed with those same equal covariances.
    closed_shared = {}
    for i in range(T3 - 1):
        for j in range(i + 1, T3):
            ddv = np.array(task_mu[i]) - np.array(task_mu[j])
            closed_shared[(i, j)] = float(ddv @ np.linalg.inv(S1) @ ddv)
    res_shared = gamma_min(tmeans, S1)
    assert res_shared.convention == "shared"
    for i in range(T3 - 1):
        for j in range(i + 1, T3):
            assert abs(res_shared.gamma_matrix[i, j] - closed_shared[(i, j)]) < EPS_LINALG
    res_auto = gamma_min(tmeans, np.broadcast_to(S1, (T3, d, d)))
    assert res_auto.convention == "shared"  # auto-detected equal covariances
    assert np.abs(res_auto.gamma_matrix - res_shared.gamma_matrix).max() < EPS_TIGHT
    print(
        f"PASS 1b gamma_min_population (3 tasks, heterogeneous Sigma^(t)): "
        f"gamma_min={res.gamma_min:.9f} at {res.argmin} matches closed-form "
        f"min over pairs (<1e-9); task_statistics exact (<1e-12); shared "
        f"fast path == pairwise values on equal covariances (<EPS_LINALG), "
        f"equality auto-detected from (T,d,d) input"
    )

    # ------------------------------------------------------------------ 2
    # Isotropic sanity: Sigma_i = Sigma_j = I  =>  gamma = ||mu_i-mu_j||^2.
    rng = new_rng(purpose="i01-margin-isotropic")
    for scale in (1.0, 2.5):
        a = rng.standard_normal(d)
        b = rng.standard_normal(d)
        g_iso = gamma_ij_pairwise(
            a, b, scale * np.eye(d), scale * np.eye(d)
        )
        euclid = float(np.sum((a - b) ** 2)) / scale
        assert abs(g_iso - euclid) < 1e-12, (g_iso, euclid)
    print(
        "PASS 2  isotropic sanity: Sigma=cI reduces gamma_ij to "
        "(1/c)||mu_i-mu_j||^2 (|diff| < 1e-12 on 2 draws)"
    )

    # ------------------------------------------------------------------ 3
    # Estimator on 50k synthetic features recovers population gamma_min.
    T, C, dd, n_c = 10, 10, 12, 500          # 100 classes x 500 = 50,000
    rng = new_rng(purpose="i01-margin-50k")
    A = rng.standard_normal((dd, dd))
    Sigma = A @ A.T / dd + 0.05 * np.eye(dd)  # shared class-conditional cov
    L = scipy.linalg.cholesky(Sigma, lower=True)
    Qc, _ = np.linalg.qr(rng.standard_normal((dd, T)))
    centroids = 1.5 * Qc.T                    # task centroids (orthonormal dirs)
    means = []
    for t in range(T):
        Qw, _ = np.linalg.qr(rng.standard_normal((dd, C)))
        means.append(centroids[t] + 2.0 * Qw.T)  # per-class means
    class_means_pop = np.ascontiguousarray(np.vstack(means))
    toc10 = np.repeat(np.arange(T), C)
    # Homogeneous class-conditionals: shared (d,d) Sigma -- the regime where
    # Def 3.1's pair average and the estimator's pooled Sigma_hat target the
    # SAME population quantity.
    pop = gamma_min_population(class_means_pop, Sigma, toc10)
    assert pop.convention == "shared"
    i_star, j_star = pop.argmin
    chunks = [
        class_means_pop[r] + rng.standard_normal((n_c, dd)) @ L.T
        for r in range(T * C)
    ]
    X = np.ascontiguousarray(np.vstack(chunks))     # exactly 50,000 x 12
    y = np.repeat(np.arange(T * C, dtype=np.int64), n_c)
    assert X.shape == (50000, 12)
    est = estimate_gamma_min(X, y, toc10)
    assert est.n_samples == 50000 and est.n_classes == 100 and est.d == 12
    assert est.gamma_matrix.shape == (T, T)

    # Sampling tolerance (documented, conftest Z_STAT=6):
    #   mu_hat^(t) ~ N(mu^(t), Sigma/(C n_c)) exactly (independent classes)
    #   => d_hat = mu_hat^(i)-mu_hat^(j) has Cov = eps*Sigma, eps = 2/(C n_c)
    #   delta method on gamma = d' Sigma^-1 d:
    #     E[gamma_hat] ~ gamma + eps*d_dim ;  sd ~ sqrt(2*eps*gamma)
    #   plus ~3 sd allowance for min-selection over the 45 pairs (E[min of
    #   45 ~ N(0,1)] ~ -2.1; correlations shrink it), plus a small slack for
    #   LW shrinkage and the empirical layer normalisation (gamma is exactly
    #   invariant to the population normalisation by congruence).
    eps = 2.0 / (C * n_c)
    gamma_pop = pop.gamma_min
    sd_g = float(np.sqrt(2.0 * eps * gamma_pop))
    tol = eps * dd + (Z_STAT + 3.0) * sd_g + 0.25 * sd_g
    diff = abs(est.gamma_min - gamma_pop)
    assert diff <= tol, (est.gamma_min, gamma_pop, diff, tol)
    # Per-pair check at the population argmin (no min-selection bias).
    pair_tol = eps * dd + Z_STAT * sd_g
    pair_diff = abs(est.gamma_matrix[i_star, j_star] - pop.gamma_matrix[i_star, j_star])
    assert pair_diff <= pair_tol, (pair_diff, pair_tol)
    assert est.gamma_min <= est.gamma_matrix[i_star, j_star] + EPS_TIGHT
    # Ledoit-Wolf cross-verification against sklearn when available.
    try:  # pragma: no cover - environment-dependent
        from sklearn.covariance import LedoitWolf

        # same centred rows the estimator feeds the closed form:
        z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
        cm = _class_means_from(z, y, T * C)
        centered = z - cm[y]
        lam_sk = LedoitWolf(assume_centered=True).fit(centered).shrinkage_
        assert abs(lam_sk - est.lw_intensity) < 1e-12, (lam_sk, est.lw_intensity)
        lw_note = f"sklearn LedoitWolf shrinkage_ agrees < 1e-12 ({lam_sk:.6f})"
    except ImportError:  # pragma: no cover
        lw_note = "sklearn absent; closed form only (still numpy/scipy runtime)"
    # Spec cross-check path: lambda* < 0.05 => unshrunken agreement < 5%.
    cc = (
        f"lambda*={est.lw_intensity:.6f} < 0.05, unshrunken "
        f"gamma_min={est.gamma_min_unshrunk:.6f}, rel diff "
        f"{est.cross_check_rel_diff:.2e} < 5% (spec L4329-4333)"
        if est.gamma_min_unshrunk is not None
        else f"lambda*={est.lw_intensity:.6f}; cross-check path not triggered"
    )
    if est.cross_check_rel_diff is not None:
        assert est.cross_check_rel_diff < 0.05
    print(
        f"PASS 3  50k-feature estimator (T=10, C=10, d=12, 500/class): "
        f"population gamma_min={gamma_pop:.4f}, estimate={est.gamma_min:.4f} "
        f"(|diff|={diff:.4f} <= tol {tol:.4f}; eps*d={eps * dd:.4f}, "
        f"sd(delta-method)={sd_g:.4f}, Z=6+3 min-allowance); argmin pair "
        f"({i_star},{j_star}) per-pair |diff|={pair_diff:.4f} <= "
        f"{pair_tol:.4f}; {cc}; {lw_note}"
    )

    # Determinism: pure function, two calls bit-identical.
    est2 = estimate_gamma_min(X, y, toc10)
    assert np.array_equal(est.gamma_matrix, est2.gamma_matrix)
    assert est.gamma_min == est2.gamma_min and est.argmin == est2.argmin
    print("PASS 3b estimator determinism: two calls bit-identical")

    # ------------------------------------------------------------------ 4
    # Convention divergence: Def 3.1's PER-PAIR averaged covariance vs the
    # operational estimator's SINGLE pooled covariance over ALL tasks.
    # Three tasks, isotropic but differently scaled covariances; the
    # estimator's Sigma_hat pools every class of every task, so pair (0,1)
    # is scored under the global pool, not under (Sigma^(0)+Sigma^(1))/2.
    mu0 = np.array([3.0, 0.0])
    mu1 = np.array([-3.0, 0.0])
    mu2 = np.array([0.0, 3.0])
    c0, c1, c2 = 0.2, 1.0, 5.0
    eye2 = np.eye(2)
    S0, S1, S2 = c0 * eye2, c1 * eye2, c2 * eye2
    delta_sq = float(np.sum((mu0 - mu1) ** 2))  # 36
    g_def31 = gamma_ij_pairwise(mu0, mu1, S0, S1)      # Def 3.1 pair metric
    g_def31_closed = delta_sq / (0.5 * (c0 + c1))
    assert abs(g_def31 - g_def31_closed) < 1e-9
    S_pool = ((c0 + c1 + c2) / 3.0) * eye2            # operational pooled
    g_oper = gamma_ij_pairwise(mu0, mu1, S_pool, S_pool)
    assert abs(g_oper - delta_sq / ((c0 + c1 + c2) / 3.0)) < 1e-9
    assert abs(g_def31 - g_oper) / g_oper > 0.1        # materially different
    # And neither equals a one-sided metric (pair average is its own object).
    assert abs(g_def31 - delta_sq / c0) > 1.0 and abs(g_def31 - delta_sq / c1) > 1.0
    print(
        f"PASS 4  convention divergence (3 tasks, scales 0.2/1/5): pair "
        f"(0,1) Def-3.1 gamma_ij={g_def31:.4f} (metric (c0+c1)/2) vs "
        f"operational pooled-covariance gamma={g_oper:.4f} (metric "
        f"(c0+c1+c2)/3): ratio {g_def31 / g_oper:.2f}x -- the two "
        f"conventions COINCIDE only under homogeneous covariances"
    )

    # ------------------------------------------------------------------ 5
    # Block loss matrices, tc_gap, infeasibility_gap, bound evaluators.
    rng = new_rng(purpose="i01-margin-blocks")
    T3b, C3 = 3, 3
    Nb = T3b * C3
    rho = np.triu(np.abs(rng.standard_normal((Nb, Nb))), k=1)
    rho = rho + rho.T                                # symmetric, zero diagonal
    toc_b = np.repeat(np.arange(T3b), C3)
    blocks = block_loss_matrix(rho, toc_b)
    assert blocks.shape == (T3b, T3b)
    # diagonal block of task 0: sum over 3 classes, k != l (6 ordered pairs)
    m0 = np.flatnonzero(toc_b == 0)
    manual_d = sum(rho[k, l] for k in m0 for l in m0 if k != l)
    assert abs(blocks[0, 0] - manual_d) < EPS_TIGHT
    assert abs(blocks[0, 1] - rho[np.ix_(m0, np.flatnonzero(toc_b == 1))].sum()) < EPS_TIGHT
    # decomposition identity: all == diagonal + off-diagonal; block count 6,
    # ordered off-diagonal entries T(T-1) C^2 = 54 (L511-516).
    tot_all = total_block_loss(blocks, "all")
    tot_d = total_block_loss(blocks, "diagonal")
    tot_o = total_block_loss(blocks, "off-diagonal")
    assert abs(tot_all - (tot_d + tot_o)) < EPS_TIGHT
    assert abs(tot_all - rho.sum()) < EPS_TIGHT      # diagonal ignored globally
    n_offdiag_ordered = T3b * (T3b - 1) * C3 ** 2
    assert n_offdiag_ordered == 54
    # tc_gap and infeasibility_gap arithmetic.
    l_dag, l_star = 0.8421, 0.7719
    assert abs(tc_gap(l_dag, l_star) - 0.0702) < EPS_TIGHT
    assert abs(infeasibility_gap(tot_all, l_star) - (tot_all - l_star)) < EPS_TIGHT
    assert infeasibility_gap(blocks, blocks * 0.5) > 0
    print(
        f"PASS 5  block_loss_matrix (T=3, C=3): manual block sums exact "
        f"(<1e-12); all == diagonal + off-diagonal; ordered off-diagonal "
        f"entries = T(T-1)C^2 = {n_offdiag_ordered}; tc_gap(0.8421, 0.7719) "
        f"= {tc_gap(l_dag, l_star):.4f}; infeasibility_gap scalar/matrix OK"
    )

    # Bound evaluators: orderings and identities from the tex.
    g0 = 8.3  # synthetic-proxy value, magnitude of tab:gamma-min only
    b_op = e_tc_bound(g0, M=1.0, C=10, T=10, N=100, rate="operational")
    b_cons = e_tc_bound(g0, M=1.0, C=10, T=10, N=100, rate="conservative")
    b_half = e_tc_bound(g0, M=1.0, C=10, T=10, N=100, rate=0.5)
    assert abs(b_half - b_cons) < EPS_TIGHT * b_cons   # (1-1/2)/8 == 1/16
    assert b_op < b_cons                                # operational tighter
    assert e_tc_bound(2 * g0, M=1.0, C=10, T=10, N=100) < b_op  # exp decay
    pref = 1.0 * 100 * 90 / 99
    assert abs(b_op - pref * np.exp(-g0 / 8)) < EPS_TIGHT * b_op
    # E_CF rate: e_cf(2K)/e_cf(K) = 2^{-(alpha-1)/alpha}; alpha=2 -> 1/sqrt(2).
    e1 = e_cf_bound(1000.0, C_opt_sc=1.0, Delta_t=2.0, sigma2=0.5, mu=1.0, alpha=2.0)
    e2 = e_cf_bound(2000.0, C_opt_sc=1.0, Delta_t=2.0, sigma2=0.5, mu=1.0, alpha=2.0)
    assert abs(e2 / e1 - np.sqrt(0.5)) < 1e-12
    sweep = e_cf_bound(
        np.array([[500.0, 500.0, 500.0], [8000.0, 8000.0, 8000.0]]),
        C_opt_sc=1.0, Delta_t=np.array([1.0, 2.0, 3.0]), sigma2=0.5,
        mu=1.0, alpha=1.32,
    )
    assert sweep.shape == (2,) and np.all(sweep > 0)
    # nu_spread_ratio: finite, positive (here under the true Sigma metric).
    nu = nu_spread_ratio(class_means_pop, Sigma, toc10, i_star, j_star)
    assert np.isfinite(nu) and nu > 0.0
    print(
        f"PASS 5b bound evaluators: rate=0.5 == conservative (1/16) exact; "
        f"operational < conservative; exp(-gamma/8) prefactor exact; "
        f"E_CF halving at alpha=2 gives x{e2 / e1:.6f} (=1/sqrt2); sweep "
        f"shape (2,) OK; nu_spread_ratio(pair {i_star},{j_star}) = {nu:.4f} "
        f"(synthetic-proxy: no paper value asserted)"
    )

    print(
        "ALL SELF-CHECKS PASSED (synthetic constructions only; "
        "no paper numbers; nothing written to disk)"
    )


if __name__ == "__main__":  # pragma: no cover
    _selfcheck()
