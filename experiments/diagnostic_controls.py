"""experiments.diagnostic_controls -- CI-3 controls on the 20NG bias-direction
diagnostic (Claude Q7; fleet position 28 of 40).

QUESTION UNDER TEST. The H07 smoke (results/runs/B/nonvision/
B_nonvision_smoke.json) reports cos = 1.000000 and rel = 1.000000 for every
ordered task pair and all five seeds (backend MiniLM-L6-v2, d=384). Claude's
bet: these are CODE IDENTITIES (the quantities compared are made collinear by
construction), not measurements of Theorem 4.1 alignment.

ALGEBRAIC DIAGNOSIS (derived, then verified below by direct controls).

DeltaW_hat (H04 delta_w_hat) is the mean of the C^2 cross-pair residuals
B_kl = (w_k - w_l) - Si(mu_k - mu_l), k in task i, l in task j. The block
mean collapses exactly to

    DeltaW_hat(i,j) = (w_bar_i - w_bar_j) - Si(mu_bar_i - mu_bar_j),

with w_bar_t the mean trained weight row of task t. The softmax gauge makes
the first term identically zero for the WT arm: summing the within-task CE +
decay gradient over the task's C classes kills the data term
(sum_k P_nk = 1 = sum_k 1{y_n = k} for every in-task sample), so at any
interior optimum lam * sum_{k in t} w_k = 0, i.e. w_bar_t = 0 EXACTLY (L-BFGS
drives max|grad| to ~1e-8; observed ||w_bar_t|| ~ 1e-14 in the self-check).
Therefore

    DeltaW_hat(i,j) = -Si(mu_bar_i - mu_bar_j) = DeltaW_pred(i,j)

-- the empirical and predicted vectors are the SAME vector, so cos = 1 and
rel = 1 in ANY norm, for ANY data, including a head whose within-task rows
are random noise. The metric sees the trained W only through w_bar, and the
gauge pins w_bar = 0. (rel additionally IS defined through the same
Sigma^{-1} inner product as cos -- core.diagnostic.relative_magnitude with
sigma_inv uses sigma_inner_norm -- but that coupling is not the driver:
dh = dp forces both to 1 regardless of norm.)

CONTROLS (Claude's three, run first on the cached pipeline; 200 docs/task,
seeds 0-4, canonical grouping, training data only -- the diagnostic protocol
itself is training-side):
  C1 RANDOM HEAD: (a) raw Gaussian W scaled to the trained head's Frobenius
      norm; (b) the same random W re-centered per task (w_bar_t = 0). If
      cos(b) ~= 1 while cos(a) does not, the metric is blind to everything
      in W except the task-mean gauge -> vacuous by identity.
  C2 PERMUTED PAIRING: (a) pair-shift -- compare dh(i,j) against
      dp(pi(i), pi(j)), pi a cyclic task shift; under the identity this
      reduces to cos between two DIFFERENT predicted offsets (pure centroid
      geometry, no W); (b) regroup -- randomly reassign classes to tasks
      (permuted task_of_class) and recompute dh', dp' from scratch; cos
      0.3-0.7 would indicate real surviving structure, ~1 saturation.
  C3 EUCLIDEAN BESIDE WEIGHTED: plain cosine / norm-ratio next to the
      Sigma^{-1} ones. dh = dp as vectors makes BOTH exactly 1; disagreement
      would locate the artifact in the weighting.

LAWS honored: no downloads (sklearn cache probed with download_if_missing=
False; MiniLM loaded offline-only through experiments.nonvision's embedder);
this module's only artifacts are the results JSON it writes and stdout.

Writes: results/runs/B/nonvision/CI3_diagnostic_controls.json.
Run: python experiments/diagnostic_controls.py [--seeds 0,1,2,3,4]
     [--cap-per-task 200] [--mode run|selfcheck]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import nonvision as nv  # H07 machinery (loaders/embedders)
from core.diagnostic import (  # H04 algebra under test
    delta_w_hat as _dh,
    delta_w_pred as _dp,
    symmetric_inverse as _sym_inv,
)

EXPERIMENT = "diagnostic_controls"
DRIFT_NOTES: List[str] = list(nv.DRIFT_NOTES)
DRIFT_NOTES.append(
    "CI-3 controls: H07 loaders/embedders + H04 core.diagnostic imported "
    "unchanged; controls live in this module only."
)

T = nv.T_TASKS
N_CLASSES = nv.N_CLASSES
C = nv.C_CLASSES


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cos(u: np.ndarray, v: np.ndarray, si: Optional[np.ndarray] = None) -> float:
    """Plain cosine (si=None) or Sigma^{-1}-inner-product cosine."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    if si is None:
        nu, nv_ = float(u @ u), float(v @ v)
        cross = float(u @ v)
    else:
        si = np.asarray(si, float)
        nu, nv_ = float(u @ si @ u), float(v @ si @ v)
        cross = float(u @ si @ v)
    if nu <= 0.0 or nv_ <= 0.0:
        return float("nan")
    return cross / np.sqrt(nu * nv_)


def _ratio(u: np.ndarray, v: np.ndarray, si: Optional[np.ndarray] = None) -> float:
    """||u|| / ||v|| in the same (weighted or plain) norm."""
    si2 = None if si is None else np.asarray(si, float)
    if si is None:
        nu, nv_ = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    else:
        nu = float(np.sqrt(u @ si2 @ u))
        nv_ = float(np.sqrt(v @ si2 @ v))
    return float("nan") if nv_ <= 0.0 else nu / nv_


def _norm(u: np.ndarray, si: Optional[np.ndarray] = None) -> float:
    if si is None:
        return float(np.linalg.norm(u))
    si = np.asarray(si, float)
    return float(np.sqrt(max(u @ si @ u, 0.0)))


class _Head:  # H04 head protocol (duck-typed; b unused by the diagnostic)
    def __init__(self, W, mu_hat, task_of_class):
        self.W = np.asarray(W, float)
        self.b = np.zeros(self.W.shape[0])
        self.mu_hat = np.asarray(mu_hat, float)
        self.task_of_class = np.asarray(task_of_class, np.int64)


# ---------------------------------------------------------------------------
# Control battery on one fitted seed
# ---------------------------------------------------------------------------


def pair_metrics(head: _Head, si: np.ndarray, toc: np.ndarray
                 ) -> Dict[str, Any]:
    """All controls' per-pair quantities for one head / one class grouping."""
    out: Dict[str, Any] = {}
    si_inv = np.asarray(si, float)

    # C2a task permutation for the pair-shift control (fixed cyclic shift:
    # pi(t) = t+1 mod T; never maps a task to itself for T > 1).
    shift = lambda t: (t + 1) % T

    rows = []
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            dh = _dh(head, i, j, si_inv)
            dp = _dp(head, i, j, si_inv)
            # per-task mean weights (the gauge term the block mean retains)
            ki = np.flatnonzero(toc == i)
            kj = np.flatnonzero(toc == j)
            wbi = head.W[ki].mean(axis=0)
            wbj = head.W[kj].mean(axis=0)
            row = dict(
                i=i, j=j,
                # baseline (trained head), weighted + Euclidean (C3)
                cos_si=_cos(dh, dp, si_inv),
                rel_si=_ratio(dh, dp, si_inv),
                cos_euc=_cos(dh, dp),
                rel_euc=_ratio(dh, dp),
                # direct identity probes
                gap_si=_norm(dh - dp, si_inv) / max(_norm(dp, si_inv), 1e-300),
                gap_euc=_norm(dh - dp) / max(_norm(dp), 1e-300),
                gauge_si=_norm(wbi - wbj, si_inv) / max(_norm(dp, si_inv), 1e-300),
                algebra_maxdev=float(np.abs((dh - dp) - (wbi - wbj)).max()),
                # C2a pair-shift: dh(i,j) vs dp of a DIFFERENT pair
                cos_shift=_cos(dh, _dp(head, shift(i), shift(j), si_inv), si_inv),
            )
            rows.append(row)
    out["per_pair"] = rows
    for key in ("cos_si", "rel_si", "cos_euc", "rel_euc", "gap_si", "gap_euc",
                "gauge_si", "algebra_maxdev", "cos_shift"):
        vals = [r[key] for r in rows]
        out[key] = dict(
            mean=float(np.mean(vals)), min=float(np.min(vals)),
            max=float(np.max(vals)),
        )
    # per-task gauge magnitudes of this head
    out["task_mean_w_norms"] = [
        float(np.linalg.norm(head.W[toc == t].mean(axis=0))) for t in range(T)]
    return out


def control_battery(W: np.ndarray, mu_hat: np.ndarray, toc: np.ndarray,
                    sigma_hat: np.ndarray, seed: int) -> Dict[str, Any]:
    """C1 (random head raw + per-task-centered), C2b (regroup), baseline."""
    si = _sym_inv(np.asarray(sigma_hat, float))
    W = np.asarray(W, float)
    rng = np.random.default_rng(nv.derive_seed("ci3:randhead", seed))

    base_head = _Head(W, mu_hat, toc)
    res: Dict[str, Any] = {"baseline_trained": pair_metrics(base_head, si, toc)}

    # --- C1a: raw random head, same Frobenius norm ------------------------
    Wr = rng.standard_normal(W.shape)
    Wr *= float(np.linalg.norm(W)) / float(np.linalg.norm(Wr))
    res["c1a_random_raw"] = pair_metrics(_Head(Wr, mu_hat, toc), si, toc)

    # --- C1b: same random head, re-centered per task (gauge restored) -----
    Wc = Wr.copy()
    for t in range(T):
        m = toc == t
        Wc[m] -= Wc[m].mean(axis=0)
    res["c1b_random_taskcentered"] = pair_metrics(_Head(Wc, mu_hat, toc), si, toc)

    # --- C2b: regroup -- random class->task assignment, fresh diagnostic ---
    # NOTE: H04's delta_* read head.task_of_class, so the permuted map must
    # ride ON THE HEAD (a fresh head sharing W and mu_hat), never a side arg.
    while True:
        perm = rng.permutation(N_CLASSES)
        toc_p = toc[perm]
        if not np.array_equal(toc_p, toc):
            break
    res["c2b_regroup"] = pair_metrics(_Head(W, mu_hat, toc_p), si, toc_p)
    res["c2b_regrouptoc"] = [int(x) for x in toc_p]
    return res


# ---------------------------------------------------------------------------
# Cached-pipeline driver (smoke scale)
# ---------------------------------------------------------------------------


def run_controls(seeds: Sequence[int], cap_per_task: int, backend_pref: str = "minilm"
                 ) -> Dict[str, Any]:
    t0 = time.time()
    tr = nv.load_20ng("train")  # raises NoCacheError if absent (never downloads)
    train_texts, train_y = list(tr.data), np.asarray(tr.target)
    embedder, backend = nv.build_embedder(backend_pref)

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        part = nv.make_partition("canonical", seed)
        toc = part["task_of_class"]
        # cap FIRST (only capped subset is ever embedded); train side only --
        # the tab:bias-direction protocol is training-data-only
        tr_s, ytr_s = nv._cap_per_task(
            train_texts, train_y, part["chunks"], cap_per_task,
            f"ci3:cap:s{seed}")
        Xtr = np.asarray(embedder.fit_transform(tr_s), float)
        d = Xtr.shape[1]

        # WT arm exactly as H07: sequential within-task CE fits, train order
        W = np.zeros((N_CLASSES, d))
        b = np.zeros(N_CLASSES)
        blocks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        max_grad = 0.0
        for t in part["train_order"]:
            chunk = np.array(part["chunks"][t])
            m = np.isin(ytr_s, chunk)
            Xb, yb_local = Xtr[m], np.searchsorted(chunk, ytr_s[m])
            Wt, bt, info = nv.fit_ce_head_lbfgs(Xb, yb_local, C, nv.LAM_DEFAULT)
            W[chunk], b[chunk] = Wt, bt
            blocks[t] = (Xb, ytr_s[m])
            max_grad = max(max_grad, info["max_grad"])

        # training-side stats exactly as H07 (SPEC-5 pooled LW Sigma_hat)
        Xtr_all = np.vstack([blocks[t][0] for t in sorted(blocks)])
        ytr_all = np.concatenate([blocks[t][1] for t in sorted(blocks)])
        stats, sigma_hat, lw_int = nv.build_task_stats(
            Xtr_all, ytr_all, toc, blocks)
        mu_hat = np.stack([Xtr_all[ytr_all == k].mean(axis=0)
                           for k in range(N_CLASSES)])

        battery = control_battery(W, mu_hat, toc, sigma_hat, seed)
        battery.update(seed=seed, n_train=int(len(ytr_s)), d=int(d),
                       fit_max_grad_wt=float(max_grad),
                       lw_intensity_mean=float(lw_int),
                       partition=part["chunks"])
        per_seed.append(battery)
        b_ = battery["baseline_trained"]
        print(f"[ci3] seed {seed}: trained cos_si {b_['cos_si']['mean']:.6f} "
              f"cos_euc {b_['cos_euc']['mean']:.6f} gap_si {b_['gap_si']['mean']:.2e} "
              f"| rand_raw cos_si {battery['c1a_random_raw']['cos_si']['mean']:.4f} "
              f"| rand_centered cos_si "
              f"{battery['c1b_random_taskcentered']['cos_si']['mean']:.6f} "
              f"| shift {b_['cos_shift']['mean']:.4f} "
              f"| regroup {battery['c2b_regroup']['cos_si']['mean']:.4f}",
              flush=True)

    def _agg(arm: str, keys: Sequence[str]) -> Dict[str, Any]:
        """Per-seed pair-mean of each metric under one arm of the battery."""
        out: Dict[str, Any] = {}
        for k in keys:
            vals = [float(s[arm][k]["mean"]) for s in per_seed]
            out[k] = dict(
                mean=round(float(np.mean(vals)), 6),
                sd=round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 6),
                per_seed=[round(v, 6) for v in vals])
        return out

    _COMMON = ("cos_si", "rel_si", "cos_euc", "rel_euc", "gap_si", "cos_shift")
    summary: Dict[str, Any] = {
        "baseline_trained": _agg("baseline_trained", _COMMON + (
            "gap_euc", "gauge_si", "algebra_maxdev")),
        "c1a_random_raw": _agg("c1a_random_raw", _COMMON),
        "c1b_random_taskcentered": _agg("c1b_random_taskcentered", _COMMON),
        "c2b_regroup": _agg("c2b_regroup", _COMMON),
    }
    # verdict inputs, mechanical (Claude's decision rule, encoded)
    base = summary["baseline_trained"]
    c1a = summary["c1a_random_raw"]
    c1b = summary["c1b_random_taskcentered"]
    c2b = summary["c2b_regroup"]
    c2a_vals = [s["baseline_trained"]["cos_shift"]["mean"] for s in per_seed]
    c2a_mean = float(np.mean(c2a_vals))
    identity_conditions = dict(
        baseline_cos_si_is_1=abs(base["cos_si"]["mean"] - 1.0) < 1e-6,
        baseline_rel_si_is_1=abs(base["rel_si"]["mean"] - 1.0) < 1e-6,
        gap_si_near_zero=base["gap_si"]["mean"] < 1e-3,
        euclidean_agrees=abs(base["cos_euc"]["mean"] - 1.0) < 1e-6,
        random_centered_cos_si_is_1=abs(
            c1b["cos_si"]["mean"] - 1.0) < 1e-3,
        random_raw_breaks=abs(
            c1a["cos_si"]["mean"] - base["cos_si"]["mean"]) > 1e-3,
    )
    verdict = dict(
        identity=all(identity_conditions.values()),
        conditions=identity_conditions,
        c2a_pair_shift_cos_mean=round(c2a_mean, 6),
        c2b_regroup_cos_mean=c2b["cos_si"]["mean"],
        rule=("IDENTITY iff: baseline cos_si == rel_si == 1 exactly, "
              "dh-dp gap ~ 0 in BOTH norms, and a PER-TASK-CENTERED RANDOM "
              "head still gives cos == 1 (metric blind to W beyond the task "
              "mean); structure would need the raw random head to matter "
              "and permutation to land in 0.3-0.7"),
    )
    return dict(
        backend=backend, d=int(d), seeds=list(seeds),
        cap_per_task=cap_per_task, grouping="canonical",
        lam=nv.LAM_DEFAULT, n_pairs=int(T * (T - 1)),
        per_seed=[
            {k: v for k, v in s.items()} for s in per_seed],
        summary=summary, verdict=verdict,
        algebra=("DeltaW_hat(i,j) = (w_bar_i - w_bar_j) - Si(mu_bar_i - mu_bar_j); "
                 "softmax gauge pins w_bar_t = 0 at any interior optimum of the "
                 "decayed within-task CE objective, so DeltaW_hat == DeltaW_pred "
                 "identically (verified per pair: algebra_maxdev is the max "
                 "|(dh - dp) - (w_bar_i - w_bar_j)|)."),
        runtime_s=round(time.time() - t0, 1),
        drift_notes=DRIFT_NOTES,
    )


# ---------------------------------------------------------------------------
# Self-check (data-free; proves the identity mechanism on synthetic data)
# ---------------------------------------------------------------------------


def _selfcheck() -> int:
    rng = np.random.default_rng(nv.derive_seed("ci3:selfcheck"))
    d, n_pc = 20, 400
    means = rng.standard_normal((N_CLASSES, d))
    A = rng.standard_normal((d, d))
    cov = A @ A.T / d + 0.1 * np.eye(d)
    L = np.linalg.cholesky(cov)
    X = np.vstack([means[k] + rng.standard_normal((n_pc, d)) @ L.T
                   for k in range(N_CLASSES)])
    y = np.repeat(np.arange(N_CLASSES), n_pc)
    toc = np.repeat(np.arange(T), C)
    W = np.zeros((N_CLASSES, d))
    for t in range(T):
        idx = np.flatnonzero(toc == t)
        m = np.isin(y, idx)
        Wt, _, _ = nv.fit_ce_head_lbfgs(X[m], np.searchsorted(idx, y[m]),
                                        C, nv.LAM_DEFAULT)
        W[idx] = Wt
    mu_hat = np.stack([X[y == k].mean(axis=0) for k in range(N_CLASSES)])
    bat = control_battery(W, mu_hat, toc, cov, seed=0)
    base = bat["baseline_trained"]
    assert max(base["task_mean_w_norms"]) < 1e-8, base["task_mean_w_norms"]
    print(f"PASS gauge: max_t ||w_bar_t|| = "
          f"{max(base['task_mean_w_norms']):.2e} (< 1e-8; softmax gauge)")
    assert base["gap_si"]["max"] < 1e-6, base["gap_si"]
    print(f"PASS identity: max over pairs ||dh - dp||_si / ||dp||_si = "
          f"{base['gap_si']['max']:.2e} (< 1e-6) on a converged WT head")
    assert abs(base["cos_si"]["mean"] - 1.0) < 1e-9
    assert abs(base["cos_euc"]["mean"] - 1.0) < 1e-9
    assert abs(base["rel_si"]["mean"] - 1.0) < 1e-9
    print("PASS trained head: cos_si == cos_euc == rel_si == 1 (any norm)")
    c1b = bat["c1b_random_taskcentered"]
    assert c1b["gap_si"]["max"] < 1e-9, c1b["gap_si"]
    assert abs(c1b["cos_si"]["mean"] - 1.0) < 1e-9
    print("PASS C1b: RANDOM task-centered head also gives cos_si == 1, "
          "gap ~ 0 -- the metric is blind to W beyond the task mean")
    c1a = bat["c1a_random_raw"]
    assert abs(c1a["cos_si"]["mean"] - 1.0) > 1e-3 or c1a["gap_si"]["max"] > 1e-3
    print(f"PASS C1a: raw random head breaks it (cos_si mean "
          f"{c1a['cos_si']['mean']:.4f}, gap_si max {c1a['gap_si']['max']:.2e})")
    print(f"note C2a pair-shift cos mean {base['cos_shift']['mean']:.4f}; "
          f"C2b regroup cos mean {bat['c2b_regroup']['cos_si']['mean']:.4f} "
          "(synthetic geometry; informative only)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", default="run", choices=["run", "selfcheck"])
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--cap-per-task", type=int, default=200)
    p.add_argument("--out", default=str(_REPO_ROOT / "results" / "runs" / "B"
                                        / "nonvision"
                                        / "CI3_diagnostic_controls.json"))
    args = p.parse_args(argv)
    seeds = [int(s) for s in str(args.seeds).split(",") if s != ""]

    if args.mode == "selfcheck":
        return _selfcheck()

    started = _utcnow()
    record = run_controls(seeds, args.cap_per_task)
    record = dict(
        run_id="CI3_diagnostic_controls", paper="B", experiment=EXPERIMENT,
        provenance="ci3-controls-local-cached (Claude Q7; H07 machinery, "
                   "H04 algebra, smoke scale)",
        created_utc=_utcnow(), executed_utc=started,
        config=dict(seeds=seeds, cap_per_task=args.cap_per_task,
                    grouping="canonical", lam=nv.LAM_DEFAULT,
                    dataset="20newsgroups-by-date (train side only)"),
        env=dict(python=platform.python_version(), numpy=np.__version__,
                 platform=platform.platform()),
        code_version=hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()[:16],
        **record,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    v = record["verdict"]
    print(f"\n[ci3] VERDICT inputs: identity={v['identity']}")
    for k, val in v["conditions"].items():
        print(f"  {k}: {val}")
    print(f"  c2a pair-shift cos mean: {v['c2a_pair_shift_cos_mean']} | "
          f"c2b regroup cos mean: {v['c2b_regroup_cos_mean']}")
    print(f"[ci3] record -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
