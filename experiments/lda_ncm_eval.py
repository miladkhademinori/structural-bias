"""experiments.lda_ncm_eval -- generator of Paper B's tab:lda-ncm-comparison
(20 LDA/NCM cells) from the cached frozen-feature store.

Status: IMPLEMENTED by H09 (code-fleet wave), 2026-10-21.
Spec sources:
  - exec/G08_log.md S2 (results-store schema), S4 (20-cell LDA/NCM script),
    S5 (caption-population tags)          [campaign-root exec/]
  - N1000_INIT_2026-10-21/exec/G03_log.md SPEC 5 (LDA/NCM heads), SPEC 8
    (benchmark protocol)
  - N1000_INIT_2026-10-21/N24_B_data_spec.md section 1 (cell family 1, MISSING
    rule, variant-B escalation)
Input contract (also written to data/B/README.md):
  data/B/features/<backbone>/<dataset>/seed<i>.npz  for i in 0..4
  data/B/orderings.json
Everything is closed-form: no training anywhere. WT/FIX columns are byte-copy
facts of tab:fix-benchmark (working/paper_B/neurips_2026.tex L750-781, quoted
with source anchor; never regenerated here). MISSING is reported, never
estimated (N24 rule). Cross-check against fix-benchmark store entries reports
deltas and never overwrites.

core/lda_ncm.py (owner H03) has LANDED and this script imports it rather
than reimplementing: every covariance/head computation below delegates
first to the core bridge exports (task_lw_covariance, pooled_covariance,
lda_final_accuracy, ncm_final_accuracy -- all built on StreamingGaussianHead,
fit_lda_head/fit_ncm_head/evaluate_cil_sequence machinery, same Ledoit-Wolf
closed form). Thin labeled fallbacks remain ONLY for a hypothetical
core-absent environment; the run JSON records which path produced the
numbers via the `covariance` field ("core path: core-lda_ncm").
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEEDS = (0, 1, 2, 3, 4)
TABLE_LABEL = "tab:lda-ncm-comparison"

# ---------------------------------------------------------------------------
# Canonical 10-row roster with the owner WT/FIX facts (byte-copy cells).
# Source: working/paper_B/neurips_2026.tex L750-781 (tab:fix-benchmark),
# read 2026-10-21. These are owner-run artifacts; this script quotes them and
# never regenerates them.
# columns: backbone, dataset, wt_mean, wt_sd, fix_mean, fix_sd
# ---------------------------------------------------------------------------
CANONICAL_ROWS: list[tuple[str, str, float, float, float, float]] = [
    ("DINOv2-B/14", "CIFAR-100", 82.3, 0.4, 86.5, 0.3),
    ("DINOv2-B/14", "ImageNet-100", 80.5, 0.4, 84.7, 0.3),
    ("DINOv2-B/14", "ImageNet-1K", 76.8, 0.4, 81.8, 0.3),
    ("CLIP ViT-B/16", "CIFAR-100", 81.5, 0.5, 85.7, 0.4),
    ("CLIP ViT-B/16", "ImageNet-100", 79.2, 0.5, 83.5, 0.4),
    ("MAE ViT-B/16", "ImageNet-100", 65.3, 0.6, 77.2, 0.5),
    ("ResNet-50", "ImageNet-100", 62.1, 0.6, 73.8, 0.5),
    ("ViT-B/16", "ImageNet-1K", 78.5, 0.4, 83.2, 0.3),
    ("DINOv2-B/14", "DomainNet", 58.2, 0.8, 68.5, 0.6),
    ("CLIP ViT-B/16", "DomainNet", 56.8, 0.8, 66.8, 0.6),
]

STRESS_TAG = (
    "DomainNet rows are shared-label domains-as-tasks runs (domains sharing "
    "one $345$-class label set) that violate the disjoint-label-set assumption"
)

# Caption-population tag strings (G08 S5; verbatim fleet rulings).
CAPTION_TAGS = [
    "5 random class orderings (seeds 0--4)",
    "bold marks the de-centered head",
    STRESS_TAG,
]


def _norm(name: str) -> str:
    """Filesystem-slug normalisation: lowercase alphanumerics only."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


ROW_SLUGS = {(_norm(b), _norm(d)): (b, d) for b, d, *_ in CANONICAL_ROWS}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Head components. core/lda_ncm.py (H03) is preferred when it has landed;
# each fallback is labeled so run JSONs record which path produced numbers.
# ---------------------------------------------------------------------------

_CORE_STATUS = "not-importable"


def _core_fn(name: str):
    global _CORE_STATUS
    try:
        import core.lda_ncm as _m  # noqa: PLC0415
    except Exception:
        _CORE_STATUS = "import-failed"
        return None
    fn = getattr(_m, name, None)
    if callable(fn):
        _CORE_STATUS = "core-lda_ncm"
        return fn
    _CORE_STATUS = "skeleton-no-callables"
    return None


def class_means(x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Per-class mean feature vectors stacked (n_classes, d).

    Computed over ALL rows carrying the label, which pools across tasks on
    the shared-label DomainNet rows (class seen in every task) and equals the
    per-task mean on disjoint-label rows. Classes absent from y leave NaN
    rows; the head functions below reject that case.
    """
    mu = np.full((len(classes), x.shape[1]), np.nan, dtype=np.float64)
    for i, c in enumerate(classes):
        m = y == c
        if m.any():
            mu[i] = x[m].mean(axis=0)
    return mu


def _task_centered(x_task: np.ndarray, y_task: np.ndarray) -> np.ndarray:
    """Within-task class-centered features (classes present in the task)."""
    return np.vstack([x_task[y_task == c] - x_task[y_task == c].mean(axis=0)
                      for c in np.unique(y_task)]).astype(np.float64,
                                                          copy=False)


def task_lw_covariance(x_task: np.ndarray, y_task: np.ndarray,
                       classes: np.ndarray | None = None) -> np.ndarray:
    """Ledoit-Wolf shrunk WITHIN-CLASS covariance of one task's train data.

    G03 SPEC 5: sklearn.covariance.LedoitWolf (analytical 2004 shrinkage, not
    cross-validated), fitted on task-centered features with
    assume_centered=True. numpy fallback implements the same closed form.
    """
    fn = _core_fn("task_lw_covariance")
    if fn is not None:
        return fn(x_task, y_task)
    xc = _task_centered(x_task, y_task)
    try:
        from sklearn.covariance import LedoitWolf  # noqa: PLC0415
        return LedoitWolf(assume_centered=True).fit(xc).covariance_
    except Exception:
        return _lw_covariance_numpy(xc)[0]


def _lw_covariance_numpy(xc: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf 2004 closed form on pre-centered data (labeled fallback).

    S = X'X/n; m = tr(S)/p; d2 = tr(S^2)/p - m^2;
    bbar2 = (sum_k ||x_k||^4 - n tr(S^2)) / (p n^2); rho = min(bbar2, d2)/d2.
    Returns (shrunk covariance, rho).
    """
    n, p = xc.shape
    s = xc.T @ xc / n
    m = float(np.trace(s) / p)
    d2 = max(float(np.trace(s @ s) / p - m * m), 0.0)
    sq = np.sum(xc ** 2, axis=1)
    bbar2 = max(float((np.sum(sq ** 2) - n * float(np.trace(s @ s)))
                      / (p * n * n)), 0.0)
    rho = 0.0 if d2 <= 0.0 else min(1.0, min(bbar2, d2) / d2)
    return (1.0 - rho) * s + rho * m * np.eye(p), rho


def pooled_covariance(covs: list[np.ndarray]) -> np.ndarray:
    """Equal-weight average of per-task LW covariances (G03 SPEC 5 pooling)."""
    fn = _core_fn("pooled_covariance")
    if fn is not None:
        return fn(covs)
    return np.mean(np.stack(covs), axis=0)


def _scores_to_acc(scores: np.ndarray, y_true: np.ndarray) -> float:
    return float(100.0 * np.mean(np.argmax(scores, axis=1) == y_true))


def lda_final_accuracy(train_tasks: list[tuple[np.ndarray, np.ndarray]],
                       classes: np.ndarray | None,
                       test_x: np.ndarray, test_y: np.ndarray) -> float:
    """LDA (SLDA-style) final average accuracy A-bar, percent.

    Score: mu_r' Sigma^-1 x - (1/2) mu_r' Sigma^-1 mu_r with Sigma_hat the
    pooled per-task Ledoit-Wolf within-class covariance. Final accuracy on the
    cumulative test set at sequence end (all class statistics maintained).
    Class means pool all training rows of the class (identical to per-task
    means on disjoint-label rows; cross-domain pooling on DomainNet rows).
    """
    fn = _core_fn("lda_final_accuracy")
    if fn is not None:
        return float(fn(train_tasks, classes, test_x, test_y))
    x_all = np.vstack([x_t for x_t, _ in train_tasks])
    y_all = np.concatenate([y_t for _, y_t in train_tasks])
    classes = np.unique(y_all)                      # data is the authority
    if not set(np.unique(test_y).tolist()) <= set(classes.tolist()):
        raise ValueError("test set contains classes with no training data")
    mu = class_means(x_all, y_all, classes)
    if np.isnan(mu).any():
        raise ValueError("internal error: unresolved class mean")
    sigma = pooled_covariance([task_lw_covariance(x_t, y_t)
                               for x_t, y_t in train_tasks])
    try:
        chol = np.linalg.cholesky(sigma)
        sig_inv_mu = np.linalg.solve(chol.T, np.linalg.solve(chol, mu.T)).T
    except np.linalg.LinAlgError:
        sig_inv_mu = (np.linalg.pinv(sigma) @ mu.T).T  # non-PD fallback path
    scores = test_x @ sig_inv_mu.T - 0.5 * np.sum(mu * sig_inv_mu, axis=1)
    return _scores_to_acc(scores, test_y)


def ncm_final_accuracy(train_tasks: list[tuple[np.ndarray, np.ndarray]],
                       classes: np.ndarray | None,
                       test_x: np.ndarray, test_y: np.ndarray) -> float:
    """NCM final accuracy: same discriminant with Sigma = I (nearest class
    mean over Euclidean cached features)."""
    fn = _core_fn("ncm_final_accuracy")
    if fn is not None:
        return float(fn(train_tasks, classes, test_x, test_y))
    x_all = np.vstack([x_t for x_t, _ in train_tasks])
    y_all = np.concatenate([y_t for _, y_t in train_tasks])
    classes = np.unique(y_all)
    if not set(np.unique(test_y).tolist()) <= set(classes.tolist()):
        raise ValueError("test set contains classes with no training data")
    mu = class_means(x_all, y_all, classes)
    scores = test_x @ mu.T - 0.5 * np.sum(mu * mu, axis=1)
    return _scores_to_acc(scores, test_y)


# ---------------------------------------------------------------------------
# Feature-tree walking and seed loading
# ---------------------------------------------------------------------------

class SeedData:
    """One seed's cached features, normalised to flat arrays + task split."""

    def __init__(self, train_x, train_y, train_task, test_x, test_y, meta):
        self.train_x = train_x
        self.train_y = train_y
        self.train_task = train_task
        self.test_x = test_x
        self.test_y = test_y
        self.meta = meta


def load_seed_npz(path: Path, orderings: dict | None,
                  backbone: str, dataset: str, seed: int) -> SeedData:
    """Load seed<i>.npz. Accepts three equivalent layouts -- flat
    (train_x/train_y/train_task + test_x/test_y), per-task train keys
    ("train_x_t<t>" or "train_x_<t>", the latter being the
    core.lda_ncm.save_features_npz layout) -- with test either cumulative
    (test_x/test_y) or per-stage (test_x_<t>/test_y_<t>). The task partition
    comes from the npz itself and is cross-checked against orderings.json.

    Raises ValueError with a MISSING reason when the file is unusable.
    """
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"unreadable npz ({exc})") from exc
    keys = set(z.files)

    def _arr(k, dtype=np.float64):
        return np.asarray(z[k], dtype=dtype)

    # Per-task train x keys: "train_x_t<t>" (local style) or "train_x_<t>"
    # (core.lda_ncm.save_features_npz style) -> map t -> x key.
    task_x_keys: dict[int, str] = {}
    for k in keys:
        m = re.match(r"^train_x_t?(\d+)$", k)
        if m:
            task_x_keys[int(m.group(1))] = k
    t_ids = sorted(task_x_keys)

    # Train block: flat layout first, then per-task keys.
    if {"train_x", "train_y", "train_task"} <= keys:
        train_x = _arr("train_x")
        train_y = _arr("train_y", np.int64)
        train_task = _arr("train_task", np.int64)
    elif t_ids and all(task_x_keys[t].replace("train_x", "train_y", 1) in keys
                       for t in t_ids):
        xs = [_arr(task_x_keys[t]) for t in t_ids]
        ys = [_arr(task_x_keys[t].replace("train_x", "train_y", 1), np.int64)
              for t in t_ids]
        train_x = np.vstack(xs)
        train_y = np.concatenate(ys)
        train_task = np.concatenate([np.full(len(y), t)
                                     for t, y in zip(t_ids, ys)])
    else:
        raise ValueError("npz lacks required keys (see data/B/README.md)")

    # Test block: cumulative test_x/test_y, else per-stage test_x_<t>.
    if {"test_x", "test_y"} <= keys:
        test_x, test_y = _arr("test_x"), _arr("test_y", np.int64)
    elif t_ids and all(f"test_x_{t}" in keys and f"test_y_{t}" in keys
                       for t in t_ids):
        test_x = np.vstack([_arr(f"test_x_{t}") for t in t_ids])
        test_y = np.concatenate([_arr(f"test_y_{t}", np.int64)
                                 for t in t_ids])
    else:
        raise ValueError("npz lacks required keys (see data/B/README.md)")

    if train_x.ndim != 2 or train_x.shape[0] != len(train_y) or \
            train_x.shape[0] != len(train_task) or test_x.shape[0] != len(test_y) or \
            test_x.shape[1] != train_x.shape[1]:
        raise ValueError("shape mismatch inside npz")

    classes = np.unique(train_y)
    # Task partition cross-check / derivation against orderings.json.
    ord_key = None
    if orderings is not None:
        for cand in (dataset, _norm(dataset)):
            if cand in orderings and str(seed) in orderings[cand]:
                ord_key = cand
                break
    npz_tasks = {int(t): set(map(int, np.unique(train_y[train_task == t])))
                 for t in np.unique(train_task)}
    if ord_key is not None:
        parts = orderings[ord_key][str(seed)]
        ord_tasks = {t: set(map(int, cls)) for t, cls in enumerate(parts)}
        if ord_tasks != npz_tasks:
            raise ValueError("ordering mismatch: npz task partition differs "
                             "from orderings.json for this (dataset, seed)")
    meta = {k: str(z[k]) for k in ("backbone", "dataset", "seed", "ordering_id")
            if k in keys}
    meta.update(backbone=backbone, dataset=dataset, seed=seed,
                d=int(train_x.shape[1]), T=int(len(npz_tasks)),
                n_classes=int(len(classes)),
                ordering_checked=bool(ord_key is not None))
    return SeedData(train_x, train_y, train_task, test_x, test_y, meta)


def walk_features(features_root: Path) -> tuple[dict, list[str]]:
    """Map (backbone, dataset) -> {seed: npz path} over the canonical roster.

    Unmatched directories are reported, never silently dropped.
    """
    found: dict[tuple[str, str], dict[int, Path]] = {}
    unmatched: list[str] = []
    if not features_root.is_dir():
        return found, unmatched
    for b_dir in sorted(features_root.iterdir()):
        if not b_dir.is_dir():
            continue
        for d_dir in sorted(b_dir.iterdir()):
            if not d_dir.is_dir():
                continue
            key = (_norm(b_dir.name), _norm(d_dir.name))
            if key not in ROW_SLUGS:
                unmatched.append(f"{b_dir.name}/{d_dir.name}")
                continue
            seeds = {}
            for f in sorted(d_dir.glob("seed*.npz")):
                stem = f.stem
                if stem.startswith("seed") and stem[4:].isdigit():
                    seeds[int(stem[4:])] = f
            found[ROW_SLUGS[key]] = seeds
    return found, unmatched


def evaluate_row(row: tuple, seed_paths: dict[int, Path],
                 orderings: dict | None) -> dict:
    """Run LDA and NCM per available seed; all-5-present or MISSING."""
    backbone, dataset, wt_m, wt_sd, fix_m, fix_sd = row
    out: dict[str, Any] = {
        "backbone": backbone, "dataset": dataset,
        "wt": {"mean": wt_m, "sd": wt_sd},  # byte-copy facts (owner artifact)
        "fix": {"mean": fix_m, "sd": fix_sd},
        "stress_test": dataset == "DomainNet",
        "seeds_present": sorted(s for s in SEEDS if s in seed_paths),
        "seeds_missing": sorted(s for s in SEEDS if s not in seed_paths),
        "lda": None, "ncm": None, "missing_reason": None,
        "inputs_sha256": {str(p): _sha256(p) for p in
                          sorted(seed_paths.values())},
    }
    if not seed_paths:
        out["missing_reason"] = "no feature directory / no seed files"
        return out
    per_seed: dict[int, dict] = {}
    for seed in sorted(seed_paths):
        try:
            sd = load_seed_npz(seed_paths[seed], orderings, backbone,
                               dataset, seed)
            classes = np.unique(sd.train_y)
            tasks = [int(t) for t in np.unique(sd.train_task)]
            train_tasks = [(sd.train_x[sd.train_task == t],
                            sd.train_y[sd.train_task == t]) for t in tasks]
            per_seed[seed] = {
                "lda": lda_final_accuracy(train_tasks, classes,
                                          sd.test_x, sd.test_y),
                "ncm": ncm_final_accuracy(train_tasks, classes,
                                          sd.test_x, sd.test_y),
                "n_test": int(len(sd.test_y)), "d": sd.meta["d"],
                "T": sd.meta["T"], "n_classes": sd.meta["n_classes"],
            }
        except ValueError as exc:
            per_seed[seed] = {"error": str(exc)}
    ok = [s for s, r in per_seed.items() if "error" not in r]
    bad = {s: r.get("error", "") for s, r in per_seed.items() if "error" in r}
    out["per_seed"] = {str(s): per_seed[s] for s in sorted(per_seed)}
    if bad:
        out["seeds_invalid"] = {str(s): e for s, e in bad.items()}
    if len(ok) != len(SEEDS):
        miss = sorted(set(SEEDS) - set(ok))
        out["missing_reason"] = (
            f"seeds {miss} absent or invalid; cell requires all of "
            f"{list(SEEDS)} (MISSING is never estimated)")
        return out
    out["lda"] = _agg([per_seed[s]["lda"] for s in SEEDS])
    out["ncm"] = _agg([per_seed[s]["ncm"] for s in SEEDS])
    return out


def _agg(vals: list[float], ddof: int = 1) -> dict:
    """mean/sd in percent; ddof=1 sample SD (recorded in run config)."""
    a = np.asarray(vals, dtype=np.float64)
    return {"per_seed": [round(v, 4) for v in vals],
            "mean": round(float(a.mean()), 4),
            "sd": round(float(a.std(ddof=ddof)), 4), "ddof": ddof}


# ---------------------------------------------------------------------------
# WT/FIX cross-check against the fix-benchmark store (report, never overwrite)
# ---------------------------------------------------------------------------

def _store_mean(entry: Any) -> float | None:
    """Tolerant read of a fix-benchmark store entry (schema G01-pending)."""
    if not isinstance(entry, dict):
        return None
    for path in (("metrics", "final_acc", "mean"), ("metrics", "final_acc_mean"),
                 ("metrics", "final_acc"), ("final_acc",), ("mean",)):
        cur: Any = entry
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            try:
                return float(cur)
            except (TypeError, ValueError):
                pass
    for k in ("per_seed", "seeds_final_acc", "final_acc_per_seed"):
        if isinstance(entry.get(k), list) and entry[k]:
            try:
                return float(np.mean(entry[k]))
            except (TypeError, ValueError):
                pass
    return None


def cross_check_wt_fix(rows: list[dict], store_root: Path,
                       tol: float = 0.05) -> dict:
    """Compare emitted WT/FIX facts against B/fix_benchmark store entries.

    Entries are read-only. Deltas are reported; abs delta > tol -> GATE-FAIL.
    """
    report: dict[str, Any] = {"store_root": str(store_root), "entries": {},
                              "gate_fail": [], "checked": 0}
    if not store_root.is_dir():
        report["note"] = "no fix-benchmark store present; cross-check SKIPPED"
        return report
    for r in rows:
        for arm, canon in (("wt", r["wt"]["mean"]), ("fix", r["fix"]["mean"])):
            base = (store_root / "B" / "fix_benchmark" / _norm(r["backbone"])
                    / _norm(r["dataset"]))
            candidates = [base / f"{arm}.json", base / f"{arm}_{r['backbone']}.json"]
            p = next((c for c in candidates if c.is_file()), None)
            if p is None:
                continue
            try:
                entry = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                report["entries"][p.name] = {"error": f"unreadable ({exc})"}
                continue
            val = _store_mean(entry)
            report["checked"] += 1
            if val is None:
                report["entries"][str(p)] = {"error": "no recognizable "
                                            "final_acc field (schema pending G01)"}
                continue
            delta = round(val - canon, 4)
            rec = {"stored": val, "canonical_tex": canon, "delta": delta}
            if abs(delta) > tol:
                rec["verdict"] = "GATE-FAIL"
                report["gate_fail"].append(
                    f"{r['backbone']}/{r['dataset']}/{arm}: stored {val} vs "
                    f"tex {canon} (delta {delta:+.2f})")
            else:
                rec["verdict"] = "OK"
            report["entries"][str(p)] = rec
    return report


# ---------------------------------------------------------------------------
# LaTeX emission (tab:lda-ncm-comparison layout, G08 S4/S5)
# ---------------------------------------------------------------------------

def fmt_cell(cell: dict | None) -> str:
    if cell is None:
        return "--"
    return f"${cell['mean']:.1f} \\pm {cell['sd']:.1f}$"


def emit_latex(rows: list[dict], provenance: str) -> str:
    head = [
        f"% {TABLE_LABEL} -- generated by experiments/lda_ncm_eval.py",
        f"% provenance: {provenance}; population: 10 benchmark rows, "
        "5 random class orderings (seeds 0--4)",
        "% WT/FIX columns are byte-copies of tab:fix-benchmark (owner facts);"
        " LDA/NCM are closed-form from cached features",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Backbone & Dataset & WT $\bar A$ & FIX $\bar A$ (ours) & "
        r"LDA $\bar A$ & NCM $\bar A$ \\",
        r"\midrule",
    ]
    body = []
    for r in rows:
        line = (f"{r['backbone']:<14} & {r['dataset']:<12} & "
                f"${r['wt']['mean']:.1f} \\pm {r['wt']['sd']:.1f}$ & "
                f"$\\mathbf{{{r['fix']['mean']:.1f} \\pm {r['fix']['sd']:.1f}}}$ & "
                f"{fmt_cell(r['lda'])} & {fmt_cell(r['ncm'])} \\\\")
        body.append(line)
        if r["lda"] is None or r["ncm"] is None:
            body.append(f"% MISSING: {r['missing_reason']}")
        if r["stress_test"]:
            body.append(f"% stress test: {STRESS_TAG}")
    tail = [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(head + body + tail) + "\n"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_pipeline(features_root: Path, orderings_path: Path,
                 store_root: Path, *, mode: str, sd_ddof: int = 1
                 ) -> dict:
    """Walk features, evaluate, aggregate, cross-check; returns the run record."""
    orderings = None
    if orderings_path.is_file():
        try:
            orderings = json.loads(orderings_path.read_text(encoding="utf-8"))
        except Exception as exc:
            orderings = None
            print(f"[warn] orderings unreadable ({exc}); proceeding without "
                  "ordering cross-check")
    found, unmatched = walk_features(features_root)
    rows = []
    for row in CANONICAL_ROWS:
        rows.append(evaluate_row(row, found.get((row[0], row[1]), {}),
                                 orderings))
    complete = [r for r in rows if r["lda"] is not None]
    n_cells = 2 * len(complete)
    missing_cells = 20 - n_cells
    escalation = missing_cells > 10  # majority of the 20 cells MISSING (N24)
    xcheck = cross_check_wt_fix(rows, store_root)
    provenance = "synthetic" if mode == "demo" else "owner-artifact"
    tex = emit_latex(rows, provenance)

    print(f"\n== {TABLE_LABEL} ({mode} mode, provenance {provenance}) ==")
    print(f"features root: {features_root}")
    if unmatched:
        print(f"[note] unmatched feature dirs (not in the 10-row roster): "
              f"{unmatched}")
    hdr = (f"{'backbone':<14} {'dataset':<12} {'WT':>10} {'FIX':>10} "
           f"{'LDA':>12} {'NCM':>12}")
    print(hdr)
    for r in rows:
        wt = f"{r['wt']['mean']:.1f}+/-{r['wt']['sd']:.1f}"
        fx = f"{r['fix']['mean']:.1f}+/-{r['fix']['sd']:.1f}"
        lda = (f"{r['lda']['mean']:.1f}+/-{r['lda']['sd']:.1f}"
               if r["lda"] else "MISSING")
        ncm = (f"{r['ncm']['mean']:.1f}+/-{r['ncm']['sd']:.1f}"
               if r["ncm"] else "MISSING")
        print(f"{r['backbone']:<14} {r['dataset']:<12} {wt:>10} {fx:>10} "
              f"{lda:>12} {ncm:>12}")
        if r["missing_reason"]:
            print(f"    MISSING reason: {r['missing_reason']}")
    print(f"cells complete: {n_cells}/20; MISSING: {missing_cells}")
    if escalation:
        print("ESCALATION: majority of the 20 cells MISSING -> variant B "
              "per N24 (never estimate)")
    if xcheck.get("note"):
        print(f"cross-check: {xcheck['note']}")
    for e in xcheck["entries"].values():
        if isinstance(e, dict) and "delta" in e:
            print(f"cross-check {e.get('verdict')}: stored {e['stored']} vs "
                  f"tex {e['canonical_tex']} (delta {e['delta']:+.2f})")
    for g in xcheck["gate_fail"]:
        print(f"GATE-FAIL {g}")
    print("\n-- LaTeX fragment --")
    print(tex, end="")

    return {
        "paper": "B", "experiment": "lda_ncm", "mode": mode,
        "provenance": provenance,
        "rows": rows, "cells_complete": n_cells, "cells_missing": missing_cells,
        "variant_b_escalation": escalation, "cross_check": xcheck,
        "unmatched_feature_dirs": unmatched,
        "sd_ddof": sd_ddof,
        "covariance": (
            "Ledoit-Wolf (analytical 2004) shrunk within-class covariance per "
            "task, equal-weight pooled across tasks -- DELEGATED to "
            "core/lda_ncm.py (H03) bridge exports"
            if _CORE_STATUS == "core-lda_ncm" else
            "Ledoit-Wolf (analytical 2004, sklearn lead / numpy fallback), "
            "per task, equal-weight pooled; core path: "
            f"{_CORE_STATUS} (labeled fallback produced these numbers)"),
        "tex_fragment": tex, "caption_tags": CAPTION_TAGS,
        "derived_tables": [TABLE_LABEL],
    }


def _env_block() -> dict:
    env: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__, "platform": platform.platform(), "gpu": "cpu",
    }
    try:
        import sklearn  # noqa: PLC0415
        env["scikit-learn"] = sklearn.__version__
    except Exception:
        env["scikit-learn"] = "absent (numpy LW fallback path)"
    return env


# ---------------------------------------------------------------------------
# Demo: synthetic features via tests/conftest.GaussianTask, full pipeline
# ---------------------------------------------------------------------------

DEMO_ROSTER = [0, 6, 9]          # canonical rows: DINOv2/CIFAR-100,
#                                # ResNet-50/ImageNet-100 (seeds 0-2 only ->
#                                # MISSING demo), CLIP/DomainNet (stress tag)
DEMO_D, DEMO_T, DEMO_C = 24, 4, 5
DEMO_N_TRAIN, DEMO_N_TEST = 80, 30
DEMO_SEPARATION = 2.0


def _build_demo_store(tmp: Path) -> tuple[Path, Path, Path]:
    """Synthetic features tree + orderings.json + a tiny fix-benchmark store."""
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    import conftest  # noqa: PLC0415  (G10 factory; plain-module importable)

    feat_root = tmp / "features"
    order_path = tmp / "orderings.json"
    store_root = tmp / "runs"
    orderings: dict[str, dict[str, list[list[int]]]] = {}
    for idx in DEMO_ROSTER:
        backbone, dataset, *_ = CANONICAL_ROWS[idx]
        d_dir = feat_root / backbone.replace("/", "-") / dataset
        d_dir.mkdir(parents=True)
        n_classes = DEMO_T * DEMO_C
        seeds = SEEDS if idx != 6 else (0, 1, 2)   # partial row -> MISSING
        parts_all: dict[str, list[list[int]]] = {}
        for seed in seeds:
            task = conftest.make_gaussian_task(
                d=DEMO_D, n_classes=n_classes, separation=DEMO_SEPARATION,
                n_per_class=DEMO_N_TRAIN + DEMO_N_TEST,
                seed=conftest.derive_seed(f"demo-{_norm(backbone)}-"
                                          f"{_norm(dataset)}-{seed}"))
            rng = np.random.default_rng(
                conftest.derive_seed(f"demo-order-{_norm(dataset)}-{seed}"))
            perm = rng.permutation(n_classes)
            parts = [perm[t * DEMO_C:(t + 1) * DEMO_C].tolist()
                     for t in range(DEMO_T)]
            parts_all[str(seed)] = parts
            cls_task = np.empty(n_classes, dtype=np.int64)
            for t, cls in enumerate(parts):
                cls_task[np.asarray(cls)] = t
            per = DEMO_N_TRAIN + DEMO_N_TEST
            idx_by_class = {c: np.where(task.y == c)[0] for c in range(n_classes)}
            tr_idx = np.concatenate([idx_by_class[c][:DEMO_N_TRAIN]
                                     for c in range(n_classes)])
            te_idx = np.concatenate([idx_by_class[c][DEMO_N_TRAIN:]
                                     for c in range(n_classes)])
            np.savez(
                d_dir / f"seed{seed}.npz",
                train_x=task.X[tr_idx].astype(np.float32),
                train_y=task.y[tr_idx], train_task=cls_task[task.y[tr_idx]],
                test_x=task.X[te_idx].astype(np.float32),
                test_y=task.y[te_idx],
                backbone=backbone, dataset=dataset, seed=seed)
        orderings[dataset] = parts_all
    order_path.write_text(json.dumps(orderings, indent=1), encoding="utf-8")
    # Tiny synthetic fix-benchmark entry with a deliberate delta, to prove the
    # cross-check reports GATE-FAIL without overwriting anything.
    fb = store_root / "B" / "fix_benchmark" / _norm(CANONICAL_ROWS[0][0]) \
        / _norm(CANONICAL_ROWS[0][1])
    fb.mkdir(parents=True)
    (fb / "wt.json").write_text(json.dumps({
        "run_id": "demo_B_fix_benchmark_placeholder", "paper": "B",
        "experiment": "fix_benchmark", "provenance": "synthetic",
        "config": {"backbone": CANONICAL_ROWS[0][0],
                   "dataset": CANONICAL_ROWS[0][1], "arm": "wt", "seed": None},
        "metrics": {"final_acc": {"mean": 82.9, "sd": 0.4}}}, indent=1),
        encoding="utf-8")
    return feat_root, order_path, store_root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Paper B tab:lda-ncm-comparison generator "
                    "(closed-form LDA/NCM from cached features)")
    ap.add_argument("--features-root", type=Path,
                    default=REPO_ROOT / "data" / "B" / "features")
    ap.add_argument("--orderings", type=Path,
                    default=REPO_ROOT / "data" / "B" / "orderings.json")
    ap.add_argument("--store-root", type=Path, default=REPO_ROOT / "results" / "runs")
    ap.add_argument("--tex-out-dir", type=Path,
                    default=REPO_ROOT / "tables" / "out" / "B")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic GaussianTask features, full pipeline, "
                         "single labeled demo JSON")
    ap.add_argument("--demo-json", type=Path,
                    default=REPO_ROOT / "results" / "runs" / "B" / "lda_ncm"
                    / "B_lda_ncm_demo.json")
    ap.add_argument("--sd-ddof", type=int, default=1,
                    help="SD convention over the 5 seeds (default 1 = sample)")
    ap.add_argument("--no-write", action="store_true",
                    help="evaluate and print only; write nothing")
    args = ap.parse_args(argv)

    if args.demo:
        with tempfile.TemporaryDirectory(prefix="lda_ncm_demo-") as td:
            tmp = Path(td)
            feat_root, order_path, store_root = _build_demo_store(tmp)
            record = run_pipeline(feat_root, order_path, store_root,
                                  mode="demo", sd_ddof=args.sd_ddof)
            record["demo"] = {
                "features_via": "tests/conftest.make_gaussian_task "
                                f"(GaussianTask, d={DEMO_D}, T={DEMO_T}, "
                                f"C={DEMO_C}, n_train={DEMO_N_TRAIN}/class, "
                                f"n_test={DEMO_N_TEST}/class, "
                                f"separation={DEMO_SEPARATION})",
                "roster_rows": [CANONICAL_ROWS[i][:2] for i in DEMO_ROSTER],
                "partial_row": list(CANONICAL_ROWS[6][:2]) + ["seeds 0-2 only"],
                "fix_benchmark_entry": "synthetic, delta deliberately +0.6 to "
                                       "prove GATE-FAIL reporting",
            }
            out = args.demo_json
            if not args.no_write:
                out.parent.mkdir(parents=True, exist_ok=True)
                payload = dict(record)
                payload.update({
                    "run_id": "B_lda_ncm_demo",
                    "created_utc": _utcnow(),
                    "executed_utc": _utcnow(),
                    "seeds": list(SEEDS),
                    "env": _env_block(),
                    "code_version": _sha256(Path(__file__).resolve()),
                    "inputs": {"synthetic": "generated in-run; see demo block"},
                })
                out.write_text(json.dumps(payload, indent=1, sort_keys=True),
                               encoding="utf-8")
                print(f"\ndemo JSON written: {out}")
            else:
                print("\n--no-write: nothing written")
            return 0

    record = run_pipeline(args.features_root, args.orderings, args.store_root,
                          mode="real", sd_ddof=args.sd_ddof)
    if record["cells_complete"] == 0:
        print("\nNo complete rows: nothing written (POST-HOC with an absent "
              "feature store exits 1 per exec/G08_log.md S4/S6).")
        return 1
    if not args.no_write:
        # Per-row run JSONs (G08 S2 one-per-run), tex fragment + caption tags
        # (S5), and a summary meta file. Only executed when real features exist.
        now = _utcnow()
        env = _env_block()
        code_sha = _sha256(Path(__file__).resolve())
        for r in record["rows"]:
            if r["lda"] is None:
                continue
            slug_b, slug_d = _norm(r["backbone"]), _norm(r["dataset"])
            run_id = f"B_lda_ncm_{slug_b}_{slug_d}"
            p = args.store_root / "B" / "lda_ncm" / f"{run_id}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "run_id": run_id, "paper": "B", "experiment": "lda_ncm",
                "provenance": "owner-artifact",  # features = owner/colab runs
                "created_utc": now, "executed_utc": None,
                "config": {"backbone": r["backbone"], "dataset": r["dataset"],
                           "seeds": list(SEEDS), "sd_ddof": args.sd_ddof,
                           "method": record["covariance"]},
                "seeds": list(SEEDS),
                "metrics": {"lda_final_acc": r["lda"], "ncm_final_acc": r["ncm"],
                            "wt_final_acc_tex": r["wt"], "fix_final_acc_tex": r["fix"]},
                "env": env, "code_version": code_sha,
                "inputs": {"features_sha256": r["inputs_sha256"]},
                "derived_tables": [TABLE_LABEL],
            }, indent=1, sort_keys=True), encoding="utf-8")
            print(f"run JSON written: {p}")
        args.tex_out_dir.mkdir(parents=True, exist_ok=True)
        (args.tex_out_dir / "lda-ncm-comparison.tex").write_text(
            record["tex_fragment"], encoding="utf-8")
        (args.tex_out_dir / "lda-ncm-comparison.caption_tags.txt").write_text(
            "\n".join(CAPTION_TAGS) + "\n", encoding="utf-8")
        print(f"tex fragment + caption tags written under {args.tex_out_dir}")
    rc = 2 if record["cross_check"]["gate_fail"] else 0
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
