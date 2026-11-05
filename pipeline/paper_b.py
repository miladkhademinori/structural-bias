"""pipeline.paper_b -- every real-backbone number of Paper B (12500) from feature caches.

  benchmark        tab:fix-benchmark and tab:lda-ncm-comparison: WT, FIX, J, LDA, NCM final average
                   accuracy over 5 orderings; gap recovery (fig:gap-recovery)
  bias direction   tab:bias-direction: per-pair cosine / relative magnitude / cross-pair CV in the
                   Sigma^{-1} inner product (app:experiments metric iii).  The paper's C^2-averaged
                   statistic is reported beside the per-pair one, labelled as the gauge identity it is
                   (core.diagnostic, pipeline.stats.ClassStats.bias_direction)
  rate validation  tab:rate-fits / fig:rate-validation: (gamma_ij, |P_ij|) per ordered task pair on the
                   WT head, per-backbone OLS of log |P_ij| on gamma_ij with R^2 and bootstrap CI; the
                   Theorem 6.2 reference slope -(kappa-1)^2/8 at the backbone's median kappa
  kappa            Assumption 6.1 diagnostics: per-pair kappa distribution, mean, fraction > 1.5, fraction < 1
  weights-only     app:weights-only: WT trained with decay on W only; FIX unchanged; the estimated-scalar
                   variant (Def 7.4) with c_hat from a held-out split
  five-method      tab:diagnostic needs the published methods' checkpoints; ``diagnostic_on_checkpoint``
                   consumes one (W, b) head plus the features it was trained on and produces the row

One pass per (backbone, dataset, seed) trains the WT and J heads once and derives every quantity
above from them (the benchmark, the bias direction and the rate pairs all read the same WT head).

DomainNet: domains are tasks with a shared 345-class label space.  Within-task heads are trained per
domain over all classes; de-centering uses per-domain centroids; gamma_ij is between domain centroids;
accuracy maps the (domain, class) prediction back to the class.  No rate pairs (the theory does not cover it).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np

from pipeline import backend as _backend
from pipeline import registry as R
from pipeline.features import cache_exists, load_pair
from pipeline.heads import Head, de_center, estimate_c_hat, lda_head, ncm_head, train_joint, train_within_task
from pipeline.stats import ClassStats
from pipeline.paper_a import _agg, _ms, _ols


def _split(dataset: str, seed: int, ytr, dtr, yte, dte):
    """Task structure for one ordering. Returns (ytr_v, yte_v, task_of_class, info); for domains-as-tasks the
    labels are virtual classes v = domain * N + class."""
    ds = R.DATASETS[dataset]
    if ds.domains_as_tasks:
        nd = len(ds.domains); N = ds.n_classes
        toc = np.repeat(np.arange(nd), N)
        return dtr * N + ytr, dte * N + yte, toc, dict(virtual=True, N=N, nd=nd)
    toc = R.task_of_class(dataset, R.class_ordering(dataset, seed))
    return ytr, yte, toc, dict(virtual=False, N=ds.n_classes)


def _acc(head: Head, X, ytrue, info):
    if not info["virtual"]:
        return head.accuracy(X, ytrue)
    return float(np.mean(head.predict(X) % info["N"] == np.asarray(ytrue)))


def _task_diagnostics(head: Head, X, yv, toc) -> dict:
    """Where a head loses accuracy: task_acc = fraction of test samples routed to the right task;
    til_acc = accuracy under an oracle task mask (within-task discrimination only).  Streamed over rows."""
    yv = np.asarray(yv, np.int64); task_ok = 0; til_ok = 0
    for s, L in head._blocks(X):
        yb = yv[s:s + len(L)]
        task_ok += int(np.sum(toc[np.argmax(L, axis=1)] == toc[yb]))
        masked = np.where(toc[None, :] == toc[yb][:, None], L, -np.inf)
        til_ok += int(np.sum(np.argmax(masked, axis=1) == yb))
    return dict(task_acc=task_ok / len(yv), til_acc=til_ok / len(yv))


def seed_pass(backbone: str, dataset: str, seed: int, Xtr, ytr, dtr, Xte, yte, dte, *, optimizer="adamw", lam=1e-4,
              protocol=R.HEAD_PROTOCOL["B"], decay_bias=True, device=None, with_rate_pairs=True) -> dict:
    ytr_v, yte_v, toc, info = _split(dataset, seed, ytr, dtr, yte, dte)
    st = ClassStats.from_features(Xtr, ytr_v, toc, protocol="B", n_classes=toc.size)
    tag = f"B:{backbone}:{dataset}:s{seed}"
    wt = train_within_task(Xtr, ytr_v, toc, lam=lam, optimizer=optimizer, protocol=protocol, seed_purpose=f"{tag}:wt", decay_bias=decay_bias, device=device)
    J = train_joint(Xtr, ytr_v, toc, lam=lam, optimizer=optimizer, protocol=protocol, seed_purpose=f"{tag}:j", decay_bias=decay_bias, device=device)
    fix = de_center(wt, st)
    scales = st.head_scales(wt.W); fix_c = de_center(wt, st, scale=scales)
    a_wt, a_fix, a_j = _acc(wt, Xte, yte, info), _acc(fix, Xte, yte, info), _acc(J, Xte, yte, info)
    a_fix_c = _acc(fix_c, Xte, yte, info)
    a_lda, a_ncm = _acc(lda_head(st), Xte, yte, info), _acc(ncm_head(st), Xte, yte, info)
    bd = st.bias_direction_all(wt.W)
    diag = {f"{name}_{k}": v for name, h in (("WT", wt), ("FIX", fix), ("FIX_c", fix_c), ("J", J)) for k, v in _task_diagnostics(h, Xte, yte_v, toc).items()}
    row = dict(seed=seed, WT=a_wt, FIX=a_fix, FIX_c=a_fix_c, J=a_j, LDA=a_lda, NCM=a_ncm, gap=a_j - a_wt,
               recovery=(a_fix - a_wt) / (a_j - a_wt) if abs(a_j - a_wt) > 1e-9 else float("nan"),
               recovery_c=(a_fix_c - a_wt) / (a_j - a_wt) if abs(a_j - a_wt) > 1e-9 else float("nan"),
               gamma_min=st.gamma_min()[0], lam_lw=st.lam_lw, bias_direction=bd, head_scale=float(np.nanmean(scales)),
               head_scales=[float(c) for c in scales],
               ce_wt=wt.ce_loss(Xte, yte_v), ce_fix=fix.ce_loss(Xte, yte_v), ce_j=J.ce_loss(Xte, yte_v),
               fit_wt=[dict(max_grad=i.get("max_grad"), steps=i.get("steps", i.get("nit"))) for i in wt.info.get("blocks", [])],
               fit_j=dict(max_grad=J.info.get("fit", {}).get("max_grad"), steps=J.info.get("fit", {}).get("steps", J.info.get("fit", {}).get("nit"))),
               **diag)
    if not info["virtual"]:
        row["kappa_min"] = st.kappa_min()
        if with_rate_pairs:
            G, D = st.gamma_matrix(), st.D_matrix(); pairs = []
            for i in range(st.T):
                for j in range(st.T):
                    if i == j:
                        continue
                    ki, kj = np.flatnonzero(toc == i), np.flatnonzero(toc == j)
                    blk = D[np.ix_(ki, kj)] / G[i, j]
                    pairs.append(dict(backbone=backbone, dataset=dataset, seed=seed, i=i, j=j, gamma=float(G[i, j]),
                                      P=wt.task_confusion(Xte, yte, i, j), P_fix=fix.task_confusion(Xte, yte, i, j),
                                      kappa_min=float(np.sqrt(blk.min())), kappa_mean=float(np.sqrt(blk).mean())))
            row["rate_pairs"] = pairs
    return row


def rate_fits(pairs: List[dict], n_bootstrap=2000) -> dict:
    """Per backbone: OLS of log |P_ij| on gamma_ij over the pairs with |P_ij| > 0, R^2, bootstrap CI (pairs resampled),
    and the Theorem 6.2 reference slope -(kappa_hat - 1)^2 / 8 at the median kappa_min of the backbone's pairs."""
    out = {}
    rg = R.rng("rate-boot")
    for b in sorted({p["backbone"] for p in pairs}):
        P = [p for p in pairs if p["backbone"] == b and p["P"] > 0]
        if len(P) < 3:
            continue
        x = np.array([p["gamma"] for p in P]); y = np.log(np.array([p["P"] for p in P]))
        fit = _ols(x, y); boots = []
        for _ in range(n_bootstrap):
            idx = rg.integers(0, len(P), len(P)); boots.append(_ols(x[idx], y[idx])["slope"])
        fit["ci95"] = [float(v) for v in np.percentile(boots, [2.5, 97.5])]
        kap = float(np.median([p["kappa_min"] for p in P]))
        pred = float(-(kap - 1) ** 2 / 8) if kap > 1 else None
        fit.update(n_pairs=len(P), n_pairs_zero=len([p for p in pairs if p["backbone"] == b and p["P"] <= 0]),
                   kappa_median=kap, predicted_slope=pred,
                   slope_error_pct=(abs(fit["slope"] - pred) / abs(pred) * 100 if pred else None),
                   datasets=sorted({p["dataset"] for p in P}))
        out[b] = fit
    return out


def kappa_summary(kappas: Sequence[float], thresh=1.5) -> dict:
    ks = np.asarray(list(kappas), float)
    if ks.size == 0:
        return dict(n=0)
    return dict(mean=float(ks.mean()), frac_gt=float(np.mean(ks > thresh)), frac_lt1=float(np.mean(ks < 1)),
                quantiles={str(q): float(np.quantile(ks, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}, n=int(ks.size))


def weights_only_control(backbone: str, dataset: str, root: Path, seeds=R.SEEDS, optimizer="adamw",
                         protocol=R.HEAD_PROTOCOL["B"], lam=1e-4, val_frac=0.1, device=None) -> dict:
    """app:weights-only: WT trained with decay on W only (the bias gauge floats), FIX by Definition 7.1 unchanged,
    J, the retention of the FIX-WT gain relative to the full-decay run on the same data, and the Definition 7.4
    variant with two estimators of the per-task scalar c_hat, evaluated on the same heads:

      heldout      the printed Definition 7.4: c_hat from a held-out split (val_frac of the training rows, never
                   used for the heads or the statistics), aligning each task's mean own-task maximum logit
      closed_form  the split-free estimator c_hat_t = mean_r [b_r - S_bar_t/2 + mu_r^T Sigma^{-1} mu_r / 2]
                   (heads.estimate_c_hat), which recovers the floating bias-gauge constant exactly in population

    Also reports the per-task bias sums of the weights-only head (the floating gauge) and the gauge residuals of
    both heads (the screen of docs/REPLICATION.md section 4.1)."""
    (Xtr, ytr, dtr), (Xte, yte, dte), meta = load_pair(backbone, dataset, root)
    rows = []
    for s in seeds:
        ytr_v, yte_v, toc, info = _split(dataset, s, ytr, dtr, yte, dte)
        g = R.rng(f"wo-val:{backbone}:{dataset}:{s}"); perm = g.permutation(len(ytr)); nv = int(val_frac * len(ytr))
        va, tr = perm[:nv], perm[nv:]
        st = ClassStats.from_features(Xtr[tr], ytr[tr], toc, protocol="B")
        tag = f"wo:{backbone}:{dataset}:s{s}"
        wt_full = train_within_task(Xtr[tr], ytr[tr], toc, lam=lam, optimizer=optimizer, protocol=protocol, seed_purpose=f"{tag}:full", decay_bias=True, device=device)
        wt_wo = train_within_task(Xtr[tr], ytr[tr], toc, lam=lam, optimizer=optimizer, protocol=protocol, seed_purpose=f"{tag}:wo", decay_bias=False, device=device)
        J = train_joint(Xtr[tr], ytr[tr], toc, lam=lam, optimizer=optimizer, protocol=protocol, seed_purpose=f"{tag}:j", device=device)
        fix_full, fix_wo = de_center(wt_full, st), de_center(wt_wo, st)
        # Definition 7.4 as printed: held-out logit-scale alignment
        fix_h = fix_wo.copy(); L = fix_h.logits(Xtr[va]); yv = ytr[va]
        task_mean = np.array([L[toc[yv] == t][:, toc == t].max(1).mean() for t in range(st.T)])
        c_held = task_mean - task_mean.mean(); fix_h.b = fix_h.b - c_held[toc]
        # split-free closed form (centred over tasks, since only differences affect decisions)
        c_closed = estimate_c_hat(wt_wo, st, center=True); c_closed_full = estimate_c_hat(wt_full, st, center=True)
        fix_c = fix_wo.copy(); fix_c.b = fix_c.b - c_closed[toc]
        rows.append(dict(seed=s, WT_full=wt_full.accuracy(Xte, yte), FIX_full=fix_full.accuracy(Xte, yte),
                         WT_wo=wt_wo.accuracy(Xte, yte), FIX_wo=fix_wo.accuracy(Xte, yte),
                         FIX_wo_chat=fix_h.accuracy(Xte, yte), FIX_wo_chat_closed=fix_c.accuracy(Xte, yte), J=J.accuracy(Xte, yte),
                         c_hat_heldout=[float(c) for c in c_held], c_hat_closed=[float(c) for c in c_closed],
                         c_hat_closed_full_decay=[float(c) for c in c_closed_full],
                         gauge_full=[float(x) for x in st.gauge_residuals(wt_full.W)], gauge_wo=[float(x) for x in st.gauge_residuals(wt_wo.W)],
                         bias_sum_wo=[float(wt_wo.b[toc == t].sum()) for t in range(st.T)], bias_sum_full=[float(wt_full.b[toc == t].sum()) for t in range(st.T)]))
    keys = ["WT_full", "FIX_full", "WT_wo", "FIX_wo", "FIX_wo_chat", "FIX_wo_chat_closed", "J"]
    out = dict(backbone=backbone, dataset=dataset, provenance=meta.get("provenance", "real"), val_frac=val_frac,
               c_hat_estimators=dict(heldout="Definition 7.4 as printed: held-out split, mean own-task max-logit alignment",
                                     closed_form="split-free: mean_r [b_r - S_bar_t/2 + mu_r^T Sigma^-1 mu_r / 2], centred over tasks"),
               **_agg(rows, keys), per_seed=rows)
    out["gain_full"] = _ms([r["FIX_full"] - r["WT_full"] for r in rows]); out["gain_wo"] = _ms([r["FIX_wo"] - r["WT_wo"] for r in rows])
    out["chat_increment"] = _ms([r["FIX_wo_chat"] - r["FIX_wo"] for r in rows])
    out["chat_closed_increment"] = _ms([r["FIX_wo_chat_closed"] - r["FIX_wo"] for r in rows])
    out["retention"] = (out["gain_wo"]["mean"] / out["gain_full"]["mean"]) if out["gain_full"]["mean"] else None
    return out


def diagnostic_on_checkpoint(W: np.ndarray, b: np.ndarray, Xtr, ytr, Xte, yte, task_of_class, training_config: dict = None) -> dict:
    """tab:diagnostic row for an external method's linear head (W [N, d], b [N]) evaluated on the backbone
    features the method used, preceded by the gauge screen:

      gauge_residuals   per task, ||sum_r w_r|| / mean ||w_r||: zero for any head at the softmax gauge, where the
                        block-mean cosine equals 1 identically and carries no information about the head
      decay / optimizer read off ``training_config`` when given (keys 'weight_decay_on', 'optimizer', 'epochs'),
                        recorded beside the numbers, never inferred

    then the per-pair statistics (cosine mean and min, relative magnitude, CV) beside the block-mean identity,
    the kappa statistics, and the raw and de-centered accuracies.  A head with gauge residuals near 1e-14 and a
    block-mean cosine below 1 is a contradiction and is flagged."""
    st = ClassStats.from_features(Xtr, ytr, task_of_class, protocol="B")
    head = Head(np.asarray(W, float), np.asarray(b, float), np.asarray(task_of_class))
    gauge = st.gauge_residuals(head.W)
    bd = st.bias_direction_all(head.W)
    fix = de_center(head, st)
    ks = kappa_summary([p["kappa"] for p in st.kappa_pairs()])
    at_gauge = bool(gauge.max() < 1e-6)
    return dict(gauge_residuals=[float(g) for g in gauge], at_gauge=at_gauge,
                gauge_contradiction=bool(at_gauge and bd["paper_averaged_cos"] < 0.999),
                training_config=dict(training_config or {}), head_scale=float(np.nanmean(st.head_scales(head.W))),
                cos_pair_mean=bd["pair_cos_mean"], cos_pair_min=bd["pair_cos_min"], rel_pair_mean=bd["pair_rel_mean"], cv=bd["cv"],
                cos_block_mean=bd["paper_averaged_cos"], rel_block_mean=bd["paper_averaged_rel"],
                kappa_mean=ks.get("mean"), kappa_frac_gt_1_5=ks.get("frac_gt"), acc_raw=head.accuracy(Xte, yte), acc_fix=fix.accuracy(Xte, yte))


BD_KEYS = ("pair_cos_mean", "pair_cos_min", "pair_rel_mean", "cv", "paper_averaged_cos", "paper_averaged_rel", "gauge_residual")


def run_all(root: Path, out_dir: Path, seeds=R.SEEDS, optimizer="adamw", rows=R.PAPER_B_BENCHMARK_ROWS,
            all_backbones=R.PAPER_B_BACKBONES, all_datasets=R.PAPER_B_DATASETS, weights_only_row=("clip_b16", "cifar100"),
            device=None, protocol=None, log=print) -> dict:
    """Drive every Paper B measurement over the available caches.  ``protocol`` overrides the AdamW head protocol
    (R.HEAD_PROTOCOL["B"]); the selftest shortens its epochs, the real run leaves it alone."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol or R.HEAD_PROTOCOL["B"]
    dev = _backend.describe(device); t0 = time.time()
    wanted = list(dict.fromkeys(list(rows) + [(b, d) for b in all_backbones for d in all_datasets]))
    cfgs = [(b, d) for (b, d) in wanted if cache_exists(b, d, root)]
    missing = [(b, d) for (b, d) in wanted if not cache_exists(b, d, root)]
    log(f"[B] backend {dev}; {len(cfgs)} configurations, {len(missing)} missing: {missing}")
    bench, pairs = {}, []
    for (b, d) in cfgs:
        (Xtr, ytr, dtr), (Xte, yte, dte), meta = load_pair(b, d, root)
        rws = []
        for s in seeds:
            r = seed_pass(b, d, s, Xtr, ytr, dtr, Xte, yte, dte, optimizer=optimizer, protocol=protocol, device=device)
            pairs += r.pop("rate_pairs", [])
            rws.append(r)
            log(f"[B] {b}/{d} s{s}: WT {r['WT']:.4f} FIX {r['FIX']:.4f} FIX_c {r['FIX_c']:.4f} J {r['J']:.4f} LDA {r['LDA']:.4f} scale {r['head_scale']:.2f} "
                f"pair-cos {r['bias_direction']['pair_cos_mean']:.3f} (paper-avg {r['bias_direction']['paper_averaged_cos']:.4f}) ({time.time()-t0:.0f}s)")
        base = dict(backbone=b, dataset=d, provenance=meta.get("provenance", "real"), optimizer=optimizer, backend=dev["backend"],
                    in_fix_benchmark=(b, d) in rows)
        bench[(b, d)] = dict(base, **_agg(rws, ["WT", "FIX", "FIX_c", "J", "LDA", "NCM", "gap", "recovery", "recovery_c", "gamma_min", "ce_wt", "ce_fix", "ce_j",
                                               "head_scale", "WT_task_acc", "WT_til_acc", "FIX_task_acc", "FIX_til_acc", "FIX_c_task_acc", "J_task_acc", "J_til_acc"]),
                             **{k: _ms([r["bias_direction"][k] for r in rws]) for k in BD_KEYS}, per_seed=rws)
    fits = rate_fits(pairs) if pairs else {}
    kappa = {b: kappa_summary([p["kappa_min"] for p in pairs if p["backbone"] == b]) for b in sorted({p["backbone"] for p in pairs})}
    kappa_all_pairs = {b: kappa_summary([p["kappa_mean"] for p in pairs if p["backbone"] == b]) for b in kappa}
    wo = weights_only_control(*weights_only_row, root, seeds, optimizer, protocol=protocol, device=device) if cache_exists(*weights_only_row, root) else None
    if wo:
        log(f"[B] weights-only {weights_only_row}: gain full {wo['gain_full']['mean']:.4f} wo {wo['gain_wo']['mean']:.4f} retention {wo['retention']}")

    def keyed(dct): return {f"{k[0]}|{k[1]}": v for k, v in dct.items()}
    payload = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), optimizer=optimizer, protocol=protocol, seeds=list(seeds), backend=dev,
                   configs=[f"{b}|{d}" for b, d in cfgs], missing=[f"{b}|{d}" for b, d in missing],
                   benchmark=keyed(bench), bias_direction={f"{b}|{d}": {k: bench[(b, d)][k] for k in BD_KEYS} for (b, d) in cfgs},
                   rate_pairs=pairs, rate_fits=fits, kappa=kappa, kappa_all_pairs=kappa_all_pairs, weights_only=wo,
                   provenance=sorted({v["provenance"] for v in bench.values()}))
    (out_dir / "paperB_results.json").write_text(json.dumps(payload, indent=1, default=float))
    log(f"[B] wrote {out_dir / 'paperB_results.json'} in {time.time()-t0:.0f}s")
    return payload
