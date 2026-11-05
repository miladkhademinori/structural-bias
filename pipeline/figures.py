"""pipeline.figures -- every figure regenerated from the result JSONs.

Rule (docs/REPLICATION.md section 5): no figure curve without a generating artifact.  Each figure here is
drawn from a results file this repository writes, and each one is accompanied by a sidecar ``*.data.json``
holding the exact arrays plotted, the source file and its sha256, so a reader can check a curve against the
numbers without re-running anything.

  python replicate.py figures [--results results] [--out results/figures] [--drift results/paperA_drift.json]

Paper A
  A_fig_gamma_tc        fig:gamma-tc: log TC gap against gamma_min, one point per configuration (mean over
                        orderings; faint per-ordering points), per-backbone OLS lines and the pooled common
                        slope with its CI; random-ViT anchors when present
  A_fig_ecf             E_CF(K) on log-log axes per arm (clipped, unclipped) with the fitted slopes
  A_fig_drift           gamma_min trajectory per learning rate (mean and sd over seeds) with the gamma0/2 and
                        0.1 gamma0 lines; a second panel with the fine-tuned and frozen-head accuracies
  A_fig_alpha           Hill tail index per backbone with its bootstrap CI (tab:alpha-fits)
Paper B
  B_fig1c_constructions Figure 1(c): closed-form cross-task error P_13(a) of the anti-aligned and the
                        catastrophic constructions with their Bayes baselines (no data: the curves are
                        the theorem statements; the sidecar carries the anchor values)
  B_fig_rate            rate validation: log |P_ij| against gamma_ij per backbone, the OLS line and the
                        Theorem 6.2 reference slope -(kappa_median - 1)^2 / 8
  B_fig_gap_recovery    per-row gap recovery (FIX - WT) / (J - WT) with FIX_c beside it

Matplotlib only; Agg backend; no seaborn.  Fonts and sizes are left at defaults so the PDFs are plain.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm as _norm

from pipeline import registry as R

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    plt = None
    _MPL_ERR = e


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label(b: str) -> str:
    return R.BACKBONES[b].label if b in R.BACKBONES else b


def _dlabel(d: str) -> str:
    return R.DATASETS[d].label if d in R.DATASETS else d


def _save(fig, out_dir: Path, name: str, data: dict, sources: Dict[str, Path]) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=160 if ext == "png" else None)
        paths.append(p)
    plt.close(fig)
    side = out_dir / f"{name}.data.json"
    side.write_text(json.dumps(dict(figure=name, sources={k: dict(path=str(v), sha256=_sha(v)) for k, v in sources.items() if v.exists()},
                                    data=data), indent=1, default=float), encoding="utf-8")
    paths.append(side)
    return paths


# ---------------------------------------------------------------------------
# Paper A
# ---------------------------------------------------------------------------

BACKBONE_COLOR = {"vit_in21k": "#1f77b4", "dinov2_b14": "#d62728", "clip_b16": "#2ca02c", "mae_b16": "#9467bd", "resnet50": "#8c564b", "random_vit": "#7f7f7f"}
DATASET_MARKER = {"cifar100": "o", "imagenet_r": "s", "cub200": "^", "omnibenchmark": "D", "imagenet100": "v", "domainnet": "P"}
CR_WIDTH_IN = 5.5          # NeurIPS text width; the figure is included at 0.8\linewidth


def fig_gamma_tc(A: dict, out_dir: Path, src: Path, include_anchors: bool = False, n_band: int = 60) -> List[Path]:
    """fig:gamma-tc as the caption specifies (FIGURE_SPECS.md, section 1): marker shape = dataset, colour =
    backbone; faint per-ordering points and solid configuration means; per-backbone OLS lines with shaded 95
    percent bootstrap bands (percentiles of the bootstrap lines, from slopes.per_backbone.<b>.boot_slopes and
    boot_intercepts); the dashed grey reference of slope -1/8 through the pooled fit's centroid; the pooled
    annotation with its population tag; an inset with the bootstrap distribution of the pooled slope. The
    random-ViT anchors are off by default (the caption reports them in a table)."""
    tc = A.get("tc") or {}
    pts = []
    for key, v in tc.items():
        b, d = key.split("|")
        for p in v.get("per_seed", []):
            if p["tc_gap"] > 0:
                pts.append(dict(backbone=b, dataset=d, seed=p["seed"], gamma_min=p["gamma_min"], log_tc_gap=float(np.log(p["tc_gap"]))))
    if not pts:
        return []
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(CR_WIDTH_IN * 0.8, CR_WIDTH_IN * 0.8 * 0.68))
    data = dict(points=pts, means=[], lines={}, bands={}, reference=None, inset=None, anchors=[])
    xs_all = np.array([p["gamma_min"] for p in pts]); xx = np.linspace(xs_all.min(), xs_all.max(), n_band)
    slopes = A.get("slopes") or {}
    for b in sorted({p["backbone"] for p in pts}):
        P = [p for p in pts if p["backbone"] == b]; col = BACKBONE_COLOR.get(b, "k")
        for d in sorted({p["dataset"] for p in P}):
            Q = [p for p in P if p["dataset"] == d]; mk = DATASET_MARKER.get(d, "o")
            ax.scatter([p["gamma_min"] for p in Q], [p["log_tc_gap"] for p in Q], s=9, alpha=0.25, marker=mk, color=col, linewidths=0)
            gx = float(np.mean([p["gamma_min"] for p in Q])); gy = float(np.mean([p["log_tc_gap"] for p in Q]))
            data["means"].append(dict(backbone=b, dataset=d, gamma_min=gx, log_tc_gap=gy, n=len(Q)))
            ax.scatter([gx], [gy], s=40, marker=mk, color=col, edgecolor="k", linewidth=0.5, zorder=3)
        f = (slopes.get("per_backbone") or {}).get(b)
        if f and f.get("slope") is not None:
            ax.plot(xx, f["intercept"] + f["slope"] * xx, lw=1.2, color=col, label=f"{_label(b)}: slope {f['slope']:.3f}")
            data["lines"][b] = dict(slope=f["slope"], intercept=f["intercept"], ci95=f.get("ci95"), r2=f.get("r2"))
            bs, bi = f.get("boot_slopes"), f.get("boot_intercepts")
            if bs and bi and len(bs) == len(bi) and len(bs) >= 20:
                Y = np.asarray(bi)[:, None] + np.asarray(bs)[:, None] * xx[None, :]
                lo, hi = np.percentile(Y, [2.5, 97.5], axis=0)
            elif f.get("ci95"):   # fallback: band from the slope CI pivoting at the fitted centroid
                xm = float(np.mean([p["gamma_min"] for p in P])); ym = f["intercept"] + f["slope"] * xm
                lo = np.minimum(ym + f["ci95"][0] * (xx - xm), ym + f["ci95"][1] * (xx - xm)); hi = np.maximum(ym + f["ci95"][0] * (xx - xm), ym + f["ci95"][1] * (xx - xm))
            else:
                lo = hi = None
            if lo is not None:
                ax.fill_between(xx, lo, hi, color=col, alpha=0.12, linewidth=0)
                data["bands"][b] = dict(x=xx.tolist(), lo=np.asarray(lo).tolist(), hi=np.asarray(hi).tolist(), source="bootstrap lines" if bs else "slope CI about the centroid")
    pooled = slopes.get("pooled12")
    if pooled and pooled.get("slope") is not None:
        ci = pooled.get("ci95") or [None, None]
        xm, ym = float(xs_all.mean()), float(np.mean([p["log_tc_gap"] for p in pts]))
        ref = (slopes.get("reference") or {}).get("operational", -1 / 8)
        ax.plot(xx, ym + ref * (xx - xm), ls="--", lw=1.0, color="0.55", label=r"reference $\widehat m = -1/8$")
        data["reference"] = dict(slope=ref, through=[xm, ym])
        tag = f"pooled ({pooled.get('n', len(data['means']))} pretrained): slope {pooled['slope']:.3f}"
        if ci[0] is not None:
            tag += f", 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"
        ax.text(0.98, 0.03, tag, transform=ax.transAxes, fontsize=7.5, ha="right", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5))
        data["lines"]["pooled12"] = dict(slope=pooled["slope"], ci95=pooled.get("ci95"), intercepts=pooled.get("intercepts"), r2=pooled.get("r2"), n=pooled.get("n"))
        bs = pooled.get("boot_slopes")
        if bs and len(bs) >= 20:
            ins = ax.inset_axes([0.62, 0.62, 0.35, 0.33])
            ins.hist(bs, bins=30, color="0.3", alpha=0.85)
            ins.axvline(ref, color="0.55", ls="--", lw=0.8); ins.axvline(pooled["slope"], color="k", lw=0.8)
            ins.set_title("bootstrap: pooled slope", fontsize=6.5, pad=2); ins.tick_params(labelsize=5.5); ins.set_yticks([])
            data["inset"] = dict(boot_slopes=[float(v) for v in bs], reference=ref)
    if include_anchors:
        for d, v in (A.get("anchors") or {}).items():
            if v.get("log_tc_gap", {}).get("mean") is not None:
                ax.scatter([v["gamma_min"]["mean"]], [v["log_tc_gap"]["mean"]], marker="x", s=40, color=BACKBONE_COLOR["random_vit"])
                data["anchors"].append(dict(dataset=d, gamma_min=v["gamma_min"]["mean"], log_tc_gap=v["log_tc_gap"]["mean"]))
    # dataset legend (marker shapes) as a second legend
    from matplotlib.lines import Line2D
    h1, l1 = ax.get_legend_handles_labels()
    leg1 = ax.legend(h1, l1, loc="upper left", frameon=False, fontsize=6.5); ax.add_artist(leg1)
    dsets = sorted({p["dataset"] for p in pts})
    h2 = [Line2D([0], [0], marker=DATASET_MARKER.get(d, "o"), color="0.3", lw=0, markersize=5) for d in dsets]
    ax.legend(h2, [_dlabel(d) for d in dsets], loc="lower left", bbox_to_anchor=(0.0, 0.13), frameon=False, fontsize=6.5, title="dataset", title_fontsize=6.5)
    ax.set_xlabel(r"$\widehat\gamma_{\min}(\phi_0)$"); ax.set_ylabel(r"$\log(\widehat{\mathrm{TC\ gap}})$")
    prov = ",".join(A.get("provenance", [])); arm = A.get("optimizer", "?")
    if prov != "real":
        ax.set_title(f"[{arm} arm; provenance: {prov}]", fontsize=7, color="0.4")
    return _save(fig, out_dir, "A_fig_gamma_tc", data, {"paperA_results.json": src})


def fig_ecf(A: dict, out_dir: Path, src: Path) -> List[Path]:
    ecf = A.get("ecf") or {}
    if not ecf:
        return []
    fig, axes = plt.subplots(1, len(ecf), figsize=(3.6 * len(ecf), 3.2), squeeze=False)
    data = {}
    for ax, (b, v) in zip(axes[0], sorted(ecf.items())):
        data[b] = {}
        for arm, color in (("clipped", "C0"), ("unclipped", "C3")):
            res = v["arms"].get(arm) or {}
            Ks = [int(k) for k in res.get("E_CF", {})]
            ms = [res["E_CF"][str(K)]["mean"] for K in Ks]; sds = [res["E_CF"][str(K)].get("sd") or 0.0 for K in Ks]
            pos = [(K, m, s) for K, m, s in zip(Ks, ms, sds) if m is not None and m > 0]
            data[b][arm] = dict(K=Ks, E_CF_mean=ms, E_CF_sd=sds, fit=res.get("fit"), excluded=res.get("excluded"))
            if pos:
                ax.errorbar([p[0] for p in pos], [p[1] for p in pos], yerr=[p[2] for p in pos], fmt="o-", ms=4, lw=1, color=color, capsize=2,
                            label=f"{arm}" + (f" (slope {res['fit']['slope']:.2f})" if res.get("fit", {}).get("slope") is not None else ""))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("K (iterations per task)"); ax.set_ylabel(r"$E_{\mathrm{CF}}$")
        ax.set_title(_label(b), fontsize=9); ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Paper A E_CF sweep  [{A.get('optimizer','?')} reference; provenance: {','.join(A.get('provenance', []))}]", fontsize=8)
    return _save(fig, out_dir, "A_fig_ecf", data, {"paperA_results.json": src})


def fig_alpha(A: dict, out_dir: Path, src: Path) -> List[Path]:
    al = A.get("alpha") or {}
    if not al:
        return []
    bs = sorted(al); fig, ax = plt.subplots(figsize=(4.2, 3.0)); data = {}
    for i, b in enumerate(bs):
        v = al[b]; a = v["alpha"]["mean"]; ci = v.get("ci") or [None, None]
        lo = (a - ci[0]) if ci and ci[0] is not None else 0; hi = (ci[1] - a) if ci and ci[1] is not None else 0
        ax.errorbar([i], [a], yerr=[[max(lo, 0)], [max(hi, 0)]], fmt="o", color="C0", capsize=3)
        data[b] = dict(alpha=a, ci=ci, k=v.get("k"))
    ax.set_xticks(range(len(bs))); ax.set_xticklabels([_label(b) for b in bs], fontsize=8); ax.set_ylabel(r"Hill $\hat\alpha$ (k = %s)" % (al[bs[0]].get("k", "?")))
    ax.axhline(2.0, color="gray", lw=0.8, ls=":"); ax.set_title("Paper A tab:alpha-fits", fontsize=9)
    return _save(fig, out_dir, "A_fig_alpha", data, {"paperA_results.json": src})


def fig_drift(D: dict, out_dir: Path, src: Path) -> List[Path]:
    """fig:drift-trajectory as the caption specifies (FIGURE_SPECS.md, section 2): gamma_min(phi_t) against the
    stage for the three learning rates (mean, shaded sd over seeds); dotted gamma0 baseline, dashed gamma0/2
    critical threshold, solid floor at 0; green operational band above gamma0/2, pink collapsed band below;
    per-rate pre-floor OLS slope annotated with its stage window and the t* crossing marked with a down
    triangle; for the saturated rate the stages 0 to 2 window shaded, the floor-saturated secants -gamma0/w
    (w = 1, 2) drawn from the baseline to the floor, and the time-to-floor marked with a cross; inset: |m|
    against the learning rate on log-log axes with a slope-1 reference line and the fitted log-log slope
    (no correlation coefficient is printed)."""
    runs = D.get("runs") or []
    if not runs:
        return []
    summ = D.get("summary") or {}
    lrs = sorted({r["lr"] for r in runs}); g0 = float(np.mean([r["gamma0"] for r in runs]))
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(CR_WIDTH_IN * 0.8, CR_WIDTH_IN * 0.8 * 0.7))
    data = dict(gamma0=g0, lrs={}, inset=None)
    tmax = 1.0
    for i, lr in enumerate(lrs):
        rs = [r for r in runs if r["lr"] == lr]; col = f"C{i}"
        tr0 = rs[0]["trajectory"]
        ts = np.array([p["t"] for p in tr0], float) if tr0 and isinstance(tr0[0], dict) else np.arange(len(tr0), dtype=float)
        G = []
        for r in rs:
            tr_ = r["trajectory"]
            if tr_ and isinstance(tr_[0], dict):
                t_r = np.array([p["t"] for p in tr_], float); g_r = np.array([p["gamma"] for p in tr_], float)
            else:
                g_r = np.asarray(tr_, float); t_r = np.arange(len(g_r), dtype=float)
            G.append(np.interp(ts, t_r, g_r))
        G = np.array(G); m = G.mean(0); sd = G.std(0, ddof=1) if len(G) > 1 else np.zeros_like(m); tmax = max(tmax, float(ts.max()))
        ax.plot(ts, m, color=col, lw=1.3, label=rf"$\eta_\phi = 10^{{{int(round(np.log10(lr)))}}}$ (n={len(rs)})")
        ax.fill_between(ts, m - sd, m + sd, color=col, alpha=0.15, linewidth=0)
        s_lr = summ.get(str(lr)) or summ.get(repr(lr)) or {}
        entry = dict(t=ts.tolist(), gamma_mean=m.tolist(), gamma_sd=sd.tolist(), n_seeds=len(rs), summary=s_lr)
        sl = (s_lr.get("slope_prefloor") or {}).get("mean"); win = s_lr.get("window_stages"); tstar = (s_lr.get("t_star_observed") or {}).get("mean")
        ttf = (s_lr.get("time_to_floor") or {}).get("mean")
        if sl is not None and win:
            k = int(np.median(win))
            xk = min(k, tmax) * 0.8
            ax.annotate(rf"$\widehat m = {sl:.3g}$ per stage (stages 1 to {k})", xy=(xk, float(np.interp(xk, ts, m)) if ts.size else g0),
                        xytext=(4, 8), textcoords="offset points", fontsize=6.5, color=col)
        if tstar is not None and np.isfinite(tstar):
            ax.plot([tstar], [g0 / 2], marker="v", color=col, ms=6, zorder=4); ax.annotate(rf"$\hat t^\star = {tstar:.2f}$", xy=(tstar, g0 / 2), xytext=(4, -12), textcoords="offset points", fontsize=6.5, color=col)
        if i == len(lrs) - 1 and ttf is not None and np.isfinite(ttf):    # the saturated (largest) rate
            ax.axvspan(0, 2, color="0.85", alpha=0.35, lw=0, zorder=0)
            for w in (1, 2):
                ax.plot([0, w], [g0, 0], ls="-", lw=0.8, color="0.4"); ax.annotate(rf"$-\widehat\gamma_{{\min}}(\phi_0)/{w}$", xy=(w, 0), xytext=(2, 4 + 8 * (w - 1)), textcoords="offset points", fontsize=6, color="0.4")
            ax.plot([ttf], [0.1 * g0], marker="x", color=col, ms=7, zorder=4); ax.annotate(f"time-to-floor {ttf:.2f}", xy=(ttf, 0.1 * g0), xytext=(4, 6), textcoords="offset points", fontsize=6.5, color=col)
        data["lrs"][str(lr)] = entry
    ax.axhline(g0, ls=":", color="0.3", lw=0.9); ax.axhline(g0 / 2, ls="--", color="0.3", lw=0.9); ax.axhline(0, ls="-", color="k", lw=0.9)
    ax.axhspan(g0 / 2, g0 * 1.08, color="#2ca02c", alpha=0.07, lw=0); ax.axhspan(-0.04 * g0, g0 / 2, color="#d62728", alpha=0.06, lw=0)
    ax.set_xlim(0, tmax); ax.set_ylim(-0.04 * g0, g0 * 1.08)
    ax.set_xlabel("stage"); ax.set_ylabel(r"$\widehat\gamma_{\min}(\phi_t)$"); ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax.text(0.01, 0.985, rf"$\widehat\gamma_{{\min}}(\phi_0) = {g0:.2f}$ (dotted); $\widehat\gamma_{{\min}}(\phi_0)/2$ (dashed)", transform=ax.transAxes, va="top", fontsize=6.5, color="0.3")
    # inset: |m| against eta on log-log with the slope-1 reference
    pts = [(float(lr), abs((summ.get(str(lr)) or {}).get("slope_prefloor", {}).get("mean") or np.nan)) for lr in lrs]
    pts = [(a, b) for a, b in pts if np.isfinite(b) and b > 0]
    if len(pts) >= 2:
        ins = ax.inset_axes([0.40, 0.40, 0.27, 0.27]); xa = np.array([p[0] for p in pts]); ya = np.array([p[1] for p in pts])
        ins.loglog(xa, ya, "o", color="k", ms=3.5)
        c = np.log10(ya[0]) - np.log10(xa[0]); xr = np.array([xa.min() / 2, xa.max() * 2]); ins.loglog(xr, 10 ** (np.log10(xr) + c), ls="--", color="0.5", lw=0.8)
        fit = np.polyfit(np.log10(xa), np.log10(ya), 1) if len(pts) >= 2 else [np.nan, np.nan]
        ins.set_title(rf"$|\widehat m|$ vs $\eta_\phi$; log-log slope {fit[0]:.2f}", fontsize=6, pad=2); ins.tick_params(labelsize=5.5)
        data["inset"] = dict(eta=xa.tolist(), abs_slope=ya.tolist(), loglog_slope=float(fit[0]), reference_slope=1.0)
    return _save(fig, out_dir, "A_fig_drift", data, {"paperA_drift.json": src})


# ---------------------------------------------------------------------------
# Paper B
# ---------------------------------------------------------------------------

def fig1c_constructions(out_dir: Path) -> List[Path]:
    """Figure 1(c) of Paper B from the closed forms of the two negative-result theorems.

    Anti-aligned (mu_1 = e1, mu_2 = -e1, mu_3 = a e1 + e2, mu_4 = a e1 - e2, Sigma = I): the per-task CF heads are
    w_1 = e1, w_2 = -e1, w_3 = e2, w_4 = -e2 with zero biases, so for x ~ N(e1, I) the score (w_1 - w_3)^T x is
    N(1, 2) and P_13 = Phi(-1/sqrt 2) for every a; Bayes: Phi(-(1/2) sqrt((a-1)^2 + 1)).
    Catastrophic (mu_1 = e1, mu_2 = -e1, mu_3 = e2, mu_4 = -e2, mu_5 = a e1): task-2 centroid (a/3, 0),
    S_bar^(2) = (2 + a^2)/3, w_3 = (-a/3, 1), b_3 = -1/2 + (2 + a^2)/6, w_1 = e1, b_1 = 0; the class-1 score has
    mean (-a^2 + 2a + 7)/6 and variance (1 + a/3)^2 + 1, so P_13 = Phi(-mean/sd) -> 1; the pair (1, 3) Bayes
    baseline is Phi(-1/sqrt 2) since mu_1 and mu_3 do not move with a."""
    a = np.linspace(0.0, 12.0, 481)
    anti = np.full_like(a, _norm.cdf(-1 / np.sqrt(2)))
    anti_bayes = _norm.cdf(-0.5 * np.sqrt((a - 1) ** 2 + 1))
    mean = (-a ** 2 + 2 * a + 7) / 6; var = (1 + a / 3) ** 2 + 1
    cat = _norm.cdf(-mean / np.sqrt(var)); cat_bayes = np.full_like(a, _norm.cdf(-1 / np.sqrt(2)))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    axes[0].plot(a, anti, color="C3", lw=1.5, label=r"CF head $P_{13}$ (Thm anti-aligned)"); axes[0].plot(a, anti_bayes, color="C0", lw=1.2, ls="--", label="Bayes")
    axes[0].set_title("anti-aligned construction", fontsize=9)
    axes[1].plot(a, cat, color="C3", lw=1.5, label=r"CF head $P_{13}$ (Thm catastrophic)"); axes[1].plot(a, cat_bayes, color="C0", lw=1.2, ls="--", label="Bayes (pair 1,3)")
    axes[1].set_title("catastrophic construction", fontsize=9)
    for ax in axes:
        ax.set_xlabel("a"); ax.set_ylabel(r"$P_{13}$"); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=7, frameon=False)
    anchors = {str(v): dict(anti=float(_norm.cdf(-1 / np.sqrt(2))), anti_bayes=float(_norm.cdf(-0.5 * np.sqrt((v - 1) ** 2 + 1))),
                            catastrophic=float(_norm.cdf(-((-v ** 2 + 2 * v + 7) / 6) / np.sqrt((1 + v / 3) ** 2 + 1)))) for v in (1, 2, 3, 5, 10)}
    data = dict(a=a.tolist(), anti=anti.tolist(), anti_bayes=anti_bayes.tolist(), catastrophic=cat.tolist(), catastrophic_bayes_pair13=cat_bayes.tolist(), anchors=anchors,
                closed_forms=dict(anti="Phi(-1/sqrt 2)", anti_bayes="Phi(-(1/2) sqrt((a-1)^2+1))", catastrophic="Phi(-((-a^2+2a+7)/6) / sqrt((1+a/3)^2+1))"))
    return _save(fig, out_dir, "B_fig1c_constructions", data, {})


def fig_rate(B: dict, out_dir: Path, src: Path) -> List[Path]:
    pairs = B.get("rate_pairs") or []; fits = B.get("rate_fits") or {}
    if not pairs:
        return []
    bs = sorted({p["backbone"] for p in pairs})
    fig, axes = plt.subplots(1, len(bs), figsize=(3.4 * len(bs), 3.1), squeeze=False); data = {}
    for ax, b in zip(axes[0], bs):
        P = [p for p in pairs if p["backbone"] == b and p["P"] > 0]
        x = np.array([p["gamma"] for p in P]); y = np.log(np.array([p["P"] for p in P]))
        ax.scatter(x, y, s=8, alpha=0.4, color="C0")
        f = fits.get(b) or {}
        data[b] = dict(gamma=x.tolist(), logP=y.tolist(), fit=f)
        if f.get("slope") is not None and x.size:
            xx = np.linspace(x.min(), x.max(), 50); ax.plot(xx, f["intercept"] + f["slope"] * xx, color="C3", lw=1.2, label=f"OLS slope {f['slope']:.3f}")
            if f.get("predicted_slope") is not None:
                yy0 = f["intercept"] + f["slope"] * xx.mean()
                ax.plot(xx, yy0 + f["predicted_slope"] * (xx - xx.mean()), color="k", lw=1, ls="--", label=rf"$-(\hat\kappa-1)^2/8$ = {f['predicted_slope']:.3f}")
        ax.set_title(_label(b), fontsize=9); ax.set_xlabel(r"$\hat\gamma_{ij}$"); ax.set_ylabel(r"$\log |P_{ij}|$"); ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Paper B rate validation  [{B.get('optimizer','?')} arm; provenance: {','.join(B.get('provenance', []))}]", fontsize=8)
    return _save(fig, out_dir, "B_fig_rate", data, {"paperB_results.json": src})


def fig_gap_recovery(B: dict, out_dir: Path, src: Path) -> List[Path]:
    bench = B.get("benchmark") or {}
    rows = [(k, v) for k, v in bench.items() if v.get("recovery", {}).get("mean") is not None]
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(max(4.0, 0.55 * len(rows)), 3.0)); data = []
    for i, (k, v) in enumerate(rows):
        b, d = k.split("|")
        r = v["recovery"]["mean"]; rc = (v.get("recovery_c") or {}).get("mean"); sd = v["recovery"].get("sd") or 0.0
        ax.bar(i - 0.18, r, width=0.36, color="C0", yerr=sd, capsize=2, label="FIX (Def 7.1)" if i == 0 else None)
        if rc is not None:
            ax.bar(i + 0.18, rc, width=0.36, color="C2", alpha=0.7, label="FIX_c (scale-matched)" if i == 0 else None)
        data.append(dict(backbone=b, dataset=d, recovery=r, recovery_sd=sd, recovery_c=rc, head_scale=(v.get("head_scale") or {}).get("mean")))
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([f"{_label(k.split('|')[0])}\n{_dlabel(k.split('|')[1])}" for k, _ in rows], fontsize=6)
    ax.axhline(1.0, color="gray", lw=0.6, ls=":"); ax.set_ylabel("gap recovery (FIX - WT)/(J - WT)"); ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"Paper B fig:gap-recovery  [{B.get('optimizer','?')} arm; provenance: {','.join(B.get('provenance', []))}]", fontsize=8)
    return _save(fig, out_dir, "B_fig_gap_recovery", data, {"paperB_results.json": src})


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_all(results_dir: Path, out_dir: Optional[Path] = None, drift_path: Optional[Path] = None, log=print) -> List[Path]:
    if plt is None:  # pragma: no cover
        raise RuntimeError(f"matplotlib is required for the figures: {_MPL_ERR}")
    out_dir = out_dir or results_dir / "figures"
    written: List[Path] = []
    written += fig1c_constructions(out_dir)
    pa = results_dir / "paperA_results.json"
    if pa.exists():
        A = json.loads(pa.read_text(encoding="utf-8"))
        written += fig_gamma_tc(A, out_dir, pa); written += fig_ecf(A, out_dir, pa); written += fig_alpha(A, out_dir, pa)
    pb = results_dir / "paperB_results.json"
    if pb.exists():
        B = json.loads(pb.read_text(encoding="utf-8"))
        written += fig_rate(B, out_dir, pb); written += fig_gap_recovery(B, out_dir, pb)
    dp = drift_path or results_dir / "paperA_drift.json"
    if dp.exists():
        written += fig_drift(json.loads(dp.read_text(encoding="utf-8")), out_dir, dp)
    for p in written:
        log(f"[figures] {p}")
    return written
