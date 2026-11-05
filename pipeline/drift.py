"""pipeline.drift -- Paper A, Section 7: representation drift under backbone fine-tuning (GPU).

Protocol (subsec:exp-drift, app:drift-results): Split-CIFAR-100 (T = 10 stages), ViT-B/16-IN21K fine-tuned
jointly with its linear head, backbone learning rate eta_phi in {1e-5, 1e-4, 1e-3}, head learning rate 1e-3,
AdamW, batch 128, K = 1000 iterations per stage (the appendix quotes sub-stage windows "of 1000 iterations
per stage").  gamma_min(phi_t) is measured on TRAINING features of ALL classes through the current backbone
(app:gamma-computation) at the end of every stage and every ``measure_every`` iterations inside a stage, so
that the time-to-floor at 1e-3 (printed 0.48 stages) and the critical horizon t* are resolved below one
stage.  Reported per learning rate: the OLS slope of gamma_min over the pre-floor stage-end points, t*
(first crossing of gamma0/2 on the fine trajectory, linearly interpolated, and the linear extrapolation
gamma0/2 / |slope|), the time-to-floor (first crossing of 10% of gamma0), the floor-saturated secants
-gamma0/w, and the accuracy crossover: the first stage at which the fine-tuned model's class-IL accuracy on
the seen classes falls below a head trained by the same procedure (same optimizer, batches, budget and masked
CE) on the FROZEN backbone's features.

Cost on a T4: about 0.3 s per training iteration (10,000 per run) plus one gamma pass per checkpoint over
gamma_subsample images per class; with the defaults (measure_every 200, 50 images per class) a run is
about 70 minutes and the full 3 x 5 grid about 17 GPU-hours, resumable per (lr, seed).  ``iters_per_stage``,
``measure_every`` and ``gamma_subsample`` trade fidelity for time and are recorded in the output.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import numpy as np

from pipeline import registry as R
from pipeline.stats import ClassStats

SETTINGS = ("iters_per_stage", "measure_every", "gamma_subsample", "head_lr", "batch")


def gamma_min_from_features(X, y, toc) -> float:
    return ClassStats.from_features(X, y, toc, protocol="A").gamma_min()[0]


def _resume(out_path: Path, settings: dict, log) -> List[dict]:
    if not out_path.exists():
        return []
    try:
        prev = json.loads(out_path.read_text())
    except Exception as e:  # a corrupt file is not a reason to stop
        log(f"[drift] could not read {out_path} ({e}); starting afresh"); return []
    if all(prev.get(k) == settings[k] for k in SETTINGS):
        log(f"[drift] resuming: {len(prev.get('runs', []))} finished runs in {out_path}"); return prev.get("runs", [])
    log(f"[drift] {out_path} was made with other settings; starting afresh"); return []


def run_drift(data_root: Path, out_path: Path, *, lrs=R.DRIFT_LRS, seeds=R.SEEDS, iters_per_stage=1000, measure_every=200,
              head_lr=1e-3, batch=128, gamma_subsample=50, device="cuda", log=print) -> dict:
    import torch, torch.nn as nn
    import torch.nn.functional as Fnn
    from transformers import ViTModel, AutoImageProcessor
    from pipeline.features import build_dataset, _assert_download_allowed
    _assert_download_allowed("replicate.py drift (dataset + backbone weights)")   # owner storage law: no downloads off Colab
    ds = R.DATASETS["cifar100"]; N = ds.n_classes
    splits = build_dataset("cifar100", data_root); tr, te = splits["train"], splits["test"]
    proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1); std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
    use_cuda = (device == "cuda")

    def to_tensor(il, idx):
        x = torch.from_numpy(il.in_memory[idx]).permute(0, 3, 1, 2).float() / 255.0
        x = Fnn.interpolate(x, size=224, mode="bilinear", align_corners=False)
        return (x - mean) / std

    @torch.no_grad()
    def features(model, il, idx):
        model.eval(); out = []
        for s in range(0, len(idx), 256):
            xb = to_tensor(il, idx[s:s + 256]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                out.append(model(pixel_values=xb).last_hidden_state[:, 0].float().cpu().numpy())
        return np.concatenate(out)

    settings = dict(iters_per_stage=iters_per_stage, measure_every=measure_every, gamma_subsample=gamma_subsample, head_lr=head_lr, batch=batch)
    out_path = Path(out_path)
    results = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **settings, runs=_resume(out_path, settings, log))
    done = {(r["lr"], r["seed"]) for r in results["runs"]}
    g_idx = np.arange(len(tr)) if gamma_subsample is None else np.concatenate(
        [R.rng("drift-gamma-subsample").choice(np.flatnonzero(tr.labels == c), min(gamma_subsample, int((tr.labels == c).sum())), replace=False) for c in range(N)])
    F0_all = te_feat0 = None
    for lr in lrs:
        for seed in seeds:
            if (lr, seed) in done:
                continue
            torch.manual_seed(seed)
            ordering = R.class_ordering("cifar100", seed); toc = R.task_of_class("cifar100", ordering); tasks = R.tasks_from_ordering("cifar100", ordering)
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", add_pooling_layer=False).to(device)
            head = nn.Linear(768, N).to(device)
            opt = torch.optim.AdamW([{"params": model.parameters(), "lr": lr}, {"params": head.parameters(), "lr": head_lr}], weight_decay=1e-4)
            if F0_all is None:   # frozen-backbone features, once (identical for every lr and seed)
                F0_all = features(model, tr, np.arange(len(tr))); te_feat0 = features(model, te, np.arange(len(te)))
            gamma0 = gamma_min_from_features(F0_all[g_idx], tr.labels[g_idx], toc)
            traj = [dict(t=0.0, gamma=gamma0)]; acc_ft, acc_frozen = [], []
            # frozen-backbone reference: the same head, optimizer, budget and masked CE on phi_0 features
            torch.manual_seed(seed); head_fr = nn.Linear(768, N).to(device)
            opt_fr = torch.optim.AdamW(head_fr.parameters(), lr=head_lr, weight_decay=1e-4)
            F0_t = torch.from_numpy(F0_all).to(device); te0_t = torch.from_numpy(te_feat0).to(device)
            seen = np.zeros(0, np.int64); t0 = time.time()
            for t, cls in enumerate(tasks):
                seen = np.union1d(seen, cls)
                idx = np.flatnonzero(np.isin(tr.labels, cls))
                g = np.random.default_rng(R.derive_seed(f"drift:{lr}:{seed}:{t}"))
                mask = torch.full((N,), float("-inf"), device=device); mask[torch.from_numpy(seen).to(device)] = 0.0
                model.train(); head.train(); head_fr.train()
                for k in range(iters_per_stage):
                    bi = g.choice(idx, batch, replace=True)
                    xb = to_tensor(tr, bi).to(device); yb = torch.from_numpy(tr.labels[bi]).long().to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                        logits = head(model(pixel_values=xb).last_hidden_state[:, 0].float())
                    loss = nn.functional.cross_entropy(logits.float() + mask, yb)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    loss_fr = nn.functional.cross_entropy(head_fr(F0_t[bi]) + mask, yb)        # same batch, frozen features
                    opt_fr.zero_grad(set_to_none=True); loss_fr.backward(); opt_fr.step()
                    if (k + 1) % measure_every == 0 and (k + 1) < iters_per_stage:
                        traj.append(dict(t=t + (k + 1) / iters_per_stage, gamma=gamma_min_from_features(features(model, tr, g_idx), tr.labels[g_idx], toc)))
                        model.train(); head.train()
                traj.append(dict(t=float(t + 1), gamma=gamma_min_from_features(features(model, tr, g_idx), tr.labels[g_idx], toc)))
                mte = np.isin(te.labels, seen); mte_idx = np.flatnonzero(mte)
                with torch.no_grad():
                    Fte = features(model, te, mte_idx); head.eval(); head_fr.eval()
                    lg = head(torch.from_numpy(Fte).to(device)).float().cpu().numpy()
                    lg_fr = head_fr(te0_t[torch.from_numpy(mte_idx).to(device)]).float().cpu().numpy()
                unseen = ~np.isin(np.arange(N), seen); lg[:, unseen] = -np.inf; lg_fr[:, unseen] = -np.inf
                acc_ft.append(float(np.mean(np.argmax(lg, 1) == te.labels[mte])))
                acc_frozen.append(float(np.mean(np.argmax(lg_fr, 1) == te.labels[mte])))
                log(f"[drift] lr {lr:g} seed {seed} stage {t+1}: gamma_min {traj[-1]['gamma']:.3f} (gamma0 {gamma0:.3f}) acc ft {acc_ft[-1]:.3f} frozen {acc_frozen[-1]:.3f} ({time.time()-t0:.0f}s)")
            results["runs"].append(dict(lr=lr, seed=seed, gamma0=gamma0, trajectory=traj, acc_finetuned=acc_ft, acc_frozen_head=acc_frozen))
            out_path.parent.mkdir(parents=True, exist_ok=True); out_path.write_text(json.dumps(results, indent=1))
            del model, head, opt, head_fr, opt_fr, F0_t, te0_t
            if use_cuda:
                torch.cuda.empty_cache()
    results["summary"] = summarize_drift(results["runs"])
    out_path.parent.mkdir(parents=True, exist_ok=True); out_path.write_text(json.dumps(results, indent=1))
    return results


def _crossing(ts, gs, level):
    """First time at which the trajectory drops below ``level``, linearly interpolated; nan if never."""
    below = np.flatnonzero(gs < level)
    if not below.size:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(ts[0])
    return float(ts[i - 1] + (gs[i - 1] - level) / (gs[i - 1] - gs[i]) * (ts[i] - ts[i - 1]))


def summarize_drift(runs: List[dict]) -> dict:
    """Per lr: gamma0; the OLS slope over the pre-floor stage-end points (stage ends with gamma above 10% of
    gamma0, at least three points); t* observed (first crossing of gamma0/2 on the fine trajectory) and linear
    (gamma0/2 divided by |slope|); the time-to-floor (first crossing of 10% of gamma0); the floor-saturated
    secants -gamma0/w; the accuracy crossover (first stage at which the fine-tuned model is below the frozen-
    feature head, linearly interpolated between stage ends).  Trajectories may be stage-end only (old format)."""
    out = {}
    for lr in sorted({r["lr"] for r in runs}):
        rs = [r for r in runs if r["lr"] == lr]
        slopes, tstar, t_lin, floors, cross, windows, slopes_rel = [], [], [], [], [], [], []
        for r in rs:
            tr_ = r["trajectory"]
            if tr_ and isinstance(tr_[0], dict):
                ts = np.array([p["t"] for p in tr_], float); gs = np.array([p["gamma"] for p in tr_], float)
            else:
                gs = np.asarray(tr_, float); ts = np.arange(len(gs), dtype=float)
            g0 = r["gamma0"]
            ends = np.isclose(ts, np.round(ts)); te_, ge = ts[ends], gs[ends]
            above = np.flatnonzero(ge > 0.1 * g0); k = int(above.max()) if above.size else 0
            windows.append(k)
            if k >= 2:
                sl = float(np.polyfit(te_[:k + 1], ge[:k + 1], 1)[0]); slopes.append(sl); slopes_rel.append(sl / g0)
                if sl < 0:
                    t_lin.append(-0.5 * g0 / sl)
            tstar.append(_crossing(ts, gs, g0 / 2)); floors.append(_crossing(ts, gs, 0.1 * g0))
            a_ft = np.asarray(r["acc_finetuned"], float); a_fr = np.asarray(r.get("acc_frozen_head", r.get("acc_frozen_ncm")), float)
            diff = a_ft - a_fr; below = np.flatnonzero(diff < 0)
            if below.size:
                i = int(below[0])
                cross.append(float(i + 1) if i == 0 else float(i + diff[i - 1] / (diff[i - 1] - diff[i])))
            else:
                cross.append(float("nan"))

        def ms(v):
            v = np.asarray([x for x in v if np.isfinite(x)], float)
            return dict(mean=float(v.mean()) if v.size else None, sd=float(v.std(ddof=1)) if v.size > 1 else 0.0, n=int(v.size))

        g0s = [r["gamma0"] for r in rs]
        lag = [c - t for c, t in zip(cross, tstar) if np.isfinite(c) and np.isfinite(t)]
        out[str(lr)] = dict(gamma0=ms(g0s), slope_prefloor=ms(slopes), window_stages=windows, t_star_observed=ms(tstar), t_star_linear=ms(t_lin),
                            time_to_floor=ms(floors), secants={"w=1": -float(np.mean(g0s)), "w=2": -float(np.mean(g0s)) / 2},
                            accuracy_crossover=ms(cross), crossover_minus_tstar=ms(lag),
                            # scale-free forms (invariant under a common rescaling of gamma): t* and the floor in stages,
                            # the pre-floor slope in units of gamma0 per stage
                            slope_prefloor_over_gamma0=ms(slopes_rel))
    fitted = sorted((float(k), abs(v["slope_prefloor"]["mean"])) for k, v in out.items() if v["slope_prefloor"]["mean"])
    out["fitted_lrs"] = [l for l, _ in fitted]
    if len(fitted) >= 2:
        # the log-log slope of |m| against the learning rate; 1.0 is the linear-in-eta prediction of the drift theorem
        ll = np.polyfit(np.log([l for l, _ in fitted]), np.log([m for _, m in fitted]), 1)[0]
        out["loglog_slope_abs_m_vs_lr"] = dict(value=float(ll), n_points=len(fitted))
    if len(fitted) >= 3:
        # the accepted text's correlation statistic, kept for the ledger only; three points make it uninformative
        out["corr_log_slope_log_lr"] = float(np.corrcoef(np.log([m for _, m in fitted]), np.log([l for l, _ in fitted]))[0, 1])
    elif len(fitted) == 2:
        (l0, m0), (l1, m1) = fitted
        out["slope_ratio_per_decade"] = float((m1 / m0) / (l1 / l0))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Readback: the pre-registered predictions P1..P8 and consistency checks C1..C5 of A_DRIFT_READBACK_PROTOCOL.md,
# scored from paperA_drift.json alone (C3 also reads paperA_results.json when given).  Writes DRIFT_READBACK.md.
# ----------------------------------------------------------------------------------------------------------------

ACCEPTED_DRIFT = {  # the accepted text's values, for the ledger only (never printed as regenerated)
    "gamma0": 8.20, "slope_1e-05": -0.16, "slope_0.0001": -1.73, "ttf_0.001": 0.48,
    "tstar": {"1e-05": 14.2, "0.0001": 2.35, "0.001": 0.21}, "crossover_0.0001": 3.18, "ratio_per_decade": 10.5,
}


def _traj_arrays(r):
    tr_ = r["trajectory"]
    if tr_ and isinstance(tr_[0], dict):
        return np.array([p["t"] for p in tr_], float), np.array([p["gamma"] for p in tr_], float)
    gs = np.asarray(tr_, float)
    return np.arange(len(gs), dtype=float), gs


def readback(drift_json: Path, out_path: Path, results_json: Path | None = None, log=print) -> dict:
    d = json.loads(Path(drift_json).read_text()); runs = d["runs"]; S = d.get("summary") or summarize_drift(runs)
    lrs = sorted({float(r["lr"]) for r in runs}); key = {lr: str(lr) for lr in lrs}
    rows = []

    def row(id_, text, verdict, value, band):
        rows.append(dict(id=id_, prediction=text, verdict=verdict, value=value, band=band))

    def fin(x):
        return x is not None and np.isfinite(x)

    # P1 monotone decrease up to noise: no stage-end value exceeds gamma0 by more than 2 sd of the within-run residuals
    viol = 0
    for r in runs:
        ts, gs = _traj_arrays(r); ends = np.isclose(ts, np.round(ts)); ge = gs[ends]
        resid = np.diff(ge); sd = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
        viol += int(np.any(ge[1:] > r["gamma0"] + 2 * sd))
    row("P1", "gamma decreases monotonically in t up to noise (no stage end above gamma0 + 2 sd), all runs", "HIT" if viol == 0 else "MISS", f"{viol} of {len(runs)} runs violate", "0 violations")
    # P2 ordering of horizons
    ts_means = {lr: (S[key[lr]]["t_star_observed"]["mean"]) for lr in lrs}
    vals = [ts_means[lr] if fin(ts_means[lr]) else float("inf") for lr in lrs]   # NaN (never crossed) counts as the largest horizon
    ok = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)) if len(vals) > 1 else None
    row("P2", "t*(1e-5) > t*(1e-4) > t*(1e-3), seed means (not crossed = largest)", "NOT-RUN" if ok is None else ("HIT" if ok else "MISS"), {key[lr]: ts_means[lr] for lr in lrs}, "strict ordering")
    # P3 collapse at the largest rate inside stage 1
    lr_max = max(lrs); ttf = S[key[lr_max]]["time_to_floor"]["mean"]
    row("P3", f"time-to-floor at eta = {lr_max:g} below 1.0 stage", "NOT-RUN" if not fin(ttf) and S[key[lr_max]]["time_to_floor"]["n"] == 0 else ("HIT" if fin(ttf) and ttf < 1.0 else "MISS"), ttf, "< 1.0")
    # P4 half-margin not reached inside the run at the smallest rate
    lr_min = min(lrs); tmin = S[key[lr_min]]["t_star_observed"]; n_stages = max(len(r["acc_finetuned"]) for r in runs)
    reached = tmin["n"]
    row("P4", f"at eta = {lr_min:g} the half-margin is not reached inside the {n_stages} stages (t* NaN or > {n_stages})", "HIT" if reached == 0 or (fin(tmin["mean"]) and tmin["mean"] > n_stages) else "MISS", dict(mean=tmin["mean"], n_crossed=reached, n_runs=len([r for r in runs if float(r["lr"]) == lr_min])), "not reached")
    # P5 crossover lags the geometric horizon at the middle rate (the rate with both quantities finite)
    mid = [lr for lr in lrs if fin(S[key[lr]]["crossover_minus_tstar"]["mean"])]
    if mid:
        # the protocol names the middle rate (1e-4); fall back to the finite rate nearest the geometric middle
        centre = float(np.exp(np.mean(np.log(lrs)))); lr_mid = min(mid, key=lambda lr: abs(np.log(lr) - np.log(centre)))
        lag = S[key[lr_mid]]["crossover_minus_tstar"]["mean"]
        row("P5", f"accuracy crossover minus t* at eta = {lr_mid:g} in (0, 3) stages", "HIT" if 0 < lag < 3 else "MISS", lag, "(0, 3)")
    else:
        row("P5", "accuracy crossover minus t*: no rate has both quantities finite", "NOT-RUN", None, "(0, 3)")
    # P6 log-log slope of |m| against eta
    ll = S.get("loglog_slope_abs_m_vs_lr")
    row("P6", "log-log slope of |m| against eta in [0.7, 1.3] (linear-in-eta prediction)", "NOT-RUN" if not ll else ("HIT" if 0.7 <= ll["value"] <= 1.3 else "MISS"), ll, "[0.7, 1.3]; n_points reported")
    # P7 relative pre-floor slope at the middle rate of the accepted order (-1.73 / 8.20 = -0.21 per stage)
    rel = {key[lr]: S[key[lr]]["slope_prefloor_over_gamma0"]["mean"] for lr in lrs}
    cand = [v for v in rel.values() if fin(v)]
    hit7 = any(-0.35 <= v <= -0.10 for v in cand) if cand else None
    row("P7", "some fitted rate has pre-floor slope / gamma0 in [-0.35, -0.10] per stage (the accepted order at 1e-4)", "NOT-RUN" if hit7 is None else ("HIT" if hit7 else "MISS"), rel, "[-0.35, -0.10]")
    # P8 frozen-head comparator coincides across learning rates per seed
    by_seed = {}
    for r in runs:
        by_seed.setdefault(r["seed"], {})[float(r["lr"])] = np.asarray(r.get("acc_frozen_head", r.get("acc_frozen_ncm")), float)
    maxdiff = 0.0; n_cmp = 0
    for seed, m in by_seed.items():
        if len(m) > 1:
            A = np.stack([v for v in m.values()]); maxdiff = max(maxdiff, float(np.max(A.max(0) - A.min(0)))); n_cmp += 1
    row("P8", "frozen-head comparator curves coincide across lr per seed (max abs difference < 0.005)", "NOT-RUN" if n_cmp == 0 else ("HIT" if maxdiff < 0.005 else "MISS"), maxdiff, "< 0.005")

    checks = []

    def chk(id_, text, verdict, value):
        checks.append(dict(id=id_, check=text, verdict=verdict, value=value))

    # C1 linear horizon against observed at the rate(s) with both
    for lr in lrs:
        a, b = S[key[lr]]["t_star_linear"]["mean"], S[key[lr]]["t_star_observed"]["mean"]
        if fin(a) and fin(b) and b > 0:
            chk("C1", f"t*_linear against t*_observed at eta = {lr:g} within 15 percent", "PASS" if abs(a - b) / b <= 0.15 else "FLAG", dict(linear=a, observed=b))
    # C2 the accepted 14.2 against the accepted slope and gamma0
    acc_lin = ACCEPTED_DRIFT["gamma0"] / (2 * abs(ACCEPTED_DRIFT["slope_1e-05"]))
    chk("C2", "accepted t*(1e-5) = 14.2 against gamma0/(2|m|) from the accepted 8.20 and -0.16", "FLAG" if abs(acc_lin - ACCEPTED_DRIFT["tstar"]["1e-05"]) > 0.15 * acc_lin else "PASS", dict(accepted=14.2, linear_from_accepted=round(acc_lin, 2)))
    # C3 gamma0 against the table value for the same configuration
    g0 = S[key[lrs[0]]]["gamma0"]["mean"]; tab = None
    if results_json and Path(results_json).exists():
        try:
            res = json.loads(Path(results_json).read_text())
            gm = res.get("gamma_min") or {}
            tab = (gm.get("vit_in21k") or {}).get("cifar100")
            tab = tab.get("mean") if isinstance(tab, dict) else tab
        except Exception:   # a results file of another shape: the check is simply not run
            tab = None
    chk("C3", "drift gamma0 against tab:gamma-min ViT-IN21K/CIFAR-100 (same protocol, subsampled against full pass)", "NOT-RUN" if tab is None else ("PASS" if abs(g0 - tab) / max(abs(tab), 1e-12) <= 0.10 else "FLAG"), dict(drift_gamma0=g0, table=tab, accepted_drift=8.20, accepted_table=8.3))
    # C4 secants are the identity
    sec = S[key[lr_max]]["secants"]; chk("C4", "secants equal -gamma0/w", "PASS" if abs(sec["w=1"] + g0) < 1e-9 and abs(sec["w=2"] + g0 / 2) < 1e-9 else "FLAG", sec)
    # C5 fitted rates are those with >= 3 pre-floor stage ends
    fitted = S.get("fitted_lrs", []); expect = [lr for lr in lrs if any(k >= 2 for k in S[key[lr]]["window_stages"])]
    chk("C5", "fitted_lrs equals the rates with at least three pre-floor stage ends; no correlation printed from fewer than five points", "PASS" if sorted(fitted) == sorted(expect) else "FLAG", dict(fitted=fitted, expected=expect, corr_present=("corr_log_slope_log_lr" in S)))

    hits = sum(r["verdict"] == "HIT" for r in rows); misses = sum(r["verdict"] == "MISS" for r in rows)
    world = "H" if all(r["verdict"] == "HIT" for r in rows if r["id"] in ("P1", "P2", "P3")) and not any(r["verdict"] == "MISS" for r in rows if r["id"] == "P6") else (
        "H-minus" if all(r["verdict"] == "HIT" for r in rows if r["id"] in ("P1", "P2", "P3")) else "D")
    lines = ["# Drift readback (A_DRIFT_READBACK_PROTOCOL.md, sections 2 and 3)", "",
             f"Source: `{Path(drift_json)}` ({len(runs)} runs; rates {', '.join(f'{lr:g}' for lr in lrs)}). Predictions: {hits} HIT, {misses} MISS, "
             f"{len(rows) - hits - misses} NOT-RUN. World by section 4: **{world}** (H: P1 to P3 HIT and P6 in band; H-minus: P6 out of band; D: a P1 to P3 MISS).", "",
             "| # | prediction | verdict | value | band |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['prediction']} | {r['verdict']} | {json.dumps(r['value'])} | {r['band']} |")
    lines += ["", "| # | check | verdict | value |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['id']} | {c['check']} | {c['verdict']} | {json.dumps(c['value'])} |")
    lines += ["", "## Bracket values for the T10 TeX (section 1 of the protocol)", "", "| bracket | value |", "|---|---|"]
    for lr in lrs:
        s = S[key[lr]]
        lines.append(f"| t* at eta = {lr:g} | {s['t_star_observed']['mean']} +- {s['t_star_observed']['sd']} (n = {s['t_star_observed']['n']}) |")
        lines.append(f"| time-to-floor at eta = {lr:g} | {s['time_to_floor']['mean']} +- {s['time_to_floor']['sd']} (n = {s['time_to_floor']['n']}) |")
        lines.append(f"| accuracy crossover at eta = {lr:g} | {s['accuracy_crossover']['mean']} (n = {s['accuracy_crossover']['n']}); lag over t*: {s['crossover_minus_tstar']['mean']} |")
        lines.append(f"| pre-floor slope at eta = {lr:g} | {s['slope_prefloor']['mean']} +- {s['slope_prefloor']['sd']} over stage windows {s['window_stages']}; in units of gamma0: {s['slope_prefloor_over_gamma0']['mean']} |")
    lines.append(f"| gamma0 | {S[key[lrs[0]]]['gamma0']['mean']} +- {S[key[lrs[0]]]['gamma0']['sd']} (n = {S[key[lrs[0]]]['gamma0']['n']}) |")
    lines.append(f"| log-log slope of abs m against eta | {json.dumps(S.get('loglog_slope_abs_m_vs_lr'))} |")
    lines.append(f"| secants at the largest rate | {json.dumps(S[key[lr_max]]['secants'])} |")
    lines += ["", "Accepted values (ledger only, never printed as regenerated): " + json.dumps(ACCEPTED_DRIFT)]
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True); out_path.write_text("\n".join(lines) + "\n")
    log(f"[drift-readback] {hits} HIT / {misses} MISS; world {world}; wrote {out_path}")
    return dict(predictions=rows, checks=checks, world=world)
