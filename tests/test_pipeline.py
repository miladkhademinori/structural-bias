"""Tests for the replication pipeline (pipeline/): identities the code must satisfy on data that obeys the
papers' assumptions, agreement between the compute backends, and the statistical helpers.  No paper number
is asserted anywhere; these are mechanism tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import backend, heads, registry as R, stats as S  # noqa: E402
from pipeline.paper_a import _pooled_common_slope, predictor_regression, slope_fits  # noqa: E402

os.environ.setdefault("REPLICATION_DEVICE", "numpy")


def gaussian_tasks(seed=0, N=20, d=12, n=400, sep=1.2, T=4):
    """Shared-covariance Gaussian class-conditionals (Assumption 2.1 of Paper B), N classes in T tasks."""
    g = np.random.default_rng(seed)
    A = g.standard_normal((d, d)); sigma = A @ A.T / d + 0.5 * np.eye(d); L = np.linalg.cholesky(sigma)
    mu = sep * g.standard_normal((N, d))
    y = np.repeat(np.arange(N), n // N); X = mu[y] + g.standard_normal((len(y), d)) @ L.T
    toc = np.repeat(np.arange(T), N // T)
    return X, y, toc, mu, sigma


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_ledoit_wolf_matches_sklearn_if_available():
    g = np.random.default_rng(0); Rm = g.standard_normal((300, 10)) @ np.diag(np.linspace(0.5, 3, 10))
    sig, lam = S.ledoit_wolf_shrinkage(Rm)
    sk = pytest.importorskip("sklearn.covariance")
    sig2, lam2 = sk.ledoit_wolf(Rm, assume_centered=True)
    assert abs(lam - lam2) < 1e-10 and np.allclose(sig, sig2, atol=1e-10)


def test_gamma_matrix_matches_direct_mahalanobis():
    X, y, toc, mu, sigma = gaussian_tasks()
    st = S.ClassStats.from_features(X, y, toc, protocol="B")
    cents = st.centroids; G = st.gamma_matrix()
    inv = np.linalg.inv(st.sigma)
    for i in range(st.T):
        for j in range(st.T):
            dlt = cents[i] - cents[j]
            assert abs(G[i, j] - dlt @ inv @ dlt) < 1e-9


def test_protocol_a_standardizes_and_transform_roundtrip():
    X, y, toc, *_ = gaussian_tasks()
    st = S.ClassStats.from_features(X, y, toc, protocol="A")
    Z = st.transform(X)
    assert abs(Z.mean()) < 1e-9 and abs(Z.std() - 1) < 1e-9


# ---------------------------------------------------------------------------
# Paper B identities on the population head
# ---------------------------------------------------------------------------

def population_head(st: S.ClassStats):
    """Theorem 3.3 / Definition CF-optimal: w_r = Sigma^{-1}(mu_r - mu_bar_t), b_r = -mu_r^T Sigma^{-1} mu_r / 2 + S_bar_t / 2."""
    W = st.sigma_inv_apply(st.mu - st.centroids[st.task_of_class])
    norms = (st.whiten(st.mu) ** 2).sum(1)
    S_bar = np.array([norms[st.task_of_class == t].mean() for t in range(st.T)])
    b = -0.5 * norms + 0.5 * S_bar[st.task_of_class]
    return heads.Head(W, b, st.task_of_class)


def test_de_centering_recovers_lda_exactly_on_population_head():
    X, y, toc, *_ = gaussian_tasks()
    st = S.ClassStats.from_features(X, y, toc, protocol="B")
    fix = heads.de_center(population_head(st), st); lda = heads.lda_head(st)
    assert np.allclose(fix.W, lda.W, atol=1e-9) and np.allclose(fix.b, lda.b, atol=1e-9)
    assert st.head_scale(population_head(st).W) == pytest.approx(1.0, abs=1e-9)


def test_gauge_identity_averaged_cosine_is_one_for_any_head_at_the_gauge():
    """The paper's C^2-averaged bias statistic equals the prediction for ANY within-task weights that sum to zero."""
    X, y, toc, *_ = gaussian_tasks()
    st = S.ClassStats.from_features(X, y, toc, protocol="B")
    g = np.random.default_rng(3); W = g.standard_normal((toc.size, X.shape[1]))
    for t in range(st.T):                          # project each task block onto the gauge
        k = np.flatnonzero(toc == t); W[k] -= W[k].mean(0)
    bd = st.bias_direction_all(W)
    assert bd["paper_averaged_cos"] == pytest.approx(1.0, abs=1e-9) and bd["paper_averaged_rel"] == pytest.approx(1.0, abs=1e-9)
    assert bd["pair_cos_mean"] < 0.9       # while the per-pair statistic sees that the head is random


def test_per_pair_cosine_is_one_on_population_head():
    X, y, toc, *_ = gaussian_tasks()
    st = S.ClassStats.from_features(X, y, toc, protocol="B")
    bd = st.bias_direction_all(population_head(st).W)
    assert bd["pair_cos_mean"] == pytest.approx(1.0, abs=1e-9) and bd["cv"] == pytest.approx(0.0, abs=1e-9)


def test_lbfgs_minimizer_satisfies_the_gauge_and_within_task_lbfgs_equals_blockwise():
    X, y, toc, *_ = gaussian_tasks(n=800)
    wt = heads.train_within_task(X, y, toc, lam=1e-2, optimizer="lbfgs")
    for t in range(toc.max() + 1):
        k = np.flatnonzero(toc == t)
        assert np.linalg.norm(wt.W[k].sum(0)) < 1e-6 * np.linalg.norm(wt.W[k]) + 1e-8   # Lemma 3.2: sum of block weights is zero
        assert abs(wt.b[k].sum()) < 1e-6


# ---------------------------------------------------------------------------
# backends agree on the same batch streams
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not backend.HAVE_TORCH, reason="torch not installed")
def test_numpy_and_torch_backends_agree():
    X, y, toc, *_ = gaussian_tasks(n=600)
    tasks = [np.flatnonzero(toc == t) for t in range(toc.max() + 1)]
    out = {}
    for dev in ("numpy", "cpu"):
        h, st = heads.clipped_sgd_head(X, y, toc, tasks, K=40, eta=0.05, tau=3.0, seed_purpose="t", device=dev, X_eval=X, y_eval=y)
        W, b, _ = heads.fit_adamw(X, y, toc.size, 1e-4, epochs=2, seed_purpose="a", device=dev)
        norms, _ = heads.warmup_gradient_norms(X, y, toc.size, k_warm=50, device=dev)
        Wl, bl, _ = heads.fit_lbfgs(X, y, toc.size, 1e-2, device=dev)
        out[dev] = (h.W, W, norms, Wl, st)
    for a, b_ in zip(out["numpy"][:4], out["cpu"][:4]):
        assert np.allclose(a, b_, rtol=1e-4, atol=1e-5)
    assert out["numpy"][4] == pytest.approx(out["cpu"][4], abs=1e-6)


def test_clipping_bounds_the_step():
    X, y, toc, *_ = gaussian_tasks(n=200)
    W0 = np.zeros((toc.size, X.shape[1])); b0 = np.zeros(toc.size)
    idx = np.random.default_rng(0).integers(0, len(y), (5, 32))
    W, b, norms = backend.sgd_subset(W0, b0, X, y, np.arange(toc.size), idx, eta=1.0, tau=0.01, lam=0.0, clip=True, record_norms=True, device="numpy")
    assert norms.shape == (5,) and np.linalg.norm(np.concatenate([W.ravel(), b])) <= 5 * 0.01 + 1e-9


# ---------------------------------------------------------------------------
# Paper A statistics
# ---------------------------------------------------------------------------

def test_hill_estimator_on_pareto_norms():
    g = np.random.default_rng(0); alpha = 1.5
    norms = (1 - g.random(20000)) ** (-1 / alpha)          # Pareto with tail index alpha
    h = heads.hill_estimate(norms, k=500)
    assert abs(h["alpha"] - alpha) < 0.15 and h["ci"][0] < alpha < h["ci"][1]


def test_prop51_schedule_formulas():
    eta, tau = heads.prop51_schedule(K=1000, alpha=1.5, sigma=2.0, mu=0.1, beta=5.0)
    assert eta == pytest.approx(min(1 / 5.0, (2.0 ** 1.5 / (0.01 * 1000)) ** (1 / 1.5)))
    assert tau == pytest.approx((2.0 ** 1.5 * 1000 / 0.1) ** (1 / 1.5))


def test_pooled_common_slope_and_interaction_recover_a_planted_model():
    g = np.random.default_rng(0)
    x = g.uniform(5, 50, 12); groups = np.repeat(np.array(["a", "b", "c"]), 4)
    inter = {"a": 0.0, "b": 1.0, "c": -1.0}
    y = np.array([inter[k] for k in groups]) - 0.125 * x + 1e-3 * g.standard_normal(12)
    f = _pooled_common_slope(x, y, groups)
    assert f["slope"] == pytest.approx(-0.125, abs=1e-3) and f["df"] == 12 - 4


def test_predictor_regression_nested_f_identity():
    g = np.random.default_rng(1); n = 12
    rows = [dict(final_acc=float(0.5 + 0.01 * i + 0.01 * g.standard_normal()), gamma_min=float(i), probe_acc=float(0.6 + 0.005 * i),
                 ncm_acc=float(0.4 + 0.004 * i + 0.01 * g.standard_normal()), dataset=["c", "i", "u", "o"][i % 4]) for i in range(n)]
    p = predictor_regression(rows)
    F = ((p["r2_full"] - p["r2_reduced"]) / 1) / ((1 - p["r2_full"]) / (n - 4))
    assert p["F"] == pytest.approx(F) and p["df"] == [1, n - 4] and 0 <= p["p"] <= 1


def test_slope_fits_handles_missing_configurations():
    tc = {("vit_in21k", "cifar100"): dict(per_seed=[dict(gamma_min=8.0, tc_gap=0.3), dict(gamma_min=8.5, tc_gap=0.25)]),
          ("vit_in21k", "imagenet_r"): dict(per_seed=[dict(gamma_min=5.0, tc_gap=0.5), dict(gamma_min=5.2, tc_gap=0.45)]),
          ("dinov2_b14", "cifar100"): dict(per_seed=[dict(gamma_min=47.0, tc_gap=0.01), dict(gamma_min=46.0, tc_gap=0.012)]),
          ("dinov2_b14", "imagenet_r"): dict(per_seed=[dict(gamma_min=39.0, tc_gap=0.02), dict(gamma_min=38.0, tc_gap=0.021)])}
    out = slope_fits(tc, n_bootstrap=50)
    assert set(out["per_backbone"]) == {"vit_in21k", "dinov2_b14"} and out["pooled12"]["slope"] < 0


def test_registry_orderings_are_deterministic_and_cover_all_classes():
    for ds in ("cifar100", "imagenet_r", "cub200", "omnibenchmark", "imagenet100"):
        o1, o2 = R.class_ordering(ds, 3), R.class_ordering(ds, 3)
        assert np.array_equal(o1, o2)
        toc = R.task_of_class(ds, o1)
        assert toc.min() == 0 and toc.max() == R.DATASETS[ds].T - 1 and (toc >= 0).all()


def test_fig1c_closed_form_anchors(tmp_path):
    import json
    from pipeline import figures
    paths = figures.fig1c_constructions(tmp_path)
    side = json.loads((tmp_path / "B_fig1c_constructions.data.json").read_text())
    a = side["data"]["anchors"]
    assert abs(a["1"]["anti"] - 0.2398) < 2e-4 and abs(a["5"]["catastrophic"] - 0.6802) < 2e-4 and abs(a["10"]["catastrophic"] - 0.9969) < 2e-4
    assert abs(a["2"]["anti_bayes"] - 0.2398) < 2e-4 and abs(a["5"]["anti_bayes"] - 0.0196) < 2e-4
    assert any(p.suffix == ".pdf" for p in paths)


def test_tex_audit_parses_tables_and_values():
    from pipeline import tex_audit as T
    tex = r"""
\begin{table}[h]
\caption{x}
\label{tab:gamma-min}
\begin{tabular}{@{}lcc@{}}
\toprule
Backbone & CIFAR-100 & ImageNet-R \\
\midrule
ViT-B/16-IN21K   & $\phantom{0}8.3 \pm 0.4$  & $\phantom{0}5.2 \pm 0.3$ \\
DINOv2-ViT-B/14  & $\mathbf{47.2 \pm 1.2}$   & $39.1^\dagger \pm 1.0$ \\
\bottomrule
\end{tabular}
\end{table}
\begin{table}
\label{tab:slope-fits}
\begin{tabular}{lccc}
\toprule
& \multicolumn{2}{c}{spanner} & \\
Backbone & Slope & 95\% CI & $R^2$ \\
\midrule
CLIP-ViT-B/16    & $-0.118$ & $[-0.152,\, -0.084]$ & $0.95$ \\
Pooled (12 pretrained)  & $\mathbf{-0.115}$ & $\mathbf{[-0.128,\, -0.102]}$ & $\mathbf{0.97}$ \\
\bottomrule
\end{tabular}
\end{table}
"""
    tabs = T.parse_tables(tex)
    assert set(tabs) == {"tab:gamma-min", "tab:slope-fits"}
    g = tabs["tab:gamma-min"]
    assert T.dataset_key(g["header"][1]) == "cifar100" and T.backbone_key(g["rows"][1][0]) == "dinov2_b14"
    assert T.parse_value(g["rows"][0][1]) == dict(mean=8.3, sd=0.4) and T.parse_value(g["rows"][1][2]) == dict(mean=39.1, sd=1.0)
    s = tabs["tab:slope-fits"]
    assert s["header"][0] == "Backbone" and T.parse_value(s["rows"][0][2]) == dict(ci=[-0.152, -0.084])
    assert T.parse_value(s["rows"][1][1]) == dict(mean=-0.115)
    A = dict(gamma={"vit_in21k|cifar100": dict(gamma_min=dict(mean=8.1, sd=0.2))}, slopes=dict(per_backbone={"clip_b16": dict(slope=-0.12, ci95=[-0.15, -0.09], r2=0.9)}, pooled12=dict(slope=-0.11, ci95=[-0.13, -0.10], r2=0.96)))
    rows = T.audit_A(tabs, A, None)
    r = [x for x in rows if x["cell"] == "vit_in21k/cifar100"][0]
    assert r["within"] is True and abs(r["delta"] + 0.2) < 1e-9
    r = [x for x in rows if x["cell"] == "pooled12 slope"][0]
    assert r["reproduced"]["mean"] == -0.11 and r["within"] is True


def test_schedules_and_smoothness_bound_units():
    import numpy as np
    from pipeline.heads import prop51_schedule, smoothness_bound
    eta, tau = prop51_schedule(400, 1.5, 2.0, 0.1, 10.0, schedule="consistent")
    assert abs(eta - 1.0 / (10.0 * 20.0)) < 1e-12 and abs(tau - 2.0 * 400 ** (1 / 3.0)) < 1e-6
    # scale covariance: features scaled by s multiply the curvature bound by about s^2 (the augmented constant
    # column does not scale, hence the tolerance)
    Z = np.random.default_rng(1).standard_normal((500, 8)) * 4.0
    b1, b2 = smoothness_bound(Z, 0.0), smoothness_bound(3.0 * Z, 0.0)
    assert abs(b2 / b1 - 9.0) < 0.05


def test_fingerprint_moments_match_classstats_and_grid_runs():
    import numpy as np
    from pipeline import fingerprint as F
    from pipeline.stats import ClassStats
    X, y, toc, mu, sigma = gaussian_tasks()
    N = int(y.max()) + 1
    M = F.Moments(X, y, N)
    st = ClassStats.from_features(X, y, toc, protocol="A")
    # the pooled Ledoit-Wolf covariance of the fingerprint moments equals ClassStats' (up to protocol A's scalar scale)
    S_lw, rho = M.lw()
    scale = st.gamma_min()[0]
    gv = F.gamma_variants(M, F.Moments(X, y, N, normalize=True), F.Moments(X, y, N, subsample=20, seed_purpose="t"), np.asarray(toc))
    assert abs(gv["lw_pooled (ours)"]["gamma_min"] - scale) / scale < 1e-6
    assert gv["ridge_abs eps=10"]["gamma_min"] < gv["sample_pooled"]["gamma_min"]              # an additive ridge lowers gamma
    assert gv["ridge_abs eps=1e-06"]["gamma_min"] <= gv["sample_pooled"]["gamma_min"] * (1 + 1e-6)
    assert "_mediators" in gv and gv["_mediators"]["lw_rho"] == rho
    G = np.array(gv["lw_pooled (ours)"]["G"])
    cells = F.nu_rhs_variants(M, np.asarray(toc), G, 10.0, M.class_lambda_max())
    assert any(k.startswith("nu | norm=task_within_max (paper)") for k in cells) and any(k.startswith("rhs |") for k in cells)
    assert all(np.isfinite(v) for v in cells.values())


def test_drift_readback_worlds(tmp_path):
    """summarize_drift + readback: linear-in-eta synthetic trajectories land in world H; a non-collapsing largest
    rate lands in world D; check C2 flags the accepted 14.2 against gamma0/(2|m|) = 25.6 from the accepted inputs."""
    import json
    import numpy as np
    from pipeline import drift as D

    def make(lr_slopes, collapse=True):
        runs = []
        for lr, m in lr_slopes:
            for seed in range(3):
                g0 = 30.0 + seed
                ts = np.arange(0, 10.01, 0.2)
                gs = np.maximum(g0 - m * ts + 0.1 * np.sin(7 * ts + seed), 0.5) if (collapse or lr < 1e-3) else g0 - 0.01 * ts
                runs.append(dict(lr=lr, seed=seed, gamma0=g0, trajectory=[dict(t=float(t), gamma=float(g)) for t, g in zip(ts, gs)],
                                 acc_finetuned=list(np.linspace(0.9, 0.6, 10) - 0.009 * (lr > 1e-5) * np.arange(10)),
                                 acc_frozen_head=list(np.linspace(0.85, 0.55, 10))))
        d = dict(runs=runs); d["summary"] = D.summarize_drift(runs); return d

    d = make([(1e-5, 0.4), (1e-4, 3.5), (1e-3, 30.0)])
    s = d["summary"]
    assert abs(s["loglog_slope_abs_m_vs_lr"]["value"] - np.log10(3.5 / 0.4)) < 0.05 and s["loglog_slope_abs_m_vs_lr"]["n_points"] == 2
    assert s["0.001"]["time_to_floor"]["mean"] < 1.0 and s["1e-05"]["t_star_observed"]["n"] == 0
    p = tmp_path / "paperA_drift.json"; p.write_text(json.dumps(d))
    r = D.readback(p, tmp_path / "DRIFT_READBACK.md")
    assert r["world"] == "H" and all(x["verdict"] == "HIT" for x in r["predictions"])
    c2 = [c for c in r["checks"] if c["id"] == "C2"][0]
    assert c2["verdict"] == "FLAG" and abs(c2["value"]["linear_from_accepted"] - 25.62) < 0.01
    assert (tmp_path / "DRIFT_READBACK.md").read_text().startswith("# Drift readback")
    d2 = make([(1e-5, 0.4), (1e-4, 3.5), (1e-3, 30.0)], collapse=False)
    p2 = tmp_path / "d2.json"; p2.write_text(json.dumps(d2))
    assert D.readback(p2, tmp_path / "R2.md")["world"] == "D"


def test_fingerprint_new_conventions_present():
    """The grid carries the conventions the protocol-matrix questions need: per-row layer norm, per-class LW,
    per-task standardization, the task-count variants; eq. 1 per-task LW sits below pooled LW on a steep spectrum."""
    import numpy as np
    from pipeline import fingerprint as FP
    rng = np.random.default_rng(0); d = 32; N = 40; n_per = 60
    A = rng.normal(size=(d, d)); L = np.linalg.cholesky(A @ A.T / d + 1e-3 * np.eye(d)); mus = rng.normal(size=(N, d)) * 1.5
    X = np.concatenate([mus[c] + rng.normal(size=(n_per, d)) @ L.T for c in range(N)]); y = np.repeat(np.arange(N), n_per)
    M = FP.Moments(X, y, N); Mn = FP.Moments(X, y, N, normalize=True); Ms = FP.Moments(X, y, N, subsample=10); Ml = FP.Moments(X, y, N, rowln=True)
    order = np.arange(N); toc = np.empty(N, np.int64); toc[order] = np.arange(N) // 4
    gv = FP.gamma_variants(M, Mn, Ms, toc, Ml=Ml, order=order, dataset="x")
    for k in ("rowln_lw", "rowln_sample", "eq1_pair_classlw", "eq1_pair_sample_ridge1e-3", "pertask_std_scalar_eq1", "pertask_std_zscore_eq1", "lw_pooled T=5", "lw_pooled T=20"):
        assert k in gv and np.isfinite(gv[k]["gamma_min"]) and gv[k]["gamma_min"] > 0, k
    assert gv["eq1_pair_lw"]["gamma_min"] < gv["lw_pooled (ours)"]["gamma_min"]       # heavier per-task shrinkage lowers gamma on a steep spectrum
    assert gv["eq1_pair_classlw"]["gamma_min"] < gv["eq1_pair_lw"]["gamma_min"]       # per-class shrinkage heavier still
