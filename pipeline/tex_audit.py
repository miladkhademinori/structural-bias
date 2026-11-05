"""pipeline.tex_audit -- the mechanical claim ledger: every number printed in the papers' tables against the
number this repository regenerates.

  python replicate.py audit-tex --paper A --tex path/to/paper_A/neurips_2026.tex [--results results]
  python replicate.py audit-tex --paper B --tex path/to/paper_B/neurips_2026.tex

Reads the LaTeX source, parses every ``\\begin{table}`` block that carries a ``\\label{tab:...}``, cleans the
cells (``$``, ``\\phantom``, ``\\mathbf``, ``\\underline``, ``^\\dagger``, ``\\pm``, intervals), maps row and
column labels to the registry's backbone and dataset keys, and looks the matching quantity up in
``results/paperA_results.json`` / ``results/paperB_results.json`` (and ``paperA_drift.json``).  Writes
``results/TEX_AUDIT_<paper>.md`` and ``.json`` with one row per printed cell: printed value, regenerated
value, the difference, whether the printed value sits inside the regenerated mean +- 2 sd (or the
regenerated CI), and a note.  Tables without a mapping are still listed, so the ledger is complete; inline
claims (the pooled slope, the 0.998, the 90.5 percent, the cosine floor) are matched by regular
expressions and reported the same way.

The ledger is the accepted-versus-camera-ready claim register for numeric content: it says, cell by cell,
which printed numbers the released code reproduces and which it does not.  It never edits the TeX.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pipeline import registry as R

NUM = r"[-+]?\d+(?:\.\d+)?"


# ---------------------------------------------------------------------------
# Cell cleaning and parsing
# ---------------------------------------------------------------------------

def _strip_cmd(s: str, cmd: str) -> str:
    """Replace \\cmd{inner} by inner (one level of braces)."""
    pat = re.compile(r"\\" + cmd + r"\s*\{([^{}]*)\}")
    while True:
        s2 = pat.sub(r"\1", s)
        if s2 == s:
            return s
        s = s2


def clean_cell(c: str) -> str:
    c = c.strip()
    c = re.sub(r"%.*$", "", c)
    c = re.sub(r"\\phantom\{[^{}]*\}", "", c)
    for cmd in ("mathbf", "underline", "textbf", "emph", "text", "mathrm", "scriptsize", "small"):
        c = _strip_cmd(c, cmd)
    c = re.sub(r"\\cite[tp]?\{[^{}]*\}", "", c)
    c = re.sub(r"\\multirow\{[^{}]*\}\{[^{}]*\}\{([^{}]*)\}", r"\1", c)
    c = re.sub(r"\\multicolumn\{[^{}]*\}\{[^{}]*\}\{([^{}]*)\}", r"\1", c)
    c = c.replace("\\,", "").replace("\\;", " ").replace("\\!", "").replace("~", " ").replace("\\%", "%")
    c = re.sub(r"\^\{?\\dagger\}?", "", c)
    c = re.sub(r"\{\\pm\}|\\pm", "±", c)
    c = re.sub(r"\{,\}", ",", c)
    c = c.replace("$", "").replace("{", "").replace("}", "")
    c = re.sub(r"\s+", " ", c).strip()
    return c


def parse_value(c: str) -> dict:
    """'8.3 ± 0.4' -> mean/sd; '[-0.179, -0.103]' -> lo/hi; '-0.312 [-0.358, -0.267]' -> value + ci;
    '12,000' or '2,000' -> 2000; '81.42' -> value; anything else -> text."""
    s = c.replace("−", "-")
    m = re.fullmatch(rf"({NUM})\s*±\s*({NUM})", s)
    if m:
        return dict(mean=float(m.group(1)), sd=float(m.group(2)))
    m = re.fullmatch(rf"({NUM})\s*\[\s*({NUM})\s*,\s*({NUM})\s*\]", s)
    if m:
        return dict(mean=float(m.group(1)), ci=[float(m.group(2)), float(m.group(3))])
    m = re.fullmatch(rf"\[\s*({NUM})\s*,\s*({NUM})\s*\]", s)
    if m:
        return dict(ci=[float(m.group(1)), float(m.group(2))])
    m = re.fullmatch(rf"({NUM})\s*%?", s.replace(",", "") if re.fullmatch(r"[\d,]+", s) else s)
    if m:
        return dict(mean=float(m.group(1)))
    m = re.fullmatch(r"10\^(-?\d+)", s)
    if m:
        return dict(mean=10.0 ** int(m.group(1)))
    return dict(text=s)


def parse_tables(tex: str) -> Dict[str, dict]:
    out = {}
    for m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, flags=re.S):
        block = m.group(1)
        lab = re.search(r"\\label\{(tab:[^}]+)\}", block)
        tab = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", block, flags=re.S)
        if not lab or not tab:
            continue
        body = tab.group(1)
        body = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", body)
        rows: List[List[str]] = []
        for raw in re.split(r"\\\\", body):
            raw = raw.strip()
            raw = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("%"))
            raw = re.sub(r"\\(toprule|midrule|bottomrule|hline|addlinespace)", "", raw).strip()
            if not raw:
                continue
            cells = [clean_cell(c) for c in raw.split("&")]
            if all(c == "" for c in cells):
                continue
            rows.append(cells)
        if not rows:
            continue
        # carry a multirow first cell down
        last = ""
        for r in rows:
            if r[0] == "" and last:
                r[0] = last
            elif r[0]:
                last = r[0]
        cap = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\\begin)", block, flags=re.S)
        # the header is the widest of the first three rows (a \multicolumn spanner row above it is narrower)
        hi = max(range(min(3, len(rows))), key=lambda i: (len([c for c in rows[i] if c]), -i))
        out[lab.group(1)] = dict(label=lab.group(1), header=rows[hi], rows=rows[hi + 1:], line=tex[:m.start()].count("\n") + 1,
                                 caption=(cap.group(1)[:200] if cap else ""))
    return out


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


BACKBONE_ALIASES = {
    "vitb16in21k": "vit_in21k", "vitin21k": "vit_in21k", "vitb16": "vit_in21k", "vitb16imagenet21k": "vit_in21k",
    "dinov2vitb14": "dinov2_b14", "dinov2b14": "dinov2_b14", "dinov2": "dinov2_b14",
    "clipvitb16": "clip_b16", "clipb16": "clip_b16", "clip": "clip_b16",
    "maevitb16": "mae_b16", "maeb16": "mae_b16", "mae": "mae_b16",
    "resnet50": "resnet50", "supervisedresnet50": "resnet50",
    "randomvitb16": "random_vit", "randomvit": "random_vit", "random": "random_vit",
}
DATASET_ALIASES = {
    "cifar100": "cifar100", "splitcifar100": "cifar100", "imagenetr": "imagenet_r", "cub200": "cub200", "cub2002011": "cub200",
    "omnibenchmark": "omnibenchmark", "imagenet100": "imagenet100", "imagenet1k": "imagenet1k", "domainnet": "domainnet",
}


def backbone_key(label: str) -> Optional[str]:
    n = _norm(label)
    if n in BACKBONE_ALIASES:
        return BACKBONE_ALIASES[n]
    for k, v in R.BACKBONES.items():
        if _norm(v.label) == n or _norm(k) == n:
            return k
    return None


def dataset_key(label: str) -> Optional[str]:
    n = _norm(label)
    if n in DATASET_ALIASES:
        return DATASET_ALIASES[n]
    for k, v in R.DATASETS.items():
        if _norm(v.label) == n or _norm(k) == n:
            return k
    return None


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _row(table, cell, printed: dict, repro: Optional[dict], note="") -> dict:
    pm = printed.get("mean"); rm = (repro or {}).get("mean")
    delta = (rm - pm) if (pm is not None and rm is not None) else None
    within = None
    if pm is not None and repro:
        if repro.get("ci"):
            within = bool(repro["ci"][0] <= pm <= repro["ci"][1])
        elif repro.get("sd") is not None and rm is not None:
            within = bool(abs(pm - rm) <= 2 * max(repro["sd"], 1e-12))
    return dict(table=table, cell=cell, printed=printed, reproduced=repro, delta=delta, within=within, note=note)


def _ms(v: Optional[dict], scale=1.0, nd=None) -> Optional[dict]:
    if not v or v.get("mean") is None:
        return None
    out = dict(mean=v["mean"] * scale)
    if v.get("sd") is not None:
        out["sd"] = v["sd"] * scale
    return out


def _fmt(v: Optional[dict]) -> str:
    if not v:
        return "--"
    if "text" in v:
        return v["text"]
    s = ""
    if v.get("mean") is not None:
        s = f"{v['mean']:.4g}"
        if v.get("sd") is not None:
            s += f" ± {v['sd']:.2g}"
    if v.get("ci"):
        s += f" [{v['ci'][0]:.4g}, {v['ci'][1]:.4g}]"
    return s or "--"


# ---------------------------------------------------------------------------
# Paper A audits
# ---------------------------------------------------------------------------

def audit_A(tables: Dict[str, dict], A: Optional[dict], D: Optional[dict]) -> List[dict]:
    rows: List[dict] = []
    A = A or {}
    t = tables.get("tab:gamma-min")
    if t:
        dsets = [dataset_key(h) for h in t["header"][1:]]
        for r in t["rows"]:
            b = backbone_key(r[0])
            for d, c in zip(dsets, r[1:]):
                g = A.get("gamma", {}).get(f"{b}|{d}") if b and d else None
                rows.append(_row("tab:gamma-min", f"{b}/{d}", parse_value(c), _ms(g["gamma_min"]) if g else None,
                                 note=("eq. 1 per-task-cov variant " + _fmt(_ms(g.get("gamma_min_eq1")))) if g and g.get("gamma_min_eq1") else ""))
    t = tables.get("tab:residual-check")
    if t:
        for r in t["rows"]:
            if len(r) < 5:
                continue
            b, d = backbone_key(r[0]), dataset_key(r[1]); g = A.get("gamma", {}).get(f"{b}|{d}") if b and d else None
            for col, key in ((2, "gamma_min"), (3, "rhs"), (4, "nu")):
                note = ""
                if key == "nu" and g and g.get("nu_pooled") is not None:
                    note = f"isotropic metric (eq:nu-def); pooled-metric nu {g['nu_pooled']['mean']:.3f}" if isinstance(g.get("nu_pooled"), dict) else ""
                if key == "nu" and g and isinstance(g.get("condition_S"), dict) and g["condition_S"].get("delta_over_gamma"):
                    note += f"; delta/gamma {g['condition_S']['delta_over_gamma']['mean']:.3f}; clause holds {g.get('condition_S_clause_holds', {}).get('clause_holds')}"
                rows.append(_row("tab:residual-check", f"{b}/{d} {key}", parse_value(r[col]), _ms(g.get(key)) if g else None, note))
    t = tables.get("tab:random-anchor")
    if t:
        for r in t["rows"]:
            d = dataset_key(r[0]); a = A.get("anchors", {}).get(d) if d else None
            for col, key in ((1, "gamma_min"), (2, "tc_gap"), (3, "log_tc_gap")):
                if col < len(r):
                    rows.append(_row("tab:random-anchor", f"random_vit/{d} {key}", parse_value(r[col]), _ms(a.get(key)) if a else None))
    t = tables.get("tab:slope-fits")
    if t:
        s = A.get("slopes") or {}
        for r in t["rows"]:
            if len(r) < 4:
                continue
            name = r[0]; b = backbone_key(name)
            f = s.get("per_backbone", {}).get(b) if b else (s.get("pooled16") if "16" in name or "anchor" in name.lower() else s.get("pooled12") if "pooled" in name.lower() else None)
            cell = b or ("pooled16" if ("16" in name or "anchor" in name.lower()) else "pooled12")
            rep = dict(mean=f["slope"], ci=f.get("ci95")) if f and f.get("slope") is not None else None
            rows.append(_row("tab:slope-fits", f"{cell} slope", parse_value(r[1]), rep))
            rows.append(_row("tab:slope-fits", f"{cell} CI", parse_value(r[2]), dict(ci=f["ci95"]) if f and f.get("ci95") else None))
            rows.append(_row("tab:slope-fits", f"{cell} R2", parse_value(r[3]), dict(mean=f["r2"]) if f and f.get("r2") is not None else None))
    t = tables.get("tab:ecf-rates")
    if t:
        arm = None
        for r in t["rows"]:
            if len(r) == 1 or (len(r) >= 1 and ("clipped" in r[0].lower() and len([c for c in r if c]) == 1)):
                arm = "unclipped" if "unclipped" in r[0].lower() else "clipped"
                continue
            b = backbone_key(r[0]); e = (A.get("ecf") or {}).get(b) if b else None
            res = e["arms"].get(arm) if e and arm else None
            Ks = [int(re.sub(r"[^\d]", "", h)) for h in t["header"][1:-1] if re.sub(r"[^\d]", "", h)]
            for K, c in zip(Ks, r[1:1 + len(Ks)]):
                rows.append(_row("tab:ecf-rates", f"{b} {arm} K={K}", parse_value(c), _ms((res or {}).get("E_CF", {}).get(str(K))) if res else None))
            if len(r) > 1 + len(Ks):
                f = (res or {}).get("fit") or {}
                rows.append(_row("tab:ecf-rates", f"{b} {arm} slope", parse_value(r[1 + len(Ks)]), dict(mean=f["slope"], ci=f.get("ci95")) if f.get("slope") is not None else None))
    t = tables.get("tab:alpha-fits")
    if t:
        for r in t["rows"]:
            b = backbone_key(r[0]); a = (A.get("alpha") or {}).get(b) if b else None
            rep = dict(mean=a["alpha"]["mean"], ci=a.get("ci")) if a else None
            rows.append(_row("tab:alpha-fits", f"{b} alpha", parse_value(r[1]), rep, note="Hill k=%s; finite-k bias not corrected" % (a.get("k") if a else "?")))
            if len(r) > 2:
                rows.append(_row("tab:alpha-fits", f"{b} CI", parse_value(r[2]), dict(ci=a["ci"]) if a and a.get("ci") else None))
    for lab in ("tab:main", "tab:headline-comparison"):
        t = tables.get(lab)
        if not t:
            continue
        dsets = [dataset_key(h) for h in t["header"][1:]]
        mmap = {"ours": "ours", "diagnosticguidedclippedsgdours": "ours", "sequentialft": "sequential_head", "sequentialftlower": "sequential_head",
                "jointtraining": "joint", "jointtrainingupper": "joint", "slda": "slda", "ncm": "ncm", "fecam": "fecam", "ranpac": "ranpac",
                "withintasksoftmax": "within_task", "withintask": "within_task"}
        for r in t["rows"]:
            key = mmap.get(_norm(r[0]))
            if key is None:
                for d, c in zip(dsets, r[1:]):
                    rows.append(_row(lab, f"{r[0]} / {d}", parse_value(c), None, note="published-method row: not reproduced here (D4)"))
                continue
            for d, c in zip(dsets, r[1:]):
                mt = None
                for b in R.PAPER_A_BACKBONES:
                    m_ = (A.get("method") or {}).get(f"{b}|{d}")
                    if m_ and m_.get(key):
                        mt = m_[key]; bb = b; break
                rows.append(_row(lab, f"{key} / {d}", parse_value(c), _ms(mt, 100.0) if mt else None, note=(f"backbone {bb}" if mt else "")))
    t = tables.get("tab:drift-slopes")
    if t and D:
        summ = D.get("summary") or {}
        for r in t["rows"]:
            m = re.fullmatch(r"10\^\{?(-?\d+)\}?|1e(-?\d+)", r[0].replace(" ", ""))
            if not m:
                continue
            lr = 10.0 ** int(m.group(1) or m.group(2)); s = None
            for k, v in summ.items():
                try:
                    if abs(float(k) - lr) / lr < 1e-6:
                        s = v
                except Exception:
                    pass
            if s is None:
                continue
            rows.append(_row("tab:drift-slopes", f"lr {lr:g} slope", parse_value(r[1]), _ms(s.get("slope_prefloor")), note=f"window stages {s.get('window_stages')}"))
            if len(r) > 2:
                rows.append(_row("tab:drift-slopes", f"lr {lr:g} t_star", parse_value(r[2]), _ms(s.get("t_star_observed")), note=f"linear {_fmt(_ms(s.get('t_star_linear')))}"))
    # generic listing of unmapped tables
    mapped = {"tab:gamma-min", "tab:residual-check", "tab:random-anchor", "tab:slope-fits", "tab:ecf-rates", "tab:alpha-fits", "tab:main", "tab:headline-comparison", "tab:drift-slopes"}
    for lab, t in tables.items():
        if lab in mapped:
            continue
        for r in t["rows"]:
            for i, c in enumerate(r[1:], 1):
                pv = parse_value(c)
                if "text" not in pv:
                    rows.append(_row(lab, f"{r[0]} / col {i}", pv, None, note="no mapping"))
    return rows


# ---------------------------------------------------------------------------
# Paper B audits
# ---------------------------------------------------------------------------

def audit_B(tables: Dict[str, dict], B: Optional[dict]) -> List[dict]:
    rows: List[dict] = []
    B = B or {}
    t = tables.get("tab:fix-benchmark")
    if t:
        for r in t["rows"]:
            if len(r) < 5:
                continue
            b, d = backbone_key(r[0]), dataset_key(r[1]); bench = (B.get("benchmark") or {}).get(f"{b}|{d}") if b and d else None
            for col, key in ((2, "WT"), (3, "FIX"), (4, "J")):
                note = ""
                if key == "FIX" and bench and bench.get("FIX_c"):
                    note = f"FIX_c {bench['FIX_c']['mean']*100:.2f}; head scale {(bench.get('head_scale') or {}).get('mean')}"
                rows.append(_row("tab:fix-benchmark", f"{b}/{d} {key}", parse_value(r[col]), _ms(bench.get(key), 100.0) if bench else None, note))
    t = tables.get("tab:lda-ncm-comparison")
    if t:
        for r in t["rows"]:
            if len(r) < 6:
                continue
            b, d = backbone_key(r[0]), dataset_key(r[1]); bench = (B.get("benchmark") or {}).get(f"{b}|{d}") if b and d else None
            for col, key in ((2, "WT"), (3, "FIX"), (4, "LDA"), (5, "NCM")):
                rows.append(_row("tab:lda-ncm-comparison", f"{b}/{d} {key}", parse_value(r[col]), _ms(bench.get(key), 100.0) if bench else None,
                                 note=("owner cell owed (--)" if parse_value(r[col]).get("text") == "--" else "")))
    t = tables.get("tab:bias-direction")
    if t:
        for r in t["rows"]:
            if len(r) < 5:
                continue
            d, b = dataset_key(r[0]), backbone_key(r[1]); bd = (B.get("bias_direction") or {}).get(f"{b}|{d}") if b and d else None
            for col, key, rkey in ((2, "cos", "pair_cos_mean"), (3, "rel", "pair_rel_mean"), (4, "CV", "cv")):
                note = ""
                if bd and key == "cos":
                    note = f"block-mean (paper's averaged) cos {bd['paper_averaged_cos']['mean']:.4f} is the gauge identity; per-pair reported"
                rows.append(_row("tab:bias-direction", f"{b}/{d} {key}", parse_value(r[col]), _ms(bd.get(rkey)) if bd else None, note))
    t = tables.get("tab:rate-fits")
    if t:
        for r in t["rows"]:
            if len(r) < 6:
                continue
            b = backbone_key(r[0]); f = (B.get("rate_fits") or {}).get(b) if b else None
            rows.append(_row("tab:rate-fits", f"{b} kappa", parse_value(r[1]), dict(mean=f["kappa_median"]) if f else None, note="median kappa_min over pairs"))
            rows.append(_row("tab:rate-fits", f"{b} predicted slope", parse_value(r[2]), dict(mean=f["predicted_slope"]) if f and f.get("predicted_slope") is not None else None))
            rows.append(_row("tab:rate-fits", f"{b} empirical slope", parse_value(r[3]), dict(mean=f["slope"], ci=f.get("ci95")) if f else None))
            rows.append(_row("tab:rate-fits", f"{b} R2", parse_value(r[4]), dict(mean=f["r2"]) if f else None))
            rows.append(_row("tab:rate-fits", f"{b} n", parse_value(r[5]), dict(mean=f["n_pairs"]) if f else None, note=f"pairs with |P|=0 excluded: {f.get('n_pairs_zero')}" if f else ""))
    t = tables.get("tab:weights-only")
    if t:
        wo = B.get("weights_only") or {}
        amap = {"wtweightsonlydecay": "WT_wo", "fixdefinitionunmodified": "FIX_wo", "fixhatctscalarablation": "FIX_wo_chat", "fixctscalarablation": "FIX_wo_chat", "jjoint": "J"}
        for r in t["rows"]:
            key = None
            for k, v in amap.items():
                if _norm(r[0]).startswith(k[:12]):
                    key = v
            rows.append(_row("tab:weights-only", f"{r[0]} acc", parse_value(r[1]), _ms(wo.get(key), 100.0) if key and wo else None,
                             note=("closed-form c_hat arm " + _fmt(_ms(wo.get("FIX_wo_chat_closed"), 100.0))) if key == "FIX_wo_chat" and wo else ""))
    mapped = {"tab:fix-benchmark", "tab:lda-ncm-comparison", "tab:bias-direction", "tab:rate-fits", "tab:weights-only"}
    for lab, t in tables.items():
        if lab in mapped:
            continue
        note = "published-method row: not reproduced here (D4)" if lab in ("tab:diagnostic", "tab:kappa-distribution") else ("nonvision: experiments/nonvision.py (notebook 05)" if lab == "tab:nonvision" else "no mapping")
        for r in t["rows"]:
            for i, c in enumerate(r[1:], 1):
                pv = parse_value(c)
                if "text" not in pv:
                    rows.append(_row(lab, f"{r[0]} / col {i}", pv, None, note=note))
    return rows


# ---------------------------------------------------------------------------
# Inline claims
# ---------------------------------------------------------------------------

INLINE_A = [
    ("pooled slope", r"[Pp]ooled[^$]{0,80}\$\\mathbf\{(-0\.\d+)\}|yields slope \$(-0\.\d+)\$", lambda A: ((A.get("slopes") or {}).get("pooled12") or {}).get("slope")),
    ("r = 0.998 (drift)", r"r\s*=\s*\+?(0\.9\d+)", lambda A: None),
    ("ten of twelve satisfy nu <= 1/2", r"(ten) of (?:the )?twelve", lambda A: None),
    ("alpha_xi bound 1/16", r"\\tfrac\{1\}\{16\}", lambda A: None),
]
INLINE_B = [
    ("gap recovery mean 90.5", r"mean \$(9\d\.\d)\\%\$", lambda B: np.mean([v["recovery"]["mean"] for v in (B.get("benchmark") or {}).values() if v.get("recovery", {}).get("mean") is not None]) * 100 if B.get("benchmark") else None),
    ("gap recovery range 82-94", r"recovers \$(8\d)\$--\$(9\d)\\%\$", lambda B: None),
    ("cosine at least 0.92", r"cosine at least \$(0\.9\d)\$", lambda B: min([v["pair_cos_mean"]["mean"] for v in (B.get("bias_direction") or {}).values()]) if B.get("bias_direction") else None),
    ("nonvision gain +7.30", r"\+(7\.\d\d)\$? points", lambda B: None),
    ("nonvision cosine 0.93", r"cosine \$(0\.9\d)\$, cross-pair CV", lambda B: None),
    ("slopes within 5 percent", r"within \$(\d)\\%\$", lambda B: max([f.get("slope_error_pct") or 0 for f in (B.get("rate_fits") or {}).values()]) if B.get("rate_fits") else None),
]


def audit_inline(tex: str, claims, results) -> List[dict]:
    rows = []
    for name, pat, fn in claims:
        ms = list(re.finditer(pat, tex))
        printed = []
        for m in ms[:6]:
            vals = [g for g in m.groups() if g]
            printed.append(" ".join(vals) if vals else m.group(0))
        try:
            rep = fn(results or {})
        except Exception:
            rep = None
        rows.append(dict(table="inline", cell=name, printed=dict(text=", ".join(dict.fromkeys(printed)) if printed else "not found"),
                         reproduced=(dict(mean=float(rep)) if isinstance(rep, (int, float)) and rep is not None and np.isfinite(rep) else None),
                         delta=None, within=None, note=f"{len(ms)} occurrence(s)"))
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def render(rows: List[dict], paper: str, tex_path: Path, results_dir: Path) -> str:
    n = len(rows); comp = [r for r in rows if r["reproduced"] is not None and r["printed"].get("mean") is not None]
    inside = sum(1 for r in comp if r["within"]); outside = sum(1 for r in comp if r["within"] is False)
    L = [f"# TeX audit, Paper {paper}", "", f"Source: `{tex_path}`; results: `{results_dir}`.", "",
         f"{n} printed numbers parsed; {len(comp)} have a regenerated counterpart; {inside} printed values sit inside the regenerated "
         f"mean ± 2 sd or CI, {outside} do not, {len(comp) - inside - outside} have no spread to judge by.", "",
         "Provenance of the regenerated numbers is in the results JSON (`provenance`: real or synthetic). A synthetic provenance means the "
         "comparison exercises the ledger, not the paper.", "",
         "| table | cell | printed | reproduced | delta | inside | note |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        d = f"{r['delta']:+.4g}" if r["delta"] is not None else ""
        w = "" if r["within"] is None else ("yes" if r["within"] else "**no**")
        L.append(f"| {r['table']} | {r['cell']} | {_fmt(r['printed'])} | {_fmt(r['reproduced'])} | {d} | {w} | {r['note']} |")
    return "\n".join(L) + "\n"


def run(paper: str, tex_path: Path, results_dir: Path, out_dir: Optional[Path] = None, log=print) -> Path:
    tex = tex_path.read_text(encoding="utf-8", errors="replace")
    tables = parse_tables(tex)
    out_dir = out_dir or results_dir
    if paper.upper() == "A":
        pa = results_dir / "paperA_results.json"; pd_ = results_dir / "paperA_drift.json"
        A = json.loads(pa.read_text(encoding="utf-8")) if pa.exists() else None
        D = json.loads(pd_.read_text(encoding="utf-8")) if pd_.exists() else None
        rows = audit_A(tables, A, D) + audit_inline(tex, INLINE_A, A)
    else:
        pb = results_dir / "paperB_results.json"
        B = json.loads(pb.read_text(encoding="utf-8")) if pb.exists() else None
        rows = audit_B(tables, B) + audit_inline(tex, INLINE_B, B)
    md = render(rows, paper.upper(), tex_path, results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"TEX_AUDIT_{paper.upper()}.md"; p.write_text(md, encoding="utf-8")
    (out_dir / f"TEX_AUDIT_{paper.upper()}.json").write_text(json.dumps(dict(tex=str(tex_path), tables=sorted(tables), rows=rows), indent=1, default=float), encoding="utf-8")
    log(md.splitlines()[4]); log(f"[audit-tex] wrote {p}")
    return p
