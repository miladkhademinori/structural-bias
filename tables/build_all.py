"""tables.build_all -- G08 tables layer: a thin, tested wrapper over pipeline.tables.

Enumerates the results store (``results/runs`` + ``results/INDEX.json`` plus any
``paperA_results.json`` / ``paperB_results.json`` payloads under ``results/``),
classifies every live table of Papers A and B per the G08 S1/COUNTS facts
(STATIC / POST-HOC / RUN-FED), calls the matching ``pipeline.tables`` emitter for
each table whose inputs the store can serve, and writes

    tables/out/<paper>/<label>.tex            tabular body (pipeline.tables output)
    tables/out/<paper>/<label>.caption_tags.txt  verbatim caption strings + population tag
    tables/out/<paper>/<label>.meta.json      provenance, inputs, gates, checksum
    tables/out/DIFF.md                        printed-vs-reproduced diff (pipeline.tables.diff_report)
    tables/out/BUILD_REPORT.txt               the full audit printed to stdout

NO table math lives here: every number is rendered by ``pipeline.tables``; this
module only decides which tables the store can serve and records why the rest
are blocked.  Synthetic-provenance payloads render with the pipeline watermark
(``% PROVENANCE ...: NOT A PAPER NUMBER``) and never without it.

Classification rules (G08 S1/S6/S7):
  STATIC   emits always (fixed rows verbatim).
  POST-HOC emits when the payload carries the needed section, at any
           provenance (synthetic previews are watermarked, not blocked).
  RUN-FED  emits ONLY when the payload section exists at real-run provenance
           (real/colab/owner-artifact feature caches); otherwise the table is
           reported FED-BLOCKED with the reason "inputs MISSING".
  A POST-HOC table that needs per-pair records the store does not carry
  (tab:sens-tasks, G08 S7.3) is reported GATED, never silently dropped.

Exit codes (fleet convention, see experiments/lda_ncm_eval.py):
  0  every live table regenerated, FED-BLOCKED, or GATED with a recorded
     reason, and no gate failed;
  1  store regression: a real-provenance payload is missing a POST-HOC
     section it is supposed to carry;
  2  GATE-FAIL: a numeric gate evaluated on real-run data failed (predicted
     exit path on real inputs; the synthetic store never trips it).

Spec source: exec/G08_log.md (campaign root, read-only).  Tests:
tests/test_tables.py (agent W6-8).  Owner: U6-1, 2026-10-21.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import tables as PT  # noqa: E402  (thin wrapper target; no local table math)

EXIT_OK, EXIT_STORE, EXIT_GATE = 0, 1, 2

#: provenance values that may feed RUN-FED tables (G08 S2 vocabulary plus the
#: pipeline feature-cache value "real" written by pipeline.paper_a/paper_b).
REAL_PROVENANCE = frozenset({"real", "colab", "owner-artifact"})

WATERMARK_NOTE = "synthetic preview: NOT A PAPER NUMBER"

# ---------------------------------------------------------------------------
# G08 S5 caption-population tags (verbatim strings; written per fragment)
# ---------------------------------------------------------------------------

TAG_A_SEEDS = "5 random class orderings (seeds 0--4)"
TAG_A_BOOTSTRAP = "1{,}000 resamples"
TAG_A_POOLED12 = "12 pretrained configurations"
TAG_A_ANCHORS = "combined with the 12 pretrained configurations"
TAG_A_ANCHOR_POINTS = "13th--16th data point"
TAG_A_THEORY = "-1/8 = -0.125 operational"
TAG_A_THEORY_CONS = "-1/16 = -0.0625 conservative"
TAG_A_MAIN_BOLD = "within one standard deviation of the column maximum"
TAG_A_HEADLINE_BOLD = "Bold: best; underlined: best non-bold method"
TAG_B_STRESS = ("shared-label domains-as-tasks runs (domains sharing one "
                "$345$-class label set) that violate the disjoint-label-set assumption")
TAG_B_BOLD = "bold marks the de-centered head"
TAG_B_PREREG = "aggregates as pre-registered on 2 Aug 2026"
TAG_B_PAIRS = "$N = 200$ cross-task pairs"
TAG_B_LDA_ESTIMATOR = ("SLDA-style discriminant vs Sigma = I specialization "
                       "(caption already fixed)")

#: Paper A tab:diagnostic, Table 11: STATIC symptom table, rows verbatim from
#: the working TeX (app:diagnostic-table), zero data.
A_DIAGNOSTIC_ROWS = (
    ("Poor accuracy with ample data", "$E_{\\mathrm{TC}}$",
     "Upgrade pretrained backbone (raise $\\gamma_{\\min}$)"),
    ("Poor accuracy with strong $\\gamma_{\\min}$", "$E_{\\mathrm{CF}}$",
     "More iterations $K_t$, stronger clipping, smaller $\\beta$"),
    ("Heavy-tailed gradients ($\\alpha$ small)", "$E_{\\mathrm{CF}}$",
     "Tighter clipping threshold $\\tau_t$"),
    ("Long task sequence ($T$ large)", "both",
     "Backbone with larger $\\gamma_{\\min}$; shorter task chunks"),
)


class TableSpec:
    """One live table environment of Papers A/B (G08 S1 enumeration)."""

    __slots__ = ("paper", "label", "cls", "emitter", "needs", "tags", "gates", "blocked_reason")

    def __init__(self, paper, label, cls, emitter, needs, tags, gates=(), blocked_reason=""):
        self.paper = paper
        self.label = label
        self.cls = cls              # STATIC | POST-HOC | RUN-FED
        self.emitter = emitter      # pipeline.tables callable name, "static", or None
        self.needs = needs          # payload sections required
        self.tags = tags            # S5 tag constants (written verbatim)
        self.gates = gates          # named gates (G08 S7.4) recorded in meta
        self.blocked_reason = blocked_reason


#: The 19 live tables in float order (G08 S1/COUNTS; spec facts, not measured).
MANIFEST: List[TableSpec] = [
    TableSpec("A", "tab:headline-comparison", "RUN-FED", None, ("method",),
              (TAG_A_SEEDS, TAG_A_HEADLINE_BOLD), ("owner-fed subset of tab:main",),
              "inputs MISSING: owner/colab benchmark runs (A/benchmark/...); no local emitter (subset of tab:main)"),
    TableSpec("A", "tab:gamma-min", "POST-HOC", "tab_gamma_min", ("gamma",),
              (TAG_A_SEEDS,), ("8.3 vs 8.20 drift baseline (owner reconciliation)",)),
    TableSpec("A", "tab:random-anchor", "POST-HOC", "tab_random_anchor", ("gamma", "anchors"),
              (TAG_A_SEEDS, TAG_A_ANCHORS, TAG_A_ANCHOR_POINTS), ()),
    TableSpec("A", "tab:residual-check", "POST-HOC", "tab_residual_check", ("gamma",),
              (TAG_A_SEEDS,), ()),
    TableSpec("A", "tab:slope-fits", "POST-HOC", "tab_slope_fits", ("slopes",),
              (TAG_A_BOOTSTRAP, TAG_A_POOLED12, TAG_A_THEORY, TAG_A_THEORY_CONS),
              ("ratify -0.115 canonical vs -0.119 (owner ruling)",)),
    TableSpec("A", "tab:ecf-rates", "RUN-FED", "tab_ecf_rates", ("ecf",),
              (TAG_A_BOOTSTRAP,), (),
              "inputs MISSING: real per-K convergence-run logs (A/ecf/...)"),
    TableSpec("A", "tab:alpha-fits", "RUN-FED", "tab_alpha_fits", ("alpha",),
              (TAG_A_BOOTSTRAP,), (),
              "inputs MISSING: real gradient-norm logs (A/alpha/...)"),
    TableSpec("A", "tab:drift-slopes", "RUN-FED", None, ("drift",),
              (TAG_A_SEEDS,), ("8.3 vs 8.20 drift baseline (owner reconciliation)",),
              "inputs MISSING: real eta_phi stage trajectories (A/drift/...); pipeline.drift is GPU-run upstream"),
    TableSpec("A", "tab:main", "RUN-FED", "tab_method_comparison", ("method",),
              (TAG_A_SEEDS, TAG_A_MAIN_BOLD), (),
              "inputs MISSING: real feature caches / benchmark runs (A/benchmark/...)"),
    TableSpec("A", "tab:sens-tasks", "POST-HOC", None, ("gamma_pairs",),
              (TAG_A_POOLED12,), (),
              "needs per-pair gamma_pairs records (G08 S7.3); no payload in the store carries them; "
              "stays gated until the owner logs ship per-pair granularity"),
    TableSpec("A", "tab:diagnostic", "STATIC", "static", (), (), ()),
    TableSpec("B", "tab:fix-benchmark", "RUN-FED", "tab_fix_benchmark", ("benchmark",),
              (TAG_A_SEEDS, TAG_B_STRESS, TAG_B_BOLD), (),
              "inputs MISSING: trained WT/J heads from real runs (B/fix_benchmark/...)"),
    TableSpec("B", "tab:diagnostic", "RUN-FED", None, ("method_diag",),
              (TAG_B_BOLD,), (),
              "inputs MISSING: trained artifacts of the five published methods (B/method_diag/...)"),
    TableSpec("B", "tab:lda-ncm-comparison", "POST-HOC", "tab_lda_ncm", ("benchmark",),
              (TAG_A_SEEDS, TAG_B_STRESS, TAG_B_BOLD, TAG_B_LDA_ESTIMATOR), ()),
    TableSpec("B", "tab:kappa-distribution", "RUN-FED", "tab_kappa", ("kappa",),
              (TAG_B_PAIRS,), (),
              "inputs MISSING: per-pair kappa-hat distributions from method artifacts (B/method_diag/...)"),
    TableSpec("B", "tab:rate-fits", "POST-HOC", "tab_rate_fits", ("rate_fits",),
              (TAG_B_PAIRS, TAG_B_STRESS), ("ImageNet-R purge + DomainNet aggregation (owner ruling)",)),
    TableSpec("B", "tab:bias-direction", "POST-HOC", "tab_bias_direction", ("bias_direction",),
              (TAG_B_PAIRS, TAG_B_STRESS), ()),
    TableSpec("B", "tab:weights-only", "RUN-FED", "weights_only_text", ("weights_only",),
              (TAG_A_SEEDS, TAG_B_PREREG), ("two SD cells: FIX+c-hat 85.72, J 85.93 (owner)",),
              "inputs MISSING: pre-registered weights-only control runs (B/weights_only/...)"),
    TableSpec("B", "tab:nonvision", "RUN-FED", None, ("nonvision",),
              (TAG_B_PREREG,), (),
              "inputs MISSING: owner Newsgroups/MiniLM artifact (B/nonvision/...; results/nonvision_per_seed.csv gated)"),
]

#: run-record experiment slugs that support a given table label (store audit).
EXPERIMENT_TO_TABLES = {
    ("A", "slope_fits"): ("tab:slope-fits",),
    ("A", "drift"): ("tab:drift-slopes",),
    ("A", "predictor_regression"): ("predictor",),
    ("A", "synthetic_margin"): (),
    ("A", "falsification"): (),
    ("B", "lda_ncm"): ("tab:lda-ncm-comparison",),
    ("B", "nonvision"): ("tab:nonvision",),
    ("B", "diagnostic_controls"): ("tab:diagnostic",),
    ("B", "synthetic_bias"): ("tab:bias-direction",),
    ("B", "synthetic_rate"): ("tab:rate-fits",),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _emit_static_diagnostic() -> str:
    lines = ["% tab:diagnostic (Paper A, Table 11) -- STATIC per G08 S1: fixed rows emitted verbatim"]
    lines += [f"{s} & {t} & {r} \\\\" for (s, t, r) in A_DIAGNOSTIC_ROWS]
    return "\n".join(lines)


def _emit_predictor(A: dict) -> str:
    """pipeline.tables.predictor_text is a comment fragment, not a table env."""
    return PT.predictor_text(A)


EMITTERS = {
    "tab_gamma_min": PT.tab_gamma_min,
    "tab_random_anchor": PT.tab_random_anchor,
    "tab_residual_check": PT.tab_residual_check,
    "tab_slope_fits": PT.tab_slope_fits,
    "tab_ecf_rates": PT.tab_ecf_rates,
    "tab_alpha_fits": PT.tab_alpha_fits,
    "tab_method_comparison": PT.tab_method_comparison,
    "tab_fix_benchmark": PT.tab_fix_benchmark,
    "tab_lda_ncm": PT.tab_lda_ncm,
    "tab_bias_direction": PT.tab_bias_direction,
    "tab_rate_fits": PT.tab_rate_fits,
    "tab_kappa": PT.tab_kappa,
    "weights_only_text": PT.weights_only_text,
    "static": _emit_static_diagnostic,
}


def _frag_stem(spec: TableSpec) -> str:
    """Filesystem-safe fragment stem: tab:gamma-min -> tab_gamma-min."""
    return spec.label.replace(":", "_")


# ---------------------------------------------------------------------------
# store enumeration
# ---------------------------------------------------------------------------

def enumerate_store(results_root: Path) -> dict:
    """Read results/runs/**/*.json + results/INDEX.json + payload files.

    Returns {index_runs, disk_runs, dangling_index, orphan_disks,
    by_experiment, payloads, warnings}.
    """
    results_root = Path(results_root)
    index_path = results_root / "INDEX.json"
    index_runs: List[dict] = []
    if index_path.is_file():
        try:
            index_runs = json.loads(index_path.read_text(encoding="utf-8")).get("runs", [])
        except (OSError, ValueError) as exc:
            index_runs = []
            print(f"[store] WARNING: INDEX.json unreadable ({exc})")
    index_ids = {r.get("run_id") for r in index_runs}

    disk: List[dict] = []
    for p in sorted((results_root / "runs").rglob("*.json")) if (results_root / "runs").is_dir() else []:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"[store] WARNING: unreadable run JSON {p}")
            continue
        rec["_path"] = str(p.relative_to(results_root)).replace("\\", "/")
        disk.append(rec)
    disk_ids = {r.get("run_id") for r in disk}

    dangling = sorted(i for i in index_ids - disk_ids if i)
    orphans = sorted(disk_ids - index_ids)

    by_experiment: Dict[Tuple[str, str], List[dict]] = {}
    for rec in disk:
        by_experiment.setdefault((rec.get("paper", "?"), rec.get("experiment", "?")), []).append(rec)

    payloads = {}
    for name, paper in (("paperA_results.json", "A"), ("paperB_results.json", "B")):
        found = [p for p in sorted(results_root.rglob(name), key=lambda q: (q.stat().st_mtime, str(q)))]
        if not found:
            continue
        real = [p for p in found if _payload_is_real(p)]
        chosen = real[-1] if real else found[-1]  # newest real payload, else newest
        try:
            payloads[paper] = dict(path=chosen, data=json.loads(chosen.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            print(f"[store] WARNING: payload {chosen} unreadable ({exc})")

    warnings = []
    if dangling:
        warnings.append(f"{len(dangling)} INDEX entries without a run JSON: {dangling}")
    if orphans:
        warnings.append(f"{len(orphans)} run JSONs not listed in INDEX: {orphans}")
    return dict(index_runs=index_runs, disk_runs=disk, dangling_index=dangling,
                orphan_disks=orphans, by_experiment=by_experiment,
                payloads=payloads, warnings=warnings)


def _payload_is_real(path: Path) -> bool:
    try:
        prov = json.loads(path.read_text(encoding="utf-8")).get("provenance", [])
    except (OSError, ValueError):
        return False
    return bool(prov) and set(prov) <= REAL_PROVENANCE


def _provenance_of(payload: dict) -> List[str]:
    prov = payload.get("provenance", [])
    return list(prov) if isinstance(prov, list) else [str(prov)]


def _population_line(paper: str, payload: Optional[dict], results_root: Path, spec: TableSpec) -> str:
    if spec.cls == "STATIC":
        return "population: fixed text, zero data (G08 S1 STATIC)"
    if payload is None:
        return "population: (no payload)"
    rel = str(payload["path"].relative_to(Path(results_root))).replace("\\", "/")
    prov = _provenance_of(payload["data"])
    bits = [f"payload {rel}", f"provenance {prov}"]
    if paper == "A":
        bits.append(f"{len(payload['data'].get('configs', []))} of 12 roster configurations")
        n_anch = len(payload["data"].get("anchors", {}))
        if n_anch:
            bits.append(f"{n_anch} of 4 random anchors")
    else:
        bits.append(f"{len(payload['data'].get('configs', []))} of 10 benchmark rows covered")
    if not set(prov) <= REAL_PROVENANCE:
        bits.append(WATERMARK_NOTE)
    return "population: " + "; ".join(bits)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def evaluate_gates(spec: TableSpec, payload: Optional[dict]) -> List[dict]:
    """Named gates per G08 S7.4.  Local numeric evaluation happens only on
    real-run data; owner-ruling gates are recorded, never failed locally."""
    out = []
    real = payload is not None and set(_provenance_of(payload["data"])) <= REAL_PROVENANCE
    for name in spec.gates:
        if name.startswith("owner") or "owner" in name:
            out.append(dict(name=name, status="pending-owner-ruling", detail="outside the local store"))
        else:
            out.append(dict(name=name, status="not-evaluated" if not real else "pass",
                            detail="" if real else WATERMARK_NOTE))
    # the one local numeric gate: predictor cells (G08 S3 / N23 S2)
    if spec.label == "predictor":
        p = (payload or {}).get("data", {}).get("predictor")
        if p is None:
            out.append(dict(name="predictor R2/F cells (N23 S2)", status="not-evaluated",
                            detail="no predictor block in payload"))
        elif not real:
            out.append(dict(name="predictor R2/F cells (N23 S2)", status="not-evaluated",
                            detail=WATERMARK_NOTE))
        else:
            dr2 = round(p["r2_full"] - p["r2_reduced"], 2)
            ok_dr2 = dr2 == 0.06
            ok_p = p.get("p") is not None and 0.040 <= p["p"] <= 0.044
            out.append(dict(name="predictor R2/F cells (N23 S2)",
                            status="pass" if (ok_dr2 and ok_p) else "fail",
                            detail=f"delta_r2 {dr2:.2f} (want 0.06), p {p.get('p')} (want [0.040, 0.044])"))
    return out


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(argv: Optional[List[str]] = None, paper: str = "both") -> int:
    ap = argparse.ArgumentParser(
        description="Build Paper A/B table fragments from the results store "
                    "(thin wrapper over pipeline.tables; G08 spec).")
    ap.add_argument("--paper", choices=("A", "B", "both"), default=paper,
                    help="which paper's tables (default: both)")
    ap.add_argument("--check", action="store_true",
                    help="audit only: print the report, write nothing")
    ap.add_argument("--only", default="",
                    help="comma-separated tab: labels to restrict to")
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "tables" / "out")
    args = ap.parse_args(argv)

    papers = ("A", "B") if args.paper == "both" else (args.paper,)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    specs = [s for s in MANIFEST if s.paper in papers and (not only or s.label in only)]
    if only:
        unknown = only - {s.label for s in MANIFEST}
        if unknown:
            print(f"unknown labels in --only (ignored): {sorted(unknown)}")

    store = enumerate_store(args.results_root)

    # ---- store audit lines -------------------------------------------------
    report_lines: List[str] = []

    def say(s: str = "") -> None:
        print(s)
        report_lines.append(s)

    n_idx, n_disk = len(store["index_runs"]), len(store["disk_runs"])
    say(f"[store] results root {args.results_root}")
    say(f"[store] INDEX lists {n_idx} runs; {n_disk} run JSONs under runs/")
    for w in store["warnings"]:
        say(f"[store] WARNING: {w}")
    exps = sorted(f"{p}:{e}({len(rs)})" for (p, e), rs in store["by_experiment"].items())
    say(f"[store] experiments: {' '.join(exps) if exps else '(none)'}")
    for p in papers:
        pay = store["payloads"].get(p)
        if pay:
            rel = str(pay["path"].relative_to(args.results_root)).replace("\\", "/")
            say(f"[store] {p} payload {rel} provenance {_provenance_of(pay['data'])} "
                f"configs {len(pay['data'].get('configs', []))} "
                f"missing {len(pay['data'].get('missing', []))}")
        else:
            say(f"[store] {p} payload: NONE (no paper{p}_results.json under results/)")

    regenerated, blocked, gated = [], [], []
    fragments: List[Tuple[TableSpec, str, dict]] = []   # spec, tex, meta
    regressions: List[str] = []
    gate_failures: List[str] = []

    for spec in specs:
        payload = store["payloads"].get(spec.paper)
        data = payload["data"] if payload else None
        sections_ok = data is not None and all(data.get(n) for n in spec.needs)
        prov_real = payload is not None and set(_provenance_of(data)) <= REAL_PROVENANCE
        support = sorted(r.get("run_id") for r in store["disk_runs"]
                         if (spec.paper, r.get("experiment")) in EXPERIMENT_TO_TABLES
                         and spec.label in EXPERIMENT_TO_TABLES[(spec.paper, r.get("experiment"))])

        if spec.cls == "STATIC":
            tex = EMITTERS["static"]()
            meta = dict(g08_class=spec.cls, emitter="static rows (G08 S1)", provenance=["static"],
                        supporting_runs=support, gates=evaluate_gates(spec, None))
            regenerated.append((spec, _frag_stem(spec), meta)); fragments.append((spec, tex, meta))
        elif spec.cls == "RUN-FED":
            if spec.emitter is None:
                blocked.append((spec, spec.blocked_reason))
            elif sections_ok and prov_real:
                tex = EMITTERS[spec.emitter](data)
                meta = dict(g08_class=spec.cls, emitter=f"pipeline.tables.{spec.emitter}",
                            provenance=_provenance_of(data), supporting_runs=support,
                            gates=evaluate_gates(spec, payload))
                regenerated.append((spec, _frag_stem(spec), meta)); fragments.append((spec, tex, meta))
            else:
                reason = spec.blocked_reason
                if sections_ok and not prov_real:
                    reason += "; synthetic payload present but not eligible for a RUN-FED table"
                blocked.append((spec, reason))
        else:  # POST-HOC
            if sections_ok:
                tex = EMITTERS[spec.emitter](data)
                meta = dict(g08_class=spec.cls, emitter=f"pipeline.tables.{spec.emitter}",
                            provenance=_provenance_of(data), supporting_runs=support,
                            gates=evaluate_gates(spec, payload))
                regenerated.append((spec, _frag_stem(spec), meta)); fragments.append((spec, tex, meta))
            elif spec.emitter is None:
                gated.append((spec, spec.blocked_reason))
            elif data is not None and prov_real:
                regressions.append(f"{spec.paper} {spec.label}: real payload missing section(s) "
                                   f"{[n for n in spec.needs if not data.get(n)]}")
                blocked.append((spec, "inputs MISSING: real payload lost a required section (store regression)"))
            else:
                gated.append((spec, f"payload section(s) {[n for n in spec.needs]} absent from every store payload"))

    # predictor text fragment (Paper A Section 8 cells; not a live table env)
    predictor_frag = None
    if "A" in papers and not only:
        payload = store["payloads"].get("A")
        if payload and payload["data"].get("predictor"):
            gates = evaluate_gates(TableSpec("A", "predictor", "POST-HOC", None, ("predictor",), ()), payload)
            gate_failures += [f"predictor: {g['name']} {g['detail']}" for g in gates if g["status"] == "fail"]
            ci1 = [r.get("run_id") for r in store["disk_runs"]
                   if r.get("experiment") == "predictor_regression"]
            meta = dict(g08_class="POST-HOC text fragment (Section 8, four R2/F cells)",
                        emitter="pipeline.tables.predictor_text",
                        provenance=_provenance_of(payload["data"]),
                        supporting_runs=ci1,
                        store_note="real predictor inputs MISSING per A_predictor_regression_CI1 "
                                   "(suite_status INPUTS_MISSING); this fragment is the payload preview",
                        gates=gates)
            predictor_frag = (_emit_predictor(payload["data"]), meta)

    gate_failures += [f"{s.paper} {s.label}: {g['name']} {g['detail']}"
                      for (s, _, m) in fragments for g in m["gates"] if g["status"] == "fail"]

    # ---- write (unless --check; a failed gate writes nothing, G08 S3) -------
    written: List[Path] = []
    if args.check:
        pass
    elif gate_failures:
        say("writes suppressed: a numeric gate failed on real-run data (G08 S3: write nothing, report)")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for p in papers:
            (args.out_dir / p).mkdir(parents=True, exist_ok=True)
        for (spec, tex, meta) in fragments:
            d = args.out_dir / spec.paper
            stem = _frag_stem(spec)
            (d / f"{stem}.tex").write_text(tex + "\n", encoding="utf-8")
            tag_lines = [t for t in spec.tags] + [_population_line(spec.paper, store["payloads"].get(spec.paper), args.results_root, spec)]
            (d / f"{stem}.caption_tags.txt").write_text("\n".join(tag_lines) + "\n", encoding="utf-8")
            meta.update(label=spec.label, paper=spec.paper, created_utc=_utcnow(),
                        tex_sha256=_sha256_text(tex), mode="build")
            (d / f"{stem}.meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
            written += [d / f"{stem}.tex", d / f"{stem}.caption_tags.txt", d / f"{stem}.meta.json"]
        if predictor_frag is not None:
            tex, meta = predictor_frag
            (args.out_dir / "A" / "predictor.tex").write_text(tex + "\n", encoding="utf-8")
            meta.update(label="predictor", paper="A", created_utc=_utcnow(),
                        tex_sha256=_sha256_text(tex), mode="build")
            (args.out_dir / "A" / "predictor.meta.json").write_text(
                json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
            written += [args.out_dir / "A" / "predictor.tex", args.out_dir / "A" / "predictor.meta.json"]
        # DIFF.md via pipeline.tables.diff_report, with a provenance banner
        A = store["payloads"]["A"]["data"] if "A" in store["payloads"] else None
        B = store["payloads"]["B"]["data"] if "B" in store["payloads"] else None
        banner = ["<!-- tables/build_all.py via pipeline.tables.diff_report -->",
                  f"<!-- payload provenance A: {_provenance_of(A) if A else None}; "
                  f"B: {_provenance_of(B) if B else None}; synthetic rows are NOT paper numbers -->", ""]
        (args.out_dir / "DIFF.md").write_text("\n".join(banner) + PT.diff_report(A, B) + "\n", encoding="utf-8")
        written.append(args.out_dir / "DIFF.md")
        # sweep: this layer owns tab_*/predictor files only; drop stale ones
        keep = {w.name for w in written}
        for p in papers:
            d = args.out_dir / p
            for pat in ("tab_*.tex", "tab_*.meta.json", "tab_*.caption_tags.txt", "predictor.tex", "predictor.meta.json"):
                for old in d.glob(pat):
                    if old.name not in keep:
                        old.unlink()
                        say(f"swept stale fragment {old.relative_to(args.out_dir)}")

    # ---- report -------------------------------------------------------------
    say()
    for (spec, stem, meta) in regenerated:
        prov = ",".join(meta.get("provenance") or ["static"])
        mark = "" if set(prov.split(",")) <= REAL_PROVENANCE or prov == "static" else f" [{WATERMARK_NOTE}]"
        say(f"{spec.paper}  {spec.label:<26} {spec.cls:<8} REGENERATED  out/{spec.paper}/{stem}.tex "
            f"(provenance {prov}){mark}")
    for (spec, reason) in blocked:
        say(f"{spec.paper}  {spec.label:<26} {spec.cls:<8} FED-BLOCKED  {reason}")
    for (spec, reason) in gated:
        say(f"{spec.paper}  {spec.label:<26} {spec.cls:<8} GATED        {reason}")
    n_pred = 1 if predictor_frag is not None else 0
    final = (f"REGENERATED ({len(regenerated)}) | "
             f"FED-BLOCKED ({len(blocked)}, {' '.join(s.label for s, _ in blocked)}) | "
             f"GATED ({len(gated)}, {' '.join(s.label for s, _ in gated)})")
    say()
    say(final)
    if n_pred:
        say(f"text fragments: {n_pred} (out/A/predictor.tex, Section 8 R2/F cells; not a table env)")
    if args.check:
        say("check mode: no writes")
    else:
        say(f"wrote {len(written)} files under {args.out_dir}")
        (args.out_dir / "BUILD_REPORT.txt").write_text(
            f"# tables build report ({_utcnow()}; argv={' '.join(sys.argv[1:]) or 'default'})\n\n"
            + "\n".join(report_lines) + "\n", encoding="utf-8")
        say(f"report: {args.out_dir / 'BUILD_REPORT.txt'}")
    for r in regressions:
        say(f"STORE REGRESSION: {r}")

    if gate_failures:
        for g in gate_failures:
            print(f"GATE-FAIL {g}")
        return EXIT_GATE
    if regressions:
        return EXIT_STORE
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    return build(argv)


if __name__ == "__main__":
    raise SystemExit(build())
