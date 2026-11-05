#!/usr/bin/env python3
"""replicate.py -- one entry point for the whole replication.

  python replicate.py features  --backbones vit_in21k,dinov2_b14,clip_b16,random_vit --datasets cifar100,imagenet_r,cub200,omnibenchmark
  python replicate.py features  --paper B                 # the Paper B set (adds mae_b16, resnet50; imagenet100 needs --imagenet100-root)
  python replicate.py analyze   --paper A|B|AB [--optimizer adamw|lbfgs] [--seeds 0,1,2,3,4]
  python replicate.py tables                               # LaTeX rows + DIFF.md from results/paper{A,B}_results.json
  python replicate.py condition-s                          # Condition (S) clauses (nu, delta_ij, a_ij) from the feature caches alone
  python replicate.py fingerprint                          # which convention reproduces the printed gamma_min / nu / RHS (caches only)
  python replicate.py features-timm --data-root /content/data --out-root features_timm   # the recovered timm protocol, separate cache root
  python replicate.py figures                              # every figure from results/*.json, each with a .data.json sidecar
  python replicate.py audit-tex --paper A --tex <neurips_2026.tex>   # printed-versus-regenerated ledger, cell by cell
  python replicate.py drift     --data-root /content/data  # Paper A drift (GPU; resumable per (lr, seed))
  python replicate.py drift-readback                        # P1..P8 and C1..C5 scored from paperA_drift.json -> DRIFT_READBACK.md
  python replicate.py nonvision --allow-download           # Paper B 20 Newsgroups full run (Colab)
  python replicate.py synthetic                            # the synthetic verification suites + tolerance scoreboard
  python replicate.py selftest                             # end-to-end on synthetic caches (no GPU, ~minutes)

Feature caches live under $REPLICATION_FEATURES (default ./features); results under ./results.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline import registry as R  # noqa: E402


def _log(msg):
    print(msg, flush=True)


def cmd_features(a):
    from pipeline.features import extract, determinism_check
    dev = a.device if a.device not in (None, "auto", "numpy") else "cuda"
    bbs = a.backbones.split(",") if a.backbones else (list(R.PAPER_A_BACKBONES) + ["random_vit"] if a.paper == "A" else list(R.PAPER_B_BACKBONES))
    dss = a.datasets.split(",") if a.datasets else (list(R.PAPER_A_DATASETS) if a.paper == "A" else list(R.PAPER_B_DATASETS))
    root = Path(a.features)
    for b in bbs:
        _log(json.dumps(determinism_check(b, dev)))
        for d in dss:
            if b == "random_vit" and d not in R.PAPER_A_DATASETS:
                continue
            try:
                extract(b, d, Path(a.data_root), root, batch_size=a.batch, device=dev, imagenet100_root=a.imagenet100_root)
            except Exception as e:  # keep going; the manifest records what is missing
                _log(f"[features] FAILED {b}/{d}: {type(e).__name__}: {e}")
    manifest = {f"{b}|{d}": bool((root / b / d / "train.npz").exists() and (root / b / d / "test.npz").exists()) for b in bbs for d in dss}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1)); _log(json.dumps(manifest, indent=1))


def cmd_analyze(a):
    from pipeline import paper_b
    try:
        from pipeline import paper_a
    except ImportError:  # B-only release: the A driver is not part of this repository
        paper_a = None
    root = Path(a.features); out = Path(a.results); seeds = tuple(int(s) for s in a.seeds.split(","))
    if "A" in a.paper:
        if not hasattr(paper_a, "run_all"):
            raise SystemExit("paper A driver not included in this repository (B-only release)")
        paper_a.run_all(root, out, seeds=seeds, optimizer=a.optimizer, K_method=a.K, skip_ecf=a.skip_ecf, ranpac_M=a.ranpac_M,
                        device=a.device, schedule=a.schedule, log=_log)
    if "B" in a.paper:
        paper_b.run_all(root, out, seeds=seeds, optimizer=a.optimizer, device=a.device, log=_log)
    cmd_tables(a)


def cmd_condition_s(a):
    from pipeline import paper_a
    root = Path(a.features); out = Path(a.results); seeds = tuple(int(s) for s in a.seeds.split(","))
    bb = tuple(a.backbones.split(",")) if a.backbones else R.PAPER_A_BACKBONES
    dd = tuple(a.datasets.split(",")) if a.datasets else R.PAPER_A_DATASETS
    paper_a.condition_S_survey(root, out / "paperA_conditionS.json", seeds=seeds, backbones=bb, datasets=dd, log=_log)


def cmd_figures(a):
    from pipeline import figures
    out = Path(a.results)
    figures.make_all(out, Path(a.out) if a.out else None, Path(a.drift) if a.drift else None, log=_log)


def cmd_audit_tex(a):
    from pipeline import tex_audit
    tex_audit.run(a.paper, Path(a.tex), Path(a.results), Path(a.out) if a.out else None, log=_log)


def cmd_fingerprint(a):
    from pipeline import fingerprint
    root = Path(a.features); out = Path(a.results); seeds = tuple(int(s) for s in a.seeds.split(","))
    bb = tuple(a.backbones.split(",")) if a.backbones else R.PAPER_A_BACKBONES
    dd = tuple(a.datasets.split(",")) if a.datasets else R.PAPER_A_DATASETS
    fingerprint.run(root, out / "FINGERPRINT.md", seeds=seeds, backbones=bb, datasets=dd, subsample=a.subsample, log=_log)


def cmd_features_timm(a):
    from pipeline import features_timm as FT
    out_root = Path(a.out_root); data_root = Path(a.data_root)
    bb = tuple(a.backbones.split(",")) if a.backbones else tuple(list(R.PAPER_A_BACKBONES) + ["random_vit"])
    dd = tuple(a.datasets.split(",")) if a.datasets else R.PAPER_A_DATASETS
    dev = a.device if a.device not in (None, "auto", "numpy") else "cuda"
    for b in bb:
        _log(json.dumps(FT.determinism_check_timm(b, dev)))
        for d in dd:
            FT.extract_timm(b, d, data_root, out_root, batch_size=a.batch_size, device=dev, num_workers=a.workers,
                            imagenet100_root=Path(a.imagenet100_root) if a.imagenet100_root else None, fp16=a.fp16,
                            transform=a.transform, img_size=a.img_size, log=_log)
    _log(f"[features-timm] manifest {FT.manifest(out_root)}")


def cmd_tables(a):
    from pipeline import tables
    out = Path(a.results)
    A = json.loads((out / "paperA_results.json").read_text()) if (out / "paperA_results.json").exists() else None
    B = json.loads((out / "paperB_results.json").read_text()) if (out / "paperB_results.json").exists() else None
    if getattr(a, "check", False):
        # Freeze idempotence test. The canonical generator is tables/build_all.py (the G08 layer: per-paper fragments with
        # caption tags and provenance meta, DIFF.md with a banner) when it exists; it is rerun into a temporary folder and
        # compared with tables/out: .tex, .caption_tags.txt and DIFF.md byte for byte, .meta.json on every key except the
        # build timestamp. Without that layer the pipeline.tables emitters are compared directly.
        builder = ROOT / "tables" / "build_all.py"
        if builder.exists():
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                r = subprocess.run([sys.executable, str(builder), "--out-dir", td, "--results-root", str(ROOT / "results")], capture_output=True, text=True)
                if r.returncode != 0:
                    _log(r.stdout[-2000:]); _log(r.stderr[-2000:]); _log("[tables --check] FAIL: tables/build_all.py did not run"); raise SystemExit(1)
                tmp = Path(td); out_dir = ROOT / "tables" / "out"; bad = []
                for q in sorted(tmp.rglob("*")):
                    if not q.is_file():
                        continue
                    rel = q.relative_to(tmp); tgt = out_dir / rel
                    if q.name == "BUILD_REPORT.txt":
                        continue                                    # the audit transcript carries paths and timings
                    if not tgt.exists():
                        bad.append(f"{rel} (missing in tables/out)"); continue
                    if q.suffix == ".json" and q.name.endswith(".meta.json"):
                        ja, jb = json.loads(q.read_text()), json.loads(tgt.read_text())
                        ja.pop("created_utc", None); jb.pop("created_utc", None)
                        if ja != jb:
                            bad.append(f"{rel} (meta differs beyond the timestamp)")
                    elif q.read_bytes().replace(b"\r\n", b"\n") != tgt.read_bytes().replace(b"\r\n", b"\n"):   # line endings are not content
                        bad.append(f"{rel} (differs)")
                for q in sorted(out_dir.rglob("*")):
                    if q.is_file() and q.name != "BUILD_REPORT.txt" and not (tmp / q.relative_to(out_dir)).exists() and q.suffix in (".tex", ".md", ".txt", ".json"):
                        bad.append(f"{q.relative_to(out_dir)} (in tables/out but not regenerated)")
            for b_ in bad:
                _log(f"[tables --check] {b_}")
            if bad:
                _log(f"[tables --check] FAIL: {len(bad)} difference(s) between tables/out and a regeneration by tables/build_all.py"); raise SystemExit(1)
            _log("[tables --check] PASS: tables/build_all.py regenerates tables/out (tex, caption tags, DIFF.md byte for byte; meta up to the timestamp)")
            return
        status = tables.check_all(ROOT / "tables" / "out", A, B)
        for name, st in status.items():
            _log(f"[tables --check] {name}: {st}")
        bad = [n for n, st in status.items() if st != "same"]
        if bad:
            _log(f"[tables --check] FAIL: {len(bad)} file(s) differ from a regeneration: {', '.join(bad)}")
            raise SystemExit(1)
        _log("[tables --check] PASS: every emitted table and DIFF.md regenerates byte for byte")
        return
    for p in tables.write_all(ROOT / "tables" / "out", A, B):
        _log(f"[tables] {p}")


def cmd_freeze_manifest(a):
    """sha256 of every versioned file (the freeze record) plus one hash over the sorted list of (path, sha256)."""
    import hashlib
    skip_dirs = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", ".ipynb_checkpoints"}
    skip_prefix = ("results/_selftest",)
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if any(part in skip_dirs for part in p.relative_to(ROOT).parts) or rel.startswith(skip_prefix) or p.suffix in (".pyc",) or rel.startswith("FREEZE_MANIFEST"):
            continue
        rows.append(dict(path=rel, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size))
    tree = hashlib.sha256("\n".join(f"{r['path']} {r['sha256']}" for r in rows).encode()).hexdigest()
    man = dict(generated_utc=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()), n_files=len(rows), tree_sha256=tree, files=rows)
    out = ROOT / "FREEZE_MANIFEST.json"; out.write_text(json.dumps(man, indent=1))
    (ROOT / "FREEZE_MANIFEST.md").write_text("# Freeze manifest\n\n" + f"{len(rows)} files; tree sha256 `{tree}`; generated {man['generated_utc']}\n\n| path | bytes | sha256 |\n|---|---|---|\n" +
                                            "\n".join(f"| {r['path']} | {r['bytes']} | {r['sha256']} |" for r in rows) + "\n")
    _log(f"[freeze-manifest] {len(rows)} files, tree sha256 {tree}; wrote {out} and FREEZE_MANIFEST.md")


def cmd_drift_readback(a):
    from pipeline.drift import readback
    res = Path(a.results)
    readback(Path(a.drift) if a.drift else res / "paperA_drift.json", res / "DRIFT_READBACK.md",
             Path(a.results_json) if a.results_json else None, log=_log)


def cmd_drift(a):
    from pipeline.drift import run_drift
    seeds = tuple(int(s) for s in a.seeds.split(",")); lrs = tuple(float(x) for x in a.lrs.split(","))
    run_drift(Path(a.data_root), Path(a.results) / "paperA_drift.json", lrs=lrs, seeds=seeds, iters_per_stage=a.iters_per_stage,
              measure_every=a.measure_every, gamma_subsample=(None if a.gamma_subsample <= 0 else a.gamma_subsample),
              device=(a.device or "cuda"), log=_log)


def cmd_nonvision(a):
    args = [sys.executable, str(ROOT / "experiments" / "nonvision.py"), "--mode", "full", "--allow-local-full", "--update-index",
            "--out-root", a.results]
    if a.allow_download: args.append("--allow-download")
    if a.grouping: args += ["--grouping", a.grouping]
    subprocess.run(args, check=True)


def cmd_synthetic(a):
    from experiments import scoreboard
    scoreboard.main((["--run"] if a.run else []) + (["--quick"] if a.quick else []))


def cmd_selftest(a):
    """End-to-end run of both drivers and the table emitters on small synthetic feature caches (overlapping
    Gaussian classes so that accuracies are not saturated).  Exercises every code path the real run uses, with
    the optimizer and backend of the real run (AdamW, auto device), at a fraction of the sizes."""
    from pipeline import backend, features as F, paper_b, tables
    try:
        from pipeline import paper_a
    except ImportError:  # B-only release: the A leg of the selftest is skipped
        paper_a = None
    root = ROOT / "results" / "_selftest_features"; out = ROOT / "results" / "_selftest"
    for b in ("vit_in21k", "dinov2_b14", "clip_b16", "random_vit"):
        for d in ("cifar100", "imagenet_r"):
            F.write_synthetic_cache(b, d, root, d=32, n_train_per_class=30, n_test_per_class=10, sep=(0.9 if b != "random_vit" else 0.3))
    for b in ("mae_b16", "resnet50"):
        F.write_synthetic_cache(b, "cifar100", root, d=32, n_train_per_class=30, n_test_per_class=10, sep=0.9)
    t0 = time.time(); _log(f"[selftest] backend {backend.describe(a.device)}")
    proto_a = dict(R.HEAD_PROTOCOL["A"], epochs=60); proto_b = dict(R.HEAD_PROTOCOL["B"], epochs=60)
    if hasattr(paper_a, "run_all"):
        A = paper_a.run_all(root, out, seeds=(0, 1), optimizer=a.optimizer, K_method=100, skip_ecf=a.skip_ecf, K_sweep=(50, 100, 200, 400),
                            datasets=("cifar100", "imagenet_r"), ranpac_M=500, device=a.device, protocol=proto_a, log=_log)
    else:
        A = None; _log("[selftest] paper A driver not included (B-only release; shared helpers only); A leg skipped")
    B = paper_b.run_all(root, out, seeds=(0, 1), optimizer=a.optimizer, rows=(("dinov2_b14", "cifar100"), ("clip_b16", "cifar100"), ("mae_b16", "cifar100"), ("resnet50", "cifar100")),
                        all_datasets=("cifar100", "imagenet_r"), device=a.device, protocol=proto_b, log=_log)
    for p in tables.write_all(out / "tables", A, B):
        _log(f"[selftest] {p}")
    assert (A is None or (A["slopes"] and A["predictor"])) and B["rate_fits"], "selftest produced no fits"
    _log(f"[selftest] OK in {time.time()-t0:.0f}s (synthetic provenance; nothing here is a paper number)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=os.environ.get("REPLICATION_FEATURES", "features"))
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--device", default=None, help="numpy | cpu | cuda | auto (default: $REPLICATION_DEVICE or auto)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("features"); s.add_argument("--paper", default="A"); s.add_argument("--backbones", default=None); s.add_argument("--datasets", default=None)
    s.add_argument("--data-root", default="/content/data"); s.add_argument("--imagenet100-root", default=None); s.add_argument("--batch", type=int, default=128)
    s.set_defaults(fn=cmd_features)   # the torch device for extraction is the top-level --device (default cuda here)
    s = sub.add_parser("analyze"); s.add_argument("--paper", default="AB"); s.add_argument("--optimizer", default="adamw", choices=["adamw", "lbfgs"])
    s.add_argument("--seeds", default="0,1,2,3,4"); s.add_argument("--K", type=int, default=2000); s.add_argument("--skip-ecf", action="store_true")
    s.add_argument("--ranpac-M", type=int, default=10000, help="RanPAC random-projection width (10000 as in RanPAC; lower it on CPU)")
    s.add_argument("--schedule", default="printed", choices=["printed", "consistent"], help="Algorithm 1 stage-0 schedule for the clipped-SGD arms (Paper A)")
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("tables"); s.add_argument("--check", action="store_true", help="freeze check: regenerate in memory and compare with tables/out byte for byte; exit 1 on any difference")
    s.set_defaults(fn=cmd_tables)
    s = sub.add_parser("freeze-manifest", help="sha256 of every versioned file plus a tree hash, to FREEZE_MANIFEST.json and .md (the Oct 31 freeze record)"); s.set_defaults(fn=cmd_freeze_manifest)
    s = sub.add_parser("figures", help="every figure from the result JSONs, with a sidecar .data.json per figure (no curve without an artifact)")
    s.add_argument("--out", default=None, help="output folder (default <results>/figures)"); s.add_argument("--drift", default=None, help="drift JSON (default <results>/paperA_drift.json)")
    s.set_defaults(fn=cmd_figures)
    s = sub.add_parser("audit-tex", help="claim ledger: every number printed in the paper's tables against the regenerated results (reads the TeX, never edits it)")
    s.add_argument("--paper", required=True, choices=["A", "B"]); s.add_argument("--tex", required=True, help="path to the paper's neurips_2026.tex")
    s.add_argument("--out", default=None, help="output folder (default <results>)"); s.set_defaults(fn=cmd_audit_tex)
    s = sub.add_parser("features-timm", help="re-extract Paper A features under the recovered timm protocol into a separate cache root")
    s.add_argument("--data-root", default="/content/data"); s.add_argument("--out-root", default="features_timm")
    s.add_argument("--backbones", default=None); s.add_argument("--datasets", default=None); s.add_argument("--batch-size", type=int, default=128)
    s.add_argument("--workers", type=int, default=4); s.add_argument("--imagenet100-root", default=None); s.add_argument("--fp16", action="store_true")
    s.add_argument("--transform", default="timm", choices=["timm", "archaeology"], help="timm: the model's own eval transform (recovers the printed ViT values); archaeology: per-dataset transforms")
    s.add_argument("--img-size", type=int, default=224, help="input size passed to timm.create_model and the transform (DINOv2 native is 518)")
    s.set_defaults(fn=cmd_features_timm)
    s = sub.add_parser("fingerprint", help="which estimator / formula convention reproduces the printed gamma_min, nu and RHS (feature caches only, minutes)")
    s.add_argument("--backbones", default=None); s.add_argument("--datasets", default=None); s.add_argument("--seeds", default="0")
    s.add_argument("--subsample", type=int, default=50, help="rows per class for the subsampled-covariance variant"); s.set_defaults(fn=cmd_fingerprint)
    s = sub.add_parser("condition-s", help="Condition (S) clauses (nu, delta, a_ij) on every cached Paper A configuration; feature caches only, no training")
    s.add_argument("--backbones", default=None); s.add_argument("--datasets", default=None); s.add_argument("--seeds", default="0,1,2,3,4"); s.set_defaults(fn=cmd_condition_s)
    s = sub.add_parser("drift"); s.add_argument("--data-root", default="/content/data"); s.add_argument("--seeds", default="0,1,2,3,4"); s.add_argument("--lrs", default="1e-5,1e-4,1e-3")
    s.add_argument("--iters-per-stage", type=int, default=1000, help="training iterations per stage (paper: 1000)")
    s.add_argument("--measure-every", type=int, default=200, help="gamma_min checkpoints inside a stage, in iterations")
    s.add_argument("--gamma-subsample", type=int, default=50, help="training images per class for the gamma pass; 0 = all"); s.set_defaults(fn=cmd_drift)
    s = sub.add_parser("drift-readback", help="score the pre-registered drift predictions P1..P8 and checks C1..C5 from paperA_drift.json")
    s.add_argument("--drift", default=None, help="drift JSON (default <results>/paperA_drift.json)")
    s.add_argument("--results-json", default=None, help="paperA_results.json for check C3 (optional)"); s.set_defaults(fn=cmd_drift_readback)
    s = sub.add_parser("nonvision"); s.add_argument("--allow-download", action="store_true"); s.add_argument("--grouping", default=None); s.set_defaults(fn=cmd_nonvision)
    s = sub.add_parser("synthetic"); s.add_argument("--run", action="store_true", help="re-run every synthetic suite before scoring")
    s.add_argument("--quick", action="store_true", help="reduced grids"); s.set_defaults(fn=cmd_synthetic)
    s = sub.add_parser("selftest"); s.add_argument("--skip-ecf", action="store_true"); s.add_argument("--optimizer", default="adamw", choices=["adamw", "lbfgs"]); s.set_defaults(fn=cmd_selftest)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
