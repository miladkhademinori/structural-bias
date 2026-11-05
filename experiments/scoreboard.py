"""experiments.scoreboard -- run the synthetic verification suites and score them in two tiers.

    python -m experiments.scoreboard            # score the JSON records already under results/runs
    python -m experiments.scoreboard --run      # re-run every suite first (about 30 to 60 CPU-minutes)
    python -m experiments.scoreboard --run --quick

Each suite (experiments/synthetic_bias.py E1-E4, synthetic_rate.py E5-E6, synthetic_margin.py EX1-EX5,
falsification.py G1-G3, slope_fits.py S4/EX6-EX8, drift.py EX9-EX11) writes one envelope JSON per run
under results/runs/<paper>/<experiment>/ with a list of assertion records.  The scoreboard reads them
all and reports every assertion with its tier (core.gates.tier_of) and verdict:

  PASS                    the assertion held
  MECHANISM-FAIL          an algebraic or population identity failed, a finite-sample arm missed by more than
                          the battery's bound, or the spec design cannot exhibit the claim at any n ([design])
  PAPER-QUESTION          a deterministic finding about a printed paper number; carried to the owner's Q-list
  CALIBRATION-EXCURSION   a Monte-Carlo or finite-sample estimate missed a hand-set absolute tolerance;
                          the size of the miss is in the record, and the fix is a standard-error gate
                          (core.gates.z_battery / binomial_battery / one_sided_band), not a wider tolerance

The suites' own tolerances are left untouched by this module (their re-gating is a separate task); the
scoreboard is the honest reading of what they currently say.  Output: results/SCOREBOARD.md and
results/SCOREBOARD.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import gates  # noqa: E402

SUITES = [  # (module, argv builder)
    ("experiments.synthetic_bias", lambda q: ["--suites", "E1,E2,E3,E4"]),
    ("experiments.synthetic_rate", lambda q: ["--suites", "E5,E6"] + (["--budget-s", "120"] if q else ["--full-grid"])),
    ("experiments.synthetic_margin", lambda q: ["--suites", "EX1,EX2,EX3,EX4,EX5"]),
    ("experiments.falsification", lambda q: ["--suites", "G1,G2,G3"]),
    ("experiments.slope_fits", lambda q: ["--suites", "S4,EX6,EX7,EX8"]),
    ("experiments.drift", lambda q: (["--quick"] if q else [])),
]


def run_suites(quick: bool, log=print) -> Dict[str, int]:
    import importlib
    codes = {}
    for mod, argv in SUITES:
        t0 = time.time(); log(f"[scoreboard] running {mod} {' '.join(argv(quick))}")
        try:
            m = importlib.import_module(mod)
            codes[mod] = int(m.main(argv(quick)) or 0)
        except SystemExit as e:  # some mains call sys.exit
            codes[mod] = int(e.code or 0)
        except Exception as e:  # keep scoring the others
            log(f"[scoreboard] {mod} crashed: {type(e).__name__}: {e}"); codes[mod] = 99
        log(f"[scoreboard] {mod}: exit {codes[mod]} in {time.time()-t0:.0f}s")
    return codes


def _assertions(payload: dict) -> List[dict]:
    rec = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    out = []
    for a in rec.get("assertions", []) or []:
        a = dict(a); a.setdefault("desc", a.get("description", "")); out.append(a)
    return out


def collect(results_root: Path) -> List[dict]:
    rows = []
    for p in sorted((results_root / "runs").rglob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for a in _assertions(payload):
            a["run"] = str(p.relative_to(results_root))
            rows.append(dict(run=str(p.relative_to(results_root)), paper=payload.get("paper"), experiment=payload.get("experiment"),
                             provenance=payload.get("provenance", "synthetic"), id=a.get("id"), arm=a.get("arm", ""),
                             desc=a.get("desc", ""), status=a.get("status", "PASS"), value=a.get("value"), threshold=a.get("threshold"),
                             diagnosis=a.get("diagnosis", ""), tier=gates.tier_of(a), verdict=gates.verdict(a)))
    return rows


def claims(rows: List[dict]) -> List[dict]:
    """Claim-level roll-up: the verdict unit is the claim (core.gates.claim_verdict)."""
    by = {}
    for r in rows:
        by.setdefault((r["run"].split("/")[1] if "/" in r["run"] else r["paper"], gates.claim_of(r["id"])), []).append(r)
    out = []
    for (paper, cid), arms in sorted(by.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        out.append(dict(paper=paper, claim=cid, n_arms=len(arms), verdict=gates.claim_verdict(arms),
                        arms=[dict(id=a["id"], arm=a["arm"], verdict=a["verdict"]) for a in arms]))
    return out


def render(rows: List[dict]) -> str:
    n = len(rows); npass = sum(r["verdict"] == "PASS" for r in rows)
    mech = [r for r in rows if r["verdict"].startswith("MECHANISM-FAIL")]; cal = [r for r in rows if r["verdict"] == "CALIBRATION-EXCURSION"]
    pq = [r for r in rows if r["verdict"] == "PAPER-QUESTION"]
    cl = claims(rows); ncl = len(cl)
    counts = {w: sum(c["verdict"].split(" ")[0] == w for c in cl) for w in ("PASS", "MECHANISM-FAIL", "CALIBRATION-EXCURSION", "PAPER-QUESTION")}
    L = ["# Synthetic verification scoreboard", "",
         f"{n} assertions from {len({r['run'] for r in rows})} suite records: {npass} PASS, {len(mech)} MECHANISM-FAIL, {len(cal)} CALIBRATION-EXCURSION, {len(pq)} PAPER-QUESTION.",
         f"Claims (the verdict unit): {ncl} claims: {counts['PASS']} PASS, {counts['MECHANISM-FAIL']} MECHANISM-FAIL, {counts['CALIBRATION-EXCURSION']} CALIBRATION-EXCURSION, {counts['PAPER-QUESTION']} PAPER-QUESTION.", "",
         "Vocabulary (docs/VERDICTS.md): MECHANISM-FAIL, the mechanism the claim asserts is not exhibited (an exact arm misses its numerical tolerance, "
         "a finite-sample arm misses by more than the battery's family-wise bound, or the spec design cannot exhibit the claim at any n, marked [design] with the variant arm printed beside it); "
         "CALIBRATION-EXCURSION, the mechanism is exhibited and one gate number is missed within the battery's bound, size in standard errors reported; "
         "PAPER-QUESTION, a deterministic finding about a printed paper number, carried to the owner's Q-list with the measured value.", "",
         "## Claims", "", "| paper | claim | arms | verdict |", "|---|---|---|---|"]
    for c in cl:
        if c["verdict"] != "PASS":
            L.append(f"| {c['paper']} | {c['claim']} | {'; '.join(f'{a['id']} ({a['arm']}): {a['verdict']}' for a in c['arms'])} | **{c['verdict']}** |")
    L += ["", "PASS: " + ", ".join(c["claim"] for c in cl if c["verdict"] == "PASS"), ""]
    L += ["Tier 1 (mechanism: algebraic and population identities, exact in all inputs) must be all PASS for the code to be trusted.",
          "Tier 2 (calibration: Monte-Carlo and finite-sample estimates, including exact functionals of finite-sample heads) lists excursions with the suites' own diagnosis; "
          "they are re-gated by standard error (tests/zbattery.py and core.gates), never by widening tolerances.", ""]
    for tier, title in (("MECHANISM", "## Tier 1: mechanism"), ("CALIBRATION", "## Tier 2: calibration")):
        rs = [r for r in rows if r["tier"] == tier]
        L += [title, "", f"{sum(r['verdict']=='PASS' for r in rs)}/{len(rs)} PASS", "", "| run | id | arm | verdict | value | threshold |", "|---|---|---|---|---|---|"]
        for r in rs:
            if r["verdict"] != "PASS":
                L.append(f"| {r['run']} | {r['id']} | {r['arm']} | **{r['verdict']}** | {str(r['value'])[:80]} | {str(r['threshold'])[:60]} |")
        passing = [f"{r['id']}" for r in rs if r["verdict"] == "PASS"]
        L += ["", f"PASS: {', '.join(passing)}" if passing else "PASS: none", ""]
    if cal or pq:
        L += ["## Diagnoses recorded by the suites for the excursions and paper questions", ""]
        for r in cal + pq:
            if r["diagnosis"]:
                L.append(f"- {r['run']} {r['id']}: {r['diagnosis']}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", action="store_true", help="re-run every suite before scoring")
    p.add_argument("--quick", action="store_true", help="reduced budgets where a suite supports them")
    p.add_argument("--results", default=str(ROOT / "results"))
    a = p.parse_args(argv)
    root = Path(a.results)
    if a.run:
        run_suites(a.quick)
    rows = collect(root)
    (root / "SCOREBOARD.json").write_text(json.dumps(dict(assertions=rows, claims=claims(rows)), indent=1, default=str), encoding="utf-8")
    md = render(rows); (root / "SCOREBOARD.md").write_text(md, encoding="utf-8")
    print(md.splitlines()[2]); print(f"[scoreboard] wrote {root / 'SCOREBOARD.md'}")
    return 0 if not any(r["verdict"].startswith("MECHANISM-FAIL") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
