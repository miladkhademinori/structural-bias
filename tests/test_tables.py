"""tests.test_tables -- integration tests for the tables/ layer (agent W6-8).

Spec source: ``exec/G08_log.md`` (tables layer spec, 2026-10-21), read-only.
Implements the W6-8 task order: INTEGRATION tier coverage of
  (1) ``tables/build_all.py --check`` runs clean as a subprocess;
  (2) every TeX fragment under ``tables/out/`` parses balanced
      (unescaped braces, \\begin/\\end environment stack);
  (3) the blocked report lists EXACTLY the expected RUN-FED/owner tables,
      counted against the G08 S1 enumeration (A 11 live tables, 5 RUN-FED;
      B 8 live tables, 5 RUN-FED; 19 live + 1 folded histogram artifact);
  (4) results-store round-trip through the conftest fixtures
      (results_store / store_write_json / store_read_json), G08-S2 envelope.

All tiers below are INTEGRATION per the conftest test policy. No paper
number is asserted anywhere (NO REAL-BACKBONE RULE); the RUN-FED label set
is a SPEC fact from G08 S1/COUNTS, not a measurement.

Runner: stdlib unittest (pytest absent, probed 2026-10-21)::

    python repo/tests/test_tables.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

# conftest lives next to this file; make it importable under plain unittest
# and under `python -m unittest discover -s repo/tests`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import conftest as ct  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_ALL = REPO_ROOT / "tables" / "build_all.py"
OUT_DIR = REPO_ROOT / "tables" / "out"

#: Subprocess budget: the full --check audit walks every table; 10 minutes
#: is two orders of magnitude above the observed build time.
SUBPROC_TIMEOUT_S = 600

# ---------------------------------------------------------------------------
# G08 S1/COUNTS expectations (spec facts, not measurements)
# ---------------------------------------------------------------------------

#: Paper A live table environments in float order (G08 S1; 11 total).
G08_A_LIVE = (
    "tab:headline-comparison",  # Table 1, RUN-FED
    "tab:gamma-min",            # Table 2, POST-HOC
    "tab:random-anchor",        # Table 3, POST-HOC
    "tab:residual-check",       # Table 4, POST-HOC
    "tab:slope-fits",           # Table 5, POST-HOC
    "tab:ecf-rates",            # Table 6, RUN-FED
    "tab:alpha-fits",           # Table 7, RUN-FED
    "tab:drift-slopes",         # Table 8, RUN-FED
    "tab:main",                 # Table 9, RUN-FED
    "tab:sens-tasks",           # Table 10, POST-HOC
    "tab:diagnostic",           # Table 11, STATIC
)
#: Paper B live table environments (G08 S1; 8 total, +1 folded histograms).
G08_B_LIVE = (
    "tab:fix-benchmark",        # Table 1, RUN-FED
    "tab:diagnostic",           # Table 2, RUN-FED  (B's diagnostic != A's)
    "tab:lda-ncm-comparison",   # Table 3, POST-HOC (20 gated cells)
    "tab:kappa-distribution",   # Table 4, RUN-FED
    "tab:rate-fits",            # Table 5, POST-HOC given heads, else RUN-FED
    "tab:bias-direction",       # Table 6, POST-HOC given heads, else RUN-FED
    "tab:weights-only",         # Table 7, RUN-FED
    "tab:nonvision",            # Table 8, RUN-FED (owner artifact)
)
#: The folded ninth B table: owed as histogram artifacts, not a table env.
G08_B_FOLDED = "tab:kappa-histogram-l2p"

#: G08 COUNTS line: RUN-FED = A 5 + B 5 = 10 exactly. The two B
#: "POST-HOC given heads, else RUN-FED" entries (rate-fits, bias-direction)
#: are counted POST-HOC there because heads/features ship with the store.
G08_A_RUNFED = frozenset({
    "tab:headline-comparison",
    "tab:ecf-rates",
    "tab:alpha-fits",
    "tab:drift-slopes",
    "tab:main",
})
G08_B_RUNFED = frozenset({
    "tab:fix-benchmark",
    "tab:diagnostic",
    "tab:kappa-distribution",
    "tab:weights-only",
    "tab:nonvision",
})
G08_RUNFED = G08_A_RUNFED | G08_B_RUNFED


def _build_all_is_skeleton() -> bool:
    """True while tables/build_all.py is the G09 placeholder (no logic)."""
    try:
        text = BUILD_ALL.read_text(encoding="utf-8")
    except OSError:
        return True
    return "NOT_IMPLEMENTED_YET" in text or "SKELETON PLACEHOLDER" in text


def _run_build_all(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUILD_ALL), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=SUBPROC_TIMEOUT_S,
    )


def _tex_source_lines(text: str) -> list[str]:
    """Strip TeX comments (unescaped % to end-of-line) from each line."""
    kept = []
    for line in text.splitlines():
        cut = None
        for m in re.finditer(r"%", line):
            # unescaped % (even number of preceding backslashes) starts a comment
            pre = line[: m.start()]
            if (len(pre) - len(pre.rstrip("\\"))) % 2 == 0:
                cut = m.start()
                break
        kept.append(line if cut is None else line[:cut])
    return kept


def _brace_balance(lines: list[str]) -> int:
    """Net unescaped-brace count; raises ValueError if it ever goes negative."""
    depth = 0
    for line in lines:
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\":  # skip the escaped character (\{ \} \\ ... )
                i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    raise ValueError("unbalanced TeX: '}' before matching '{'")
            i += 1
    return depth


def _environment_stack(lines: list[str]) -> list[str]:
    """Validate \\begin{env}/\\end{env} nesting; raise on mismatch.

    Returns the environments seen (for diagnostics). A fragment need not
    wrap its rows in a float environment, but whatever it opens it must
    close, in order.
    """
    stack: list[str] = []
    seen: list[str] = []
    for line in lines:
        for m in re.finditer(r"\\(begin|end)\{([^{}]*)\}", line):
            kind, env = m.group(1), m.group(2)
            if kind == "begin":
                stack.append(env)
                seen.append(env)
            else:
                if not stack or stack[-1] != env:
                    top = stack[-1] if stack else "<nothing open>"
                    raise ValueError(
                        f"unbalanced TeX: \\end{{{env}}} closes {{{top}}}"
                    )
                stack.pop()
    if stack:
        raise ValueError(f"unbalanced TeX: unclosed environments {stack}")
    return seen


def _blocked_labels(report_text: str) -> frozenset[str]:
    """Extract tab: labels from the FED-BLOCKED segment of a build report.

    G08 S6 final line shape:
    ``REGENERATED (n) | FED-BLOCKED (n, labels) | GATED (n, labels)``.
    Returns every ``tab:...`` token inside the FED-BLOCKED segment only.
    """
    m = re.search(r"FED-BLOCKED\s*\(([^)]*)\)", report_text)
    if m is None:
        raise AssertionError(
            "no FED-BLOCKED segment in build_all report; G08 S6 requires "
            "the final REGENERATED | FED-BLOCKED | GATED line. Report was:\n"
            + report_text[-2000:]
        )
    return frozenset(re.findall(r"tab:[A-Za-z0-9_-]+", m.group(1)))


class TestBuildAllCheck(unittest.TestCase):
    """INTEGRATION: tables/build_all.py --check as a subprocess, exit clean."""

    def test_check_runs_clean(self):
        """INTEGRATION: ``python tables/build_all.py --check`` exits 0.

        G08 S6: exit code 0 is only legal when every table is regenerated
        or explicitly FED-BLOCKED with owner-artifact provenance in the
        store. A traceback or a nonzero exit fails here. (While build_all
        is the G09 skeleton this passes vacuously -- the placeholder has
        no logic and cannot fail; the fragment/blocked-report tests carry
        the real coverage once the implementation lands.)
        """
        self.assertTrue(BUILD_ALL.is_file(), f"missing {BUILD_ALL}")
        proc = _run_build_all("--check")
        combined = (proc.stdout or "") + (proc.stderr or "")
        self.assertEqual(
            proc.returncode, 0,
            f"build_all --check exited {proc.returncode}:\n{combined[-2000:]}",
        )
        self.assertNotIn("Traceback", combined)
        self.assertNotIn("GATE-FAIL", combined)


class TestOutFragmentsBalanced(unittest.TestCase):
    """INTEGRATION: every fragment in tables/out/ parses balanced."""

    def test_balance_checker_contract(self):
        """INTEGRATION (checker self-test): the balance validators flag
        exactly the malformed inputs they must, and accept well-formed
        TeX including escaped braces, comments and nested environments.
        Keeps this file's coverage meaningful even while tables/out/ is
        still empty (vacuous fragment set)."""
        good = [
            r"\begin{tabular}{lcc} a & 1 & \pm 0.4 \\ \end{tabular}",
            r"escaped \{ and \} do not count \\ literal backslash",
            "comment % { unbalanced in comment\nreal { nested { deep } } text",
            r"\begin{small} \begin{center} x \end{center} \end{small}",
        ]
        for g in good:
            lines = _tex_source_lines(g)
            self.assertEqual(_brace_balance(lines), 0, msg=g)
            _environment_stack(lines)  # must not raise
        # NB: r"{ one \{ two }" is BALANCED (escaped \{ does not open), so
        # it must NOT be here; the third case is a real escaped-open leak.
        bad_braces = ["{ unclosed", "} backwards", r"open { escaped \{ no close"]
        for b in bad_braces:
            with self.assertRaises((ValueError, AssertionError), msg=b):
                lines = _tex_source_lines(b)
                depth = _brace_balance(lines)
                self.assertEqual(depth, 0, msg=b)
        bad_envs = [
            r"\begin{tabular} \end{center}",
            r"\begin{tabular} no end",
        ]
        for b in bad_envs:
            with self.assertRaises(ValueError, msg=b):
                _environment_stack(_tex_source_lines(b))

    def test_every_fragment_balanced(self):
        """INTEGRATION: each tables/out/**/*.tex has balanced unescaped
        braces and a matched \\begin/\\end stack; sibling .meta.json (when
        present) parses as JSON; sibling .caption_tags.txt (when present)
        decodes as UTF-8. G08 target layout: out/<paper>/<label>.tex plus
        <label>.caption_tags.txt and <label>.meta.json."""
        fragments = sorted(OUT_DIR.rglob("*.tex")) if OUT_DIR.is_dir() else []
        for frag in fragments:
            with self.subTest(fragment=str(frag.relative_to(REPO_ROOT))):
                text = frag.read_text(encoding="utf-8")
                lines = _tex_source_lines(text)
                self.assertEqual(
                    _brace_balance(lines), 0,
                    f"{frag}: unbalanced braces",
                )
                _environment_stack(lines)  # raises with a precise message
                meta = frag.with_suffix("").with_name(
                    frag.stem + ".meta.json"
                )
                if meta.is_file():
                    json.loads(meta.read_text(encoding="utf-8"))
                tags = frag.with_name(frag.stem + ".caption_tags.txt")
                if tags.is_file():
                    tags.read_text(encoding="utf-8")
        # Visible in verbose runs: 0 fragments is a legitimate (vacuous)
        # pass today; the count records it.
        print(f"\n[tables-out] {len(fragments)} .tex fragments checked")


class TestBlockedReport(unittest.TestCase):
    """INTEGRATION: blocked report vs the G08 S1/COUNTS enumeration."""

    def test_g08_runfed_expectation_selfconsistent(self):
        """INTEGRATION (spec side): the G08 expectation embedded above is
        internally consistent -- A 11 live labels of which 5 RUN-FED;
        B 8 live labels (+1 folded, not a table) of which 5 RUN-FED;
        10 RUN-FED total; RUN-FED disjoint from the live POST-HOC/STATIC
        remainder. Counted against exec/G08_log.md S1 and its COUNTS line.
        """
        self.assertEqual(len(G08_A_LIVE), 11)   # G08 S1: A has 11 tables
        self.assertEqual(len(G08_B_LIVE), 8)    # G08 S1: B has 8 live
        self.assertEqual(len(G08_A_RUNFED), 5)  # G08 COUNTS: A RUN-FED 5
        self.assertEqual(len(G08_B_RUNFED), 5)  # G08 COUNTS: B RUN-FED 5
        self.assertEqual(len(G08_RUNFED), 10)   # G08 COUNTS: 10 RUN-FED
        self.assertTrue(G08_A_RUNFED <= frozenset(G08_A_LIVE))
        self.assertTrue(G08_B_RUNFED <= frozenset(G08_B_LIVE))
        a_rest = frozenset(G08_A_LIVE) - G08_A_RUNFED   # 5 POST-HOC + 1 STATIC
        b_rest = frozenset(G08_B_LIVE) - G08_B_RUNFED   # 3 POST-HOC
        self.assertEqual(len(a_rest), 6)
        self.assertEqual(len(b_rest), 3)
        # Paper A and Paper B both have a tab:diagnostic; they are distinct
        # tables (A Table 11 STATIC symptom table; B Table 2 RUN-FED method
        # diagnostics) -- the per-paper sets keep them separable.
        self.assertIn("tab:diagnostic", G08_A_LIVE)
        self.assertIn("tab:diagnostic", G08_B_LIVE)
        # The folded B histogram promise is NOT a live table and must never
        # appear as either regenerated or blocked TABLE output.
        self.assertNotIn(G08_B_FOLDED, G08_B_LIVE)

    def test_blocked_report_lists_exactly_the_runfed_tables(self):
        """INTEGRATION: build_all's FED-BLOCKED segment lists EXACTLY the
        10 RUN-FED/owner tables of G08 S1 -- no more (a POST-HOC table
        reported blocked means the store regressed), no fewer (a RUN-FED
        table reported regenerable means a fake fill). Skipped with an
        explicit reason while tables/build_all.py is still the G09
        skeleton: the placeholder emits no report at all.
        """
        if _build_all_is_skeleton():
            self.skipTest(
                "tables/build_all.py is still the NOT_IMPLEMENTED_YET "
                "skeleton (G09, 2026-10-21); it emits no blocked report. "
                "The G08 expectation is asserted in "
                "test_g08_runfed_expectation_selfconsistent; this test "
                "activates when the tables implementer lands."
            )
        proc = _run_build_all("--check")
        combined = (proc.stdout or "") + (proc.stderr or "")
        self.assertEqual(proc.returncode, 0, combined[-2000:])
        blocked = _blocked_labels(combined)
        self.assertEqual(
            blocked, G08_RUNFED,
            f"FED-BLOCKED set mismatch; expected exactly the 10 G08 "
            f"RUN-FED labels, got {sorted(blocked)}",
        )


class TestResultsStoreRoundTrip(unittest.TestCase):
    """INTEGRATION: results-store round-trip via the conftest fixtures."""

    @staticmethod
    def _sample_run_record(run_id: str) -> dict:
        """A G08-S2-envelope-shaped synthetic record (no paper numbers)."""
        return {
            "run_id": run_id,
            "paper": "A",
            "experiment": "synthetic_margin",
            "provenance": "synthetic",
            "created_utc": "2026-10-22T05:30:30Z",
            "executed_utc": "2026-10-22T05:30:29Z",
            "schema": "G08-S2-proposal",
            "config": {
                "T": 10, "d": 32, "draws": 200,
                "gammas": [5.2, 12.1, 18.9, 31.4, 39.1, 51.8],
                "ladder": [0.12, 0.13, 0.15, 0.17, 0.19, 0.23,
                           0.26, 0.42, 0.49, 0.51, 0.586, 0.73],
                "nested": {"flags": [True, False], "label": "round-trip"},
            },
            "seeds": [0, 1, 2, 3, 4],
            "metrics": {
                "suite": "EX-ROUNDTRIP",
                "status": "PASS",
                "assertions": [
                    {"id": "X.1", "status": "PASS", "measured": -0.1250000001,
                     "band": [-0.135, -0.115], "diagnosis": ""},
                    {"id": "X.2", "status": "PASS",
                     "measured": {"gap": 0.0, "ratio": 6.4411e-16}},
                ],
                "summary": {
                    "gap_min": 64.6062704941537,
                    "gap_max": 304.53471005411143,
                    "feas_max": 0.0,
                    "note": "tiny-exponent float 6.441149361011384e-16",
                    "tiny": 6.441149361011384e-16,
                },
                "runtime_s": 0.7,
            },
            "env": {"python": "3.12.10", "numpy": "2.5.2"},
            "code_version": "0" * 16,
            "inputs": {"margin": "e33ef149e8f256ee"},
            "derived_tables": [],
        }

    def test_store_json_round_trip_exact(self):
        """INTEGRATION: a G08-S2-envelope record written through
        conftest.store_write_json and read back through
        conftest.store_read_json recovers IDENTICAL content (EPS_EXACT:
        JSON text round-trip is exact for int/float/str/list/dict/bool --
        float64 round-off epsilon does not apply to serialize/deserialize
        of decimal text, conftest EPS_TIGHT documentation).
        """
        with ct.results_store(prefix="n26-test-tables") as store:
            rec = self._sample_run_record("A_synthetic_margin_ROUNDTRIP")
            path = ct.store_write_json(store, "A_synthetic_margin_ROUNDTRIP", rec)
            self.assertTrue(path.is_file())
            self.assertEqual(path, store / "A_synthetic_margin_ROUNDTRIP.json")
            back = ct.store_read_json(store, "A_synthetic_margin_ROUNDTRIP")
            self.assertEqual(back, rec)
            # spot-load-bearing fields, so a failure names the field
            self.assertEqual(back["run_id"], rec["run_id"])
            self.assertEqual(
                back["metrics"]["summary"]["tiny"],
                rec["metrics"]["summary"]["tiny"],
            )
            self.assertEqual(back["config"]["nested"]["flags"], [True, False])
            self.assertEqual(
                back["metrics"]["assertions"][0]["measured"],
                -0.1250000001,
            )

    def test_store_write_deterministic_bytes(self):
        """INTEGRATION: rewriting the same object produces byte-identical
        JSON (conftest.store_write_json sorts keys); two different
        run_ids coexist in one store directory."""
        with ct.results_store(prefix="n26-test-tables") as store:
            rec = self._sample_run_record("A_synthetic_margin_RT1")
            p1 = ct.store_write_json(store, "rt_probe", rec)
            first = p1.read_bytes()
            rec2 = dict(rec)  # same content, fresh container object
            p2 = ct.store_write_json(store, "rt_probe", rec2)
            self.assertEqual(first, p2.read_bytes())
            ct.store_write_json(store, "rt_other", {"run_id": "other"})
            names = sorted(p.name for p in store.glob("*.json"))
            self.assertEqual(names, ["rt_other.json", "rt_probe.json"])

    def test_results_store_auto_cleaned(self):
        """INTEGRATION: the results_store context manager removes its
        directory on exit (no repo/results pollution; the fixture is the
        ONLY sanctioned place for test artifacts)."""
        with ct.results_store(prefix="n26-test-tables") as store:
            marker = store / "sentinel.json"
            ct.store_write_json(store, "sentinel", {"ok": True})
            self.assertTrue(marker.is_file())
            self.assertEqual(ct.store_read_json(store, "sentinel"), {"ok": True})
        self.assertFalse(store.exists(), f"store not cleaned: {store}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
