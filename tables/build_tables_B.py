"""tables.build_tables_B -- Paper B tables CLI; delegates to tables.build_all.

Thin per-paper front end: every rule, emitter call and report line lives in
``tables/build_all.py`` (shared core over ``pipeline.tables``); this module
only pins ``--paper=B``.  Flags are passed through unchanged (``--check``,
``--only``, ``--results-root``, ``--out-dir``); exit codes follow the core:
0 all-well, 1 store regression, 2 GATE-FAIL.

    python tables/build_tables_B.py [--check] [--only tab:label,...]

Spec: exec/G08_log.md S6.  Owner: U6-1, 2026-10-21.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import build_all  # noqa: E402  (same directory; shared core)


def main(argv=None) -> int:
    return build_all.build(argv, paper="B")


if __name__ == "__main__":
    raise SystemExit(main())
