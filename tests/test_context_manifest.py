"""Checksummed paper-context dicts (W7-1, 2026-10-21; owner Q8/R4 ruling).

Every dict of PRINTED paper numbers that the code quotes as context and NEVER
asserts (the diff-against-paper inputs) is hashed against a committed
manifest, tests/context_manifest.json, so a transcription error in a paper
number is caught the first time anyone edits it (the audit's stated purpose).

Tier: INTEGRATION (imports the owning modules; no computation, no network,
no repo results files).  The manifest records one sha256 per dict over a
canonical serialisation (see ``_canon``); floats are pinned exactly by
``repr`` (no rounding), int-vs-float distinction is preserved by a type tag,
and dict order is normalised away by sorting on the canonical key encoding.

Regeneration (deliberate, human-approved edits only): run this module with
``REPLICATION_REGEN_CONTEXT_MANIFEST=1`` (or ``python tests/test_context_manifest.py
--regen``); the manifest is rewritten from the current store and MUST be
re-reviewed in the same change that touched a printed number.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST_PATH = Path(__file__).resolve().parent / "context_manifest.json"

#: Every never-asserted printed-context dict in the store, as
#: (import path, attribute).  Adding a new context dict means adding it here
#: and regenerating the manifest in the same change.
CONTEXT_DICTS = (
    ("pipeline.registry", "PRINTED"),
    ("experiments.falsification", "PAPER_CONTEXT_QUOTED"),
    ("experiments.slope_fits", "PAPER_CONTEXT_QUOTED"),
    ("experiments.nonvision", "PAPER_CONTEXT"),
    ("experiments.predictor_regression", "GAMMA_MIN_PRINTED"),
    ("experiments.predictor_regression", "FINAL_ACC_PRINTED"),
    ("experiments.predictor_regression", "PRINTED_TARGETS"),
)

CANON_VERSION = "v1 (typed tags, repr floats, key-sorted dicts)"


def _canon(o) -> str:
    """Canonical exact string for hashing. Deterministic across platforms:
    no locale, no dict-order dependence, repr() floats (shortest round-trip).
    """
    if isinstance(o, bool):
        return f"bool:{o}"
    if isinstance(o, int):
        return f"int:{o}"
    if isinstance(o, float):
        return f"float:{o!r}"
    if isinstance(o, str):
        return f"str:{o}"
    if isinstance(o, dict):
        items = sorted((_canon(k), _canon(v)) for k, v in o.items())
        return "dict{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(o, (list, tuple)):
        return "seq[" + ",".join(_canon(v) for v in o) + "]"
    return f"{type(o).__name__}:{o!r}"


def digest(obj) -> str:
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


def collect() -> dict:
    """{(module, attr): sha256} for every dict in CONTEXT_DICTS."""
    mods = {}
    out = {}
    for mod_name, attr in CONTEXT_DICTS:
        if mod_name not in mods:
            mods[mod_name] = __import__(mod_name, fromlist=["__name__"])
        obj = getattr(mods[mod_name], attr)
        if not isinstance(obj, dict):
            raise TypeError(f"{mod_name}.{attr} is not a dict ({type(obj)})")
        out[f"{mod_name}.{attr}"] = digest(obj)
    return out


def write_manifest() -> Path:
    payload = {
        "_comment": ("sha256 of every never-asserted printed-context dict; "
                     "regenerate deliberately via REPLICATION_REGEN_CONTEXT_MANIFEST=1 "
                     "and re-review alongside the edit that changed a paper number"),
        "canonicalization": CANON_VERSION,
        "dicts": dict(sorted(collect().items())),
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                             encoding="utf-8")
    return MANIFEST_PATH


class ContextManifestTests(unittest.TestCase):

    def test_printed_context_dicts_match_committed_manifest(self):
        """INTEGRATION: every never-asserted printed-context dict hashes to
        its committed sha256.  Epsilon: none (exact digests; the underlying
        numbers are quoted paper values, not measurements, so any drift is a
        transcription and must be reviewed, never tolerated).
        """
        if os.environ.get("REPLICATION_REGEN_CONTEXT_MANIFEST") == "1":
            p = write_manifest()
            print(f"[manifest] REGENERATED {p}")
            return
        if not MANIFEST_PATH.exists():
            self.fail(f"committed manifest missing: {MANIFEST_PATH} "
                      "(regenerate with REPLICATION_REGEN_CONTEXT_MANIFEST=1)")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        committed = manifest["dicts"]
        current = collect()
        self.assertEqual(sorted(committed), sorted(current),
                         "the set of context dicts changed; update CONTEXT_DICTS "
                         "and regenerate the manifest in the same change")
        changed = [name for name in current
                   if committed.get(name) != current[name]]
        self.assertEqual(changed, [],
                         "printed-context dicts diverged from the committed "
                         f"manifest: {changed}. If the edit is deliberate, re-run "
                         "with REPLICATION_REGEN_CONTEXT_MANIFEST=1 and re-review "
                         "the changed number in the same change.")


if __name__ == "__main__":
    if "--regen" in sys.argv or os.environ.get("REPLICATION_REGEN_CONTEXT_MANIFEST") == "1":
        print(f"[manifest] wrote {write_manifest()}")
    else:
        unittest.main()
