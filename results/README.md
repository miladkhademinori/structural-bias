# results/ -- the results store

This directory is the single results store for the replication pipeline.

- Experiment runners (`experiments/`, `colab/` notebooks) write their raw
  outputs (JSON/CSV per run, keyed by experiment, dataset, and seed) here.
- Table builders (`tables/`) read ONLY from this store -- they never re-run
  experiments and never invent numbers.
- The regeneration command for every family, and what each family holds:
  `docs/REPLICATION.md`. The verdict vocabulary and the per-table verdict
  summary: `docs/VERDICTS.md`.
- Large raw result files may be excluded via the optional `results/*.json`
  line in the root `.gitignore` (currently left commented; enable when the
  store grows).

## Contents

Populated by the real runs of the replication waves:

- `runs/B/` -- the Paper B suite records (`synthetic_bias`, `synthetic_rate`,
  `nonvision`, `lda_ncm`), one self-describing JSON envelope per run.
- `INDEX.json` -- the run registry: run id, path, provenance, suite status,
  and notes.
- `SCOREBOARD.md` / `SCOREBOARD.json` -- the claim-by-claim verification
  scoreboard over the suite records.
- `CORRECTIONS.md` -- the corrections ledger (the CI5 ledger closes and the
  Paper B withdrawal rows of the camera-ready change summary).
- `_selftest/`, `_selftest_features/` -- the selftest outputs on synthetic
  caches (`python replicate.py selftest`); synthetic provenance, not paper
  numbers.
