# results/ -- the results store

SKELETON NOTE (G09, 2026-10-21) -- NOT_IMPLEMENTED_YET.

This directory is the single results store for the replication pipeline.

- Experiment runners (`experiments/`, `colab/` notebooks) write their raw
  outputs (JSON/CSV per run, keyed by experiment, dataset, and seed) here.
- Table builders (`tables/`) read ONLY from this store -- they never re-run
  experiments and never invent numbers.
- Exact on-disk layout (filenames, schema, nesting) is pending spec from the
  G02/G03/G06/G07/G08 exec logs; nothing is fabricated in the skeleton wave.
- `.gitkeep` keeps the empty directory in version control. Large raw result
  files may be excluded via the optional `results/*.json` line in the root
  `.gitignore` (currently left commented; enable when the store grows).

## Contents

Empty in the skeleton wave. Populate only via real runs in the H/I waves.
