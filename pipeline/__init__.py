"""pipeline -- the real-data replication pipeline for Paper B (12500).

Written by Claude on 2026-10-22 for the NeurIPS 2026 camera-ready replication.

Design
------
Every number printed in the paper that depends on a real backbone is produced
by one of two stages:

1. ``features``  (GPU, Colab): one forward pass per (backbone, dataset, split),
   cached as ``features/<backbone>/<dataset>/<split>.npz``.  Extraction is a
   fixed-order forward pass, so caches are seed-independent and idempotent.
2. ``analyze``   (CPU, anywhere): every head, statistic, fit and table is computed
   from the caches with numpy/scipy only, seeded by (SEED, purpose) strings.

The only experiment that needs a GPU beyond feature extraction is the
five-method diagnostic of Paper B (external checkpoints; see
``docs/REPLICATION.md``).

B-only release: the companion Paper A driver is released separately at
github.com/miladkhademinori/mahalanobis-margin.  ``paper_a`` here is a shim
that carries only the statistical helpers shared by both drivers.

Modules
-------
registry   backbones, datasets, task splits, seeds, paper-printed context values
features   Colab-side extraction (torch/transformers), cache format, synthetic caches
stats      class means, Ledoit-Wolf pooled covariance, gamma/D/kappa/nu, bias direction
heads      linear heads (within-task, joint, diagonal-joint), AdamW/L-BFGS/clipped SGD,
           post-hoc de-centering, LDA/SLDA, NCM, FeCAM-style, RanPAC-style
paper_a    shared-helpers shim only (_agg/_ms/_ols); the full A driver is separate
paper_b    WT/FIX/J benchmark, LDA/NCM, per-pair bias direction, rate validation,
           kappa distribution, weights-only control
figures    every figure from the result JSONs
tables     LaTeX emitters and the paper-vs-reproduced diff
tex_audit  printed-versus-regenerated claim ledger (Paper B tables)
"""

__version__ = "2026.10.22"
