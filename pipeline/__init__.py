"""pipeline -- the real-data replication pipeline for Papers A (4828) and B (12500).

Written by Claude on 2026-10-22 for the NeurIPS 2026 camera-ready replication.

Design
------
Every number printed in either paper that depends on a real backbone is produced
by one of two stages:

1. ``features``  (GPU, Colab): one forward pass per (backbone, dataset, split),
   cached as ``features/<backbone>/<dataset>/<split>.npz``.  Extraction is a
   fixed-order forward pass, so caches are seed-independent and idempotent.
2. ``analyze``   (CPU, anywhere): every head, statistic, fit and table is computed
   from the caches with numpy/scipy only, seeded by (SEED, purpose) strings.

The only experiments that need a GPU beyond feature extraction are the drift
trajectories of Paper A (``pipeline.drift``, fine-tunes the backbone) and the
five-method diagnostic of Paper B (external checkpoints; see ``docs/REPLICATION.md``).

Modules
-------
registry   backbones, datasets, task splits, seeds, paper-printed context values
features   Colab-side extraction (torch/transformers), cache format, synthetic caches
stats      class means, Ledoit-Wolf pooled covariance, gamma/D/kappa/nu, bias direction
heads      linear heads (within-task, joint, diagonal-joint), AdamW/L-BFGS/clipped SGD,
           post-hoc de-centering, LDA/SLDA, NCM, FeCAM-style, RanPAC-style
paper_a    gamma table, anchors, Condition (S), TC gap, slopes, predictor regression,
           E_CF sweep, Hill alpha, Algorithm 1 ("ours") and the frozen-feature baselines
paper_b    WT/FIX/J benchmark, LDA/NCM, per-pair bias direction, rate validation,
           kappa distribution, weights-only control
drift      Paper A drift trajectories (GPU)
tables     LaTeX emitters and the paper-vs-reproduced diff
"""

__version__ = "2026.10.22"
