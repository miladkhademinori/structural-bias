"""pipeline.registry -- backbones, datasets, splits, seeds and paper context.

Everything here is a declaration; nothing downloads, loads or computes.
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

SEED = 0
SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)   # five random class orderings, both papers


def derive_seed(purpose: str, salt: int = 0, base: int = SEED) -> int:
    """SHA-256 child seed from (base, purpose, salt); never Python's hash()."""
    h = hashlib.sha256(f"{base}|{purpose}|{salt}".encode("utf-8")).hexdigest()
    return int(h, 16) % (2 ** 63)


def rng(purpose: str, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng(derive_seed(purpose, salt))


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Backbone:
    key: str                 # cache directory name
    label: str               # how the papers print it
    hf_id: Optional[str]     # HuggingFace model id (None for the random anchor)
    kind: str                # "vit" | "dinov2" | "clip" | "mae" | "resnet" | "random_vit"
    dim: int
    papers: Tuple[str, ...]
    note: str = ""


BACKBONES: Dict[str, Backbone] = {
    "vit_in21k": Backbone("vit_in21k", "ViT-B/16-IN21K", "google/vit-base-patch16-224-in21k",
                          "vit", 768, ("A", "B"), "CLS token of the last layer (pooler is a tanh head; not used)"),
    "dinov2_b14": Backbone("dinov2_b14", "DINOv2-ViT-B/14", "facebook/dinov2-base",
                           "dinov2", 768, ("A", "B"), "CLS token, 224x224 input (patch 14 -> 16x16 grid)"),
    "clip_b16": Backbone("clip_b16", "CLIP-ViT-B/16", "openai/clip-vit-base-patch16",
                         "clip", 768, ("A", "B"), "vision tower pooled output (768, pre-projection), as the paper's d=768"),
    "mae_b16": Backbone("mae_b16", "MAE ViT-B/16", "facebook/vit-mae-base",
                        "mae", 768, ("B",), "encoder CLS token with mask_ratio=0"),
    "resnet50": Backbone("resnet50", "ResNet-50", "microsoft/resnet-50",
                         "resnet", 2048, ("B",), "global-average-pooled last stage"),
    "random_vit": Backbone("random_vit", "Random ViT-B/16", None,
                           "random_vit", 768, ("A",), "ViT-B/16 config, random init with torch seed 0; the anchor of tab:random-anchor"),
}

PAPER_A_BACKBONES = ("vit_in21k", "dinov2_b14", "clip_b16")
PAPER_B_BACKBONES = ("dinov2_b14", "clip_b16", "mae_b16", "resnet50", "vit_in21k")


# ---------------------------------------------------------------------------
# Datasets and task splits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    key: str
    label: str
    n_classes: int
    T: int                       # tasks
    C: int                       # classes per task (0 for domains-as-tasks)
    papers: Tuple[str, ...]
    domains_as_tasks: bool = False
    domains: Tuple[str, ...] = ()
    note: str = ""


DATASETS: Dict[str, Dataset] = {
    "cifar100": Dataset("cifar100", "CIFAR-100", 100, 10, 10, ("A", "B"),
                        note="torchvision CIFAR100; 32x32 resized to 224"),
    "imagenet_r": Dataset("imagenet_r", "ImageNet-R", 200, 10, 20, ("A", "B"),
                          note="Hendrycks ImageNet-R (30k images); 80/20 stratified train/test split, seed 0 (the RevisitingCIL split is 24k/6k)"),
    "cub200": Dataset("cub200", "CUB-200", 200, 10, 20, ("A",),
                      note="CUB_200_2011 official train/test lists"),
    "omnibenchmark": Dataset("omnibenchmark", "OmniBenchmark", 300, 10, 30, ("A",),
                             note="RevisitingCIL 300-class OmniBenchmark release"),
    "imagenet100": Dataset("imagenet100", "ImageNet-100", 100, 10, 10, ("B",),
                           note="PILOT 100-class ILSVRC-2012 subset; owner-gated data"),
    "imagenet1k": Dataset("imagenet1k", "ImageNet-1K", 1000, 10, 100, ("B",),
                          note="ILSVRC-2012; descoped from the camera-ready run (gated, 150 GB)"),
    "domainnet": Dataset("domainnet", "DomainNet", 345, 6, 0, ("B",), domains_as_tasks=True,
                         domains=("clipart", "infograph", "painting", "quickdraw", "real", "sketch"),
                         note="six domains as tasks, shared 345-class label space (assumption-violating stress test)"),
}

PAPER_A_DATASETS = ("cifar100", "imagenet_r", "cub200", "omnibenchmark")
PAPER_B_DATASETS = ("cifar100", "imagenet100", "imagenet_r", "domainnet")   # imagenet1k descoped

# Paper B's tab:fix-benchmark rows (backbone, dataset), in print order.
PAPER_B_BENCHMARK_ROWS: Tuple[Tuple[str, str], ...] = (
    ("dinov2_b14", "cifar100"), ("dinov2_b14", "imagenet100"), ("dinov2_b14", "imagenet1k"),
    ("clip_b16", "cifar100"), ("clip_b16", "imagenet100"),
    ("mae_b16", "imagenet100"), ("resnet50", "imagenet100"),
    ("vit_in21k", "imagenet1k"),
    ("dinov2_b14", "domainnet"), ("clip_b16", "domainnet"),
)


def paper_a_configs() -> List[Tuple[str, str]]:
    return [(b, d) for b in PAPER_A_BACKBONES for d in PAPER_A_DATASETS]


def class_ordering(dataset: str, seed: int) -> np.ndarray:
    """Random class ordering for one seed (seeds 0-4), or the canonical order for seed=None."""
    n = DATASETS[dataset].n_classes
    if seed is None:
        return np.arange(n)
    return rng(f"ordering:{dataset}", seed).permutation(n)


def tasks_from_ordering(dataset: str, ordering: np.ndarray) -> List[np.ndarray]:
    """Consecutive chunks of C classes; the remainder (if any) joins the last task."""
    ds = DATASETS[dataset]
    if ds.domains_as_tasks:
        raise ValueError("domains-as-tasks datasets split by domain, not by class")
    chunks = [np.sort(ordering[t * ds.C:(t + 1) * ds.C]) for t in range(ds.T)]
    rem = ordering[ds.T * ds.C:]
    if rem.size:
        chunks[-1] = np.sort(np.concatenate([chunks[-1], rem]))
    return chunks


def task_of_class(dataset: str, ordering: np.ndarray) -> np.ndarray:
    toc = np.full(DATASETS[dataset].n_classes, -1, dtype=np.int64)
    for t, cls in enumerate(tasks_from_ordering(dataset, ordering)):
        toc[cls] = t
    return toc


def override_task_split(overrides: Optional[str] = None) -> List[dict]:
    """Apply per-dataset task-split overrides in place (tab:sens-tasks).

    Spec: comma-separated ``dataset=T`` or ``dataset=TxC`` entries, e.g.
    ``"cifar100=5x20,imagenet_r=20x10"``.  ``T`` replaces the task count and
    ``C`` the classes-per-task of the ``Dataset`` record; the ``T``-only form
    requires ``n_classes % T == 0`` and derives ``C = n_classes // T``
    (otherwise give both).  Validation: dataset known, not domains-as-tasks
    (DomainNet splits by domain), ``1 <= T <= n_classes``, ``C >= 1`` and
    ``T*C <= n_classes`` (a remainder joins the last task per
    ``tasks_from_ordering``).

    Because every consumer (``class_ordering`` / ``tasks_from_ordering`` /
    ``task_of_class`` / the analysis drivers) reads ``DATASETS`` at call
    time, this mutation is the whole pass-through.  Backward compatible:
    ``None`` / ``""`` returns ``[]`` and leaves ``DATASETS`` untouched.
    Returns one record per entry (old/new T and C) for logging.
    """
    if not overrides:
        return []
    records: List[dict] = []
    for item in overrides.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--tasks entry {item!r}: expected dataset=T or dataset=TxC")
        name, _, spec = item.partition("=")
        name = name.strip()
        if name not in DATASETS:
            raise ValueError(f"--tasks: unknown dataset {name!r} (known: {sorted(DATASETS)})")
        ds = DATASETS[name]
        if ds.domains_as_tasks:
            raise ValueError(f"--tasks: {name} splits by domain, not by class; T/C do not apply")
        if "x" in spec:
            t_s, _, c_s = spec.partition("x")
            T, C = int(t_s), int(c_s)
        else:
            T, C = int(spec), None
        if not (1 <= T <= ds.n_classes):
            raise ValueError(f"--tasks {name}: T={T} outside [1, {ds.n_classes}]")
        if C is None:
            if ds.n_classes % T:
                raise ValueError(f"--tasks {name}: n_classes {ds.n_classes} not divisible by T={T}; give dataset=TxC")
            C = ds.n_classes // T
        if C < 1 or T * C > ds.n_classes:
            raise ValueError(f"--tasks {name}: C={C} with T={T} invalid for {ds.n_classes} classes")
        DATASETS[name] = dataclasses.replace(ds, T=T, C=C)
        records.append({"dataset": name, "T_old": ds.T, "C_old": ds.C, "T": T, "C": C})
    return records


# ---------------------------------------------------------------------------
# Head-training protocols (as printed)
# ---------------------------------------------------------------------------

HEAD_PROTOCOL = {
    # Paper B app:experiments: AdamW lr 1e-3, batch 128, wd 1e-4 on W and b, cosine to 1e-6, 50 epochs/task
    "B": dict(optimizer="adamw", lr=1e-3, batch=128, lam=1e-4, epochs=50, eta_min=1e-6, decay_bias=True),
    # Paper A app:tc-gap: AdamW wd 1e-4, 20 epochs, lr 1e-3
    "A": dict(optimizer="adamw", lr=1e-3, batch=128, lam=1e-4, epochs=20, eta_min=1e-6, decay_bias=True),
}

# Paper A Algorithm 1 / app:alpha-estimation constants
K_WARM = 1000
HILL_K = 50
N_BOOTSTRAP = 1000
K_SWEEP = (500, 2000, 8000, 32000)
DRIFT_LRS = (1e-5, 1e-4, 1e-3)


# ---------------------------------------------------------------------------
# Printed context (never asserted; used only for the paper-vs-reproduced diff)
# ---------------------------------------------------------------------------

PRINTED: Dict[str, object] = {
    "A.tab:gamma-min": {("vit_in21k", "cifar100"): (8.3, 0.4), ("vit_in21k", "imagenet_r"): (5.2, 0.3),
                        ("vit_in21k", "cub200"): (12.1, 0.7), ("vit_in21k", "omnibenchmark"): (9.6, 0.5),
                        ("dinov2_b14", "cifar100"): (47.2, 1.2), ("dinov2_b14", "imagenet_r"): (39.1, 1.0),
                        ("dinov2_b14", "cub200"): (51.8, 1.4), ("dinov2_b14", "omnibenchmark"): (44.6, 1.1),
                        ("clip_b16", "cifar100"): (23.1, 0.8), ("clip_b16", "imagenet_r"): (18.9, 0.6),
                        ("clip_b16", "cub200"): (31.4, 0.9), ("clip_b16", "omnibenchmark"): (26.3, 0.7)},
    "A.tab:random-anchor": {"cifar100": (0.31, 0.189), "imagenet_r": (0.28, 0.211),
                            "cub200": (0.58, 0.159), "omnibenchmark": (0.42, 0.174)},
    "A.slopes": dict(per_backbone={"vit_in21k": -0.141, "dinov2_b14": -0.099, "clip_b16": -0.118},
                     pooled12=(-0.115, -0.128, -0.102), pooled16=(-0.119, -0.134, -0.105),
                     r2_per_backbone={"vit_in21k": 0.93, "dinov2_b14": 0.96, "clip_b16": 0.95}, r2_pooled12=0.97,
                     interaction_F=(2.14, 2, 6, 0.199)),
    "A.predictor": dict(r2_single={"gamma": 0.88, "probe": 0.85, "ncm": 0.72}, delta_r2=0.06, p=0.042),
    "A.alpha": {"vit_in21k": 1.32, "dinov2_b14": 1.28, "clip_b16": 1.35},
    "A.ecf_slopes": dict(clipped=-0.31, unclipped=-0.10),
    "A.drift": dict(slopes={1e-5: -0.164, 1e-4: -1.73}, t_star={1e-5: 14.2, 1e-4: 2.35, 1e-3: 0.21},
                    time_to_floor=0.48, gamma0=8.20, secants=(-8.20, -4.10), corr=0.998, crossover=3.18),
    "B.tab:fix-benchmark": {("dinov2_b14", "cifar100"): (82.3, 86.5, 86.8), ("dinov2_b14", "imagenet100"): (80.5, 84.7, 85.0),
                            ("dinov2_b14", "imagenet1k"): (76.8, 81.8, 82.3), ("clip_b16", "cifar100"): (81.5, 85.7, 86.0),
                            ("clip_b16", "imagenet100"): (79.2, 83.5, 83.8), ("mae_b16", "imagenet100"): (65.3, 77.2, 78.5),
                            ("resnet50", "imagenet100"): (62.1, 73.8, 75.2), ("vit_in21k", "imagenet1k"): (78.5, 83.2, 83.5),
                            ("dinov2_b14", "domainnet"): (58.2, 68.5, 70.8), ("clip_b16", "domainnet"): (56.8, 66.8, 68.5)},
    "B.bias_direction": dict(cos_range_natural=(0.92, 0.98), cv_range_natural=(0.05, 0.16),
                             cos_range_domainnet=(0.78, 0.86), cv_range_domainnet=(0.22, 0.32)),
    "B.nonvision": dict(WT=64.20, FIX=71.50, J=73.84, gain=7.30, cos=0.93, cv=0.12),
}
