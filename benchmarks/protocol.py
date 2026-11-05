"""benchmarks.protocol -- the class-incremental-learning (CIL) protocol engine.

Implemented by agent H08 (implementation wave), 2026-10-21, replacing the
G09 skeleton placeholder. Spec sources (binding):
  - exec/G03_log.md SPEC 8 (paper B benchmark protocol: datasets, orderings
    seeds 0-4, head-training recipe), SPEC 9 (20 Newsgroups nonvision).
  - exec/G02_log.md S2/S3 (paper A: 12 pretrained configs = 3 backbones x
    {Split-CIFAR-100, Split-ImageNet-R, Split-CUB-200, Split-OmniBenchmark};
    4 random-anchor configs = random-init ViT-B/16 on the same datasets) and
    OWNER-ASK 5/12 (canonical-vs-random class-ordering tension).
  - exec/G01_log.md port map items 1/4 (bank partition idiom: shuffle over
    the class list with numpy RandomState seeds, then in-order consecutive
    task blocks, remainder to the last task).
  - exec/G04_log.md (dataset dossier backing benchmarks/datasets.py).

Scope: task construction from class orderings, split definitions (T x C
registry), incremental stage schedules, evaluation hooks, aggregation, and
the seeds policy. Pixel/text loading lives in benchmarks/datasets.py;
backbone loading lives in benchmarks/backbones.py; head training lives in
core/heads.py. This module is pure python+numpy: no network, no disk I/O,
no torch -- importable on the bare local environment (numpy 2.5.2).

ORDERING-MODE LAW (G02 OWNER-ASK 5 + G03 SPEC 8 tension; coordinator
ruling: implement BOTH modes, default per paper):
  - mode "random":    np.random.RandomState(seed).shuffle over
                      list(range(n_classes)), then consecutive blocks of
                      classes per task. Seeds 0-4 = the paper-standard five
                      random class orderings; tables report mean +- SD.
  - mode "canonical": PILOT/RanPAC canonical orderings. The actual external
                      ordering lists are NOT on this machine (G02 OWNER-ASK
                      12), so canonical mode uses the identity order 0..N-1
                      (the standard PILOT in-order convention, same as the
                      bank arms' partition="inorder") and sets the flag
                      `external_lists_owed` on the ordering object. When the
                      owner supplies the real lists, register them in
                      CANONICAL_ORDERINGS below; nothing else changes.
  Per-paper defaults (DEFAULT_ORDERING_MODE): paper A body text says
  "task partitions following the canonical PILOT/RanPAC class orderings"
  (tab:gamma-min), paper B appendix says random class orderings seeds 0-4,
  so A defaults to canonical and B to random. Both modes are always
  constructible for every dataset.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SEEDS",
    "ORDERING_MODES",
    "DEFAULT_ORDERING_MODE",
    "SplitSpec",
    "SPLITS",
    "split_spec",
    "ClassOrdering",
    "Task",
    "StagePlan",
    "StageHook",
    "AccuracyMatrix",
    "canonical_ordering",
    "random_ordering",
    "make_ordering",
    "iter_orderings",
    "build_tasks",
    "seen_classes",
    "iter_stages",
    "run_incremental_schedule",
    "mean_sd",
    "HEAD_TRAINING",
    "PAPER_A_BACKBONES",
    "PAPER_B_BACKBONES",
    "PAPER_A_DATASETS",
    "PAPER_B_DATASETS",
    "paper_a_configs",
    "derive_seed",
    "ordering_fingerprint",
]

# ---------------------------------------------------------------------------
# Seeds policy
# ---------------------------------------------------------------------------

#: The paper-standard five random class orderings (G03 SPEC 8 "seeds 0-4
#: (5 orderings); mean +- SD"; G02 S2 "mean +- std, 5 orderings"). These
#: index class permutations and are the only seed axis the papers report
#: over. Bank-style robustness seeds outside 0-4 (e.g. 101-105) remain
#: constructible via random_ordering(dataset, seed).
SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)


def derive_seed(base: int, purpose: str, salt: int = 0) -> int:
    """SHA-256 child-seed derivation for auxiliary streams (anchor init
    noise, synthetic data, within-task shuffling).

    Mirrors tests/conftest.py::derive_seed (G10 policy: Python's builtin
    hash() is process-salted and forbidden). conftest fixes base=SEED=0 for
    the test suite; production callers here pass their ordering seed as
    `base` when a per-seed stream is needed. Deterministic across
    processes and platforms.
    """
    digest = hashlib.sha256(f"{base}|{purpose}|{salt}".encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**63)


# ---------------------------------------------------------------------------
# Split definitions (T x C registry)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitSpec:
    """Definition of one benchmark split.

    Class-disjoint CIL (shared_labels=False): n_tasks blocks of
    classes_per_task classes drawn from the class ordering; a nonzero
    remainder (n_classes % n_tasks) joins the LAST task (bank idiom, e.g.
    the pets T=9/C=5 arm where the last task holds 2 classes).

    Shared-label domain-incremental (shared_labels=True, task_axis=
    "domain"; DomainNet): every task carries the FULL class set and tasks
    are domains -- paper B's stress configuration that violates the
    disjoint-label assumption (G03 SPEC 8(d)).
    """

    name: str
    n_classes: int
    n_tasks: int
    classes_per_task: int
    #: Allowed T variants (paper-A sensitivity tab:sens-tasks T in
    #: {5,10,20}; DomainNet T in {5,6} hedge; bank arms add 2/50).
    task_grid: Tuple[int, ...] = ()
    #: Native resolution and the resize target every backbone pipeline
    #: uses (G03 SPEC 8: CIFAR-100 32x32 resized to 224x224).
    native_size: Optional[Tuple[int, int]] = None
    resize: Optional[int] = None
    shared_labels: bool = False
    task_axis: str = "classes"   # "classes" | "domain"
    kind: str = "image"          # "image" | "text"
    expected_train: Optional[int] = None   # paper-recorded sizes (provenance)
    expected_test: Optional[int] = None
    papers: Tuple[str, ...] = ()
    note: str = ""

    @property
    def is_disjoint(self) -> bool:
        return not self.shared_labels

    def check(self) -> None:
        if self.shared_labels:
            if self.classes_per_task != self.n_classes:
                raise ValueError(
                    f"{self.name}: shared-label split must carry the full "
                    f"class set per task "
                    f"({self.classes_per_task} != {self.n_classes})"
                )
        else:
            if self.n_classes < self.n_tasks:
                raise ValueError(f"{self.name}: fewer classes than tasks")
            if self.n_tasks * self.classes_per_task > self.n_classes:
                raise ValueError(
                    f"{self.name}: {self.n_tasks}x{self.classes_per_task} "
                    f"exceeds {self.n_classes} classes"
                )


def _registry() -> Dict[str, SplitSpec]:
    """Per-paper split definitions. Sources: G03 SPEC 8 (a)-(d) + L879
    ImageNet-R 10x20; G02 S2 (paper-A dataset quartet); G03 SPEC 9 (20NG
    5x4); PILOT/RanPAC standard splits for CUB (10x20) and OmniBenchmark
    (10x30) as cited by G02 S2 / G04; paper-A sensitivity grid T in
    {5,10,20} on CIFAR-100 (G01 architecture item 13)."""
    specs = [
        SplitSpec(
            name="split-cifar100", n_classes=100, n_tasks=10,
            classes_per_task=10, task_grid=(2, 5, 10, 20, 50),
            native_size=(32, 32), resize=224, expected_train=50_000,
            expected_test=10_000, papers=("A", "B"),
            note="G03 SPEC 8(a): 10 tasks x 10 classes, 50k train / 10k "
                 "test, 32x32 resized to 224x224. task_grid covers the "
                 "paper-A sensitivity study T in {5,10,20} (2/50 kept from "
                 "the bank arms).",
        ),
        SplitSpec(
            name="split-imagenet-r", n_classes=200, n_tasks=10,
            classes_per_task=20, resize=224, papers=("A", "B"),
            note="G03 SPEC 8 open integrity flag + L879: T = 10 x 20; "
                 "paper-A quartet member (G02 S2) and paper-B "
                 "rate-validation; ~30k images, no local copy (G04).",
        ),
        SplitSpec(
            name="split-cub-200", n_classes=200, n_tasks=10,
            classes_per_task=20, resize=224, expected_train=5_994,
            expected_test=5_794, papers=("A",),
            note="CUB-200-2011, PILOT/RanPAC 10 x 20; 11,788 images total; "
                 "paper-A quartet member (G02 S2, G04 table).",
        ),
        SplitSpec(
            name="split-omnibenchmark", n_classes=300, n_tasks=10,
            classes_per_task=30, resize=224, papers=("A",),
            note="OmniBenchmark 300-class CIL split, PILOT/RanPAC 10 x 30; "
                 "paper-A quartet member (G02 S2, G04 table).",
        ),
        SplitSpec(
            name="split-imagenet-100", n_classes=100, n_tasks=10,
            classes_per_task=10, resize=224, expected_train=130_000,
            papers=("B",),
            note="G03 SPEC 8(b): ILSVRC-2012 100-class subset, ~130k train; "
                 "PILOT class lists; owner-gated derivation per G04.",
        ),
        SplitSpec(
            name="split-imagenet-1k", n_classes=1000, n_tasks=10,
            classes_per_task=100, resize=224, expected_test=50_000,
            papers=("B",),
            note="G03 SPEC 8(c): 10 tasks x 100 classes; ~150GB, gated "
                 "(owner-only per G04).",
        ),
        SplitSpec(
            name="domainnet", n_classes=345, n_tasks=6, classes_per_task=345,
            task_grid=(5, 6), resize=224, shared_labels=True,
            task_axis="domain", papers=("B",),
            note="G03 SPEC 8(d): domains-as-tasks shared-label stress test "
                 "(violates the disjoint-label assumption); T in {5,6} is "
                 "the recorded hedge -- default 6 domains; task_grid "
                 "records both (G03 OWNER-ASK 11 owed).",
        ),
        SplitSpec(
            name="twenty-newsgroups", n_classes=20, n_tasks=5,
            classes_per_task=4, kind="text", papers=("B",),
            note="G03 SPEC 9: 20 classes -> 5 disjoint-label tasks x 4 "
                 "classes; BY-DATE split, headers/footers/quotes removed; "
                 "frozen all-MiniLM-L6-v2 features (backbones.py domain).",
        ),
    ]
    table: Dict[str, SplitSpec] = {}
    for s in specs:
        s.check()
        table[s.name] = s
    return table


#: The split registry. Names are the canonical dataset keys accepted by
#: benchmarks/datasets.py::get_dataset.
SPLITS: Dict[str, SplitSpec] = _registry()


def split_spec(name: str) -> SplitSpec:
    try:
        return SPLITS[name]
    except KeyError:
        raise KeyError(
            f"unknown split '{name}'; registered: {sorted(SPLITS)}"
        ) from None


# ---------------------------------------------------------------------------
# Class orderings
# ---------------------------------------------------------------------------

ORDERING_MODES: Tuple[str, ...] = ("canonical", "random")

#: Per-paper ordering-mode defaults (H08 ruling from the two spec logs; the
#: owner can flip either key -- both modes are always constructible):
#:   paper A (4828): tab:gamma-min body text names the canonical
#:     PILOT/RanPAC class orderings -> default "canonical".
#:   paper B (12500): app:experiments uses 5 random class orderings,
#:     seeds 0-4 -> default "random".
DEFAULT_ORDERING_MODE: Dict[str, str] = {"A": "canonical", "B": "random"}

#: Registry for the external PILOT/RanPAC canonical ordering lists once the
#: owner supplies them (G02 OWNER-ASK 12): dataset name -> verbatim
#: class-index sequence. Empty = identity order stands in and ordering
#: objects carry external_lists_owed=True.
CANONICAL_ORDERINGS: Dict[str, Sequence[int]] = {}


@dataclass(frozen=True)
class ClassOrdering:
    """A total order over class indices, sliced into tasks by build_tasks."""

    dataset: str
    mode: str                    # "canonical" | "random"
    order: Tuple[int, ...]       # permutation of range(n_classes)
    seed: Optional[int] = None   # set iff mode == "random"
    external_lists_owed: bool = False
    source: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ORDERING_MODES:
            raise ValueError(f"mode must be one of {ORDERING_MODES}")
        order = list(self.order)
        if sorted(order) != list(range(len(order))):
            raise ValueError("order must be a permutation of range(n_classes)")
        if (self.seed is None) != (self.mode == "canonical"):
            raise ValueError("seed set iff mode == 'random'")
        spec = split_spec(self.dataset)
        if len(order) != spec.n_classes:
            raise ValueError(
                f"{self.dataset} ordering has {len(order)} classes, "
                f"registry says {spec.n_classes}"
            )

    @property
    def n_classes(self) -> int:
        return len(self.order)

    def task_blocks(self, n_tasks: Optional[int] = None) -> List[Tuple[int, ...]]:
        """Slice the order into per-task class blocks.

        Consecutive blocks along the ordering (bank idiom: in-order blocks
        after the shuffle); a remainder (n_classes % n_tasks) joins the
        LAST block. Shared-label splits repeat the full set per task.
        """
        spec = split_spec(self.dataset)
        t = spec.n_tasks if n_tasks is None else n_tasks
        if t < 1:
            raise ValueError("n_tasks must be >= 1")
        if spec.shared_labels:
            return [tuple(self.order) for _ in range(t)]
        base = spec.n_classes // t
        blocks = [tuple(self.order[i * base:(i + 1) * base]) for i in range(t)]
        remainder = tuple(self.order[t * base:])
        if remainder:
            blocks[-1] = tuple(blocks[-1]) + remainder
        return blocks

    def fingerprint(self) -> str:
        """Short stable id for logs/manifests (process-independent)."""
        payload = (
            f"{self.dataset}|{self.mode}|{self.seed}|"
            + ",".join(map(str, self.order))
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def ordering_fingerprint(ordering: ClassOrdering) -> str:
    return ordering.fingerprint()


def canonical_ordering(dataset: str) -> ClassOrdering:
    """Canonical PILOT/RanPAC ordering.

    The true external lists are owed (G02 OWNER-ASK 12); until they are
    registered in CANONICAL_ORDERINGS this returns the identity order
    0..N-1 (the PILOT in-order convention used by the bank arms) with
    external_lists_owed=True so downstream consumers can flag the
    substitution in results-store `flags`.
    """
    spec = split_spec(dataset)
    external = CANONICAL_ORDERINGS.get(dataset)
    if external is not None:
        order = tuple(int(c) for c in external)
        owed, source = False, (
            "PILOT/RanPAC canonical list, owner-supplied (registered)"
        )
    else:
        order = tuple(range(spec.n_classes))
        owed, source = True, (
            "identity 0..N-1 standing in for the PILOT/RanPAC canonical "
            "list (external lists owed, G02 OWNER-ASK 12)"
        )
    return ClassOrdering(
        dataset=dataset, mode="canonical", order=order, seed=None,
        external_lists_owed=owed, source=source,
    )


def random_ordering(dataset: str, seed: int) -> ClassOrdering:
    """Random class ordering for one seed.

    Bank idiom (G01 port item 4; stage_a / cub_partition_variant lineage):
    np.random.RandomState(seed).shuffle over list(range(n_classes)).
    RandomState (MT19937) is pinned deliberately -- the bank's recorded
    partitions used it. Generator/PCG64 is used only for auxiliary streams
    via derive_seed(), matching the G10 conftest policy. Paper seeds are
    0-4; bank-style robustness seeds (101-105) also work.
    """
    spec = split_spec(dataset)
    order = list(range(spec.n_classes))
    np.random.RandomState(seed).shuffle(order)
    return ClassOrdering(
        dataset=dataset, mode="random", order=tuple(int(c) for c in order),
        seed=int(seed),
        source=f"RandomState({seed}).shuffle over range({spec.n_classes})",
    )


def make_ordering(
    dataset: str, mode: Optional[str] = None, *,
    seed: Optional[int] = None, paper: Optional[str] = None,
) -> ClassOrdering:
    """Dispatch on mode. `paper` ("A" | "B") selects the per-paper default
    when mode is None; the bare default (no mode, no paper) is "random"
    (paper B appendix convention)."""
    if mode is None:
        mode = DEFAULT_ORDERING_MODE.get(paper or "", "random")
    if mode == "random":
        if seed is None:
            raise ValueError("random ordering requires an explicit seed")
        return random_ordering(dataset, seed)
    if mode == "canonical":
        if seed is not None:
            raise ValueError("canonical ordering takes no seed")
        return canonical_ordering(dataset)
    raise ValueError(f"unknown ordering mode '{mode}'")


def iter_orderings(
    dataset: str, mode: Optional[str] = None, *,
    seeds: Sequence[int] = SEEDS, paper: Optional[str] = None,
) -> List[ClassOrdering]:
    """The orderings an experiment aggregates over.

    canonical mode yields ONE ordering (deterministic; nothing varies), so
    spreads measured under canonical mode come from training-stream seeds
    alone -- recorded here so no caller silently mistakes n=1 for n=5.
    """
    if mode is None:
        mode = DEFAULT_ORDERING_MODE.get(paper or "", "random")
    if mode == "canonical":
        return [canonical_ordering(dataset)]
    return [random_ordering(dataset, s) for s in seeds]


# ---------------------------------------------------------------------------
# Tasks and incremental stages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """One incremental stage's training population."""

    task_id: int                      # 0-based
    classes: Tuple[int, ...]          # class indices carried by this task
    domain: Optional[str] = None      # domain label when task_axis == "domain"

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def build_tasks(
    ordering: ClassOrdering, n_tasks: Optional[int] = None
) -> Tuple[Task, ...]:
    """Materialize the T tasks from an ordering.

    Validates disjointness and exhaustiveness for class-disjoint splits
    (the CIL invariant the theory needs) and full-set identity for
    shared-label domain splits.
    """
    spec = split_spec(ordering.dataset)
    t = spec.n_tasks if n_tasks is None else n_tasks
    if t < 1:
        raise ValueError("n_tasks must be >= 1")
    if spec.shared_labels:
        full = tuple(ordering.order)
        return tuple(
            Task(task_id=i, classes=full, domain=f"domain_{i}")
            for i in range(t)
        )
    blocks = ordering.task_blocks(t)
    tasks = tuple(Task(task_id=i, classes=b) for i, b in enumerate(blocks))
    flat = [c for b in blocks for c in b]
    if len(flat) != spec.n_classes or set(flat) != set(range(spec.n_classes)):
        raise RuntimeError("task blocks must cover every class exactly once")
    return tasks


def seen_classes(tasks: Sequence[Task], upto: int) -> Tuple[int, ...]:
    """Union of class indices of tasks 0..upto (inclusive), first-seen
    order -- the evaluation label space after stage `upto`."""
    out: List[int] = []
    seen: set = set()
    for task in tasks[: upto + 1]:
        for c in task.classes:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return tuple(out)


@dataclass(frozen=True)
class StagePlan:
    """One incremental stage: train on task t, evaluate on everything seen.

    eval_classes = union of seen tasks' classes for class-disjoint splits
    (standard CIL), or the full shared label space from stage 0 for
    domain-incremental splits (G03 SPEC 8(d) stress protocol).
    """

    stage: int                 # == task_id of this stage's training task
    train_task: Task
    seen_tasks: Tuple[Task, ...]
    eval_classes: Tuple[int, ...]

    @property
    def n_train_classes(self) -> int:
        return self.train_task.n_classes

    @property
    def n_eval_classes(self) -> int:
        return len(self.eval_classes)


def _is_shared_label(tasks: Sequence[Task]) -> bool:
    """Infer shared-label semantics from the task set itself: every task
    carries the identical multi-class set."""
    if not tasks:
        return False
    first = set(tasks[0].classes)
    return len(first) > 1 and all(set(t.classes) == first for t in tasks)


def iter_stages(
    tasks: Sequence[Task], spec: Optional[SplitSpec] = None
) -> Iterable[StagePlan]:
    """Yield the T stage plans in sequence. Evaluation happens after each
    stage's training (paper protocol: accuracy matrix over all seen
    classes; every stage trains ONLY its own task's data)."""
    shared = spec.shared_labels if spec is not None else _is_shared_label(tasks)
    for i, task in enumerate(tasks):
        if shared:
            eval_classes = tuple(dict.fromkeys(
                c for t in tasks for c in t.classes
            ))  # full shared label space from stage 0
        else:
            eval_classes = seen_classes(tasks, i)
        yield StagePlan(
            stage=i, train_task=task, seen_tasks=tuple(tasks[: i + 1]),
            eval_classes=eval_classes,
        )


# Evaluation hook interface ------------------------------------------------

#: A hook fires after each stage with the report dict: protocol guarantees
#: 'stage', 'train_classes', 'eval_classes' and passes trainer-produced
#: metrics through untouched.
StageHook = Callable[[Dict], None]


class AccuracyMatrix:
    """T x T accuracy recorder (rows = stages, cols = seen task blocks).

    matrix[s, j] = accuracy on task j's classes measured after stage s;
    unmeasured cells hold NaN (task not yet seen). `final_row()` is the
    last-stage vector; `class_weighted_average()` is the standard CIL
    average accuracy (classes weighted equally, so task blocks weighted by
    their class counts).
    """

    def __init__(self, tasks: Sequence[Task]) -> None:
        self.tasks = tuple(tasks)
        self.n = len(self.tasks)
        self.matrix: np.ndarray = np.full((self.n, self.n), np.nan)
        self._counts = np.array([t.n_classes for t in self.tasks], dtype=float)

    def record(self, stage: int, per_task: Sequence[float]) -> None:
        """Record stage `stage` accuracies over the seen task blocks
        (per_task[j] for j <= stage; trailing NaN padding is fine)."""
        if not 0 <= stage < self.n:
            raise IndexError(f"stage {stage} out of range 0..{self.n - 1}")
        row = np.full(self.n, np.nan)
        row[: len(per_task)] = per_task
        self.matrix[stage] = row

    def final_row(self) -> np.ndarray:
        return self.matrix[-1]

    def class_weighted_average(self, row: Optional[np.ndarray] = None) -> float:
        r = self.matrix[-1] if row is None else np.asarray(row, dtype=float)
        mask = ~np.isnan(r)
        if not mask.any():
            return float("nan")
        return float(
            np.sum(r[mask] * self._counts[mask]) / np.sum(self._counts[mask])
        )


def run_incremental_schedule(
    tasks: Sequence[Task],
    train_stage: Callable[[Task, int], Dict],
    eval_stage: Callable[[StagePlan], Sequence[float]],
    hooks: Sequence[StageHook] = (),
    spec: Optional[SplitSpec] = None,
) -> AccuracyMatrix:
    """Drive one incremental run end to end.

    `train_stage(task, stage)` trains that stage's head (core/heads.py owns
    the recipe: HEAD_TRAINING below) and returns an opaque metrics dict;
    `eval_stage(plan)` returns per-task-block accuracies over the plan's
    seen tasks (length plan.stage + 1 for disjoint splits, n for
    shared-label). Hooks fire after each stage. Pure orchestration: no
    pixels, no torch, no disk.
    """
    matrix = AccuracyMatrix(tasks)
    for plan in iter_stages(tasks, spec=spec):
        metrics = train_stage(plan.train_task, plan.stage)
        per_task = list(eval_stage(plan))
        expected = len(plan.seen_tasks)
        if len(per_task) != expected:
            raise ValueError(
                f"stage {plan.stage}: eval returned {len(per_task)} "
                f"per-task values, expected {expected} (seen tasks)"
            )
        matrix.record(plan.stage, per_task)
        report: Dict = {
            "stage": plan.stage,
            "train_classes": plan.train_task.classes,
            "eval_classes": plan.eval_classes,
        }
        report.update(metrics)
        for hook in hooks:
            hook(report)
    return matrix


# ---------------------------------------------------------------------------
# Aggregation over seeds
# ---------------------------------------------------------------------------


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    """mean +- SD over the five orderings (G02 S2/S3, G03 SPEC 8).
    Sample SD (ddof=1), the convention for the n=5 reported spreads."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("no values")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


# ---------------------------------------------------------------------------
# Head-training protocol constants (G03 SPEC 8, L2034-2048)
# ---------------------------------------------------------------------------

HEAD_TRAINING: Dict[str, object] = {
    "head": "linear over frozen features",
    "optimizer": "AdamW",
    "lr": 1e-3,
    "batch_size": 128,
    "weight_decay": 1e-4,       # lambda; applied to weights AND biases
                                # (selects the canonical gauge; G03 SPEC 1)
    "schedule": "cosine",
    "eta_min": 1e-6,
    "warmup_epochs": 0,
    "epochs_per_task": 50,
    "augmentation": "none",
    "normalization": "per-backbone",
    "joint_baseline": "eq:ce-full on the union of all tasks, same settings",
}

#: Paper B backbone HF ids (G03 SPEC 8 L1983-1997; d = feature dim).
PAPER_B_BACKBONES: Dict[str, Dict[str, object]] = {
    "dinov2-vitb14": {"hf_id": "facebook/dinov2-base", "d": 768},
    "vitb16-in21k": {"hf_id": "google/vit-base-patch16-224-in21k", "d": 768},
    "clip-vitb16": {"hf_id": "openai/clip-vit-base-patch16", "d": 768},
    "mae-vitb16": {"hf_id": "facebook/vit-mae-base", "d": 768},
    "resnet-50": {"hf_id": "microsoft/resnet-50", "d": 2048},
}

#: Paper A backbones (G02 S2/S3): the pretrained trio plus the random-anchor
#: recipe (random-init ViT-B/16, no pretraining, zero download per G04).
PAPER_A_BACKBONES: Dict[str, Dict[str, object]] = {
    "vitb16-in21k": {"hf_id": "google/vit-base-patch16-224-in21k", "d": 768},
    "dinov2-vitb14": {"hf_id": "facebook/dinov2-base", "d": 768},
    "clip-vitb16": {"hf_id": "openai/clip-vit-base-patch16", "d": 768},
    "random-vitb16": {
        "hf_id": None, "d": 768, "random_init": True,
        "note": "randomly initialised ViT-B/16 anchor; timm "
                "create_model(pretrained=False); init seed via "
                "derive_seed(ordering_seed, 'anchor')",
    },
}

#: Paper-A dataset quartet (G02 S2).
PAPER_A_DATASETS: Tuple[str, ...] = (
    "split-cifar100", "split-imagenet-r", "split-cub-200", "split-omnibenchmark",
)
#: Paper-B benchmark datasets (G03 SPEC 8; DomainNet = shared-label stress).
PAPER_B_DATASETS: Tuple[str, ...] = (
    "split-cifar100", "split-imagenet-100", "split-imagenet-1k",
    "split-imagenet-r", "domainnet",
)


def paper_a_configs() -> List[Dict[str, object]]:
    """The 16 paper-A configurations: 12 pretrained (3 backbones x 4
    datasets) + 4 random-anchor (random-init ViT-B/16 x the same
    datasets), each carrying its per-paper ordering default."""
    configs: List[Dict[str, object]] = []
    for bb in ("vitb16-in21k", "dinov2-vitb14", "clip-vitb16"):
        for ds in PAPER_A_DATASETS:
            configs.append({
                "paper": "A", "backbone": bb, "dataset": ds,
                "ordering_mode": DEFAULT_ORDERING_MODE["A"],
                "seeds": SEEDS, "anchor": False,
            })
    for ds in PAPER_A_DATASETS:
        configs.append({
            "paper": "A", "backbone": "random-vitb16", "dataset": ds,
            "ordering_mode": DEFAULT_ORDERING_MODE["A"],
            "seeds": SEEDS, "anchor": True,
        })
    return configs
