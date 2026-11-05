"""benchmarks.datasets -- dataset registry and mode-aware loaders.

Implemented by agent H08 (implementation wave), 2026-10-21, replacing the
G09 skeleton placeholder. Spec sources (binding):
  - exec/G03_log.md SPEC 8 (paper-B datasets: Split-CIFAR-100 10x10
    50k/10k 32->224, ImageNet variants, DomainNet domains-as-tasks) and
    SPEC 9 (20 Newsgroups BY-DATE, headers/footers/quotes removed,
    T=5 x C=4).
  - exec/G02_log.md S2/S3 (paper-A quartet: Split-CIFAR-100,
    Split-ImageNet-R, Split-CUB-200, Split-OmniBenchmark).
  - exec/G04_log.md dossier (what physically exists on this machine:
    CIFAR-100 at C:\\Users\\milad\\Projects\\pap3\\data, sklearn 20NG
    cache at C:\\Users\\milad\\scikit_learn_data; everything else absent;
    Colab fetch commands for the notebooks).

OWNER STORAGE LAW (binding, enforced below):
  1. NO downloads ever happen in local mode. Local mode is strictly
     read-only access to caches that ALREADY exist on this machine.
     A missing local dataset is a hard error (LocalDatasetMissingError),
     never a fetch.
  2. Colab mode DOCUMENTS the fetch (commands, URLs, licences) for the
     notebooks in repo/colab/; this module itself still never executes a
     download -- the notebook runs the plan, then local-style read-only
     loading applies to the fetched files.
  3. Synthetic mode (DEFAULT) builds deterministic synthetic data for the
     test suite; no network, no disk reads.
Enforcement: every code path that could conceivably fetch routes through
enforce_no_fetch() (raises StorageLawViolation); audit_no_fetch() scans
this module's own source for executable network/download calls and is run
in the self-check. The only external-data reader is sklearn's
fetch_20newsgroups called once, pinned to download_if_missing=False.

Split semantics live in benchmarks/protocol.py (SPLITS registry: T x C,
shared-label DomainNet, remainder-to-last-task blocks). This module owns
pixels/text only; features belong to benchmarks/features.py, backbones to
benchmarks/backbones.py.
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # allow direct-script execution
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.protocol import (  # noqa: E402
    SPLITS,
    SplitSpec,
    build_tasks,
    derive_seed,
    random_ordering,
    split_spec,
)

__all__ = [
    "MODES",
    "DEFAULT_MODE",
    "StorageLawViolation",
    "LocalDatasetMissingError",
    "ColabDataNotFetchedError",
    "enforce_no_fetch",
    "audit_no_fetch",
    "FetchPlan",
    "fetch_plan",
    "ArraySplit",
    "DatasetBundle",
    "get_dataset",
    "probe_local",
    "registry_status",
    "DATASETS",
    "ALIASES",
]

# ---------------------------------------------------------------------------
# Modes and the storage-law guard
# ---------------------------------------------------------------------------

MODES: Tuple[str, ...] = ("local", "colab", "synthetic")
DEFAULT_MODE = "synthetic"   # tests run everywhere; local/colab need caches


class StorageLawViolation(RuntimeError):
    """Raised when any code path would fetch data (OWNER STORAGE LAW)."""


class LocalDatasetMissingError(StorageLawViolation):
    """Local mode: the dataset cache does not exist. Never fetched."""


class ColabDataNotFetchedError(StorageLawViolation):
    """Colab mode: the notebook fetch plan has not been run yet."""


def enforce_no_fetch(mode: str, detail: str) -> None:
    """OWNER STORAGE LAW tripwire. Any operation that could fetch data
    must call this first; it always raises (this module never fetches)."""
    raise StorageLawViolation(
        f"FETCH ATTEMPTED IN {mode!r} MODE -- forbidden by the OWNER "
        f"STORAGE LAW. This module never downloads anything. Detail: "
        f"{detail}. Colab fetching is documented in fetch_plan(name) and "
        f"executed by the notebooks in repo/colab/, not here."
    )


#: Executable network/download constructs that must never appear in this
#: module's source. Fetch-plan DOCUMENTATION strings name shell commands
#: for the notebooks; those are data, not calls -- the patterns below are
#: python-executable forms only.
_FORBIDDEN_SOURCE_PATTERNS: Tuple[str, ...] = (
    "urllib" + ".request",                  # audit-exempt: pattern definition
    "urlretriev" + "e(",                    # audit-exempt: pattern definition
    "request" + "s.",                       # audit-exempt: pattern definition
    "download" + "=True",                   # audit-exempt: pattern definition
    "download_if_missing" + "=True",        # audit-exempt: pattern definition
    "socke" + "t.",                         # audit-exempt: pattern definition
    "subproces" + "s.",                     # audit-exempt: pattern definition
    "shutil" + ".which",                    # audit-exempt: pattern definition
)

_AUDIT_EXEMPT_MARK = "# audit-exempt: "


def audit_no_fetch(path: Optional[str] = None) -> List[str]:
    """Self-audit: scan this file for executable fetch calls. Returns the
    list of violations (empty = clean); the self-check asserts empty."""
    src_path = Path(path) if path else Path(__file__)
    lines = src_path.read_text(encoding="utf-8").splitlines()
    violations: List[str] = []
    for lineno, line in enumerate(lines, 1):
        if _AUDIT_EXEMPT_MARK in line:
            continue
        for pattern in _FORBIDDEN_SOURCE_PATTERNS:
            if pattern in line:
                violations.append(f"line {lineno}: {pattern!r} in {line.strip()!r}")
    return violations


# ---------------------------------------------------------------------------
# Local cache registry (verified by G04; probed read-only at call time)
# ---------------------------------------------------------------------------

#: Candidate roots probed per dataset in local mode. First existing root
#: wins; none existing = LocalDatasetMissingError (never a fetch).
LOCAL_PATHS: Dict[str, Tuple[str, ...]] = {
    "split-cifar100": (
        r"C:\Users\milad\Projects\pap3\data\cifar-100-python",   # G04-verified
        str(_REPO_ROOT / "data" / "cifar-100-python"),
    ),
    "twenty-newsgroups": (
        r"C:\Users\milad\scikit_learn_data",                     # G04-verified
        str(_REPO_ROOT / "data" / "scikit_learn_data"),
    ),
    # No verified local copies (G04 table): probe repo-local drop points
    # so an owner-supplied copy is picked up automatically, read-only.
    "split-imagenet-r": (str(_REPO_ROOT / "data" / "imagenet-r"),),
    "split-cub-200": (str(_REPO_ROOT / "data" / "cub200"),),
    "split-omnibenchmark": (str(_REPO_ROOT / "data" / "omnibenchmark"),),
    "split-imagenet-100": (str(_REPO_ROOT / "data" / "imagenet100"),),
    "split-imagenet-1k": (str(_REPO_ROOT / "data" / "imagenet1k"),),
    "domainnet": (str(_REPO_ROOT / "data" / "domainnet"),),
}

#: Colab staging root: where the notebooks put fetched data (Colab /content).
COLAB_STAGING = os.environ.get("CIL_COLAB_DATA", "/content/data")

ALIASES: Dict[str, str] = {
    "split-cub200": "split-cub-200",
    "cifar100": "split-cifar100",
    "20ng": "twenty-newsgroups",
    "twentynews": "twenty-newsgroups",
}


def _canonical(name: str) -> str:
    name = ALIASES.get(name, name)
    if name not in SPLITS:
        raise KeyError(f"unknown dataset '{name}'; registered: {sorted(SPLITS)}")
    return name


def probe_local(name: str) -> Optional[str]:
    """First existing local root for `name`, or None. Pure os.path probes
    (read-only, no filesystem writes, no fetch)."""
    name = _canonical(name)
    for root in LOCAL_PATHS.get(name, ()):
        if os.path.isdir(root):
            return root
    return None


# ---------------------------------------------------------------------------
# Colab fetch plans (DOCUMENTATION ONLY -- never executed here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchPlan:
    """What a colab notebook must fetch for this dataset. G04 dossier
    sources; commands are copied verbatim into repo/colab/ notebooks."""

    name: str
    commands: Tuple[str, ...]
    urls: Tuple[str, ...] = ()
    license_note: str = ""
    size_note: str = ""
    fetched_by: str = "colab notebook (this module never fetches)"

    def as_text(self) -> str:
        lines = [f"FETCH PLAN for {self.name} (executed in colab only):"]
        lines += [f"  $ {c}" for c in self.commands]
        if self.urls:
            lines.append("  sources: " + ", ".join(self.urls))
        for note in (self.license_note, self.size_note, self.fetched_by):
            if note:
                lines.append(f"  {note}")
        return "\n".join(lines)


def _plans() -> Dict[str, FetchPlan]:
    return {
        "split-cifar100": FetchPlan(
            name="split-cifar100",
            commands=(
                'python -c "import torchvision.datasets as d; '
                "d.CIFAR100('/content/data', "
                'download=True)"',  # audit-exempt: colab doc string, not code
            ),
            urls=("cs.toronto.edu/~kriz/cifar.html",),
            license_note="licence: Toronto permissive",
            size_note="~170MB",
        ),
        "split-imagenet-r": FetchPlan(
            name="split-imagenet-r",
            commands=(
                "pip install gdown && gdown 1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR "
                "-O /content/data/imagenet-r.zip && unzip -q "
                "/content/data/imagenet-r.zip -d /content/data",
                "# or raw: wget people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
            ),
            urls=(
                "github.com/hendrycks/imagenet-r",
                "RevisitingCIL preprocessed split (G04)",
            ),
            license_note="licence: repo MIT; images CC/Flickr/wiki (non-comm.)",
            size_note="30k imgs, ~2GB",
        ),
        "split-cub-200": FetchPlan(
            name="split-cub-200",
            commands=(
                "pip install gdown && gdown 1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb "
                "-O /content/data/cub200.zip && unzip -q "
                "/content/data/cub200.zip -d /content/data",
            ),
            urls=("data.caltech.edu/records/65de6-vp158", "RevisitingCIL (G04)"),
            license_note="licence: non-commercial research (Caltech)",
            size_note="11,788 imgs, 1.1GB",
        ),
        "split-omnibenchmark": FetchPlan(
            name="split-omnibenchmark",
            commands=(
                "pip install gdown && gdown 1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u "
                "-O /content/data/omnibenchmark.zip && unzip -q "
                "/content/data/omnibenchmark.zip -d /content/data",
            ),
            urls=("github.com/ZhangYuanhan-AI/OmniBenchmark", "RevisitingCIL (G04)"),
            license_note="licence: research use",
            size_note="300 classes, 21-dataset suite",
        ),
        "split-imagenet-100": FetchPlan(
            name="split-imagenet-100",
            commands=(
                "# OWNER-GATED (G04): ILSVRC2012 terms, no redistribution;",
                "# owner builds the PILOT 100-class subset and uploads it to the session.",
            ),
            urls=("image-net.org",),
            license_note="licence: ILSVRC2012 research terms",
            size_note="~130k train images",
        ),
        "split-imagenet-1k": FetchPlan(
            name="split-imagenet-1k",
            commands=(
                "# OWNER-ONLY (G04): ~150GB, HF-gated (login + accepted terms).",
                "# huggingface-cli login; datasets ILSVRC/imagenet-1k",
            ),
            urls=("image-net.org", "huggingface.co/datasets/ILSVRC/imagenet-1k"),
            license_note="licence: ILSVRC2012; gated",
            size_note="~150GB train",
        ),
        "domainnet": FetchPlan(
            name="domainnet",
            commands=(
                "for d in clipart infograph painting quickdraw real sketch; do",
                "  wget http://csr.bu.edu/3dzhabi/DomainNet/$d.zip -P /content/data; done",
                "# clean version (~5GB) preferred: ai.bu.edu/MFD/",
            ),
            urls=("ai.bu.edu/MFD/",),
            license_note="licence: CC-BY-4.0",
            size_note="600k imgs, 6 domains",
        ),
        "twenty-newsgroups": FetchPlan(
            name="twenty-newsgroups",
            commands=(
                'python -c "from sklearn.datasets import fetch_20newsgroups as f; '
                "f(data_home='/content/data/scikit_learn_data', subset='all')\"",
            ),
            urls=("qwone.com/~jason/20Newsgroups",),
            license_note="licence: free distribution",
            size_note="14MB",
        ),
    }


_DATASET_PLANS = _plans()


def fetch_plan(name: str) -> FetchPlan:
    """The documented colab fetch plan for a dataset (never executed)."""
    return _DATASET_PLANS[_canonical(name)]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArraySplit:
    """One split's arrays.

    x: uint8 images (n, H, W, 3) for kind="image"; object-dtype numpy
    array of str documents for kind="text". y: (n,) int64 global labels
    in 0..n_classes-1. domain: (n,) int64 domain index, present iff the
    split is shared-label domain-incremental (DomainNet).
    """

    x: np.ndarray
    y: np.ndarray
    domain: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("x/y row mismatch")
        if self.domain is not None and self.domain.shape[0] != self.y.shape[0]:
            raise ValueError("domain/y row mismatch")

    def select(self, classes: Sequence[int]) -> "ArraySplit":
        """Read-only subset carrying exactly `classes` (order-preserving)."""
        mask = np.isin(self.y, np.asarray(list(classes), dtype=np.int64))
        return ArraySplit(
            x=self.x[mask],
            y=self.y[mask],
            domain=None if self.domain is None else self.domain[mask],
        )


@dataclass
class DatasetBundle:
    """Loaded dataset + provenance. `spec` is the protocol SplitSpec."""

    name: str
    mode: str
    spec: SplitSpec
    train: ArraySplit
    test: ArraySplit
    label_names: Optional[Tuple[str, ...]] = None
    provenance: Dict[str, object] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    @property
    def n_train(self) -> int:
        return int(self.train.y.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.test.y.shape[0])

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.train.y, minlength=self.spec.n_classes)

    def task_counts(self, seed: int = 0) -> Tuple[int, ...]:
        """Sample counts per task block under the seed-s random ordering.
        Callers with a different ordering slice their own counts via
        train.select(task.classes)."""
        tasks = build_tasks(random_ordering(self.name, seed))
        if self.spec.shared_labels:
            return tuple(
                int(np.sum(self.train.domain == t.task_id)) for t in tasks
            )
        return tuple(int(self.train.select(t.classes).y.shape[0]) for t in tasks)

    def verify_against_spec(self) -> List[str]:
        """Consistency checks vs the registry. Structural mismatches
        raise; the paper-recorded expected_* sizes only append flags
        (synthetic bundles legitimately differ from them)."""
        problems: List[str] = []
        for tag, split in (("train", self.train), ("test", self.test)):
            present = np.unique(split.y)
            if len(present) != self.spec.n_classes:
                problems.append(
                    f"{tag}: {len(present)} classes present, "
                    f"{self.spec.n_classes} expected"
                )
        if self.spec.shared_labels:
            n_dom = (
                len(np.unique(self.train.domain))
                if self.train.domain is not None else 0
            )
            if n_dom != self.spec.n_tasks:
                problems.append(
                    f"train: {n_dom} domains, {self.spec.n_tasks} expected"
                )
        if problems:
            raise ValueError("; ".join(problems))
        flags: List[str] = []
        if self.spec.expected_train and self.n_train != self.spec.expected_train:
            flags.append(
                f"train size {self.n_train} != paper-recorded "
                f"{self.spec.expected_train}"
            )
        if self.spec.expected_test and self.n_test != self.spec.expected_test:
            flags.append(
                f"test size {self.n_test} != paper-recorded "
                f"{self.spec.expected_test}"
            )
        self.flags.extend(flags)
        return flags


# ---------------------------------------------------------------------------
# Local (read-only) loaders
# ---------------------------------------------------------------------------


def _load_cifar100(root: str) -> DatasetBundle:
    """Read the torchvision-format CIFAR-100 pickles directly, read-only
    (open(..., 'rb') only; no torchvision import, no download kwarg)."""
    spec = split_spec("split-cifar100")

    def _read(split_file: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(os.path.join(root, split_file), "rb") as fh:
            entry = pickle.load(fh, encoding="latin1")
        data = entry[b"data"] if b"data" in entry else entry["data"]
        labels = (
            entry[b"fine_labels"] if b"fine_labels" in entry
            else entry["fine_labels"]
        )
        raw = np.asarray(data, dtype=np.uint8)
        n = raw.shape[0]
        if raw.shape[1] != 3072:
            raise ValueError(f"CIFAR-100 {split_file}: bad row width {raw.shape[1]}")
        x = raw.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)  # CHW -> HWC
        y = np.asarray(labels, dtype=np.int64)
        # CIFAR-100 fine labels are already 0..99 indices into the meta
        # names; remap defensively only if not.
        if y.min() != 0 or y.max() != spec.n_classes - 1:
            _, y = np.unique(y, return_inverse=True)
        return np.ascontiguousarray(x), y

    with open(os.path.join(root, "meta"), "rb") as fh:
        meta = pickle.load(fh, encoding="latin1")
    names_key = (
        b"fine_label_names" if b"fine_label_names" in meta else "fine_label_names"
    )
    label_names = tuple(
        str(s, "utf-8") if isinstance(s, bytes) else str(s)
        for s in meta[names_key]
    )

    x_tr, y_tr = _read("train")
    x_te, y_te = _read("test")
    bundle = DatasetBundle(
        name="split-cifar100", mode="local", spec=spec,
        train=ArraySplit(x=x_tr, y=y_tr),
        test=ArraySplit(x=x_te, y=y_te),
        label_names=label_names,
        provenance={
            "source_root": os.path.abspath(root),
            "read_only": True,
            "verified_by": "G04 dossier (pap3 copy)",
        },
        flags=["local-read-only"],
    )
    bundle.verify_against_spec()
    return bundle


def _load_20ng(data_home: str) -> DatasetBundle:
    """20 Newsgroups, BY-DATE split, headers/footers/quotes removed
    (G03 SPEC 9). sklearn cache read with download_if_missing=False --
    the single sanctioned external reader; a missing cache raises
    IOError from sklearn and is converted to LocalDatasetMissingError."""
    spec = split_spec("twenty-newsgroups")
    try:
        from sklearn.datasets import fetch_20newsgroups  # cache read only
    except ImportError as exc:  # pragma: no cover - bare environment
        raise LocalDatasetMissingError(
            "scikit-learn not installed; 20NG local mode unavailable"
        ) from exc
    kwargs = dict(
        data_home=data_home,
        remove=("headers", "footers", "quotes"),
        download_if_missing=False,   # audit-exempt: pinned to False
        shuffle=True,
        random_state=0,
    )
    try:
        tr = fetch_20newsgroups(subset="train", **kwargs)
        te = fetch_20newsgroups(subset="test", **kwargs)
    except Exception as exc:  # sklearn raises IOError when cache missing
        raise LocalDatasetMissingError(
            f"20NG cache missing under {data_home!r} (original: {exc!r})"
        ) from exc
    bundle = DatasetBundle(
        name="twenty-newsgroups", mode="local", spec=spec,
        train=ArraySplit(
            x=np.array(tr.data, dtype=object),
            y=np.asarray(tr.target, dtype=np.int64),
        ),
        test=ArraySplit(
            x=np.array(te.data, dtype=object),
            y=np.asarray(te.target, dtype=np.int64),
        ),
        label_names=tuple(tr.target_names),
        provenance={
            "source_root": os.path.abspath(data_home),
            "split": "by-date",
            "removed": ("headers", "footers", "quotes"),
            "read_only": True,
            "verified_by": "G04 dossier (skikit_learn_data cache)",
        },
        flags=["local-read-only"],
    )
    bundle.verify_against_spec()
    return bundle


#: Loader per dataset for whatever copy exists at a given root. Datasets
#: with no machine-readable local format yet (RevisitingCIL image-folder
#: splits) are G04-absent by probe; if an owner drops a copy in, wire the
#: reader here and it lights up -- until then the probe says missing.
_LOCAL_READERS = {
    "split-cifar100": _load_cifar100,
    "twenty-newsgroups": _load_20ng,
}


def _local_missing_reader(name: str):
    def _reader(root: str) -> DatasetBundle:
        raise LocalDatasetMissingError(
            f"a directory for '{name}' exists at {root!r} but no read-only "
            f"reader is wired for this format yet (G04: no verified local "
            f"copy). Fetch plan:\n" + fetch_plan(name).as_text()
        )
    return _reader


def _load_local(name: str) -> DatasetBundle:
    root = probe_local(name)
    if root is None:
        raise LocalDatasetMissingError(
            f"local mode: no existing copy of '{name}' on this machine "
            f"(probed: {list(LOCAL_PATHS.get(name, ()))}). Per the OWNER "
            f"STORAGE LAW nothing is fetched. Colab plan:\n"
            + fetch_plan(name).as_text()
        )
    reader = _LOCAL_READERS.get(name)
    if reader is None:
        return _local_missing_reader(name)(root)
    return reader(root)


# ---------------------------------------------------------------------------
# Synthetic mode
# ---------------------------------------------------------------------------


def _within_class_counter(y: np.ndarray) -> List[int]:
    counts: Dict[int, int] = {}
    out: List[int] = []
    for c in y.tolist():
        counts[c] = counts.get(c, 0) + 1
        out.append(counts[c])
    return out


def _synthetic_bundle(
    name: str,
    seed: int,
    per_class_train: int,
    per_class_test: int,
    size: int,
) -> DatasetBundle:
    """Deterministic synthetic data matching the split's registry shape.

    Images: uint8 (n, size, size, 3) from a PCG64 Generator seeded via
    protocol.derive_seed (G10 policy: no global RNG state, no builtin
    hash()). Text: synthetic document strings. Shared-label splits get
    n_tasks domains each carrying the full class set.
    """
    spec = split_spec(name)
    rng = np.random.default_rng(derive_seed(seed, f"synthetic-{name}"))
    n = spec.n_classes
    if spec.shared_labels:
        domain = np.repeat(
            np.arange(spec.n_tasks, dtype=np.int64), n * per_class_train
        )
        domain_te = np.repeat(
            np.arange(spec.n_tasks, dtype=np.int64), n * per_class_test
        )
        reps_tr = reps_te = spec.n_tasks
    else:
        domain = domain_te = None
        reps_tr = reps_te = 1
    y_tr = np.tile(np.arange(n, dtype=np.int64), reps_tr * per_class_train)
    y_te = np.tile(np.arange(n, dtype=np.int64), reps_te * per_class_test)
    # sort by label so class blocks are contiguous, like real caches
    order_tr = np.argsort(y_tr, kind="stable")
    order_te = np.argsort(y_te, kind="stable")
    y_tr, y_te = y_tr[order_tr], y_te[order_te]
    if domain is not None:
        domain, domain_te = domain[order_tr], domain_te[order_te]
    if spec.kind == "text":
        x_tr = np.array(
            [f"synthetic-{name}-class{c}-train{i}" for c, i in
             zip(y_tr.tolist(), _within_class_counter(y_tr))], dtype=object)
        x_te = np.array(
            [f"synthetic-{name}-class{c}-test{i}" for c, i in
             zip(y_te.tolist(), _within_class_counter(y_te))], dtype=object)
    else:
        x_tr = rng.integers(
            0, 256, size=(y_tr.shape[0], size, size, 3), dtype=np.uint8)
        x_te = rng.integers(
            0, 256, size=(y_te.shape[0], size, size, 3), dtype=np.uint8)
    bundle = DatasetBundle(
        name=name, mode="synthetic", spec=spec,
        train=ArraySplit(x=x_tr, y=y_tr, domain=domain),
        test=ArraySplit(x=x_te, y=y_te, domain=domain_te),
        provenance={
            "generator": "numpy PCG64 via protocol.derive_seed",
            "seed": seed,
            "per_class_train": per_class_train,
            "per_class_test": per_class_test,
            "synthetic_size": size,
        },
        flags=["synthetic-data"],
    )
    bundle.verify_against_spec()
    return bundle


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_COLAB_SUBDIRS: Dict[str, str] = {
    "split-cifar100": "cifar-100-python",
    "twenty-newsgroups": "scikit_learn_data",
    "split-imagenet-r": "imagenet-r",
    "split-cub-200": "cub200",
    "split-omnibenchmark": "omnibenchmark",
    "split-imagenet-100": "imagenet100",
    "split-imagenet-1k": "imagenet1k",
    "domainnet": "domainnet",
}


def get_dataset(
    name: str,
    mode: str = DEFAULT_MODE,
    *,
    seed: int = 0,
    synthetic_per_class: int = 4,
    synthetic_test_per_class: int = 2,
    synthetic_size: Optional[int] = None,
) -> DatasetBundle:
    """Load a registered dataset under one of the three modes.

    mode="synthetic" (default): deterministic synthetic bundle; runs
      anywhere, for tests (G10 tiers UNIT/VERIFICATION/INTEGRATION).
    mode="local": read-only load of the verified machine caches
      (CIFAR-100 via the pap3 pickles; 20NG via the sklearn cache with
      download_if_missing=False). Missing copy -> LocalDatasetMissingError
      carrying the colab plan. NOTHING IS EVER FETCHED (storage law).
    mode="colab": load read-only from the colab staging root
      ($CIL_COLAB_DATA, default /content/data) AFTER the notebook has run
      fetch_plan(name); if staging is absent -> ColabDataNotFetchedError
      with the plan. The fetch itself lives in repo/colab/ notebooks.

    `seed` keys the synthetic generator (protocol.derive_seed policy).
    `synthetic_size` overrides the synthetic image edge (default: the
    spec's native size, else 32).
    """
    canonical = _canonical(name)
    spec = split_spec(canonical)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "synthetic":
        size = synthetic_size or (spec.native_size[0] if spec.native_size else 32)
        return _synthetic_bundle(
            canonical, seed, synthetic_per_class, synthetic_test_per_class, size
        )
    if mode == "local":
        return _load_local(canonical)
    # colab: the staging root plays the role of the local cache; never fetch
    staging = os.path.join(COLAB_STAGING, _COLAB_SUBDIRS[canonical])
    if not os.path.isdir(staging):
        raise ColabDataNotFetchedError(
            f"colab mode: staging dir {staging!r} absent -- the notebook "
            f"must run the fetch plan first (this module will not "
            f"fetch):\n" + fetch_plan(canonical).as_text()
        )
    bundle = _LOCAL_READERS.get(canonical, _local_missing_reader(canonical))(
        staging
    )
    bundle.mode = "colab"
    return bundle


# ---------------------------------------------------------------------------
# Registry table (name x mode status)
# ---------------------------------------------------------------------------


def registry_status() -> List[Dict[str, object]]:
    """Probe every (dataset, mode) cell for the status table. Probes are
    read-only existence checks; synthetic is always OK; colab cells
    report the staging plan (never fetch)."""
    rows: List[Dict[str, object]] = []
    for name in sorted(SPLITS):
        spec = SPLITS[name]
        root = probe_local(name)
        if root and name in _LOCAL_READERS:
            local = f"OK ({root})"
        elif root:
            local = f"DIR-NO-READER ({root})"
        else:
            local = "MISSING (clear error, no fetch)"
        rows.append({
            "dataset": name,
            "T x C": f"{spec.n_tasks}x{spec.classes_per_task}"
                     + (" (shared-label domains)" if spec.shared_labels else ""),
            "kind": spec.kind,
            "papers": "/".join(spec.papers),
            "local": local,
            "colab": "PLAN (fetch documented, notebook executes)",
            "synthetic": "OK",
        })
    return rows


#: The dataset registry: canonical name -> registry status row (see SPLITS
#: in benchmarks/protocol.py for the authoritative split definitions).
DATASETS: Dict[str, Dict[str, object]] = {
    r["dataset"]: r for r in registry_status()
}


# ---------------------------------------------------------------------------
# Self-check (task: synthetic Split-CIFAR-100 + local copy read-only)
# ---------------------------------------------------------------------------


def _selfcheck() -> int:
    failures: List[str] = []

    print("== benchmarks.datasets self-check (H08) ==")
    print(f"python: {sys.version.split()[0]}  numpy: {np.__version__}")
    print()

    # Storage-law audit first.
    violations = audit_no_fetch()
    print(f"[audit] source fetch-audit: "
          f"{'CLEAN' if not violations else violations}")
    if violations:
        failures.append("fetch audit violations")

    # Registry table.
    print("\n[registry] dataset x mode status:")
    for row in registry_status():
        print(f"  {row['dataset']:<20} {row['T x C']:<26} "
              f"local: {row['local']:<45} colab: {row['colab']} "
              f"| synthetic: {row['synthetic']}")

    # Synthetic Split-CIFAR-100: task counts + seed-0 ordering.
    print("\n[synthetic] split-cifar100:")
    b = get_dataset("split-cifar100")
    ordering = random_ordering("split-cifar100", 0)
    tasks = build_tasks(ordering)
    print(f"  spec: T={len(tasks)} tasks, "
          f"sizes={[t.n_classes for t in tasks]}")
    print(f"  train/test: x{b.train.x.shape} uint8 / x{b.test.x.shape} uint8")
    print(f"  classes present: {len(np.unique(b.train.y))} "
          f"(expected {b.spec.n_classes})")
    print(f"  seed-0 random ordering (first 20): {list(ordering.order[:20])}")
    print(f"  seed-0 task 0 classes: {tasks[0].classes}")
    print(f"  task sample counts (train): {b.task_counts(seed=0)}")
    b2 = get_dataset("split-cifar100", seed=0)
    det = np.array_equal(b.train.x[:64], b2.train.x[:64])
    print(f"  determinism (seed 0 replay): {'OK' if det else 'FAIL'}")
    if not det:
        failures.append("synthetic not deterministic")

    # Synthetic across the full registry.
    print("\n[synthetic] all datasets:")
    for name in sorted(SPLITS):
        sb = get_dataset(name)
        n_tasks = len(build_tasks(random_ordering(name, 0)))
        print(f"  {name:<20} tasks={n_tasks} train={sb.n_train} "
              f"test={sb.n_test} task_counts={sb.task_counts(seed=0)}")

    # Local copies, read-only (no writes; missing = clear error).
    print("\n[local] read-only probes:")
    for name in ("split-cifar100", "twenty-newsgroups"):
        try:
            lb = get_dataset(name, "local")
            print(f"  {name}: LOADED train={lb.n_train} test={lb.n_test} "
                  f"from {lb.provenance.get('source_root')} flags={lb.flags}")
        except StorageLawViolation as exc:
            print(f"  {name}: UNAVAILABLE -> "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
    for name in ("split-imagenet-r", "split-cub-200", "split-omnibenchmark",
                 "domainnet"):
        try:
            get_dataset(name, "local")
            failures.append(f"{name}: local unexpectedly loaded")
        except LocalDatasetMissingError as exc:
            print(f"  {name}: LocalDatasetMissingError (guard OK): "
                  f"{str(exc).splitlines()[0]}")
        except StorageLawViolation:
            failures.append(f"{name}: wrong storage-law error type")

    print("\n[result] " + ("ALL CHECKS PASSED" if not failures
                           else f"FAILURES: {failures}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
