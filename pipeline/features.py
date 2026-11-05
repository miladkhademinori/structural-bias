"""pipeline.features -- feature extraction (Colab, GPU) and the feature cache.

Cache format (one file per backbone x dataset x split):

    features/<backbone>/<dataset>/<split>.npz
        X        float32 [n, d]   backbone features (see registry.Backbone.note)
        y        int32   [n]      class index 0..n_classes-1
        domain   int16   [n]      domain index (domains-as-tasks datasets only; else all 0)
        meta     json string      backbone id, dataset, preprocessing, torch/transformers
                                  versions, extraction date, image count, sha256 of y

Extraction is a fixed-order forward pass in eval mode with no augmentation, so
a cache is independent of any seed and can be rebuilt bit-for-bit up to GPU
kernel nondeterminism (which the determinism self-check measures).

Only ``extract`` and the dataset builders need torch; everything else in the
pipeline reads caches with numpy.  Synthetic caches (``write_synthetic_cache``)
let the whole analysis stage run without a GPU and are the basis of the tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pipeline.registry import BACKBONES, DATASETS, derive_seed

CACHE_ROOT = Path(os.environ.get("REPLICATION_FEATURES", "features"))


# ---------------------------------------------------------------------------
# Cache I/O (numpy only)
# ---------------------------------------------------------------------------

def cache_path(backbone: str, dataset: str, split: str, root: Path = None) -> Path:
    root = Path(root) if root is not None else CACHE_ROOT
    return root / backbone / dataset / f"{split}.npz"


def write_cache(path: Path, X: np.ndarray, y: np.ndarray, meta: dict,
                domain: Optional[np.ndarray] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.int32)
    dom = np.zeros(len(y), np.int16) if domain is None else np.ascontiguousarray(domain, np.int16)
    meta = dict(meta)
    meta.update(n=int(len(y)), d=int(X.shape[1]), y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                X_sha256=hashlib.sha256(X.tobytes()).hexdigest(), written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    np.savez_compressed(path, X=X, y=y, domain=dom, meta=json.dumps(meta))
    return path


def read_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """X is returned as stored (float32): the statistics and heads accumulate in float64 themselves, and a float64
    copy of DomainNet's 400k x 768 train features would cost 2.5 GB."""
    with np.load(Path(path), allow_pickle=False) as z:
        X = z["X"]
        y = z["y"].astype(np.int64)
        dom = z["domain"].astype(np.int64)
        meta = json.loads(str(z["meta"]))
    return X, y, dom, meta


def cache_exists(backbone: str, dataset: str, root: Path = None) -> bool:
    return cache_path(backbone, dataset, "train", root).exists() and cache_path(backbone, dataset, "test", root).exists()


def load_pair(backbone: str, dataset: str, root: Path = None):
    """(Xtr, ytr, domtr), (Xte, yte, domte), meta for one configuration."""
    Xtr, ytr, dtr, meta = read_cache(cache_path(backbone, dataset, "train", root))
    Xte, yte, dte, _ = read_cache(cache_path(backbone, dataset, "test", root))
    return (Xtr, ytr, dtr), (Xte, yte, dte), meta


# ---------------------------------------------------------------------------
# Synthetic caches (tests and dry runs; provenance "synthetic", never a paper number)
# ---------------------------------------------------------------------------

def write_synthetic_cache(backbone: str, dataset: str, root: Path, *, d: int = 64,
                          n_train_per_class: int = 60, n_test_per_class: int = 20,
                          sep: float = 3.0, seed_purpose: str = "synthetic-cache") -> Tuple[Path, Path]:
    """Gaussian class-conditional features with a shared anisotropic covariance.

    Class means are drawn so that the between-class geometry has a planted
    scale ``sep``; backbone keys change the scale (DINOv2 > CLIP > ViT) so that
    the synthetic gamma ranking mimics the printed one.  This exists so the
    analysis stage can be exercised end to end; its outputs are labelled
    provenance="synthetic" and never enter a paper table.
    """
    ds = DATASETS[dataset]
    scale = {"dinov2_b14": 2.0, "clip_b16": 1.4, "vit_in21k": 1.0, "mae_b16": 0.8,
             "resnet50": 0.7, "random_vit": 0.15}.get(backbone, 1.0)
    g = np.random.default_rng(derive_seed(f"{seed_purpose}:{backbone}:{dataset}"))
    N = ds.n_classes
    mu = g.normal(0.0, sep * scale, (N, d))
    A = g.normal(0.0, 1.0, (d, d)) / np.sqrt(d)
    chol = np.linalg.cholesky(A @ A.T + 0.5 * np.eye(d))
    n_dom = len(ds.domains) if ds.domains_as_tasks else 1

    def draw(n_per_class):
        Xs, ys, ds_ = [], [], []
        for dom in range(n_dom):
            shift = g.normal(0.0, 0.3 * sep * scale, d) if n_dom > 1 else 0.0
            for c in range(N):
                Z = g.normal(0.0, 1.0, (n_per_class, d)) @ chol.T
                Xs.append(mu[c] + shift + Z); ys.append(np.full(n_per_class, c)); ds_.append(np.full(n_per_class, dom))
        return np.vstack(Xs), np.concatenate(ys), np.concatenate(ds_)

    meta = dict(backbone=backbone, dataset=dataset, provenance="synthetic", d=d, sep=sep * scale)
    Xtr, ytr, dtr = draw(n_train_per_class)
    Xte, yte, dte = draw(n_test_per_class)
    p1 = write_cache(cache_path(backbone, dataset, "train", root), Xtr, ytr, meta | {"split": "train"}, dtr)
    p2 = write_cache(cache_path(backbone, dataset, "test", root), Xte, yte, meta | {"split": "test"}, dte)
    return p1, p2


# ---------------------------------------------------------------------------
# Dataset acquisition (Colab only; everything below needs network and torch)
# ---------------------------------------------------------------------------

def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _assert_download_allowed(context: str) -> None:
    """Owner storage law: NO downloads on the owner's machine, ever.

    Downloads are legal only on Colab (google.colab importable) or with an
    explicit override flag set by a human who accepts the download.
    """
    import sys
    # Colab is recognised in two ways: ``google.colab`` imported in this interpreter (a notebook cell), or Colab's own
    # environment variables (COLAB_RELEASE_TAG, COLAB_GPU, COLAB_BACKEND_VERSION), which a ``subprocess.run([sys.executable,
    # "replicate.py", ...])`` launched from a notebook cell inherits while ``sys.modules`` there holds no ``google.colab``.
    # Neither exists on the owner's machine, so the law is unchanged there.
    on_colab = "google.colab" in sys.modules or any(k in os.environ for k in ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_BACKEND_VERSION"))
    if not on_colab and os.environ.get("REPLICATION_ALLOW_DOWNLOADS") != "1":
        raise RuntimeError(
            f"Download blocked ({context}): the owner storage law forbids "
            "downloads on this machine. Run this on Google Colab, or set "
            "REPLICATION_ALLOW_DOWNLOADS=1 if you explicitly accept the transfer."
        )


def _download(url, dst: Path) -> Path:
    """Download ``url`` (or the first of several candidate URLs that answers) to ``dst``; skip if present.
    A partial file from an interrupted session is removed and fetched again."""
    import urllib.error
    import urllib.request
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    _assert_download_allowed(f"urlretrieve {url}")
    urls = [url] if isinstance(url, str) else list(url)
    part = dst.with_suffix(dst.suffix + ".part")
    last = None
    for u in urls:
        try:
            print(f"[features] downloading {u} -> {dst}")
            urllib.request.urlretrieve(u, part)
            part.rename(dst)
            return dst
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last = e; part.unlink(missing_ok=True)
            print(f"[features]   {u}: {e}")
    raise FileNotFoundError(f"could not download {dst.name} from any of {urls}: {last}")


def _gdown(file_id: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    _assert_download_allowed(f"gdown {file_id}")
    import gdown  # noqa
    gdown.download(id=file_id, output=str(dst), quiet=False)
    return dst


def _extract_archive(archive: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    marker = dst / f".extracted_{archive.name}"
    if marker.exists():
        return dst
    print(f"[features] extracting {archive} -> {dst}")
    if archive.suffix in (".tar", ".tgz", ".gz") or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive) as t:
            t.extractall(dst, filter="data")
    else:
        with zipfile.ZipFile(archive) as z:
            z.extractall(dst)
    marker.touch()
    return dst


def _image_folder_listing(root: Path, exts=(".jpg", ".jpeg", ".png", ".JPEG", ".JPG")) -> Tuple[List[Path], np.ndarray, List[str]]:
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    paths, labels = [], []
    for ci, c in enumerate(classes):
        for p in sorted((root / c).rglob("*")):
            if p.suffix in exts:
                paths.append(p); labels.append(ci)
    return paths, np.asarray(labels, np.int32), classes


def _stratified_split(labels: np.ndarray, test_frac: float, seed_purpose: str) -> Tuple[np.ndarray, np.ndarray]:
    g = np.random.default_rng(derive_seed(seed_purpose))
    tr, te = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        g.shuffle(idx)
        n_te = max(1, int(round(test_frac * len(idx))))
        te.append(idx[:n_te]); tr.append(idx[n_te:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))


class ImageList:
    """A list of (path, label[, domain]) with a PIL loader; torch-free container."""

    def __init__(self, paths: Sequence[Path], labels: np.ndarray, classes: List[str],
                 domains: Optional[np.ndarray] = None, in_memory: Optional[np.ndarray] = None):
        self.paths, self.labels, self.classes = list(paths), np.asarray(labels, np.int32), classes
        self.domains = np.zeros(len(self.labels), np.int16) if domains is None else np.asarray(domains, np.int16)
        self.in_memory = in_memory   # uint8 [n, H, W, 3] for CIFAR

    def __len__(self):
        return len(self.labels)

    def image(self, i: int):
        from PIL import Image
        if self.in_memory is not None:
            return Image.fromarray(self.in_memory[i]).convert("RGB")
        return Image.open(self.paths[i]).convert("RGB")


def _cifar100(data_root: Path) -> Dict[str, ImageList]:
    """CIFAR-100 from the HuggingFace CDN with a Drive-side npz cache.

    The Toronto torchvision host throttles Colab downloads, so a cold cache on a
    connected GPU runtime burns paid compute time while idling. Order of attack:
    (1) npz cache (Drive; instant, survives VM death), (2) uoft-cs/cifar100 on
    HuggingFace (fast CDN), (3) torchvision/Toronto as a loud emergency fallback.
    The cache lands next to REPLICATION_FEATURES (the Drive root in every
    notebook), so every fresh VM after the first one downloads nothing at all.
    """
    env_feat = os.environ.get("REPLICATION_FEATURES", "")
    cache_dir = (Path(env_feat).parent / "data_cache") if env_feat else Path(data_root) / "cifar100_cache"
    cache = cache_dir / "cifar100.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {s: ImageList([None] * len(z[f"{s}_y"]), z[f"{s}_y"], list(z["classes"]), in_memory=z[f"{s}_x"])
                for s in ("train", "test")}
    try:
        from datasets import load_dataset
        hud = load_dataset("uoft-cs/cifar100")
        names = list(hud["train"].features["fine_label"].names)
        out: Dict[str, ImageList] = {}
        for split in ("train", "test"):
            _assert_download_allowed("huggingface cifar100 (cold cache)")
            x = np.stack([np.asarray(im, dtype=np.uint8) for im in hud[split]["img"]])
            y = np.asarray(hud[split]["fine_label"], np.int32)
            out[split] = ImageList([None] * len(y), y, names, in_memory=x)
    except Exception as e:
        print(f"[features] HF cifar100 FAILED ({type(e).__name__}: {e}); falling back to torchvision/Toronto (slow)")
        import torchvision
        out = {}
        for split, train in (("train", True), ("test", False)):
            _assert_download_allowed("torchvision CIFAR100 (cold cache)")
            ds = torchvision.datasets.CIFAR100(str(data_root / "cifar100"), train=train, download=True)
            out[split] = ImageList([None] * len(ds), np.asarray(ds.targets), list(ds.classes), in_memory=ds.data)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, train_x=out["train"].in_memory, train_y=out["train"].labels,
             test_x=out["test"].in_memory, test_y=out["test"].labels, classes=np.asarray(out["train"].classes))
    print(f"[features] cifar100 cached -> {cache}")
    return out


def build_dataset(dataset: str, data_root: Path, imagenet100_root: Optional[Path] = None) -> Dict[str, ImageList]:
    """Return {"train": ImageList, "test": ImageList}; downloads on first use (Colab)."""
    data_root = Path(data_root)
    if dataset == "cifar100":
        return _cifar100(data_root)
    if dataset == "imagenet_r":
        tar = _download("https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar", data_root / "imagenet-r.tar")
        root = _extract_archive(tar, data_root / "imagenet_r") / "imagenet-r"
        paths, labels, classes = _image_folder_listing(root)
        tr, te = _stratified_split(labels, 0.2, "split:imagenet_r")
        return {"train": ImageList([paths[i] for i in tr], labels[tr], classes),
                "test": ImageList([paths[i] for i in te], labels[te], classes)}
    if dataset == "cub200":
        tgz = _download(["https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",      # official record
                         "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz"],              # fastai re-host of the same archive
                        data_root / "CUB_200_2011.tgz")
        if tgz.stat().st_size < 1 << 20:   # an HTML error page saved in place of the archive (seen on the Caltech host)
            tgz.unlink(); raise FileNotFoundError("CUB_200_2011.tgz download was not an archive; rerun to try the next source")
        root = _extract_archive(tgz, data_root / "cub200") / "CUB_200_2011"
        imgs = {int(a): b for a, b in (l.split() for l in open(root / "images.txt"))}
        lab = {int(a): int(b) - 1 for a, b in (l.split() for l in open(root / "image_class_labels.txt"))}
        istr = {int(a): int(b) for a, b in (l.split() for l in open(root / "train_test_split.txt"))}
        classes = [l.split()[1] for l in open(root / "classes.txt")]
        out = {}
        for split, flag in (("train", 1), ("test", 0)):
            ids = sorted(i for i, f in istr.items() if f == flag)
            out[split] = ImageList([root / "images" / imgs[i] for i in ids], np.asarray([lab[i] for i in ids]), classes)
        return out
    if dataset == "omnibenchmark":
        root = data_root / "omnibenchmark"
        cands = [p for p in root.rglob("train") if p.is_dir()] if root.exists() else []
        if not cands:   # not provided by the owner: fetch the RevisitingCIL release (Google Drive id from the G04 log, unverified)
            zp = _gdown("1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u", data_root / "omnibenchmark.zip")
            root = _extract_archive(zp, root)
            cands = [p for p in root.rglob("train") if p.is_dir()]
        if not cands:
            raise FileNotFoundError("omnibenchmark: no train/ directory under data_root/omnibenchmark; place the RevisitingCIL "
                                    "OmniBenchmark release there (train/<class>/*.jpg, test/<class>/*.jpg) or check the Drive id")
        base = cands[0].parent
        out = {}
        for split in ("train", "test"):
            paths, labels, classes = _image_folder_listing(base / split)
            out[split] = ImageList(paths, labels, classes)
        return out
    if dataset == "imagenet100":
        if imagenet100_root is None:
            raise FileNotFoundError("imagenet100 is owner-gated: pass imagenet100_root=<folder with train/ and val/ ImageFolders>")
        out = {}
        for split, sub in (("train", "train"), ("test", "val")):
            paths, labels, classes = _image_folder_listing(Path(imagenet100_root) / sub)
            out[split] = ImageList(paths, labels, classes)
        return out
    if dataset == "domainnet":
        ds = DATASETS["domainnet"]
        base = "http://csr.bu.edu/ftp/visda/2019/multi-source/"
        root = data_root / "domainnet"
        split_lists = {"train": [], "test": []}
        classes = None
        for di, dom in enumerate(ds.domains):
            if not (root / dom).is_dir():   # the VisDA-2019 server keeps some domains under groundtruth/; try both
                zp = _download([base + f"{dom}.zip", base + f"groundtruth/{dom}.zip"], root / f"{dom}.zip")
                _extract_archive(zp, root)
            for split in ("train", "test"):
                txt = _download([base + f"txt/{dom}_{split}.txt", base + f"groundtruth/txt/{dom}_{split}.txt"], root / f"{dom}_{split}.txt")
                for line in open(txt):
                    rel, lab = line.split()
                    split_lists[split].append((root / rel, int(lab), di))
        names = sorted({p.parent.name for p, _, _ in split_lists["train"]})
        out = {}
        for split, rows in split_lists.items():
            out[split] = ImageList([r[0] for r in rows], np.asarray([r[1] for r in rows]), names,
                                   domains=np.asarray([r[2] for r in rows]))
        return out
    raise KeyError(dataset)


# ---------------------------------------------------------------------------
# Backbone loading and extraction (torch + transformers)
# ---------------------------------------------------------------------------

def load_backbone(backbone: str, device: str = "cuda"):
    """Return (model, processor, forward) where forward(pixel_values)->[B, d] features."""
    import torch
    from transformers import AutoImageProcessor, AutoModel, ViTConfig, ViTModel, CLIPVisionModel, ViTMAEModel
    _assert_download_allowed(f"backbone weights {backbone} (cold HF cache)")
    bb = BACKBONES[backbone]
    torch.manual_seed(0)
    if bb.kind == "random_vit":
        cfg = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel(cfg)                      # random init, torch seed 0
        proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        fwd = lambda pv: model(pixel_values=pv).last_hidden_state[:, 0]
    elif bb.kind == "vit":
        model = ViTModel.from_pretrained(bb.hf_id, add_pooling_layer=False)
        proc = AutoImageProcessor.from_pretrained(bb.hf_id)
        fwd = lambda pv: model(pixel_values=pv).last_hidden_state[:, 0]
    elif bb.kind == "dinov2":
        model = AutoModel.from_pretrained(bb.hf_id)
        proc = AutoImageProcessor.from_pretrained(bb.hf_id)
        fwd = lambda pv: model(pixel_values=pv).last_hidden_state[:, 0]
    elif bb.kind == "clip":
        # the vision tower's pooled output: post-layernorm CLS in the vision width (768 for ViT-B/16), BEFORE the visual
        # projection (the 512-d projected embedding lives in CLIPModel.get_image_features / CLIPVisionModelWithProjection
        # and is a different feature space); called on vision_model explicitly so no release's wrapper semantics can change it
        model = CLIPVisionModel.from_pretrained(bb.hf_id)
        proc = AutoImageProcessor.from_pretrained(bb.hf_id)
        tower = getattr(model, "vision_model", model)
        fwd = lambda pv: tower(pixel_values=pv).pooler_output
    elif bb.kind == "mae":
        model = ViTMAEModel.from_pretrained(bb.hf_id, mask_ratio=0.0)
        proc = AutoImageProcessor.from_pretrained(bb.hf_id)
        fwd = lambda pv: model(pixel_values=pv).last_hidden_state[:, 0]
    elif bb.kind == "resnet":
        model = AutoModel.from_pretrained(bb.hf_id)
        proc = AutoImageProcessor.from_pretrained(bb.hf_id)
        fwd = lambda pv: model(pixel_values=pv).pooler_output.flatten(1)
    else:
        raise KeyError(bb.kind)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    fwd.readout = READOUT[bb.kind]
    return model, proc, fwd


READOUT = {
    "random_vit": "ViTModel.last_hidden_state[:, 0] (CLS after the final layer norm; the pooler head is not used)",
    "vit": "ViTModel.last_hidden_state[:, 0] (CLS after the final layer norm; the pooler head is not used)",
    "dinov2": "Dinov2Model.last_hidden_state[:, 0] (CLS after the final layer norm)",
    "clip": "CLIPVisionModel.vision_model.pooler_output (post-layernorm CLS, vision width, before the visual projection)",
    "mae": "ViTMAEModel.last_hidden_state[:, 0] with mask_ratio 0 (CLS after the final layer norm)",
    "resnet": "ResNetModel.pooler_output (global average pool, 2048)",
}


def determinism_check(backbone: str, device: str = "cuda") -> dict:
    """Hash a fixed forward pass twice; a mismatch means kernels are nondeterministic."""
    import torch
    model, proc, fwd = load_backbone(backbone, device)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(8, 3, 224, 224, generator=g).to(device)
    with torch.no_grad():
        a = fwd(x).float().cpu().numpy(); b = fwd(x).float().cpu().numpy()
    return dict(backbone=backbone, bit_identical=bool(np.array_equal(a, b)), max_abs_diff=float(np.abs(a - b).max()))


def extract(backbone: str, dataset: str, data_root: Path, out_root: Path = None, *,
            batch_size: int = 128, device: str = "cuda", num_workers: int = 4,
            imagenet100_root: Optional[Path] = None, fp16: bool = True) -> Dict[str, Path]:
    """Extract and cache train and test features for one (backbone, dataset). Idempotent."""
    import torch
    out_root = Path(out_root) if out_root is not None else CACHE_ROOT
    done = {s: cache_path(backbone, dataset, s, out_root) for s in ("train", "test")}
    if all(p.exists() for p in done.values()):
        print(f"[features] {backbone}/{dataset}: cache present, skipping")
        return done
    splits = build_dataset(dataset, data_root, imagenet100_root)
    model, proc, fwd = load_backbone(backbone, device)
    import transformers
    meta_base = dict(backbone=backbone, hf_id=BACKBONES[backbone].hf_id, dataset=dataset,
                     feature=BACKBONES[backbone].note, processor=type(proc).__name__,
                     torch=torch.__version__, transformers=transformers.__version__,
                     device=str(torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else device),
                     provenance="real", fp16=bool(fp16), readout=getattr(fwd, "readout", None),
                     transform_note="HF image processor (model-specific resize and normalization; for ViT bilinear 224 without crop, which differs from timm's bicubic 248 + centre crop 224; see pipeline.features_timm)")

    class _DS(torch.utils.data.Dataset):
        def __init__(self, il): self.il = il
        def __len__(self): return len(self.il)
        def __getitem__(self, i):
            return proc(images=self.il.image(i), return_tensors="pt")["pixel_values"][0], i

    for split, il in splits.items():
        if done[split].exists():
            continue
        loader = torch.utils.data.DataLoader(_DS(il), batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=(device == "cuda"))
        feats = np.zeros((len(il), BACKBONES[backbone].dim), np.float32)
        _dim_checked = False
        t0 = time.time()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(fp16 and device == "cuda")):
            for pv, idx in loader:
                f = fwd(pv.to(device, non_blocking=True)).float().cpu().numpy()
                if not _dim_checked:
                    if f.shape[1] != BACKBONES[backbone].dim:
                        raise RuntimeError(f"{backbone}: readout '{getattr(fwd, 'readout', '?')}' returned d={f.shape[1]}, the registry declares d={BACKBONES[backbone].dim}; "
                                           "the loader is reading a different feature space (for CLIP: the 512-d projected embedding is not the 768-d vision-tower output)")
                    _dim_checked = True
                feats[idx.numpy()] = f
        meta = meta_base | dict(split=split, classes=il.classes, seconds=round(time.time() - t0, 1))
        write_cache(done[split], feats, il.labels, meta, il.domains)
        print(f"[features] {backbone}/{dataset}/{split}: {len(il)} images, d={feats.shape[1]}, {time.time()-t0:.0f}s -> {done[split]}")
    return done
