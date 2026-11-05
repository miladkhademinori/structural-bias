"""Runtime tests: the extraction loop end to end with a mock backbone (no network, no GPU), the cache format,
the dataset listing helpers, the notebooks' code cells, and the CLI parser.  These are the paths Colab hits first."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import features as F, registry as R  # noqa: E402

torch = pytest.importorskip("torch")


class _TinyProcessor:
    """Stands in for a HuggingFace image processor: PIL image -> {"pixel_values": [1, 3, 224, 224]}."""

    def __call__(self, images, return_tensors="pt"):
        arr = np.asarray(images.convert("RGB").resize((224, 224)), np.float32) / 255.0
        return {"pixel_values": torch.from_numpy(arr).permute(2, 0, 1)[None]}


def _mock_backbone(dim):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 8, stride=8), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, dim)).eval()
    return model, _TinyProcessor(), (lambda pv: model(pv))


def _mock_dataset(n_per_class=5, n_classes=4, seed=0):
    g = np.random.default_rng(seed)
    out = {}
    for split, n in (("train", n_per_class), ("test", 2)):
        y = np.repeat(np.arange(n_classes), n)
        imgs = (g.integers(0, 255, (len(y), 32, 32, 3))).astype(np.uint8)
        out[split] = F.ImageList([None] * len(y), y, [f"c{i}" for i in range(n_classes)], in_memory=imgs)
    return out


def test_extract_end_to_end_with_mock_backbone(tmp_path, monkeypatch):
    dim = R.BACKBONES["vit_in21k"].dim
    monkeypatch.setattr(F, "load_backbone", lambda backbone, device="cuda": _mock_backbone(dim))
    monkeypatch.setattr(F, "build_dataset", lambda dataset, data_root, imagenet100_root=None: _mock_dataset())
    done = F.extract("vit_in21k", "cifar100", tmp_path / "data", tmp_path / "feat", batch_size=8, device="cpu", num_workers=0, fp16=False)
    assert all(p.exists() for p in done.values())
    (Xtr, ytr, dtr), (Xte, yte, dte), meta = F.load_pair("vit_in21k", "cifar100", tmp_path / "feat")
    assert Xtr.shape == (20, dim) and Xte.shape == (8, dim) and ytr.max() == 3 and dtr.max() == 0
    assert meta["provenance"] == "real" and meta["split"] == "train" and meta["n"] == 20
    assert np.isfinite(Xtr).all() and np.abs(Xtr).max() > 0
    # idempotent: a second call skips the work and returns the same paths
    again = F.extract("vit_in21k", "cifar100", tmp_path / "data", tmp_path / "feat", device="cpu", num_workers=0)
    assert again == done


def test_cache_roundtrip_and_sha(tmp_path):
    X = np.random.default_rng(0).standard_normal((10, 6)).astype(np.float32); y = np.arange(10) % 3
    p = F.write_cache(tmp_path / "a" / "train.npz", X, y, dict(backbone="b", provenance="synthetic"))
    X2, y2, dom, meta = F.read_cache(p)
    assert np.allclose(X2, X) and (y2 == y).all() and dom.shape == (10,) and len(meta["X_sha256"]) == 64


def test_image_folder_listing_and_split(tmp_path):
    from PIL import Image
    for c in ("dog", "cat"):
        (tmp_path / c).mkdir()
        for i in range(6):
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / c / f"{i}.jpg")
    paths, labels, classes = F._image_folder_listing(tmp_path)
    assert classes == ["cat", "dog"] and len(paths) == 12 and labels.tolist() == [0] * 6 + [1] * 6
    tr, te = F._stratified_split(labels, 0.5, "t")
    assert len(tr) == 6 and len(te) == 6 and set(labels[te]) == {0, 1}
    il = F.ImageList([paths[i] for i in tr], labels[tr], classes)
    assert il.image(0).size == (8, 8)


def test_synthetic_cache_has_synthetic_provenance(tmp_path):
    F.write_synthetic_cache("vit_in21k", "cifar100", tmp_path, d=8, n_train_per_class=3, n_test_per_class=2)
    (_, ytr, _), _, meta = F.load_pair("vit_in21k", "cifar100", tmp_path)
    assert meta["provenance"] == "synthetic" and len(ytr) == 300


@pytest.mark.parametrize("nb", sorted((ROOT / "colab").glob("*.ipynb")))
def test_notebook_cells_compile(nb):
    doc = json.loads(nb.read_text(encoding="utf-8"))
    assert doc["cells"] and doc["cells"][0]["cell_type"] == "markdown"
    for c in doc["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(l for l in c["source"] if not l.lstrip().startswith(("%", "!")))
        ast.parse(src)


def test_cli_parses_every_subcommand():
    import replicate
    parser = replicate.build_parser()
    for argv in (["features", "--paper", "A"], ["--device", "numpy", "analyze", "--paper", "AB", "--optimizer", "lbfgs", "--ranpac-M", "100"],
                 ["tables"], ["figures"], ["audit-tex", "--paper", "B", "--tex", "p.tex"], ["nonvision", "--allow-download"],
                 ["synthetic", "--run", "--quick"], ["selftest", "--skip-ecf"]):
        a = parser.parse_args(argv)
        assert callable(a.fn)


def test_cli_rejects_the_removed_a_subcommands():
    """B-only release: the Paper A entry points (condition-s, fingerprint, features-timm, drift,
    drift-readback, audit-tex --paper A) are gone with their modules; argparse must say so."""
    import replicate
    parser = replicate.build_parser()
    for argv in (["condition-s"], ["fingerprint"], ["features-timm"], ["drift", "--lrs", "1e-5"], ["drift-readback"],
                 ["audit-tex", "--paper", "A", "--tex", "p.tex"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_load_backbone_forward_lambdas_on_tiny_offline_models(monkeypatch):
    """Every branch of load_backbone with a tiny offline model of the same architecture family: the forward lambdas
    must return [B, dim] with the registry's dim, and the default image processors must accept PIL input."""
    import transformers as T
    from PIL import Image
    from pipeline import features as F

    tiny = dict(
        vit_in21k=lambda: T.ViTModel(T.ViTConfig(hidden_size=768, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64), add_pooling_layer=False),
        random_vit=None,
        dinov2_b14=lambda: T.Dinov2Model(T.Dinov2Config(hidden_size=768, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64, patch_size=14)),
        clip_b16=lambda: T.CLIPVisionModel(T.CLIPVisionConfig(hidden_size=768, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64, patch_size=16)),
        mae_b16=lambda: T.ViTMAEModel(T.ViTMAEConfig(hidden_size=768, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64, mask_ratio=0.0)),
        resnet50=lambda: T.ResNetModel(T.ResNetConfig(embedding_size=8, hidden_sizes=[8, 16, 32, 2048], depths=[1, 1, 1, 1])),
    )
    procs = dict(vit=T.ViTImageProcessor(), dinov2=T.BitImageProcessor(), clip=T.CLIPImageProcessor(), mae=T.ViTImageProcessor(),
                 resnet=T.ConvNextImageProcessor(size={"shortest_edge": 224}, crop_pct=224 / 256), random_vit=T.ViTImageProcessor())
    monkeypatch.setattr(T.ViTConfig, "from_pretrained", classmethod(lambda cls, *a, **k: T.ViTConfig(hidden_size=768, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64)))
    for name, ctor in tiny.items():
        kind = R.BACKBONES[name].kind
        monkeypatch.setattr(T.AutoImageProcessor, "from_pretrained", classmethod(lambda cls, *a, _p=procs[kind], **k: _p))
        for klass in (T.ViTModel, T.AutoModel, T.CLIPVisionModel, T.ViTMAEModel):
            if ctor is not None:
                monkeypatch.setattr(klass, "from_pretrained", classmethod(lambda cls, *a, _c=ctor, **k: _c()))
        model, proc, fwd = F.load_backbone(name, device="cpu")
        img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (40, 48, 3)).astype(np.uint8))
        pv = proc(images=img, return_tensors="pt")["pixel_values"]
        assert pv.shape == (1, 3, 224, 224), (name, pv.shape)
        with torch.no_grad():
            out = fwd(torch.cat([pv, pv]))
        assert out.shape == (2, R.BACKBONES[name].dim), (name, out.shape)
        assert not any(p.requires_grad for p in model.parameters())


def test_extract_with_real_processor_and_workers(tmp_path, monkeypatch):
    """The DataLoader path with a real HuggingFace image processor and worker processes (the Colab configuration)."""
    import transformers as T
    from pipeline import features as F
    model = T.ViTModel(T.ViTConfig(hidden_size=768, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64), add_pooling_layer=False).eval()
    monkeypatch.setattr(F, "load_backbone", lambda backbone, device="cuda": (model, T.ViTImageProcessor(), (lambda pv: model(pixel_values=pv).last_hidden_state[:, 0])))
    monkeypatch.setattr(F, "build_dataset", lambda dataset, data_root, imagenet100_root=None: _mock_dataset(n_per_class=3, n_classes=3))
    F.extract("vit_in21k", "imagenet_r", tmp_path / "data", tmp_path / "feat", batch_size=4, device="cpu", num_workers=2)
    (Xtr, ytr, _), (Xte, yte, _), meta = F.load_pair("vit_in21k", "imagenet_r", tmp_path / "feat")
    assert Xtr.shape == (9, 768) and Xte.shape == (6, 768) and Xtr.dtype == np.float32 and meta["processor"].startswith("ViTImageProcessor")
