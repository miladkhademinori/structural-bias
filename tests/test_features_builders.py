"""Dataset-builder tests for pipeline/features.py (W7-1, 2026-10-21).

Tier: UNIT (mechanical invariants of the folder-based builders) plus one
INTEGRATION smoke of build_dataset on a synthesized ImageFolder.  Everything
runs OFFLINE against tiny fixtures under the system temp directory (%TEMP%):
no network, no torchvision, no torch, no downloads (the owner storage law).
The PNG fixture synthesizes 60 tiny RGB PNGs (6 classes; 42 train + 18 val)
with Pillow; if Pillow is absent those tests SKIP with the reason recorded
(nothing may be downloaded to build the fixture).  The zip-based and
split-helper tests need no images at all.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import features as F  # noqa: E402

try:
    from PIL import Image
except ImportError:  # probed present 2026-10-21 (Pillow 12.2.0); skip path kept
    Image = None

PIL_SKIP = "Pillow not installed: the PNG fixture cannot be synthesized "
"(downloads are forbidden on this machine); run where Pillow is available"

N_CLASSES = 6
TRAIN_PER_CLASS, VAL_PER_CLASS = 7, 3          # 42 + 18 = 60 PNGs total
IMG_SIZE = 8

CLASS_NAMES = [f"cls{i:02d}" for i in range(N_CLASSES)]


def _png_bytes(color) -> bytes:
    arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = color
    import io
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@contextlib.contextmanager
def imagefolder_root(base: Path):
    """ImageNet-100-style root: train/<class>/*.png + val/<class>/*.png."""
    root = base / "imagenet100_root"
    for split, n in (("train", TRAIN_PER_CLASS), ("val", VAL_PER_CLASS)):
        for ci, cname in enumerate(CLASS_NAMES):
            d = root / split / cname
            d.mkdir(parents=True)
            for j in range(n):
                (d / f"img_{j:02d}.png").write_bytes(
                    _png_bytes((40 * ci + 20, 255 - 30 * ci, 128)))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class ImageFolderBuilderTests(unittest.TestCase):

    def setUp(self):
        if Image is None:
            self.skipTest(PIL_SKIP)
        self.base = Path(tempfile.mkdtemp(prefix="w7_1_imagenet100-"))

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_build_dataset_imagenet100_smoke(self):
        """UNIT/INTEGRATION smoke: build_dataset on the folder-based path
        (the owner-gated ImageNet-100 builder; identical machinery to the
        other ImageFolder datasets) returns train/test ImageLists with the
        exact per-class counts, sorted class names, int labels 0..5, zero
        domains, and loadable RGB images.  Epsilon: exact counts/labels.
        """
        with imagefolder_root(self.base) as root:
            out = F.build_dataset("imagenet100", self.base, imagenet100_root=root)
            self.assertEqual(sorted(out), ["test", "train"])
            tr, te = out["train"], out["test"]
            self.assertEqual(len(tr), N_CLASSES * TRAIN_PER_CLASS)
            self.assertEqual(len(te), N_CLASSES * VAL_PER_CLASS)
            self.assertEqual(tr.classes, CLASS_NAMES)
            self.assertEqual(te.classes, CLASS_NAMES)
            self.assertEqual(tr.labels.dtype, np.int32)
            self.assertEqual(sorted(np.unique(tr.labels).tolist()), list(range(N_CLASSES)))
            for c in range(N_CLASSES):
                self.assertEqual(int((tr.labels == c).sum()), TRAIN_PER_CLASS)
                self.assertEqual(int((te.labels == c).sum()), VAL_PER_CLASS)
            self.assertTrue(np.all(tr.domains == 0) and np.all(te.domains == 0))
            img = tr.image(0)
            self.assertEqual(img.mode, "RGB")
            self.assertEqual(img.size, (IMG_SIZE, IMG_SIZE))

    def test_image_folder_listing_matches_fixture(self):
        """UNIT: _image_folder_listing enumerates every PNG exactly once,
        classes sorted, labels block-ordered by class directory.
        """
        with imagefolder_root(self.base) as root:
            paths, labels, classes = F._image_folder_listing(root / "train")
            self.assertEqual(classes, CLASS_NAMES)
            self.assertEqual(len(paths), N_CLASSES * TRAIN_PER_CLASS)
            self.assertEqual(labels.tolist(),
                             np.repeat(np.arange(N_CLASSES), TRAIN_PER_CLASS).tolist())
            self.assertTrue(all(p.suffix == ".png" and p.exists() for p in paths))


class SplitHelperTests(unittest.TestCase):
    """PIL-free: _stratified_split is pure label arithmetic."""

    def test_stratified_split_deterministic_disjoint_stratified(self):
        """UNIT: the ImageNet-R split helper is deterministic (same
        purpose-string -> bit-identical index arrays), disjoint, covering,
        sorted, and stratified (per-class test counts = round(frac * n)).
        """
        labels = np.repeat(np.arange(6), 10)
        a1 = F._stratified_split(labels, 0.3, "split:imagenet_r")
        a2 = F._stratified_split(labels, 0.3, "split:imagenet_r")
        tr1, te1 = a1
        self.assertEqual(tr1.tolist(), a2[0].tolist())
        self.assertEqual(te1.tolist(), a2[1].tolist())
        self.assertEqual(np.intersect1d(tr1, te1).size, 0)
        self.assertEqual(np.union1d(tr1, te1).size, labels.size)
        self.assertTrue(np.all(np.diff(tr1) > 0) and np.all(np.diff(te1) > 0))
        for c in range(6):
            self.assertEqual(int((labels[te1] == c).sum()), 3)
            self.assertEqual(int((labels[tr1] == c).sum()), 7)


class LocalArchiveBuilderTests(unittest.TestCase):
    """PIL-free: the OmniBenchmark builder path with a pre-placed zip.
    _gdown short-circuits when the archive already exists, so a fabricated
    zip exercises extract + listing with zero network."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="w7_1_omnibench-"))

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_build_dataset_omnibenchmark_from_preplaced_zip(self):
        """UNIT/INTEGRATION: build_dataset('omnibenchmark') on a local zip
        matching the RevisitingCIL layout (train/ and test/ ImageFolders)
        yields the right lengths, classes and labels, fully offline.
        """
        zp = self.base / "omnibenchmark.zip"
        with zipfile.ZipFile(zp, "w") as z:
            for split in ("train", "test"):
                for ci, cname in enumerate(CLASS_NAMES):
                    for j in range(3 if split == "train" else 1):
                        z.writestr(f"omnibenchmark/{split}/{cname}/f{j:03d}.jpg", b"x")
        out = F.build_dataset("omnibenchmark", self.base)
        tr, te = out["train"], out["test"]
        self.assertEqual(tr.classes, CLASS_NAMES)
        self.assertEqual(len(tr), N_CLASSES * 3)
        self.assertEqual(len(te), N_CLASSES)
        self.assertEqual(tr.labels.tolist(),
                         np.repeat(np.arange(N_CLASSES), 3).tolist())
        self.assertTrue((self.base / "omnibenchmark" / ".extracted_omnibenchmark.zip").exists())


class DownloadGuardTests(unittest.TestCase):

    def test_assert_download_allowed_blocks_local_fetch(self):
        """UNIT: the owner storage law guard. Off Colab and without the
        explicit override env var, _assert_download_allowed must raise.
        """
        env = dict(os.environ)
        env.pop("REPLICATION_ALLOW_DOWNLOADS", None)
        for k in ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_BACKEND_VERSION"):
            env.pop(k, None)
        old = os.environ
        try:
            os.environ = env
            self.assertNotIn("google.colab", sys.modules)
            with self.assertRaises(RuntimeError) as cm:
                F._assert_download_allowed("unit-test")
            self.assertIn("Download blocked", str(cm.exception))
        finally:
            os.environ = old

    def test_download_short_circuits_on_existing_file(self):
        """UNIT: _download returns an already-present file without touching
        the network (idempotent cache behaviour, offline).
        """
        dst = Path(tempfile.mkdtemp(prefix="w7_1_dl-")) / "cached.bin"
        dst.write_bytes(b"cached")
        try:
            got = F._download("http://invalid.invalid/x.bin", dst)
            self.assertEqual(got, dst)
            self.assertEqual(dst.read_bytes(), b"cached")
        finally:
            shutil.rmtree(dst.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
