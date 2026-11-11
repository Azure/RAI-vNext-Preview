# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Prepare fixed upstream sources with explicit, image-local Python 3.10 backports.

The local versions contain the complete Pillow 13.0.0/filelock 4.2.1 fixes,
not relabeled older binaries. SQLite reader/writer locks remain unavailable:
Python 3.10 cannot enforce the SQLite limits required by filelock's fix.
"""

import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# Include upstream's wheel-build helpers, native sources and regression fixtures.
SOURCES = {
    "pillow13": (
        "python-pillow/Pillow", "293218ad059d994baf1777d882911bed6a49704a",
        "0458511ac96e7d44e574484efff9c6210f8211046b3e7ec1dca2381cea0f8430",
    ),
    "filelock421": (
        "tox-dev/filelock", "60331bb89f325020b48bc51bde904c6446384e4a",
        "532567d0b6068831227a5b2eb3c0fd52d7332c3d1fba01d587c84f515af84201",
    ),
    "multibuild": (
        "multi-build/multibuild", "c893f305c6066b6d940e10d3333be8cb27fa3947",
        "e50c410a625333404ba318f109b082ff04b1edb1e18e92109e38897c9561db09",
    ),
    "pillow-depends": (
        "python-pillow/pillow-depends", "70c5bf5e940f5f1aee75ddb0421f0abfd9581261",
        "df367934ced0be7e6d899e1cc2c8582073c4a4544d3fab5bfb52490c39805d25",
    ),
    "pillow-test-images": (
        "python-pillow/test-images", "073ab5ff1895abf8cf5c8d797c1066ed394ce207",
        "2377ec3a523f70da7f305ccbd32ddbf063e6046c56487e7ca08b6ace04bf906f",
    ),
}


def extract_verified(archive, destination, expected_hash):
    """Reject changed archives before extracting any source."""
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise RuntimeError(f"Unexpected source archive hash: {archive.name}")
    with tempfile.TemporaryDirectory() as temporary:
        with tarfile.open(archive) as source:
            source.extractall(temporary, filter="data")
        roots = list(Path(temporary).iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise RuntimeError(f"Unexpected archive layout: {archive.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.rmdir()  # Git submodule placeholders may already exist, empty.
        shutil.move(str(roots[0]), str(destination))


def fetch_source(name, destination):
    """Fetch an immutable upstream revision, verified independently of its URL."""
    repository, revision, digest = SOURCES[name]
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / f"{name}.tar.gz"
        url = f"https://codeload.github.com/{repository}/tar.gz/{revision}"
        with urllib.request.urlopen(url, timeout=300) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        extract_verified(archive, destination, digest)


def replace_exact(path, old, new, count=1):
    """Fail closed if an upstream layout no longer matches the reviewed patch."""
    source = path.read_text(encoding="utf-8")
    if source.count(old) != count:
        raise RuntimeError(f"Unexpected upstream source: {path}")
    path.write_text(source.replace(old, new), encoding="utf-8")


def patch_pillow(root):
    """Restore 3.10 APIs without changing any native security fixes."""
    replace_exact(root / "pyproject.toml", 'requires-python = ">=3.11"', 'requires-python = ">=3.10"')
    replace_exact(
        root / "pyproject.toml", '  "Programming Language :: Python :: 3.11",',
        '  "Programming Language :: Python :: 3.10",\n  "Programming Language :: Python :: 3.11",',
    )
    replace_exact(root / "src/PIL/_version.py", '"13.0.0"', '"13.0.0+rai.py310.1"')
    typing = root / "src/PIL/_typing.py"
    replace_exact(typing, "from numbers import _IntegralLike as IntegralLike\n", "")
    replace_exact(
        typing, "if TYPE_CHECKING:\n",
        "if TYPE_CHECKING:\n    from numbers import _IntegralLike as IntegralLike\n",
    )
    replace_exact(root / "src/PIL/_binary.py", "(i & 255).to_bytes()", '(i & 255).to_bytes(1, "big")')
    # This upstream build-time utility also uses a Python 3.11 alias.
    replace_exact(root / ".github/generate-sbom.py", "dt.UTC", "dt.timezone.utc")


def patch_filelock(root):
    """Preserve ordinary locking, exception groups and weak-reference finalizers."""
    replace_exact(
        root / "pyproject.toml",
        'requires-python = ">=3.11"',
        'requires-python = ">=3.10"\n'
        'dependencies = ["exceptiongroup>=1.2,<2; python_version < \'3.11\'"]',
    )
    replace_exact(
        root / "pyproject.toml", '  "Programming Language :: Python :: 3.11",',
        '  "Programming Language :: Python :: 3.10",\n  "Programming Language :: Python :: 3.11",',
    )
    api = root / "src/filelock/_api.py"
    replace_exact(
        api, "from weakref import WeakKeyDictionary, WeakValueDictionary, ref\n",
        "from weakref import WeakKeyDictionary, WeakValueDictionary, ref\n\n"
        "if sys.version_info < (3, 11):\n    from exceptiongroup import BaseExceptionGroup\n",
    )
    replace_exact(
        api, "@dataclass(slots=True, weakref_slot=True)\nclass _FinalizerState:",
        'class _WeakrefSupport:\n    __slots__ = ("__weakref__",)\n\n\n'
        "@dataclass(slots=True)\nclass _FinalizerState(_WeakrefSupport):",
    )
    for module in ("_read_write.py", "_async_read_write.py"):
        replace_exact(
            root / "src/filelock" / module, "from __future__ import annotations\n",
            "from __future__ import annotations\n\n"
            "import sys\n\n"
            "if sys.version_info < (3, 11):\n"
            '    raise ImportError("RAI Python 3.10 backport disables SQLite reader/writer locks: "\n'
            '                      "the required SQLite security limits need Python 3.11")\n',
        )


def prepare(root):
    pillow = root / "pillow"
    filelock = root / "filelock"
    fetch_source("pillow13", pillow)
    fetch_source("filelock421", filelock)
    patch_pillow(pillow)
    patch_filelock(filelock)
    fetch_source("multibuild", pillow / "wheels/multibuild")
    fetch_source("pillow-depends", pillow / "build/pillow-depends-main")
    fetch_source("pillow-test-images", pillow / "test-images-main")
    shutil.copytree(pillow / "test-images-main", pillow / "Tests/images", dirs_exist_ok=True)


if __name__ == "__main__":
    prepare(Path.cwd())
