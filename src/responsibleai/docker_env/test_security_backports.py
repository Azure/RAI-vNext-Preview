# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline patch tests and build-time regression gates for the image backports."""

import ast
import asyncio
import builtins
import importlib
import importlib.metadata
import io
import os
import sys
import tarfile
import tempfile
import unittest
import warnings
from pathlib import Path

from prepare_security_backports import extract_verified, replace_exact


def uses_sqlite_locks(source):
    """Conservatively identify direct consumers of the disabled filelock APIs."""
    blocked_modules = {"filelock._read_write", "filelock._async_read_write"}
    blocked_names = {"ReadWriteLock", "AsyncReadWriteLock", "AsyncAcquireReadWriteReturnProxy",
                     "_read_write", "_async_read_write"}
    tree = ast.parse(source)
    aliases = {"filelock"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name in blocked_modules:
                    return True
                if name.name == "filelock":
                    aliases.add(name.asname or name.name)
        if isinstance(node, ast.ImportFrom):
            if node.module in blocked_modules:
                return True
            if node.module == "filelock" and any(name.name in blocked_names | {"*"} for name in node.names):
                return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in aliases and node.attr in blocked_names:
                return True
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            if node.args[0].value in blocked_modules:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id in aliases:
                if isinstance(node.args[1], ast.Constant) and node.args[1].value in blocked_names:
                    return True
    return False


def check_installed_consumers():
    """Do not silently remove an API required by an installed image dependency."""
    consumers = []
    for distribution in importlib.metadata.distributions():
        if distribution.metadata["Name"].lower() == "filelock":
            continue
        for entry in distribution.files or ():
            if entry.suffix != ".py":
                continue
            path = Path(distribution.locate_file(entry))
            source = path.read_bytes()
            if b"filelock" in source and (b"ReadWrite" in source or b"_read_write" in source or b"import *" in source):
                if uses_sqlite_locks(source):
                    consumers.append(str(path))
    if consumers:
        raise RuntimeError("Image dependencies require disabled SQLite locks: " + ", ".join(consumers))


def check_security_versions():
    """Reject dependency downgrades and incompatible NumPy/Pandas upgrades."""
    from packaging.version import Version

    for package, version in (
        ("pillow", "13.0.0+rai.py310.1"), ("filelock", "4.2.1+rai.py310.1"),
        ("mlflow", "3.17.2"), ("mlflow-skinny", "3.17.2"), ("mlflow-tracing", "3.17.2"),
        ("skops", "0.16.0"),
    ):
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"Security dependency was replaced: {package}")
    for package, upper_bound in (("numpy", "1.24"), ("pandas", "2")):
        if Version(importlib.metadata.version(package)) >= Version(upper_bound):
            raise RuntimeError(f"RAI compatibility constraint was violated: {package}<{upper_bound}")


def check_runtime():
    """Exercise native decoding, locking and unsupported APIs in the final image."""
    from PIL import Image

    check_security_versions()
    for format_name in ("PNG", "JPEG", "TIFF", "WEBP"):
        data = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(data, format=format_name)
        data.seek(0)
        with Image.open(data) as image:
            image.load()
            assert image.size == (8, 8)
    check_filelock()
    check_installed_consumers()


def check_filelock():
    """Keep normal synchronous/async locking usable while SQLite locks fail closed."""
    import filelock

    assert filelock.ReadWriteLock is None
    assert filelock.AsyncReadWriteLock is None
    for name in ("filelock._read_write", "filelock._async_read_write"):
        try:
            importlib.import_module(name)
        except ImportError as error:
            assert "SQLite security limits" in str(error)
        else:
            raise AssertionError(f"Unsafe SQLite lock API is available: {name}")
    with tempfile.TemporaryDirectory() as temporary:
        for lock_type in (filelock.FileLock, filelock.SoftFileLock):
            path = str(Path(temporary) / f"{lock_type.__name__}.lock")
            with lock_type(path, timeout=10):
                try:
                    lock_type(path).acquire(timeout=0)
                except filelock.Timeout:
                    pass
                else:
                    raise AssertionError("Independent locks did not exclude one another")

        async def asynchronous_lock():
            path = str(Path(temporary) / "async.lock")
            async with filelock.AsyncFileLock(path, timeout=10) as lock:
                assert lock.is_locked

        asyncio.run(asynchronous_lock())


class PatchTests(unittest.TestCase):
    def test_patch_rejects_missing_or_ambiguous_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.py"
            for source in ("unrelated", "old old"):
                path.write_text(source)
                with self.assertRaises(RuntimeError):
                    replace_exact(path, "old", "new")
                self.assertEqual(path.read_text(), source)
            path.write_text("before old after")
            replace_exact(path, "old", "new")
            self.assertEqual(path.read_text(), "before new after")

    def test_source_hash_is_checked_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.tar.gz"
            with tarfile.open(archive, "w:gz"):
                pass
            destination = Path(temporary) / "source"
            with self.assertRaisesRegex(RuntimeError, "archive hash"):
                extract_verified(archive, destination, "0" * 64)
            self.assertFalse(destination.exists())

    def test_detects_unsupported_consumers_without_blocking_ordinary_locks(self):
        for source in (
            "from filelock import ReadWriteLock",
            "import filelock as fl\nfl.AsyncReadWriteLock('database')",
            "from filelock._read_write import ReadWriteLock",
            "importlib.import_module('filelock._async_read_write')",
            "import filelock\ngetattr(filelock, 'ReadWriteLock')",
        ):
            self.assertTrue(uses_sqlite_locks(source), source)
        for source in (
            "from filelock import FileLock, SoftFileLock",
            "import filelock as fl\nfl.SoftReadWriteLock('marker')",
            "from unrelated import ReadWriteLock",
        ):
            self.assertFalse(uses_sqlite_locks(source), source)


class CatchWarnings(warnings.catch_warnings):
    """Adapt only upstream tests using the Python 3.11 warning-context API."""

    def __init__(self, *args, action=None, category=Warning, lineno=0, append=False, **kwargs):
        self.filter = (action, category, lineno, append)
        super().__init__(*args, **kwargs)

    def __enter__(self):
        result = super().__enter__()
        action, category, lineno, append = self.filter
        if action is not None:
            warnings.simplefilter(action, category, lineno, append)
        return result


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode == "runtime":
        check_runtime()
    elif mode in ("pillow", "filelock"):
        import pytest

        warnings.catch_warnings = CatchWarnings
        if mode == "pillow":
            os.chdir("/build/pillow")
            arguments = ["checks/check_wheel.py", "Tests", "-q", "-x"]
        else:
            from exceptiongroup import BaseExceptionGroup, ExceptionGroup

            builtins.BaseExceptionGroup = BaseExceptionGroup
            builtins.ExceptionGroup = ExceptionGroup
            check_filelock()
            os.chdir("/build/filelock")
            arguments = ["tests/test_filelock.py", "-q", "-x"]
        sys.exit(pytest.main(arguments))
    else:
        unittest.main()
