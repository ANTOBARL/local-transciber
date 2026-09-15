"""Workarounds for third-party quirks.

nagisa (imported by qwen-asr for Japanese alignment) builds a DyNet model at import
time, and DyNet cannot open files whose path contains non-ASCII characters — common on
Windows user profiles (e.g. C:\\Users\\Renè\\...). We initialise nagisa ourselves with
an ASCII-only path: the Windows 8.3 short path, or a copy under an ASCII directory.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

from scriba.utils.logging import get_logger

log = get_logger("compat")

_NAGISA_FILES = ("nagisa_v001.dict", "nagisa_v001.model", "nagisa_v001.hp")


def _short_path(path: Path) -> str | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    fn = ctypes.windll.kernel32.GetShortPathNameW
    fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    fn.restype = wintypes.DWORD
    buf = ctypes.create_unicode_buffer(1024)
    if fn(str(path), buf, len(buf)) == 0:
        return None
    return buf.value if buf.value.isascii() else None


def _ascii_data_dir(src: Path) -> Path | None:
    short = _short_path(src)
    if short:
        return Path(short)
    candidates = [Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "scriba" / "nagisa", Path(tempfile.gettempdir())]
    for base in candidates:
        if not str(base).isascii():
            continue
        try:
            base.mkdir(parents=True, exist_ok=True)
            for name in _NAGISA_FILES:
                dst = base / name
                if not dst.exists() or dst.stat().st_size != (src / name).stat().st_size:
                    shutil.copy2(src / name, dst)
            return base
        except OSError:
            continue
    return None


def ensure_nagisa_importable() -> None:
    if "nagisa" in sys.modules:
        return
    spec = importlib.util.find_spec("nagisa")
    if spec is None or not spec.submodule_search_locations:
        return
    pkg_dir = Path(list(spec.submodule_search_locations)[0])
    if str(pkg_dir).isascii():
        return

    data_dir = _ascii_data_dir(pkg_dir / "data")
    if data_dir is None:
        log.warning("nagisa lives in a non-ASCII path and no ASCII fallback was found")
        return

    # Register a bare package so submodules import without running nagisa/__init__.py.
    pkg = types.ModuleType("nagisa")
    pkg.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    pkg.__file__ = str(pkg_dir / "__init__.py")
    sys.modules["nagisa"] = pkg
    try:
        from nagisa.tagger import Tagger

        tagger = Tagger(
            vocabs=str(data_dir / "nagisa_v001.dict"),
            params=str(data_dir / "nagisa_v001.model"),
            hp=str(data_dir / "nagisa_v001.hp"),
        )
    except Exception as exc:
        del sys.modules["nagisa"]
        log.warning("nagisa workaround failed: %s", exc)
        return

    for name in ("wakati", "tagging", "filter", "extract", "postagging", "decode"):
        setattr(pkg, name, getattr(tagger, name))
    pkg.Tagger, pkg.tagger, pkg.__version__ = Tagger, tagger, "0.2.11"
    log.debug("nagisa initialised from ASCII path %s", data_dir)
