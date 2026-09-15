"""Background sampler for peak VRAM / RAM (used by `scriba benchmark`)."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading


class ResourceMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.peak_vram_mb: float | None = None
        self.peak_ram_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvidia_smi = shutil.which("nvidia-smi")
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
        except ImportError:
            self._process = None

    def __enter__(self) -> "ResourceMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        if self._process is not None:
            try:
                rss = self._process.memory_info().rss
                for child in self._process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except Exception:
                        pass
                self.peak_ram_mb = max(self.peak_ram_mb or 0.0, rss / 1024**2)
            except Exception:
                pass
        if self._nvidia_smi:
            try:
                out = subprocess.run(
                    [self._nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                used = max(float(x) for x in out.split() if x.strip())
                self.peak_vram_mb = max(self.peak_vram_mb or 0.0, used)
            except Exception:
                pass
