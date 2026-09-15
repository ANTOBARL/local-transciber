"""Transcription progress and time-to-completion estimate.

Progress units come from the real work qwen-asr does (ASR batches, then alignment batches).
Between units the bar advances smoothly using the measured pace; before the first unit the
estimate relies on the throughput of previous jobs (local cache).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from scriba.utils.logging import get_logger

log = get_logger("progress")

MIN_SECONDS_FOR_HISTORY = 60.0  # short clips are dominated by fixed overhead


def cache_dir() -> Path:
    env = os.environ.get("SCRIBA_CACHE_DIR")
    return Path(env) if env else Path.home() / ".cache" / "scriba"


class ThroughputHistory:
    """Remembers processing-seconds per audio-second for a given model/device setup."""

    def __init__(self, path: Path | None = None):
        self.path = path or cache_dir() / "throughput.json"

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def get(self, key: str) -> float | None:
        entry = self._load().get(key)
        return float(entry["rate"]) if isinstance(entry, dict) and "rate" in entry else None

    def record(self, key: str, processing_seconds: float, audio_seconds: float) -> None:
        if audio_seconds < MIN_SECONDS_FOR_HISTORY or processing_seconds <= 0:
            return
        rate = processing_seconds / audio_seconds
        data = self._load()
        previous = data.get(key, {}).get("rate")
        # Exponential moving average: adapts to tuning changes without forgetting everything.
        data[key] = {"rate": rate if previous is None else 0.6 * rate + 0.4 * float(previous),
                     "updated": time.time()}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            log.debug("Cannot write throughput cache: %s", exc)


class TranscriptionProgress:
    ASR_WEIGHT_WITH_ALIGNMENT = 0.85  # alignment is a single forward pass, much cheaper than generation

    def __init__(self, audio_seconds: float, timestamps: bool, prior_rate: float | None = None,
                 clock=time.monotonic):
        self.audio_seconds = max(audio_seconds, 0.001)
        self.timestamps = timestamps
        self.prior_rate = prior_rate
        self._clock = clock
        self._lock = threading.Lock()
        self.started = clock()
        self.phase = "asr"
        self.asr_done = self.asr_total = 0
        self.align_done = self.align_total = 0
        self.step = 1
        self._fraction = 0.0
        self._fraction_at = self.started
        self.finished = False

    # ------------------------------------------------------------------ updates
    @property
    def _asr_weight(self) -> float:
        return self.ASR_WEIGHT_WITH_ALIGNMENT if self.timestamps else 1.0

    def update(self, phase: str, done: int, total: int, step: int = 1) -> None:
        with self._lock:
            self.phase = phase
            self.step = max(1, step)
            if phase == "asr":
                self.asr_done, self.asr_total = done, total
            else:
                self.align_done, self.align_total = done, total
            fraction = self._real_fraction()
            if fraction > self._fraction:
                self._fraction, self._fraction_at = fraction, self._clock()

    def finish(self) -> None:
        with self._lock:
            self.finished = True
            self._fraction, self._fraction_at = 1.0, self._clock()

    def _real_fraction(self) -> float:
        w = self._asr_weight
        asr = self.asr_done / self.asr_total if self.asr_total else 0.0
        align = self.align_done / self.align_total if self.align_total else 0.0
        return min(1.0, w * asr + (1 - w) * align)

    def _next_boundary(self) -> float:
        """Fraction reached when the batch currently running completes."""
        w = self._asr_weight
        if self.phase == "asr" and self.asr_total:
            return min(w, w * min(self.asr_total, self.asr_done + self.step) / self.asr_total)
        if self.align_total:
            nxt = min(self.align_total, self.align_done + self.step) / self.align_total
            return min(1.0, w + (1 - w) * nxt)
        return w if self.phase == "asr" else 1.0

    # ------------------------------------------------------------------ estimate
    def estimated_total_seconds(self) -> float | None:
        prior = self.prior_rate * self.audio_seconds if self.prior_rate else None
        f = self._fraction
        if f <= 0:
            return prior
        measured = (self._fraction_at - self.started) / f
        if prior is None:
            return measured
        return f * measured + (1 - f) * prior  # trust measurements more as the job advances

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            elapsed = now - self.started
            real = self._fraction
            total = self.estimated_total_seconds()
            if self.finished:
                shown, eta = 1.0, 0.0
            elif total:
                # Smoothly move toward the next batch boundary, never past it.
                ceiling = max(real, self._next_boundary() - 0.01)
                shown = max(real, min(elapsed / total, ceiling, 0.99))
                eta = max(0.0, total - elapsed)
            else:
                shown, eta = real, None
            return {
                "phase": self.phase,
                "fraction": round(shown, 4),
                "percent": int(shown * 100),
                "eta_seconds": None if eta is None else round(eta, 1),
                "elapsed_seconds": round(elapsed, 1),
                "asr": [self.asr_done, self.asr_total],
                "align": [self.align_done, self.align_total],
                "estimated_from_history": real <= 0 and total is not None,
            }


def format_eta(seconds: float | None, lang: str = "it") -> str:
    """Human-friendly remaining time (kept in sync with frontend/src/progress.ts)."""
    it = lang == "it"
    if seconds is None:
        return "stima in corso…" if it else "estimating…"
    if seconds < 10:
        return "quasi terminato" if it else "almost done"
    s = int(round(seconds))
    if s < 60:
        return f"circa {s} s rimanenti" if it else f"about {s} s left"
    if s < 3600:
        m, sec = divmod(s, 60)
        sec = (sec // 10) * 10
        body = f"{m} min {sec:02d} s" if m < 10 and sec else f"{m} min"
        return f"circa {body} rimanenti" if it else f"about {body} left"
    h, rem = divmod(s, 3600)
    return f"circa {h} h {rem // 60:02d} min rimanenti" if it else f"about {h} h {rem // 60:02d} min left"
