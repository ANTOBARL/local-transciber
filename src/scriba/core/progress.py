"""Transcription progress and time-to-completion estimate.

The estimate is built only from measured work:
  - seconds per chunk from the ASR batches already completed,
  - seconds per chunk of alignment, measured once alignment starts (before that, the
    alignment/ASR time ratio observed in previous jobs, or a conservative default).
The bar may advance within the batch currently running, but never by more than that batch,
so it cannot run ahead of the real work. Before the first batch completes there is no ETA
unless previous jobs on the same setup provide one.
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

MIN_SECONDS_FOR_HISTORY = 60.0   # short clips are dominated by fixed overhead
DEFAULT_ALIGN_RATIO = 0.25       # alignment time / ASR time when nothing was measured yet (conservative)
INFLIGHT_CAP = 0.9               # credit at most 90% of a batch while it is still running


def cache_dir() -> Path:
    env = os.environ.get("SCRIBA_CACHE_DIR")
    return Path(env) if env else Path.home() / ".cache" / "scriba"


class ThroughputHistory:
    """Exponential moving averages of measured job metrics, per model/device setup."""

    def __init__(self, path: Path | None = None):
        self.path = path or cache_dir() / "throughput.json"
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def entry(self, key: str) -> dict[str, Any]:
        value = self._load().get(key)
        return value if isinstance(value, dict) else {}

    def get(self, key: str, field: str = "rate") -> float | None:
        value = self.entry(key).get(field)
        return float(value) if isinstance(value, (int, float)) else None

    def record(self, key: str, processing_seconds: float, audio_seconds: float, **extra: float) -> None:
        """Store a finished job. `rate` = processing seconds per audio second."""
        if audio_seconds < MIN_SECONDS_FOR_HISTORY or processing_seconds <= 0:
            return
        with self._lock:
            data = self._load()
            previous = data.get(key, {}) if isinstance(data.get(key), dict) else {}
            values = {"rate": processing_seconds / audio_seconds, **extra}
            merged: dict[str, Any] = {}
            for name, value in values.items():
                old = previous.get(name)
                # Moving average: adapts to tuning changes without forgetting everything.
                merged[name] = value if not isinstance(old, (int, float)) else 0.6 * value + 0.4 * float(old)
            merged["jobs"] = int(previous.get("jobs", 0)) + 1
            merged["audio_seconds"] = float(previous.get("audio_seconds", 0.0)) + audio_seconds
            merged["updated"] = time.time()
            data[key] = merged
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except OSError as exc:
                log.debug("Cannot write throughput cache: %s", exc)


def real_speed_key(device_name: str | None, batch: int, timestamps: bool) -> str:
    return f"real|{device_name}|bs={batch}|ts={timestamps}"


class TranscriptionProgress:
    """Progress of a windowed transcription, driven by events from WindowedRunner.

    Each window runs ASR then alignment on its chunks, so both counters advance together.
    Time per chunk is measured separately for the two phases; chunks restored from a checkpoint
    count as done but are excluded from the speed measurement.
    """

    def __init__(self, audio_seconds: float, timestamps: bool, prior_rate: float | None = None,
                 prior_align_ratio: float | None = None, clock=time.monotonic):
        self.audio_seconds = max(audio_seconds, 0.001)
        self.timestamps = timestamps
        self.prior_rate = prior_rate
        self.align_ratio = prior_align_ratio if prior_align_ratio is not None else DEFAULT_ALIGN_RATIO
        self._clock = clock
        self._lock = threading.Lock()
        self.started = clock()
        self.finished_at: float | None = None
        self.total = 0
        self.window = 1
        self.phase = "asr"
        self.asr_done = self.align_done = 0
        self.resumed = 0
        self.asr_seconds = self.align_seconds = 0.0
        self.last_event = self.started

    # ------------------------------------------------------------------ events
    def on_event(self, event: str, plan, stats) -> None:
        now = self._clock()
        with self._lock:
            self.total = plan.total_chunks
            self.window = max(1, plan.window)
            if event == "plan":
                self.resumed = stats.asr_done
            self.asr_done, self.align_done = stats.asr_done, stats.align_done
            self.asr_seconds, self.align_seconds = stats.asr_seconds, stats.align_seconds
            # What runs next: alignment right after a window's ASR, otherwise the next window's ASR.
            self.phase = "align" if event == "asr" and self.timestamps else "asr"
            self.last_event = now

    def finish(self) -> None:
        with self._lock:
            self.finished_at = self._clock()

    # ------------------------------------------------------------------ measurements
    @property
    def asr_seconds_per_chunk(self) -> float | None:
        measured = self.asr_done - self.resumed
        return self.asr_seconds / measured if measured > 0 and self.asr_seconds > 0 else None

    @property
    def align_seconds_per_chunk(self) -> float | None:
        measured = self.align_done - (self.resumed if self.timestamps else 0)
        return self.align_seconds / measured if measured > 0 and self.align_seconds > 0 else None

    def measured_align_ratio(self) -> float | None:
        asr, align = self.asr_seconds_per_chunk, self.align_seconds_per_chunk
        return align / asr if asr and align else None

    # ------------------------------------------------------------------ estimate
    def _chunk_fraction(self) -> float:
        """Share of work done judged by chunk counts alone, alignment weighted by the expected ratio."""
        if not self.total:
            return 0.0
        weight = self.align_ratio if self.timestamps else 0.0
        done = self.asr_done + self.align_done * weight
        return min(0.99, done / (self.total * (1 + weight)))

    def _estimate(self, now: float) -> tuple[float, float | None]:
        asr_spc = self.asr_seconds_per_chunk
        if asr_spc is None or not self.total:
            # Nothing measured yet (e.g. a resumed job before its first new window): the bar shows the
            # chunks already done; only previous jobs may provide an ETA.
            fraction = self._chunk_fraction()
            if self.prior_rate:
                return fraction, max(0.0, self.prior_rate * self.audio_seconds * (1 - fraction) - (now - self.started))
            return fraction, None

        align_spc = 0.0
        if self.timestamps:
            align_spc = self.align_seconds_per_chunk or asr_spc * self.align_ratio
        total_work = self.total * (asr_spc + align_spc)
        done_work = self.asr_done * asr_spc + self.align_done * align_spc

        # Credit the batch currently running, at most 90% of its expected duration.
        remaining_in_window = min(self.window, self.total - (self.asr_done if self.phase == "asr" else self.align_done))
        expected = remaining_in_window * (asr_spc if self.phase == "asr" else align_spc)
        inflight = min(max(0.0, now - self.last_event), INFLIGHT_CAP * expected)

        fraction = min(0.99, (done_work + inflight) / total_work) if total_work else 0.0
        return fraction, max(0.0, total_work - done_work - inflight)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self.finished_at or self._clock()
            fraction, eta = (1.0, 0.0) if self.finished_at is not None else self._estimate(now)
            return {
                "phase": self.phase,
                "fraction": round(fraction, 4),
                "percent": int(fraction * 100),
                "eta_seconds": None if eta is None else round(eta, 1),
                "elapsed_seconds": round(now - self.started, 1),
                "asr": [self.asr_done, self.total],
                "align": [self.align_done, self.total if self.timestamps else 0],
                "estimated_from_history": self.asr_seconds_per_chunk is None and eta is not None,
            }


def format_eta(seconds: float | None, lang: str = "it") -> str:
    """Human-friendly remaining time (kept in sync with frontend/src/progress.tsx)."""
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
