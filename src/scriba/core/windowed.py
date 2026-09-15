"""Windowed long-form transcription.

qwen-asr's `transcribe()` runs every phase over the whole recording (ASR on all chunks, then
alignment on all chunks) and keeps everything in memory until the end. For long recordings this
pushes VRAM to the limit, especially during alignment.

Here the same chunks (qwen-asr's own low-energy splitting, identical boundaries) are processed
window by window: ASR → alignment → save to disk → free memory. Peak memory depends on the window,
not on the recording length; finished windows survive crashes and allow resuming.

Only qwen-asr's building blocks are used (`split_audio_into_chunks`, `_infer_asr`,
`parse_asr_output`, `Qwen3ForcedAligner.align`), mirroring `Qwen3ASRModel.transcribe`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scriba.utils.logging import get_logger

log = get_logger("windowed")


@dataclass
class ChunkResult:
    index: int
    offset: float
    duration: float
    language: str
    text: str
    tokens: list[tuple[str, float, float]] | None  # absolute seconds; None without timestamps

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "offset": self.offset, "duration": self.duration, "language": self.language,
                "text": self.text, "tokens": self.tokens}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkResult":
        tokens = data.get("tokens")
        return cls(data["index"], data["offset"], data["duration"], data.get("language", ""), data.get("text", ""),
                   [tuple(t) for t in tokens] if tokens is not None else None)


@dataclass
class WindowPlan:
    total_chunks: int
    window: int
    offsets: list[float]
    durations: list[float]


@dataclass
class WindowStats:
    """Measured work, reported to the progress tracker after every phase of every window."""
    asr_done: int = 0
    align_done: int = 0
    asr_seconds: float = 0.0
    align_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


EventCallback = Callable[[str, "WindowPlan", "WindowStats"], None]  # (event, plan, stats)


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(exc).lower()


def _free_cuda() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class WindowedRunner:
    def __init__(self, model: Any, *, asr_batch: int, align_batch: int, min_batch: int = 1):
        self.model = model
        self.asr_batch = max(1, asr_batch)
        self.align_batch = max(1, align_batch)
        self.min_batch = min_batch

    # ------------------------------------------------------------------ planning
    @staticmethod
    def split(wav: np.ndarray, timestamps: bool) -> tuple[list[np.ndarray], WindowPlan]:
        from qwen_asr.inference.utils import (
            MAX_ASR_INPUT_SECONDS,
            MAX_FORCE_ALIGN_INPUT_SECONDS,
            SAMPLE_RATE,
            split_audio_into_chunks,
        )

        max_sec = MAX_FORCE_ALIGN_INPUT_SECONDS if timestamps else MAX_ASR_INPUT_SECONDS
        parts = split_audio_into_chunks(wav=wav, sr=SAMPLE_RATE, max_chunk_sec=max_sec)
        wavs = [p[0] for p in parts]
        offsets = [float(p[1]) for p in parts]
        ends = offsets[1:] + [len(wav) / SAMPLE_RATE]
        durations = [max(0.0, e - o) for o, e in zip(offsets, ends)]
        return wavs, WindowPlan(total_chunks=len(parts), window=0, offsets=offsets, durations=durations)

    # ------------------------------------------------------------------ execution
    def run(
        self,
        wav: np.ndarray,
        *,
        language: str | None,
        context: str,
        timestamps: bool,
        done: dict[int, ChunkResult] | None = None,
        save: Callable[[list[ChunkResult]], None] | None = None,
        on_event: EventCallback | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> tuple[list[ChunkResult], bool, WindowStats]:
        """Returns (chunk results in order, completed, stats)."""
        from qwen_asr.inference.utils import normalize_language_name, validate_language

        forced = None
        if language:
            forced = normalize_language_name(language)
            validate_language(forced)

        wavs, plan = self.split(wav, timestamps)
        plan.window = self.asr_batch
        results: dict[int, ChunkResult] = dict(done or {})
        stats = WindowStats(asr_done=len(results), align_done=len(results) if timestamps else 0)
        if results:
            log.info("Resuming: %d/%d chunks already processed", len(results), plan.total_chunks)
        self._emit(on_event, "plan", plan, stats)

        start = 0
        while start < plan.total_chunks:
            end = min(plan.total_chunks, start + self.asr_batch)
            indexes = [i for i in range(start, end) if i not in results]
            if not indexes:
                start = end
                continue
            if cancelled():
                return [results[i] for i in sorted(results)], False, stats
            try:
                window_results = self._process_window(wavs, plan, indexes, forced, context, timestamps, stats,
                                                      on_event)
            except _RetryWindow:
                continue  # batch was halved: same window again
            results.update({r.index: r for r in window_results})
            if save:
                save(window_results)
            self._emit(on_event, "window", plan, stats)
            start = end
            _free_cuda()

        return [results[i] for i in sorted(results)], True, stats

    def _process_window(self, wavs, plan, indexes, forced, context, timestamps, stats, on_event):
        from qwen_asr.inference.utils import SAMPLE_RATE, parse_asr_output

        # ---- ASR
        t0 = time.perf_counter()
        try:
            raw = self.model._infer_asr([context or ""] * len(indexes), [wavs[i] for i in indexes],
                                        [forced] * len(indexes))
        except Exception as exc:
            self._maybe_shrink(exc, "asr", stats)
            raise
        stats.asr_seconds += time.perf_counter() - t0
        stats.asr_done += len(indexes)
        parsed = [parse_asr_output(out, user_language=forced) for out in raw]
        self._emit(on_event, "asr", plan, stats)

        # ---- alignment, in its own (smaller) batches
        tokens: dict[int, list[tuple[str, float, float]]] = {}
        if timestamps:
            aligner = self.model.forced_aligner
            to_align = [(i, lang, text) for i, (lang, text) in zip(indexes, parsed) if text.strip()]
            t0 = time.perf_counter()
            for k in range(0, len(to_align), self.align_batch):
                group = to_align[k:k + self.align_batch]
                try:
                    aligned = aligner.align(audio=[(wavs[i], SAMPLE_RATE) for i, _, _ in group],
                                            text=[text for _, _, text in group],
                                            language=[lang or forced or "" for _, lang, _ in group])
                except Exception as exc:
                    stats.asr_done -= len(indexes)  # the window is redone from ASR
                    self._maybe_shrink(exc, "align", stats)
                    raise
                for (i, _, _), res in zip(group, aligned):
                    off = plan.offsets[i]
                    tokens[i] = [(str(it.text), round(float(it.start_time) + off, 3), round(float(it.end_time) + off, 3))
                                 for it in res.items]
            stats.align_seconds += time.perf_counter() - t0
            stats.align_done += len(indexes)
            self._emit(on_event, "align", plan, stats)

        return [
            ChunkResult(index=i, offset=plan.offsets[i], duration=plan.durations[i], language=lang, text=text,
                        tokens=(tokens.get(i, []) if timestamps else None))
            for i, (lang, text) in zip(indexes, parsed)
        ]

    def _maybe_shrink(self, exc: BaseException, phase: str, stats: WindowStats) -> None:
        """On out-of-memory, halve the batch of the failing phase and ask for a retry."""
        if not _is_oom(exc):
            return
        _free_cuda()
        current = self.asr_batch if phase == "asr" else self.align_batch
        if current <= self.min_batch:
            return  # cannot shrink further: let the error propagate
        new = max(self.min_batch, current // 2)
        if phase == "asr":
            self.asr_batch = new
        else:
            self.align_batch = new
        note = f"Out of GPU memory during {phase}: batch reduced from {current} to {new}"
        log.warning(note)
        stats.notes.append(note)
        raise _RetryWindow() from exc

    @staticmethod
    def _emit(on_event: EventCallback | None, event: str, plan: WindowPlan, stats: WindowStats) -> None:
        if on_event is None:
            return
        try:
            on_event(event, plan, stats)
        except Exception as exc:  # progress must never break a transcription
            log.debug("Progress callback failed: %s", exc)


class _RetryWindow(Exception):
    pass
