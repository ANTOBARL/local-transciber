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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scriba.core.quality import SpeechRate, check_text, collapse_loops, join_chunk_texts, strip_context_echo
from scriba.utils.logging import get_logger

log = get_logger("windowed")

# Sub-chunk lengths tried, in order, when a chunk's transcript looks broken (only those shorter than it).
RETRY_SECONDS = (30.0, 10.0)

# Real speech needs about 8 tokens per second. A looping decoder otherwise keeps going until
# max_new_tokens (4096) while the rest of its batch waits; with this cap a 30 s piece stops at 664.
TOKENS_PER_SECOND_CAP = 20.0
MIN_TOKEN_CAP = 64


@contextmanager
def token_budget(model: Any, seconds: float) -> Iterator[None]:
    """Temporarily lower the model's generation limit to what `seconds` of speech can need."""
    cap = int(MIN_TOKEN_CAP + TOKENS_PER_SECOND_CAP * seconds)
    restore: list[tuple[Any, str, int]] = []
    for holder, name in ((model, "max_new_tokens"), (getattr(model, "sampling_params", None), "max_tokens")):
        value = getattr(holder, name, None)
        if isinstance(value, int) and value > cap:
            try:
                setattr(holder, name, cap)
            except Exception:  # immutable backend object: keep its own limit
                continue
            restore.append((holder, name, value))
    try:
        yield
    finally:
        for holder, name, value in restore:
            setattr(holder, name, value)


@dataclass
class ChunkResult:
    index: int
    offset: float
    duration: float
    language: str
    text: str
    tokens: list[tuple[str, float, float]] | None  # absolute seconds; None without timestamps
    issues: list[dict[str, Any]] = field(default_factory=list)  # quality problems found (see _repair)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "offset": self.offset, "duration": self.duration, "language": self.language,
                "text": self.text, "tokens": self.tokens, "issues": self.issues}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkResult":
        tokens = data.get("tokens")
        return cls(data["index"], data["offset"], data["duration"], data.get("language", ""), data.get("text", ""),
                   [tuple(t) for t in tokens] if tokens is not None else None, data.get("issues") or [])


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
    def __init__(self, model: Any, *, asr_batch: int, align_batch: int, min_batch: int = 1,
                 chunk_seconds: float | None = None):
        self.model = model
        self.chunk_seconds = chunk_seconds
        self.asr_batch = max(1, asr_batch)
        self.align_batch = max(1, align_batch)
        self.min_batch = min_batch
        self.rates = SpeechRate()

    # ------------------------------------------------------------------ planning
    @staticmethod
    def split(wav: np.ndarray, timestamps: bool, max_seconds: float | None = None
              ) -> tuple[list[np.ndarray], WindowPlan]:
        from qwen_asr.inference.utils import (
            MAX_ASR_INPUT_SECONDS,
            MAX_FORCE_ALIGN_INPUT_SECONDS,
            SAMPLE_RATE,
            split_audio_into_chunks,
        )

        max_sec = MAX_FORCE_ALIGN_INPUT_SECONDS if timestamps else MAX_ASR_INPUT_SECONDS
        if max_seconds:
            max_sec = min(max_sec, max_seconds)
        parts = split_audio_into_chunks(wav=wav, sr=SAMPLE_RATE, max_chunk_sec=max_sec)
        wavs = [p[0] for p in parts]
        offsets = [float(p[1]) for p in parts]
        ends = offsets[1:] + [len(wav) / SAMPLE_RATE]
        durations = [max(0.0, e - o) for o, e in zip(offsets, ends)]
        return wavs, WindowPlan(total_chunks=len(parts), window=0, offsets=offsets, durations=durations)

    @staticmethod
    def split_sub(wav: np.ndarray, max_seconds: float) -> list[tuple[np.ndarray, float, float]]:
        """(waveform, offset, duration) pieces of one chunk, cut at low-energy points."""
        from qwen_asr.inference.utils import SAMPLE_RATE, split_audio_into_chunks

        parts = split_audio_into_chunks(wav=wav, sr=SAMPLE_RATE, max_chunk_sec=max_seconds)
        offsets = [float(p[1]) for p in parts]
        ends = offsets[1:] + [len(wav) / SAMPLE_RATE]
        return [(p[0], o, max(0.0, e - o)) for p, o, e in zip(parts, offsets, ends)]

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

        wavs, plan = self.split(wav, timestamps, self.chunk_seconds)
        plan.window = self.asr_batch
        results: dict[int, ChunkResult] = dict(done or {})
        self.rates = SpeechRate()
        for chunk in results.values():
            self.rates.add(check_text(chunk.text, chunk.duration))
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
            with token_budget(self.model, max(plan.durations[i] for i in indexes)):
                raw = self.model._infer_asr([context or ""] * len(indexes), [wavs[i] for i in indexes],
                                            [forced] * len(indexes))
        except Exception as exc:
            self._maybe_shrink(exc, "asr", stats)
            raise
        # Only the regular decode counts as speed: repairs are occasional and would distort the ETA.
        stats.asr_seconds += time.perf_counter() - t0
        stats.asr_done += len(indexes)
        parsed = [parse_asr_output(out, user_language=forced) for out in raw]
        issues: dict[int, list[dict[str, Any]]] = {}
        for (_, text), i in zip(parsed, indexes):
            self.rates.add(check_text(text, plan.durations[i]))
        reference = self.rates.reference
        for pos, i in enumerate(indexes):
            parsed[pos], issues[i] = self._repair(wavs[i], plan.offsets[i], plan.durations[i], parsed[pos],
                                                  forced, context, reference)
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
                        tokens=(tokens.get(i, []) if timestamps else None), issues=issues[i])
            for i, (lang, text) in zip(indexes, parsed)
        ]

    # ------------------------------------------------------------------ quality repair
    def _repair(self, wav, offset: float, duration: float, parsed: tuple[str, str], forced, context,
                reference: float | None = None) -> tuple[tuple[str, str], list[dict[str, Any]]]:
        """Undo a recited context first (re-decoding without it), then fix loops and dropped speech."""
        lang, text = parsed
        issues: list[dict[str, Any]] = []
        if context:
            _, echoed = strip_context_echo(text, context)
            if echoed:
                log.warning("Chunk at %.0fs: the model recited the context (%d words), re-decoding without it",
                            offset, echoed)
                try:
                    text = self._decode_pieces([(wav, 0.0, duration)], forced, "")[0]
                except Exception as exc:  # best effort: drop the recited words
                    if _is_oom(exc):
                        _free_cuda()
                    log.warning("Re-decoding the chunk at %.0fs failed: %s", offset, exc)
                    text = strip_context_echo(text, context)[0]
                context = ""
                issues.append(_issue("context", offset, duration, "redecoded", echoed))
        repaired, more = self._repair_text(wav, offset, duration, (lang, text), forced, context, reference)
        return repaired, issues + more

    def _repair_text(self, wav, offset: float, duration: float, parsed: tuple[str, str], forced, context,
                     reference: float | None = None) -> tuple[tuple[str, str], list[dict[str, Any]]]:
        """Re-decode a chunk whose transcript loops (or is nearly empty) in shorter pieces.

        Shorter inputs rarely loop, and a loop that still happens only costs its own piece. Pieces
        that keep looping are collapsed to one occurrence of the repeated phrase and reported.
        """
        lang, text = parsed
        check = check_text(text, duration, reference)
        if not (check.repetitive or check.sparse):
            return parsed, []
        kind = "repetition" if check.repetitive else "sparse"
        log.warning("Chunk at %.0fs: %s (%d words in %.0fs), re-decoding in shorter pieces",
                    offset, kind, check.words, duration)

        pieces_out: list[tuple[str, float, float, bool]] | None = None  # (text, start, duration, looping)
        for seconds in (s for s in RETRY_SECONDS if s < duration - 1):
            pieces = self.split_sub(wav, seconds)
            try:
                texts = self._decode_pieces(pieces, forced, context)
            except Exception as exc:  # best effort: keep what we already have
                if _is_oom(exc):
                    _free_cuda()
                log.warning("Re-decoding the chunk at %.0fs failed: %s", offset, exc)
                break
            pieces_out = [(t, offset + o, d, check_text(t, d).repetitive) for t, (_, o, d) in zip(texts, pieces)]
            if not any(p[3] for p in pieces_out):
                break

        if kind == "sparse":
            if pieces_out is not None:
                candidate = join_chunk_texts(p[0] for p in pieces_out)
                if not check_text(candidate, duration).repetitive and len(candidate.split()) > check.words:
                    text = candidate
            if not check_text(text, duration, reference).sparse:
                return (lang, text), [_issue("sparse", offset, duration, "redecoded")]
            # Only nearly silent pieces are worth reporting; a slow speaker is not a problem.
            if check_text(text, duration).sparse:
                return (lang, text), [_issue("sparse", offset, duration, "kept")]
            return (lang, text), []

        if pieces_out is None:
            collapsed, removed = collapse_loops(text)
            return (lang, collapsed), [_issue("repetition", offset, duration, "collapsed", removed)]

        issues: list[dict[str, Any]] = []
        texts = []
        for piece_text, start, piece_duration, looping in pieces_out:
            if looping:
                piece_text, removed = collapse_loops(piece_text)
                issues.append(_issue("repetition", start, piece_duration, "collapsed", removed))
            texts.append(piece_text)
        if not issues:
            issues.append(_issue("repetition", offset, duration, "redecoded"))
        return (lang, join_chunk_texts(texts)), issues

    def _decode_pieces(self, pieces, forced, context) -> list[str]:
        from qwen_asr.inference.utils import parse_asr_output

        with token_budget(self.model, max(p[2] for p in pieces)):
            raw = self.model._infer_asr([context or ""] * len(pieces), [p[0] for p in pieces],
                                        [forced] * len(pieces))
        texts = [parse_asr_output(out, user_language=forced)[1] for out in raw]
        # Short, quiet pieces are where the model most often recites the context: redo those without it.
        echoed = [k for k, text in enumerate(texts) if context and strip_context_echo(text, context)[1]]
        if echoed:
            again = self._decode_pieces([pieces[k] for k in echoed], forced, "")
            for k, text in zip(echoed, again):
                texts[k] = text
        return texts

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


def merge_issues(issues: list[dict[str, Any]], gap: float = 1.0) -> list[dict[str, Any]]:
    """Join consecutive ranges with the same problem (a long pause spans many chunks)."""
    merged: list[dict[str, Any]] = []
    for issue in sorted(issues, key=lambda i: i.get("start", 0.0)):
        last = merged[-1] if merged else None
        if (last is not None and last.get("kind") == issue.get("kind") and last.get("action") == issue.get("action")
                and issue.get("start", 0.0) - last.get("end", 0.0) <= gap):
            last["end"] = max(last["end"], issue["end"])
            last["words_removed"] = last.get("words_removed", 0) + issue.get("words_removed", 0)
        else:
            merged.append(dict(issue))
    return merged


def _issue(kind: str, start: float, duration: float, action: str, words_removed: int = 0) -> dict[str, Any]:
    """A time range worth checking by hand.

    kind: repetition | sparse | context; action: redecoded | collapsed | kept.
    """
    return {"kind": kind, "start": round(start, 2), "end": round(start + duration, 2), "action": action,
            "words_removed": words_removed}
