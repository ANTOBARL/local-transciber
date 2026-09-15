"""Speaker assignment and renaming. Pure functions on the canonical Transcript."""

from __future__ import annotations

import bisect
from collections.abc import Sequence

from scriba.core.aligner import build_segments
from scriba.models import SpeakerTurn, Transcript, Word


def _speaker_for(start: float, end: float, turns: Sequence[SpeakerTurn], starts: list[float]) -> str | None:
    """Speaker with the largest temporal overlap; nearest turn if there is no overlap."""
    if not turns:
        return None
    # Turns may overlap, so scan a window of candidates around the word.
    hi = bisect.bisect_right(starts, end)
    best, best_overlap = None, 0.0
    for t in turns[max(0, hi - 50):hi]:
        overlap = min(end, t.end) - max(start, t.start)
        if overlap > best_overlap:
            best, best_overlap = t.speaker, overlap
    if best is not None:
        return best
    mid = (start + end) / 2
    nearest = min(turns, key=lambda t: 0.0 if t.start <= mid <= t.end else min(abs(mid - t.start), abs(mid - t.end)))
    return nearest.speaker


def assign_speakers(
    transcript: Transcript,
    speaker_turns: Sequence[SpeakerTurn],
    max_seconds: float = 12.0,
    max_chars: int = 160,
    pause_split: float = 0.8,
) -> Transcript:
    """Return a new Transcript with a speaker on every word and segments rebuilt."""
    turns = sorted(speaker_turns, key=lambda t: t.start)
    starts = [t.start for t in turns]
    result = transcript.model_copy(deep=True)

    if not result.has_timestamps:
        # Without word timings we can only label whole segments.
        for seg in result.segments:
            seg.speaker = _speaker_for(seg.start, seg.end, turns, starts)
    else:
        words: list[Word] = [w for seg in result.segments for w in seg.words]
        for w in words:
            w.speaker = _speaker_for(w.start, max(w.end, w.start + 0.01), turns, starts)
        result.segments = build_segments(words, max_seconds=max_seconds, max_chars=max_chars, pause_split=pause_split)

    seen: list[str] = []
    for seg in result.segments:
        if seg.speaker and seg.speaker not in seen:
            seen.append(seg.speaker)
    result.speakers = sorted(seen)
    return result


def rename_speakers(transcript: Transcript, names: dict[str, str]) -> Transcript:
    """Store display names (SPEAKER_00 → Luca). Raw labels stay untouched in the data."""
    result = transcript.model_copy(deep=True)
    cleaned = {k: v.strip() for k, v in names.items() if k in result.speakers and v and v.strip() and v.strip() != k}
    result.speaker_names = cleaned
    return result
