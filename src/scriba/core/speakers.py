"""Speaker assignment and renaming. Pure functions on the canonical Transcript."""

from __future__ import annotations

import bisect
from collections.abc import Sequence

from scriba.core.aligner import _SENTENCE_END, build_segments
from scriba.models import SpeakerTurn, Transcript, Word

# Longer "sentences" usually lack punctuation and may really contain a speaker change.
MAX_SENTENCE_SECONDS = 20.0


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


def smooth_turns(turns: Sequence[SpeakerTurn], min_turn: float = 0.5, merge_gap: float = 1.0) -> list[SpeakerTurn]:
    """Drop diarization blips shorter than `min_turn` and merge what they used to split."""
    ordered = sorted(turns, key=lambda t: t.start)
    kept = [t for t in ordered if t.end - t.start >= min_turn] or ordered
    merged: list[SpeakerTurn] = []
    for t in kept:
        last = merged[-1] if merged else None
        if last is not None and last.speaker == t.speaker and t.start - last.end <= merge_gap:
            merged[-1] = SpeakerTurn(start=last.start, end=max(last.end, t.end), speaker=last.speaker)
        else:
            merged.append(t)
    return merged


def smooth_word_speakers(words: Sequence[Word], min_run: float = 1.0, majority: float = 0.7) -> None:
    """Remove implausible speaker flips, in place.

    1. A speaker change lasting less than `min_run` seconds goes to the speaker before it, unless it
       is made of whole sentences (a short "Yes." from someone else is real).
    2. When one speaker holds at least `majority` of a sentence's words, the whole sentence is theirs:
       diarization boundaries rarely match word boundaries, so first and last words tend to flip.
    """
    if not words:
        return
    if min_run > 0:
        ends = [bool(_SENTENCE_END.search(w.text)) for w in words]

        def whole(first: int, last: int) -> bool:
            return (first == 0 or ends[first - 1]) and ends[last]

        runs: list[list] = []  # [speaker, first index, last index]
        for i, w in enumerate(words):
            if runs and runs[-1][0] == w.speaker:
                runs[-1][2] = i
            else:
                runs.append([w.speaker, i, i])
        smoothed: list[list] = []
        for k, (speaker, first, last) in enumerate(runs):
            until = words[runs[k + 1][1]].start if k + 1 < len(runs) else words[last].end
            short = until - words[first].start < min_run and not whole(first, last)
            if smoothed and (short or smoothed[-1][0] == speaker):
                smoothed[-1][2] = last
            else:
                smoothed.append([speaker, first, last])
        if len(smoothed) > 1:
            span = words[smoothed[1][1]].start - words[0].start
            if span < min_run and not whole(0, smoothed[0][2]):
                smoothed[1][1] = 0
                smoothed.pop(0)
        for speaker, first, last in smoothed:
            for w in words[first:last + 1]:
                w.speaker = speaker

    if 0 < majority <= 1:
        start = 0
        for i, w in enumerate(words):
            if _SENTENCE_END.search(w.text) or i == len(words) - 1:
                sentence = words[start:i + 1]
                counts: dict[str | None, int] = {}
                for x in sentence:
                    counts[x.speaker] = counts.get(x.speaker, 0) + 1
                speaker, count = max(counts.items(), key=lambda kv: kv[1])
                short_enough = sentence[-1].end - sentence[0].start <= MAX_SENTENCE_SECONDS
                if len(counts) > 1 and short_enough and count / len(sentence) >= majority:
                    for x in sentence:
                        x.speaker = speaker
                start = i + 1


def assign_speakers(
    transcript: Transcript,
    speaker_turns: Sequence[SpeakerTurn],
    max_seconds: float = 12.0,
    max_chars: int = 160,
    pause_split: float = 0.8,
    min_turn: float = 0.0,
    min_run: float = 0.0,
    majority: float = 0.0,
) -> Transcript:
    """Return a new Transcript with a speaker on every word and segments rebuilt.

    The smoothing parameters default to off; the pipeline passes the configured values.
    """
    turns = smooth_turns(speaker_turns, min_turn) if min_turn > 0 else sorted(speaker_turns, key=lambda t: t.start)
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
        smooth_word_speakers(words, min_run, majority)
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
