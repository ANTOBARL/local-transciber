"""Convert a raw engine result into the canonical `Transcript`."""

from __future__ import annotations

from scriba.config import ExportSettings
from scriba.core.aligner import align_words, build_segments
from scriba.core.engine import RawASRResult
from scriba.core.vocabulary import Glossary
from scriba.models import Segment, Transcript
from scriba.utils.time import now_utc


def to_transcript(
    raw: RawASRResult,
    *,
    job_id: str,
    source_file: str,
    duration: float,
    model: str,
    forced_language: str | None,
    export: ExportSettings,
    metadata: dict | None = None,
    glossary: Glossary | None = None,
) -> Transcript:
    text = raw.text.strip()
    corrections: list[dict] = []
    language = forced_language or raw.language or "unknown"
    covered = duration if raw.completed else (raw.processed_seconds or 0.0)

    if raw.tokens is not None and text:
        words = align_words(text, raw.tokens)
        if glossary:
            words, corrections = glossary.correct_words(words)
            text = glossary.correct_text(text)
        segments = build_segments(
            words,
            max_seconds=export.max_segment_seconds,
            max_chars=export.max_segment_chars,
            pause_split=export.pause_split_seconds,
        )
        has_timestamps = bool(words)
    else:
        if glossary:
            text = glossary.correct_text(text)
        segments = [Segment(text=text, start=0.0, end=covered)] if text else []
        has_timestamps = False

    if raw.tail_text:
        # Transcribed before the stop but not yet aligned: keep it as one untimed block.
        start = raw.tail_start if raw.tail_start is not None else (segments[-1].end if segments else 0.0)
        segments.append(Segment(text=raw.tail_text, start=start, end=max(covered, start)))
        text = f"{text} {raw.tail_text}".strip()

    return Transcript(
        id=job_id,
        source_file=source_file,
        language=language,
        duration=duration,
        model=model,
        created_at=now_utc(),
        text=text,
        has_timestamps=has_timestamps,
        completed=raw.completed,
        processed_seconds=None if raw.completed else round(covered, 3),
        segments=segments,
        metadata={**(metadata or {}), **({"glossary_corrections": corrections} if corrections else {})},
    )
