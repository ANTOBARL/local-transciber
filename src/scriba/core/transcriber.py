"""Convert a raw engine result into the canonical `Transcript`."""

from __future__ import annotations

from scriba.config import ExportSettings
from scriba.core.aligner import align_words, build_segments
from scriba.core.engine import RawASRResult
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
) -> Transcript:
    text = raw.text.strip()
    language = forced_language or raw.language or "unknown"

    if raw.tokens is not None and text:
        words = align_words(text, raw.tokens)
        segments = build_segments(
            words,
            max_seconds=export.max_segment_seconds,
            max_chars=export.max_segment_chars,
            pause_split=export.pause_split_seconds,
        )
        has_timestamps = bool(words)
    else:
        segments = [Segment(text=text, start=0.0, end=duration)] if text else []
        has_timestamps = False

    return Transcript(
        id=job_id,
        source_file=source_file,
        language=language,
        duration=duration,
        model=model,
        created_at=now_utc(),
        text=text,
        has_timestamps=has_timestamps,
        segments=segments,
        metadata=metadata or {},
    )
