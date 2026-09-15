"""Shared rendering helpers for human-readable formats."""

from __future__ import annotations

from dataclasses import dataclass

from scriba.core.aligner import join_words
from scriba.models import Transcript, Word


@dataclass
class Block:
    start: float
    end: float
    speaker: str | None
    text: str


def speaker_blocks(transcript: Transcript, max_block_seconds: float = 60.0) -> list[Block]:
    """Readable paragraphs: consecutive segments merged while the speaker stays the same.

    Without diarization, paragraphs are cut roughly every `max_block_seconds`.
    """
    blocks: list[Block] = []
    for seg in transcript.segments:
        label = transcript.speaker_label(seg.speaker)
        if (
            blocks
            and blocks[-1].speaker == label
            and (label is not None or seg.end - blocks[-1].start <= max_block_seconds)
        ):
            last = blocks[-1]
            last.text = join_words([Word(text=last.text, start=0, end=0), Word(text=seg.text, start=0, end=0)])
            last.end = seg.end
        else:
            blocks.append(Block(start=seg.start, end=seg.end, speaker=label, text=seg.text))
    return blocks
