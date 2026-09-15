from __future__ import annotations

from pathlib import Path

from scriba.exporters.render import interruption_note
from scriba.models import Transcript
from scriba.utils.time import format_srt_time

NOTE_CUE_SECONDS = 5.0


def cue_text(transcript: Transcript, seg) -> str:
    label = transcript.speaker_label(seg.speaker)
    return f"[{label}] {seg.text}" if label else seg.text


def cues(transcript: Transcript) -> list[tuple[float, float, str]]:
    """(start, end, text) for every subtitle, including the interruption marker."""
    out = [(seg.start, max(seg.end, seg.start + 0.2), cue_text(transcript, seg)) for seg in transcript.segments]
    note = interruption_note(transcript)
    if note:
        start = max([transcript.processed_seconds or 0.0] + [end for _, end, _ in out])
        out.append((start, start + NOTE_CUE_SECONDS, f"[{note}]"))
    return out


def render_srt(transcript: Transcript) -> str:
    return "\n".join(
        f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n"
        for i, (start, end, text) in enumerate(cues(transcript), start=1)
    )


def export_srt(transcript: Transcript, path: Path) -> Path:
    path.write_text(render_srt(transcript), encoding="utf-8")
    return path
