from __future__ import annotations

from pathlib import Path

from scriba.models import Transcript
from scriba.utils.time import format_srt_time


def cue_text(transcript: Transcript, seg) -> str:
    label = transcript.speaker_label(seg.speaker)
    return f"[{label}] {seg.text}" if label else seg.text


def render_srt(transcript: Transcript) -> str:
    cues = []
    for i, seg in enumerate(transcript.segments, start=1):
        end = max(seg.end, seg.start + 0.2)
        cues.append(f"{i}\n{format_srt_time(seg.start)} --> {format_srt_time(end)}\n{cue_text(transcript, seg)}\n")
    return "\n".join(cues)


def export_srt(transcript: Transcript, path: Path) -> Path:
    path.write_text(render_srt(transcript), encoding="utf-8")
    return path
