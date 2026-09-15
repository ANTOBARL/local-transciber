from __future__ import annotations

from pathlib import Path

from scriba.models import Transcript
from scriba.utils.time import format_vtt_time


def render_vtt(transcript: Transcript) -> str:
    lines = ["WEBVTT", ""]
    for seg in transcript.segments:
        end = max(seg.end, seg.start + 0.2)
        label = transcript.speaker_label(seg.speaker)
        text = f"<v {label}>{seg.text}" if label else seg.text
        lines += [f"{format_vtt_time(seg.start)} --> {format_vtt_time(end)}", text, ""]
    return "\n".join(lines)


def export_vtt(transcript: Transcript, path: Path) -> Path:
    path.write_text(render_vtt(transcript), encoding="utf-8")
    return path
