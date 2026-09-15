from __future__ import annotations

from pathlib import Path

from scriba.exporters.render import interruption_note
from scriba.exporters.srt_exporter import NOTE_CUE_SECONDS
from scriba.models import Transcript
from scriba.utils.time import format_vtt_time


def render_vtt(transcript: Transcript) -> str:
    lines = ["WEBVTT", ""]
    last_end = 0.0
    for seg in transcript.segments:
        end = max(seg.end, seg.start + 0.2)
        last_end = max(last_end, end)
        label = transcript.speaker_label(seg.speaker)
        text = f"<v {label}>{seg.text}" if label else seg.text
        lines += [f"{format_vtt_time(seg.start)} --> {format_vtt_time(end)}", text, ""]
    note = interruption_note(transcript)
    if note:
        start = max(transcript.processed_seconds or 0.0, last_end)
        lines += [f"{format_vtt_time(start)} --> {format_vtt_time(start + NOTE_CUE_SECONDS)}", f"[{note}]", ""]
    return "\n".join(lines)


def export_vtt(transcript: Transcript, path: Path) -> Path:
    path.write_text(render_vtt(transcript), encoding="utf-8")
    return path
