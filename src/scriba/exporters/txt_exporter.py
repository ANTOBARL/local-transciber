from __future__ import annotations

from pathlib import Path

from scriba.exporters.render import speaker_blocks
from scriba.models import Transcript
from scriba.utils.time import format_clock


def render_txt(transcript: Transcript) -> str:
    if not transcript.has_timestamps and not transcript.speakers:
        return transcript.text.strip() + "\n"
    parts = []
    for block in speaker_blocks(transcript):
        header = f"[{format_clock(block.start)}]"
        if block.speaker:
            header += f" {block.speaker}"
        parts.append(f"{header}\n{block.text}")
    return "\n\n".join(parts) + "\n"


def export_txt(transcript: Transcript, path: Path) -> Path:
    path.write_text(render_txt(transcript), encoding="utf-8")
    return path
