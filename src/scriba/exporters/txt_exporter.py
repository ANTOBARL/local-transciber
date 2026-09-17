from __future__ import annotations

from pathlib import Path

from scriba.exporters.render import interruption_note, speaker_blocks
from scriba.models import Transcript
from scriba.utils.time import format_clock


def render_txt(transcript: Transcript) -> str:
    if not transcript.has_timestamps and not transcript.speakers:
        body = transcript.text.strip()
    else:
        parts = []
        for block in speaker_blocks(transcript):
            header = f"[{block.clock}]"
            if block.speaker:
                header += f" {block.speaker}"
            parts.append(f"{header}\n{block.text}")
        body = "\n\n".join(parts)

    note = interruption_note(transcript)
    if note:
        marker = f"*** {note} ***"
        body = f"{marker}\n\n{body}\n\n{marker}" if body else marker
    return body + "\n"


def export_txt(transcript: Transcript, path: Path) -> Path:
    path.write_text(render_txt(transcript), encoding="utf-8")
    return path
