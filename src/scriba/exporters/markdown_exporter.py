from __future__ import annotations

from pathlib import Path

from scriba.exporters.render import interruption_note, speaker_blocks
from scriba.models import Transcript
from scriba.utils.time import format_clock


def render_markdown(transcript: Transcript) -> str:
    lines = [
        f"# {Path(transcript.source_file).name}",
        "",
        f"- **Language:** {transcript.language}",
        f"- **Duration:** {format_clock(transcript.duration)}",
        f"- **Model:** {transcript.model}",
        f"- **Created:** {transcript.created_at:%Y-%m-%d %H:%M} UTC",
    ]
    if transcript.speakers:
        names = ", ".join(transcript.speaker_label(s) or s for s in transcript.speakers)
        lines.append(f"- **Speakers:** {names}")
    note = interruption_note(transcript)
    if note:
        lines += ["", f"> **⚠ {note}**"]
    lines += ["", "---", ""]

    if not transcript.has_timestamps and not transcript.speakers:
        lines.append(transcript.text.strip())
    else:
        for block in speaker_blocks(transcript):
            header = f"`{block.clock}`"
            if block.speaker:
                header += f" **{block.speaker}**"
            lines += [header, "", block.text, ""]
    if note:
        lines += ["", "---", "", f"> **⚠ {note}**"]
    return "\n".join(lines).rstrip() + "\n"


def export_markdown(transcript: Transcript, path: Path) -> Path:
    path.write_text(render_markdown(transcript), encoding="utf-8")
    return path
