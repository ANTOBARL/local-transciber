"""Exporters. JSON is canonical; every other format is rendered from a Transcript."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from scriba.exporters.docx_exporter import export_docx
from scriba.exporters.json_exporter import export_json, load_transcript
from scriba.exporters.markdown_exporter import export_markdown
from scriba.exporters.srt_exporter import export_srt
from scriba.exporters.txt_exporter import export_txt
from scriba.exporters.vtt_exporter import export_vtt
from scriba.models import Transcript

Exporter = Callable[[Transcript, Path], Path]

EXPORTERS: dict[str, tuple[str, Exporter]] = {
    "json": ("transcript.json", export_json),
    "txt": ("transcript.txt", export_txt),
    "markdown": ("transcript.md", export_markdown),
    "srt": ("transcript.srt", export_srt),
    "vtt": ("transcript.vtt", export_vtt),
    "docx": ("transcript.docx", export_docx),
}

TIMED_FORMATS = {"srt", "vtt"}


def export_all(transcript: Transcript, out_dir: Path, formats: list[str]) -> tuple[dict[str, Path], list[str]]:
    """Write the requested formats. Returns (written files, warnings)."""
    written: dict[str, Path] = {}
    warnings: list[str] = []
    # JSON is always written: it is the canonical source for re-exports and speaker renames.
    ordered = ["json"] + [f for f in formats if f != "json"]
    for fmt in ordered:
        if fmt not in EXPORTERS:
            warnings.append(f"Unknown export format '{fmt}' ignored")
            continue
        if fmt in TIMED_FORMATS and not transcript.has_timestamps:
            warnings.append(f"{fmt.upper()} skipped: transcript has no timestamps")
            continue
        filename, exporter = EXPORTERS[fmt]
        written[fmt] = exporter(transcript, Path(out_dir) / filename)
    return written, warnings


__all__ = ["EXPORTERS", "export_all", "load_transcript"]
