from __future__ import annotations

from pathlib import Path

from scriba.models import Transcript


def export_json(transcript: Transcript, path: Path) -> Path:
    path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_transcript(path: str | Path) -> Transcript:
    return Transcript.model_validate_json(Path(path).read_text(encoding="utf-8"))
