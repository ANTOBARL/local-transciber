from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime
from pathlib import Path

from scriba.errors import OutputDirError

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".m4b", ".flac", ".aac", ".ogg", ".oga", ".opus", ".wma", ".webm",
    ".aiff", ".aif", ".amr", ".mka", ".ac3", ".caf",
}
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".3gp"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def safe_name(text: str, max_length: int = 80) -> str:
    """Filesystem-safe name: letters (any script), digits, dot, dash, underscore; spaces become '_'."""
    import re
    import unicodedata

    text = unicodedata.normalize("NFC", text).strip()
    text = re.sub(r"\s+", "_", text)
    text = "".join(ch for ch in text if ch.isalnum() or ch in "._-")
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:max_length].rstrip("._-") or "audio"


def new_job_id(when: datetime | None = None, source: str | Path | None = None) -> str:
    """<audio_name>_YYYYMMDD_HHMMSS, or YYYYMMDD_HHMMSS_<short_uuid> without a source."""
    when = when or datetime.now()
    if source is not None:
        return f"{safe_name(Path(source).stem)}_{when:%Y%m%d_%H%M%S}"
    return f"{when:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def ensure_writable_dir(path: Path) -> Path:
    path = Path(path).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".scriba_write_test_{secrets.token_hex(4)}"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        raise OutputDirError(f"Output directory not writable: {path} ({exc})") from exc
    return path.resolve()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8-sig").strip()
