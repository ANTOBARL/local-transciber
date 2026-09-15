"""On-disk segment checkpoints: finished windows survive crashes, stops and restarts.

Layout inside a job directory:
    segments/manifest.json      fingerprint of everything that affects the chunk results
    segments/000012.json        one file per processed chunk

A later run on the same file with the same fingerprint resumes in that directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scriba.core.jobs import write_json
from scriba.core.windowed import ChunkResult
from scriba.utils.logging import get_logger
from scriba.utils.paths import safe_name

log = get_logger("checkpoint")

SEGMENTS_DIR = "segments"
MANIFEST = "manifest.json"


def fingerprint(*, sha256: str, settings: Any) -> dict[str, Any]:
    """Only what changes chunk boundaries or chunk results; batch sizes are deliberately excluded."""
    asr = settings.asr
    return {
        "version": 1,
        "sha256": sha256,
        "model": asr.model,
        "aligner": asr.forced_aligner.model if asr.timestamps_active else None,
        "timestamps": asr.timestamps_active,
        "language": asr.language,
        "context": hashlib.sha256((asr.context or "").encode("utf-8")).hexdigest(),
        "max_new_tokens": asr.max_new_tokens,
        "normalize": settings.audio.normalize,
        "sample_rate": settings.audio.sample_rate,
    }


class SegmentStore:
    def __init__(self, job_dir: Path, fp: dict[str, Any]):
        self.dir = Path(job_dir) / SEGMENTS_DIR
        self.fp = fp

    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST

    def _manifest(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def init(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest()
        if manifest is None or manifest.get("fingerprint") != self.fp:
            for stale in self.dir.glob("[0-9]*.json"):
                stale.unlink()
            write_json(self.manifest_path, {"fingerprint": self.fp, "completed": False})

    def load(self) -> dict[int, ChunkResult]:
        manifest = self._manifest()
        if manifest is None or manifest.get("fingerprint") != self.fp:
            return {}
        done: dict[int, ChunkResult] = {}
        for path in sorted(self.dir.glob("[0-9]*.json")):
            try:
                chunk = ChunkResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError) as exc:
                log.warning("Ignoring unreadable checkpoint %s: %s", path.name, exc)
                continue
            done[chunk.index] = chunk
        return done

    def save(self, chunks: list[ChunkResult]) -> None:
        for chunk in chunks:
            write_json(self.dir / f"{chunk.index:06d}.json", chunk.to_dict())

    def finish(self) -> None:
        """The job completed: checkpoints are no longer needed."""
        for path in self.dir.glob("*.json"):
            path.unlink(missing_ok=True)
        try:
            self.dir.rmdir()
        except OSError:
            pass

    @staticmethod
    def find_resumable(output_root: Path, source: Path, fp: dict[str, Any]) -> Path | None:
        """Most recent unfinished job for the same file and fingerprint, if any."""
        root = Path(output_root)
        if not root.is_dir():
            return None
        candidates = sorted(root.glob(f"{safe_name(source.stem)}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for job_dir in candidates:
            manifest_path = job_dir / SEGMENTS_DIR / MANIFEST
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if manifest.get("fingerprint") == fp and not manifest.get("completed"):
                if any((job_dir / SEGMENTS_DIR).glob("[0-9]*.json")):
                    return job_dir
        return None
