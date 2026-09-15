"""Job directories and persisted job metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scriba.models import JobRecord, JobStatus, StepStatus
from scriba.utils.paths import ensure_writable_dir, new_job_id
from scriba.utils.time import now_utc


def write_json(path: Path, data: Any) -> Path:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


class Job:
    def __init__(self, output_root: Path, source: Path, model: str, subfolder: bool = True, overwrite: bool = False,
                 existing_dir: Path | None = None):
        root = ensure_writable_dir(output_root)
        if existing_dir is not None:  # resuming an unfinished job
            self.id, self.dir = existing_dir.name, ensure_writable_dir(existing_dir)
            self.record = JobRecord(id=self.id, source=str(source), created_at=now_utc(), model=model)
            self.save()
            return
        self.id = new_job_id(source=source)
        if subfolder:
            base_id, n = self.id, 2
            while (root / self.id).exists():  # two jobs for the same file within the same second
                self.id, n = f"{base_id}_{n}", n + 1
        self.dir = ensure_writable_dir(root / self.id) if subfolder else root
        if not subfolder and not overwrite and (self.dir / "transcript.json").exists():
            from scriba.errors import OutputDirError

            raise OutputDirError(
                f"{self.dir} already contains a transcript",
                hint="Enable app.create_job_subfolder or app.overwrite.",
            )
        self.record = JobRecord(id=self.id, source=str(source), created_at=now_utc(), model=model)
        self.save()

    @property
    def log_path(self) -> Path:
        return self.dir / "processing.log"

    def path(self, name: str) -> Path:
        return self.dir / name

    def save(self) -> None:
        write_json(self.path("job.json"), self.record.model_dump(mode="json"))

    def set_status(self, status: JobStatus) -> None:
        self.record.status = status
        if status not in (JobStatus.PENDING,) and self.record.started_at is None:
            self.record.started_at = now_utc()
        self.save()

    def step(self, name: str, status: StepStatus) -> None:
        self.record.steps[name] = status
        self.save()

    def warn(self, message: str) -> None:
        self.record.warnings.append(message)
        self.save()
