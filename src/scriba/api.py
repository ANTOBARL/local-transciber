"""HTTP API + static React frontend (`scriba serve`).

Same guarantees as the Gradio UI: one model instance, one transcription at a time,
further jobs wait in a FIFO queue. No ASR logic here: everything goes through TranscriptionService.
"""

from __future__ import annotations

import queue
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scriba import APP_NAME, __version__
from scriba.config import ScribaSettings
from scriba.errors import ScribaError
from scriba.i18n import LANGUAGE_NAMES_IT, TEXTS, UI_LANGUAGES
from scriba.languages import SUPPORTED_LANGUAGES
from scriba.models import JobStatus
from scriba.utils.logging import get_logger
from scriba.utils.paths import MEDIA_EXTENSIONS
from scriba.webform import BACKEND_CHOICES, DTYPE_CHOICES, EXPORT_FORMATS, TranscriptionForm

log = get_logger("api")

UPLOAD_DIR = Path(tempfile.gettempdir()) / "scriba_uploads"
MAX_TASKS_KEPT = 50


@dataclass
class Task:
    id: str
    upload_path: Path
    filename: str
    settings: ScribaSettings
    state: str = "queued"  # queued | running | done | error
    status: str | None = None  # JobStatus value
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: Any = None


class StartTask(BaseModel):
    upload_id: str
    form: TranscriptionForm


class SpeakerNames(BaseModel):
    names: dict[str, str]


class TaskManager:
    def __init__(self, service: Any):
        self.service = service
        self.tasks: dict[str, Task] = {}
        self.uploads: dict[str, dict[str, Any]] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()  # serializes GPU work (jobs, unload, renames)
        threading.Thread(target=self._worker, daemon=True, name="scriba-worker").start()

    def queue_position(self, task_id: str) -> int:
        queued = [t.id for t in sorted(self.tasks.values(), key=lambda t: t.created) if t.state == "queued"]
        return queued.index(task_id) + 1 if task_id in queued else 0

    def submit(self, task: Task) -> None:
        self.tasks[task.id] = task
        self._prune()
        self._queue.put(task.id)

    def _prune(self) -> None:
        finished = sorted((t for t in self.tasks.values() if t.state in ("done", "error")), key=lambda t: t.created)
        for old in finished[:-MAX_TASKS_KEPT]:
            self.tasks.pop(old.id, None)

    def _worker(self) -> None:
        while True:
            task = self.tasks.get(self._queue.get())
            if task is None:
                continue
            with self._lock:
                task.state, task.started = "running", time.time()

                def progress(status: JobStatus, _message: str) -> None:
                    task.status = status.value

                try:
                    task.result = self.service.transcribe(task.upload_path, task.settings, progress=progress)
                    task.state = "done"
                except Exception as exc:
                    task.error = exc.user_message() if isinstance(exc, ScribaError) else f"{type(exc).__name__}: {exc}"
                    task.state = "error"
                    log.error("Task %s failed: %s", task.id, task.error)
                finally:
                    task.finished = time.time()

    def with_gpu_lock(self, fn, *args, **kwargs):
        with self._lock:
            return fn(*args, **kwargs)


def _file_entries(task_id: str, files: dict[str, Path]) -> list[dict[str, str]]:
    return [{"format": fmt, "name": p.name, "url": f"/api/tasks/{task_id}/files/{p.name}"} for fmt, p in files.items()]


def create_app(base: ScribaSettings, service: Any = None, frontend_dir: Path | None = None,
               default_lang: str = "it") -> FastAPI:
    from scriba.core.pipeline import TranscriptionService, apply_speaker_names
    from scriba.exporters import EXPORTERS, load_transcript
    from scriba.ui import render_preview

    service = service or TranscriptionService()
    manager = TaskManager(service)
    app = FastAPI(title=f"{APP_NAME} API", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.manager = manager

    @app.exception_handler(ScribaError)
    async def _scriba_error(_, exc: ScribaError):
        return JSONResponse(status_code=400, content={"detail": exc.user_message()})

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        info = service.engine.info
        return {"status": "ok", "version": __version__, "model_loaded": service.engine.is_loaded(),
                "device": info.device_name if info else None, "backend": info.backend if info else None}

    @app.get("/api/config")
    def config():
        return {
            "app": APP_NAME, "version": __version__, "default_lang": default_lang,
            "defaults": TranscriptionForm.from_settings(base).model_dump(),
            "languages": SUPPORTED_LANGUAGES, "export_formats": EXPORT_FORMATS,
            "dtypes": DTYPE_CHOICES, "backends": BACKEND_CHOICES,
            "media_extensions": sorted(MEDIA_EXTENSIONS),
        }

    @app.get("/api/i18n")
    def i18n():
        return {"ui_languages": UI_LANGUAGES, "texts": TEXTS, "language_names": {"it": LANGUAGE_NAMES_IT, "en": {}}}

    # ------------------------------------------------------------------ uploads
    @app.post("/api/uploads")
    def upload(file: UploadFile = File(...)):
        from scriba.audio.probe import probe_audio

        name = Path(file.filename or "audio").name
        if Path(name).suffix.lower() not in MEDIA_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type: {Path(name).suffix}")
        upload_id = uuid.uuid4().hex
        target_dir = UPLOAD_DIR / upload_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / name
        with target.open("wb") as fh:
            shutil.copyfileobj(file.file, fh, length=8 * 1024 * 1024)
        try:
            info = probe_audio(target)
        except ScribaError:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        manager.uploads[upload_id] = {"path": target, "info": info}
        return {"upload_id": upload_id, "info": info.model_dump()}

    # ------------------------------------------------------------------ tasks
    @app.post("/api/tasks")
    def start_task(body: StartTask):
        up = manager.uploads.get(body.upload_id)
        if up is None:
            raise HTTPException(404, "Upload not found, upload the file again")
        settings = body.form.to_settings(base)
        task = Task(id=uuid.uuid4().hex, upload_path=up["path"], filename=up["path"].name, settings=settings)
        manager.submit(task)
        return {"task_id": task.id}

    def _task(task_id: str) -> Task:
        task = manager.tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
        return task

    @app.get("/api/tasks/{task_id}")
    def task_status(task_id: str):
        task = _task(task_id)
        now = task.finished or time.time()
        payload: dict[str, Any] = {
            "id": task.id, "state": task.state, "status": task.status, "error": task.error,
            "filename": task.filename, "queue_position": manager.queue_position(task.id),
            "elapsed": round(now - task.started, 1) if task.started else 0.0,
            "progress": None,
        }
        if task.state == "running" and task.status == JobStatus.TRANSCRIBING.value and service.progress is not None:
            payload["progress"] = service.progress.snapshot()
        r = task.result
        if task.state == "done" and r is not None:
            payload.update({
                "elapsed": round(r.processing_seconds, 1),
                "audio_duration": r.audio.duration,
                "rtf": round(r.rtf, 4) if r.rtf else None,
                "preview": render_preview(r.transcript),
                "files": _file_entries(task.id, r.files),
                "output_folder": str(r.job_dir),
                "warnings": r.warnings,
                "speakers": r.transcript.speakers,
                "speaker_names": r.transcript.speaker_names,
                "job_id": r.job_id,
            })
        return payload

    @app.get("/api/tasks/{task_id}/files/{name}")
    def task_file(task_id: str, name: str):
        task = _task(task_id)
        if task.result is None:
            raise HTTPException(404, "No output yet")
        allowed = {filename for filename, _ in EXPORTERS.values()}
        if name not in allowed:
            raise HTTPException(404, "File not found")
        path = task.result.job_dir / name
        if not path.is_file():
            raise HTTPException(404, "File not found")
        return FileResponse(path, filename=name)

    @app.post("/api/tasks/{task_id}/speakers")
    def rename(task_id: str, body: SpeakerNames):
        task = _task(task_id)
        if task.result is None:
            raise HTTPException(409, "Task has no transcript")
        files = manager.with_gpu_lock(apply_speaker_names, task.result.job_dir, body.names)
        task.result.files = files
        task.result.transcript = load_transcript(task.result.job_dir / "transcript.json")
        return task_status(task_id)

    @app.post("/api/model/unload")
    def unload():
        manager.with_gpu_lock(service.unload)
        return {"model_loaded": False}

    # ------------------------------------------------------------------ frontend
    if frontend_dir and (frontend_dir / "index.html").is_file():
        index = frontend_dir / "index.html"
        app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            candidate = (frontend_dir / path).resolve()
            if path and candidate.is_file() and frontend_dir.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(index)
    else:
        @app.get("/", include_in_schema=False)
        def no_frontend():
            return JSONResponse({"detail": "Frontend not built. API docs at /api/docs"})

    return app


def default_frontend_dir() -> Path | None:
    import os

    env = os.environ.get("SCRIBA_FRONTEND_DIR")
    candidates = [Path(env)] if env else []
    candidates.append(Path(__file__).resolve().parents[2] / "frontend" / "dist")
    return next((c for c in candidates if (c / "index.html").is_file()), None)


def serve(settings: ScribaSettings, host: str = "127.0.0.1", port: int = 8000, preload: bool = False,
          frontend_dir: Path | None = None, lang: str = "it") -> None:
    import uvicorn

    from scriba.utils.logging import setup_logging

    setup_logging(settings.app.log_level)
    frontend_dir = frontend_dir or default_frontend_dir()
    app = create_app(settings, frontend_dir=frontend_dir, default_lang=lang)
    if preload:
        log.info("Preloading model...")
        app.state.manager.with_gpu_lock(app.state.manager.service.engine.ensure_loaded, settings.asr)
    log.info("%s on http://%s:%s (frontend: %s)", APP_NAME, "localhost" if host in ("127.0.0.1", "0.0.0.0") else host,
             port, frontend_dir or "not built")
    uvicorn.run(app, host=host, port=port, log_level="warning")
