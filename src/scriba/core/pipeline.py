"""TranscriptionService: the single entry point used by both CLI and UI.

audio → preprocess → Qwen3-ASR (+ forced aligner) → optional diarization → exports
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scriba import __version__
from scriba.audio.preprocess import load_waveform, to_mono, write_wav
from scriba.audio.probe import AudioInfo, probe_audio
from scriba.config import ScribaSettings
from scriba.core.diarizer import PyannoteDiarizer, local_model_dir
from scriba.core.engine import CancelToken, QwenASREngine
from scriba.core.checkpoint import SegmentStore
from scriba.core.checkpoint import fingerprint as checkpoint_fingerprint
from scriba.core.jobs import Job, write_json
from scriba.core.progress import ThroughputHistory, TranscriptionProgress, real_speed_key
from scriba.core.speakers import assign_speakers, rename_speakers
from scriba.core.transcriber import to_transcript
from scriba.errors import DiarizationError, ScribaError
from scriba.exporters import export_all, load_transcript
from scriba.models import JobStatus, StepStatus, Transcript
from scriba.utils.device import runtime_versions
from scriba.utils.logging import get_logger, job_log_file
from scriba.utils.paths import sha256_file
from scriba.utils.time import now_utc

log = get_logger("pipeline")

ProgressCallback = Callable[[JobStatus, str], None]

STATUS_MESSAGES = {
    JobStatus.PREPROCESSING: "Preparing audio...",
    JobStatus.LOADING_MODEL: "Loading Qwen3-ASR...",
    JobStatus.TRANSCRIBING: "Transcribing...",
    JobStatus.ALIGNING: "Generating timestamps...",
    JobStatus.DIARIZING: "Diarizing...",
    JobStatus.EXPORTING: "Exporting...",
    JobStatus.COMPLETED: "Done.",
    JobStatus.CANCELLED: "Stopped by the user: partial transcript saved.",
    JobStatus.FAILED: "Failed.",
}


@dataclass
class JobResult:
    job_id: str
    job_dir: Path
    transcript: Transcript
    files: dict[str, Path]
    audio: AudioInfo
    processing_seconds: float
    warnings: list[str] = field(default_factory=list)

    @property
    def rtf(self) -> float | None:
        return self.processing_seconds / self.audio.duration if self.audio.duration else None


class TranscriptionService:
    """Owns the model instances. One job at a time (callers must serialize)."""

    def __init__(self) -> None:
        self.engine = QwenASREngine()
        self.diarizer = PyannoteDiarizer()
        self.history = ThroughputHistory()
        # Progress of the transcription currently running (or the last one); read by the UIs.
        self.progress: TranscriptionProgress | None = None
        self._cancel: CancelToken | None = None

    def request_cancel(self) -> bool:
        """Stop the running transcription after the current batch and export what was processed."""
        token = self._cancel
        if token is None or token.cancelled:
            return False
        log.info("Stop requested: finishing the current batch")
        token.cancel()
        return True

    def unload(self) -> None:
        self.engine.unload()
        self.diarizer.unload()

    def transcribe(
        self,
        audio_path: str | Path,
        settings: ScribaSettings,
        progress: ProgressCallback | None = None,
    ) -> JobResult:
        audio_path = Path(audio_path)
        started = time.perf_counter()
        self._cancel = CancelToken()
        info = probe_audio(audio_path)  # fail fast on invalid input, before creating a job dir
        source_sha = sha256_file(audio_path)
        fp = checkpoint_fingerprint(sha256=source_sha, settings=settings)
        resume_dir = (SegmentStore.find_resumable(Path(settings.app.output_root), audio_path, fp)
                      if settings.app.create_job_subfolder else None)

        job = Job(
            settings.app.output_root,
            source=audio_path,
            model=settings.asr.model,
            subfolder=settings.app.create_job_subfolder,
            overwrite=settings.app.overwrite,
            existing_dir=resume_dir,
        )
        store = SegmentStore(job.dir, fp)
        warnings: list[str] = []

        def set_status(status: JobStatus, detail: str | None = None) -> None:
            job.set_status(status)
            message = detail or STATUS_MESSAGES.get(status, status.value)
            log.info(message)
            if progress:
                progress(status, message)

        with job_log_file(job.log_path):
            log.info("Scriba %s — job %s", __version__, job.id)
            log.info("Source: %s (%.1fs, %s, %s Hz, %s ch)", info.filename, info.duration, info.codec,
                     info.sample_rate, info.channels)
            try:
                result = self._run(job, audio_path, info, settings, set_status, warnings, store, source_sha)
            except Exception as exc:
                message = exc.user_message() if isinstance(exc, ScribaError) else f"{type(exc).__name__}: {exc}"
                log.error("Job failed: %s", message, exc_info=not isinstance(exc, ScribaError))
                job.record.error = message
                job.record.completed_at = now_utc()
                job.record.duration_processing_seconds = round(time.perf_counter() - started, 2)
                job.set_status(JobStatus.FAILED)
                if progress:
                    progress(JobStatus.FAILED, message)
                raise

        result.processing_seconds = time.perf_counter() - started
        job.record.completed_at = now_utc()
        job.record.duration_processing_seconds = round(result.processing_seconds, 2)
        job.record.real_time_factor = round(result.rtf, 4) if result.rtf else None
        job.record.warnings = warnings
        if result.transcript.completed:
            store.finish()
        # Give the temporary activation memory back to the GPU: only the model weights stay resident,
        # so other applications can use the VRAM while Scriba is idle.
        self.engine.release_cached_memory()
        self._cancel = None
        set_status(JobStatus.COMPLETED if result.transcript.completed else JobStatus.CANCELLED)
        return result

    # ------------------------------------------------------------------ internals
    def _run(self, job: Job, audio_path: Path, info: AudioInfo, settings: ScribaSettings,
             set_status: Callable, warnings: list[str], store: "SegmentStore", source_sha: str) -> JobResult:
        asr = settings.asr
        job.record.duration_audio_seconds = info.duration

        write_json(job.path("config.json"), settings.to_public_dict())
        write_json(job.path("source.json"), {
            **info.model_dump(),
            "sha256": source_sha,
        })

        # 1. Preprocess ------------------------------------------------------------
        set_status(JobStatus.PREPROCESSING)
        waveform: np.ndarray | None = None
        sr = settings.audio.sample_rate
        if settings.audio.normalize or settings.diarization.enabled:
            waveform = load_waveform(audio_path, sr, settings.audio.channels, settings.audio.ffmpeg_path)
            waveform = to_mono(waveform)  # the model consumes mono
            # Container metadata can be wrong (raw ADTS AAC, MKV): trust decoded samples.
            decoded = round(len(waveform) / sr, 3)
            if abs(decoded - info.duration) > 0.25:
                log.info("Duration from metadata %.2fs, decoded %.2fs: using decoded", info.duration, decoded)
            info.duration = decoded
            job.record.duration_audio_seconds = decoded
            if settings.app.keep_normalized_audio:
                write_wav(job.path("normalized.wav"), waveform, sr)
        asr_input = (waveform, sr) if settings.audio.normalize and waveform is not None else str(audio_path)
        job.step("preprocessing", StepStatus.COMPLETED)

        # 2. Model -----------------------------------------------------------------
        if asr.return_timestamps and not asr.forced_aligner.enabled:
            warnings.append("Timestamps requested but forced aligner disabled: exporting without timestamps")
        if not self.engine.is_loaded() or self.engine.needs_reload(asr):
            set_status(JobStatus.LOADING_MODEL)
        model_info = self.engine.ensure_loaded(asr)
        job.record.backend = model_info.backend
        job.record.device = model_info.device_name
        job.record.aligner_model = model_info.aligner_model

        # 3. ASR + alignment ---------------------------------------------------------
        timestamps = asr.timestamps_active
        history_key = "|".join(str(x) for x in (
            model_info.model, model_info.backend, model_info.device_name, model_info.dtype,
            f"ts={timestamps}", f"bs={asr.max_inference_batch_size}",
        ))
        tracker = TranscriptionProgress(info.duration, timestamps, prior_rate=self.history.get(history_key),
                                        prior_align_ratio=self.history.get(history_key, "align_ratio"))
        self.progress = tracker
        set_status(JobStatus.TRANSCRIBING,
                   "Transcribing and generating timestamps..." if timestamps else None)
        store.init()
        resumed = store.load()
        if resumed:
            log.info("Resuming job %s: %d chunks already processed", job.id, len(resumed))
            warnings.append("warn:resumed")
        raw = self.engine.transcribe(asr_input, language=asr.language, context=asr.context,
                                     return_timestamps=timestamps, align_batch=asr.align_batch_size or None,
                                     on_event=tracker.on_event, cancel=self._cancel, resume=resumed,
                                     save=store.save)
        for note in raw.notes:
            warnings.append(note)
        tracker.finish()
        transcribe_seconds = tracker.snapshot()["elapsed_seconds"]
        if raw.completed and not resumed:
            extra = {"speed": info.duration / max(transcribe_seconds, 0.001)}
            ratio = tracker.measured_align_ratio()
            if ratio is not None:
                extra["align_ratio"] = ratio
            self.history.record(history_key, transcribe_seconds, info.duration, **extra)
            # Real-world speed for this GPU + batch size, shown next to the optimizer benchmark.
            self.history.record(real_speed_key(model_info.device_name, asr.max_inference_batch_size, timestamps),
                                transcribe_seconds, info.duration, **extra)
            log.info("Transcription took %.1fs (%.3f s per audio second)", transcribe_seconds,
                     transcribe_seconds / max(info.duration, 0.001))
        if raw.completed:
            job.step("asr", StepStatus.COMPLETED)
            job.step("alignment", StepStatus.COMPLETED if timestamps else StepStatus.SKIPPED)
        else:
            covered = raw.processed_seconds or 0.0
            log.warning("Transcription stopped by the user at %.0fs of %.0fs", covered, info.duration)
            warnings.append(f"Transcription stopped by the user: {covered:.0f}s of {info.duration:.0f}s processed")
            job.step("asr", StepStatus.INTERRUPTED)
            job.step("alignment", StepStatus.INTERRUPTED if timestamps else StepStatus.SKIPPED)

        transcript = to_transcript(
            raw,
            job_id=job.id,
            source_file=str(audio_path),
            duration=info.duration,
            model=asr.model,
            forced_language=asr.language,
            export=settings.export,
            metadata=self._metadata(settings, model_info, audio_path, job),
        )
        if timestamps and not transcript.has_timestamps and transcript.text:
            warnings.append("Forced aligner returned no timestamps")

        # 4. Diarization (never fatal) ----------------------------------------------
        if settings.diarization.enabled and not transcript.completed:
            job.step("diarization", StepStatus.SKIPPED)
            warnings.append("warn:diarization_stopped")
        elif (settings.diarization.enabled and settings.diarization.hf_token is None
              and local_model_dir(settings.diarization.model) is None):
            # The pyannote model is gated: without a token the download is refused, don't even try.
            log.warning("Diarization skipped: no Hugging Face token configured")
            warnings.append("warn:diarization_no_token")
            job.step("diarization", StepStatus.SKIPPED)
        elif settings.diarization.enabled:
            set_status(JobStatus.DIARIZING)
            try:
                turns = self.diarizer.diarize(waveform, sr, settings.diarization)
                write_json(job.path("diarization.json"), [t.model_dump() for t in turns])
                transcript = assign_speakers(
                    transcript, turns,
                    max_seconds=settings.export.max_segment_seconds,
                    max_chars=settings.export.max_segment_chars,
                    pause_split=settings.export.pause_split_seconds,
                )
                write_json(job.path("speakers.json"), {s: s for s in transcript.speakers})
                job.step("diarization", StepStatus.COMPLETED)
            except (DiarizationError, Exception) as exc:
                msg = exc.user_message() if isinstance(exc, ScribaError) else str(exc)
                log.warning("Diarization failed, continuing without speakers: %s", msg)
                # Short, translatable message for the UI; full details stay in processing.log.
                warnings.append("warn:diarization_access" if "401" in msg or "gated" in msg.lower()
                                else "warn:diarization_failed")
                job.step("diarization", StepStatus.FAILED)
        else:
            job.step("diarization", StepStatus.SKIPPED)

        # 5. Export --------------------------------------------------------------------
        set_status(JobStatus.EXPORTING)
        files, export_warnings = export_all(transcript, job.dir, settings.export.enabled_formats())
        warnings.extend(export_warnings)
        for w in export_warnings:
            log.warning(w)
        job.record.outputs = {fmt: p.name for fmt, p in files.items()}
        job.step("export", StepStatus.COMPLETED)

        return JobResult(job_id=job.id, job_dir=job.dir, transcript=transcript, files=files,
                         audio=info, processing_seconds=0.0, warnings=warnings)

    @staticmethod
    def _metadata(settings: ScribaSettings, model_info, audio_path: Path, job: Job) -> dict:
        return {
            "scriba_version": __version__,
            "asr_model": model_info.model,
            "aligner_model": model_info.aligner_model,
            "backend": model_info.backend,
            "device": model_info.device_name,
            "dtype": model_info.dtype,
            "versions": runtime_versions(),
            "context": settings.asr.context,
            "language_requested": settings.asr.language,
            "input_filename": audio_path.name,
            "config": settings.to_public_dict(),
        }


def apply_speaker_names(job_dir: str | Path, names: dict[str, str], formats: list[str] | None = None) -> dict[str, Path]:
    """Rename speakers in an existing job and regenerate exports — no ASR re-run."""
    job_dir = Path(job_dir)
    transcript = load_transcript(job_dir / "transcript.json")
    transcript = rename_speakers(transcript, names)
    if formats is None:
        import json

        job_meta = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        formats = list(job_meta.get("outputs", {}).keys()) or ["json", "txt"]
    files, _ = export_all(transcript, job_dir, formats)
    write_json(job_dir / "speakers.json", {s: transcript.speaker_label(s) for s in transcript.speakers})
    return files
