"""Internal, framework-independent data model.

Nothing here depends on qwen-asr, vLLM or pyannote: if upstream APIs change,
the canonical transcript format stays stable.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class Word(BaseModel):
    text: str
    start: float
    end: float
    speaker: str | None = None


class Segment(BaseModel):
    text: str
    start: float
    end: float
    speaker: str | None = None
    words: list[Word] = Field(default_factory=list)


class SpeakerTurn(BaseModel):
    start: float
    end: float
    speaker: str


class Transcript(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str
    source_file: str
    language: str
    duration: float
    model: str
    created_at: datetime
    text: str
    has_timestamps: bool = False
    completed: bool = True                  # False when the user stopped the transcription
    processed_seconds: float | None = None  # audio covered by a partial transcript
    segments: list[Segment] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    speaker_names: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def speaker_label(self, speaker: str | None) -> str | None:
        if speaker is None:
            return None
        return self.speaker_names.get(speaker, speaker)


class JobStatus(str, Enum):
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    DIARIZING = "diarizing"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"   # stopped by the user; partial results were exported
    FAILED = "failed"


class StepStatus(str, Enum):
    SKIPPED = "skipped"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class JobRecord(BaseModel):
    id: str
    status: JobStatus = JobStatus.PENDING
    source: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_audio_seconds: float | None = None
    duration_processing_seconds: float | None = None
    real_time_factor: float | None = None
    model: str
    aligner_model: str | None = None
    backend: str | None = None
    device: str | None = None
    steps: dict[str, StepStatus] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
