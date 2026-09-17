"""Form model shared by the Gradio UI and the HTTP API (React frontend).

One place maps "what the user filled in" to `ScribaSettings`, so both frontends behave identically.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from scriba.config import ScribaSettings
from scriba.errors import ScribaError

EXPORT_FORMATS = ["json", "txt", "markdown", "srt", "vtt", "docx"]
DTYPE_CHOICES = ["auto", "bfloat16", "float16", "float32"]
BACKEND_CHOICES = ["auto", "transformers", "vllm"]


class TranscriptionForm(BaseModel):
    language: str | None = "auto"
    context: str = ""
    glossary: str = ""
    timestamps: bool = True
    diarize: bool = True
    num_speakers: int | None = 0
    min_speakers: int | None = 0
    max_speakers: int | None = 0
    formats: list[str] = Field(default_factory=lambda: ["json", "txt", "markdown", "srt"])

    output_dir: str = "./outputs"
    subfolder: bool = True
    keep_audio: bool = False
    hf_token: str = ""

    model: str = "Qwen/Qwen3-ASR-1.7B"
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    backend: str = "auto"
    device: str = "auto"
    dtype: str = "auto"
    backend_kwargs: str | dict[str, Any] = "{}"

    gpu_mem: float = 0.70
    batch: int = 32
    align_batch: int = 0
    max_tokens: int = 4096
    chunk_seconds: float | None = None  # None keeps the configured value
    sample_rate: int = 16000
    channels: int = 1
    normalize: bool = True

    @classmethod
    def from_settings(cls, s: ScribaSettings) -> "TranscriptionForm":
        return cls(
            language=s.asr.language or "auto", context=s.asr.context, glossary=s.asr.glossary,
            timestamps=s.asr.return_timestamps, chunk_seconds=s.asr.chunk_seconds,
            diarize=s.diarization.enabled, num_speakers=s.diarization.num_speakers or 0,
            min_speakers=s.diarization.min_speakers or 0, max_speakers=s.diarization.max_speakers or 0,
            formats=s.export.enabled_formats(), output_dir=str(s.app.output_root),
            subfolder=s.app.create_job_subfolder, keep_audio=s.app.keep_normalized_audio,
            model=s.asr.model, aligner_model=s.asr.forced_aligner.model, backend=s.asr.backend,
            device=s.asr.device, dtype=s.asr.dtype, backend_kwargs=json.dumps(s.asr.backend_kwargs, indent=2),
            gpu_mem=s.asr.gpu_memory_utilization, batch=s.asr.max_inference_batch_size,
            max_tokens=s.asr.max_new_tokens, align_batch=s.asr.align_batch_size, sample_rate=s.audio.sample_rate, channels=s.audio.channels,
            normalize=s.audio.normalize,
        )

    def to_settings(self, base: ScribaSettings) -> ScribaSettings:
        kwargs = self.backend_kwargs
        if isinstance(kwargs, str):
            try:
                kwargs = json.loads(kwargs or "{}")
            except json.JSONDecodeError as exc:
                raise ScribaError(f"Backend kwargs JSON: {exc}") from exc
        if not isinstance(kwargs, dict):
            raise ScribaError("Backend kwargs must be a JSON object")

        def opt_int(v: Any) -> int | None:
            return int(v) if v not in (None, "") and int(v) > 0 else None

        overrides: dict[str, Any] = {
            "app": {"output_root": self.output_dir or str(base.app.output_root),
                    "create_job_subfolder": self.subfolder, "keep_normalized_audio": self.keep_audio},
            "audio": {"normalize": self.normalize, "sample_rate": int(self.sample_rate), "channels": int(self.channels)},
            "asr": {
                "model": self.model.strip(), "backend": self.backend, "device": self.device.strip(),
                "dtype": self.dtype, "language": self.language, "context": self.context,
                "glossary": self.glossary,
                "gpu_memory_utilization": float(self.gpu_mem), "max_inference_batch_size": int(self.batch),
                "align_batch_size": int(self.align_batch),
                "max_new_tokens": int(self.max_tokens), "return_timestamps": self.timestamps,
                "forced_aligner": {"model": self.aligner_model.strip(), "dtype": self.dtype},
                "backend_kwargs": kwargs,
            },
            "diarization": {"enabled": self.diarize, "num_speakers": opt_int(self.num_speakers),
                            "min_speakers": opt_int(self.min_speakers), "max_speakers": opt_int(self.max_speakers)},
            "export": {fmt: fmt in self.formats for fmt in EXPORT_FORMATS},
        }
        if self.chunk_seconds:
            overrides["asr"]["chunk_seconds"] = float(self.chunk_seconds)
        if self.hf_token:
            overrides["diarization"]["hf_token"] = self.hf_token
        return base.with_overrides(overrides)
