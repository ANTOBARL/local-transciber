"""Optional speaker diarization (pyannote.audio).

The diarizer only produces `SpeakerTurn`s; it never touches the Transcript.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

from scriba.config import DiarizationSettings
from scriba.errors import DiarizationError
from scriba.models import SpeakerTurn
from scriba.utils import device as dev
from scriba.utils.logging import get_logger

log = get_logger("diarizer")


class PyannoteDiarizer:
    def __init__(self) -> None:
        self._pipeline: Any = None
        self._signature: tuple | None = None
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self, settings: DiarizationSettings) -> None:
        signature = (settings.model, settings.device)
        if self._pipeline is not None and self._signature == signature:
            return
        # Local-first: pyannote 4 ships opt-in telemetry; force it off before import.
        os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationError("pyannote.audio is not installed") from exc

        token = settings.hf_token.get_secret_value() if settings.hf_token else None
        log.info("Loading diarization pipeline %s", settings.model)
        try:
            try:
                pipeline = Pipeline.from_pretrained(settings.model, token=token)
            except TypeError:  # pyannote < 4
                pipeline = Pipeline.from_pretrained(settings.model, use_auth_token=token)
        except Exception as exc:
            raise DiarizationError(f"Cannot load diarization model {settings.model}: {exc}") from exc
        if pipeline is None:
            raise DiarizationError(f"Diarization model {settings.model} could not be loaded (gated? missing token?)")

        import torch

        device = dev.resolve_device(settings.device)
        pipeline.to(torch.device(device))
        self._pipeline = pipeline
        self._signature = signature

    def unload(self) -> None:
        self._pipeline = None
        self._signature = None

    def diarize(self, audio: np.ndarray, sample_rate: int, settings: DiarizationSettings) -> list[SpeakerTurn]:
        with self._lock:
            self.load(settings)
            import torch

            waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
            kwargs = {
                k: v
                for k, v in {
                    "num_speakers": settings.num_speakers,
                    "min_speakers": settings.min_speakers,
                    "max_speakers": settings.max_speakers,
                }.items()
                if v is not None
            }
            if "num_speakers" in kwargs:
                kwargs.pop("min_speakers", None)
                kwargs.pop("max_speakers", None)
            try:
                output = self._pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
            except Exception as exc:
                raise DiarizationError(f"Diarization failed: {exc}") from exc

        # pyannote 4: the "exclusive" variant has no overlapping turns, ideal for word assignment.
        annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(
            output, "speaker_diarization", output
        )
        turns = [
            SpeakerTurn(start=round(seg.start, 3), end=round(seg.end, 3), speaker=str(label))
            for seg, _, label in annotation.itertracks(yield_label=True)
        ]
        return normalize_speaker_labels(turns)


def normalize_speaker_labels(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Relabel speakers as SPEAKER_00, SPEAKER_01... in order of first appearance."""
    mapping: dict[str, str] = {}
    out = []
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        if t.speaker not in mapping:
            mapping[t.speaker] = f"SPEAKER_{len(mapping):02d}"
        out.append(SpeakerTurn(start=t.start, end=t.end, speaker=mapping[t.speaker]))
    return out
