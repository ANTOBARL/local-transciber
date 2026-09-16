"""Qwen3-ASR model engine.

Loads the model once and keeps it resident between jobs. The concrete backend
(transformers or vLLM) is an implementation detail hidden behind `QwenASREngine`.
The only object leaving this module is `RawASRResult`, never a qwen-asr type.
"""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scriba.config import ASRSettings
from scriba.core.windowed import ChunkResult, EventCallback, WindowedRunner
from scriba.errors import (
    AlignmentError,
    BackendUnavailableError,
    DeviceUnavailableError,
    ModelLoadError,
    OutOfMemoryError,
    ScribaError,
)
from scriba.utils import device as dev
from scriba.utils.logging import get_logger

log = get_logger("engine")


@dataclass
class AlignedToken:
    text: str
    start: float
    end: float


@dataclass
class RawASRResult:
    language: str
    text: str
    tokens: list[AlignedToken] | None = None
    completed: bool = True
    processed_seconds: float | None = None  # audio covered when not completed
    tail_text: str | None = None
    tail_start: float | None = None
    notes: list[str] = field(default_factory=list)  # e.g. batch reduced after an out-of-memory
    asr_batch: int | None = None
    align_batch: int | None = None


@dataclass
class LoadedModelInfo:
    backend: str
    device: str
    device_name: str
    dtype: str
    model: str
    aligner_model: str | None
    aligner_device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def local_model_source(model: str) -> str:
    """The local copy of a Qwen model if one is downloaded, else the Hugging Face id."""
    from scriba.envfile import find_local_model

    local = find_local_model(model, marker="config.json")
    return str(local) if local else model


def resolve_backend(requested: str) -> str:
    ok, reason = dev.vllm_supported()
    if requested == "auto":
        return "vllm" if ok else "transformers"
    if requested == "vllm" and not ok:
        raise BackendUnavailableError(f"vLLM backend requested but unavailable: {reason}")
    return requested


class QwenASREngine:
    def __init__(self) -> None:
        self._model: Any = None
        self._signature: tuple | None = None
        self._info: LoadedModelInfo | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ lifecycle
    @staticmethod
    def _signature_for(asr: ASRSettings, with_aligner: bool) -> tuple:
        import json

        fa = asr.forced_aligner
        return (
            asr.model, asr.backend, asr.device, asr.dtype,
            asr.gpu_memory_utilization, asr.max_inference_batch_size, asr.max_new_tokens,
            json.dumps(asr.backend_kwargs, sort_keys=True, default=str),
            (fa.model, fa.dtype, fa.device) if with_aligner else None,
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def info(self) -> LoadedModelInfo | None:
        return self._info

    def needs_reload(self, asr: ASRSettings) -> bool:
        return self._signature != self._signature_for(asr, asr.timestamps_active)

    def ensure_loaded(self, asr: ASRSettings) -> LoadedModelInfo:
        with self._lock:
            if self.is_loaded() and not self.needs_reload(asr):
                return self._info  # type: ignore[return-value]
            if self.is_loaded():
                log.info("Model settings changed, reloading")
                self.unload()
            self.load(asr)
            return self._info  # type: ignore[return-value]

    def load(self, asr: ASRSettings) -> LoadedModelInfo:
        with self._lock:
            if self.is_loaded():
                return self._info  # type: ignore[return-value]
            with_aligner = asr.timestamps_active
            backend = resolve_backend(asr.backend)
            device = dev.resolve_device(asr.device)
            dtype_name = dev.resolve_dtype_name(asr.dtype, device)
            if backend == "vllm" and not device.startswith("cuda"):
                raise DeviceUnavailableError("vLLM backend requires a CUDA device")

            model_source = local_model_source(asr.model)
            aligner_source = local_model_source(asr.forced_aligner.model) if with_aligner else None

            aligner_kwargs = None
            aligner_device = None
            if with_aligner:
                aligner_device = dev.resolve_device(
                    asr.forced_aligner.device if asr.forced_aligner.device != "auto" else device
                )
                aligner_dtype = dev.resolve_dtype_name(asr.forced_aligner.dtype, aligner_device)
                aligner_kwargs = {"dtype": dev.torch_dtype(aligner_dtype), "device_map": aligner_device}

            log.info(
                "Loading %s%s (backend=%s, device=%s, dtype=%s, aligner=%s%s)",
                asr.model, " (local copy, offline)" if model_source != asr.model else "",
                backend, device, dtype_name,
                asr.forced_aligner.model if with_aligner else "disabled",
                " (local copy, offline)" if with_aligner and aligner_source != asr.forced_aligner.model else "",
            )
            try:
                from scriba.compat import ensure_nagisa_importable

                ensure_nagisa_importable()
                from qwen_asr import Qwen3ASRModel
            except ImportError as exc:
                raise ModelLoadError(f"qwen-asr is not importable: {exc}", hint="pip install qwen-asr") from exc

            common = dict(
                forced_aligner=aligner_source if with_aligner else None,
                forced_aligner_kwargs=aligner_kwargs,
                max_inference_batch_size=asr.max_inference_batch_size,
                max_new_tokens=asr.max_new_tokens,
            )
            try:
                if backend == "vllm":
                    kwargs = dict(gpu_memory_utilization=asr.gpu_memory_utilization, dtype=dtype_name)
                    kwargs.update(asr.backend_kwargs)
                    model = Qwen3ASRModel.LLM(model=model_source, **common, **kwargs)
                else:
                    kwargs = dict(dtype=dev.torch_dtype(dtype_name), device_map=device)
                    kwargs.update(asr.backend_kwargs)
                    if isinstance(kwargs.get("dtype"), str):
                        kwargs["dtype"] = dev.torch_dtype(kwargs["dtype"])
                    model = Qwen3ASRModel.from_pretrained(model_source, **common, **kwargs)
            except ScribaError:
                raise
            except Exception as exc:
                self._free_memory()
                if _is_oom(exc):
                    raise OutOfMemoryError(f"Out of memory while loading {asr.model}: {exc}") from exc
                if backend == "vllm":
                    raise ModelLoadError(f"vLLM initialization failed: {exc}") from exc
                raise ModelLoadError(f"Failed to load {asr.model}: {exc}") from exc

            if backend == "transformers" and device.startswith("cuda"):
                _cap_cuda_memory(device)
            self._model = model
            self._signature = self._signature_for(asr, with_aligner)
            self._info = LoadedModelInfo(
                backend=backend,
                device=device,
                device_name=dev.device_display_name(device),
                dtype=dtype_name,
                model=asr.model,
                aligner_model=asr.forced_aligner.model if with_aligner else None,
                aligner_device=aligner_device,
            )
            log.info("Model loaded on %s", self._info.device_name)
            return self._info

    def set_batch_size(self, asr: ASRSettings, batch: int) -> None:
        """Change the inference batch size of the loaded model without reloading it."""
        with self._lock:
            if self._model is None:
                raise ModelLoadError("Model is not loaded")
            self._model.max_inference_batch_size = int(batch)
            updated = asr.model_copy(update={"max_inference_batch_size": int(batch)})
            self._signature = self._signature_for(updated, asr.timestamps_active)

    def release_cached_memory(self) -> None:
        """Return PyTorch's cached (unused) VRAM to the driver, keeping the model loaded."""
        with self._lock:
            self._free_memory()

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            log.info("Unloading model")
            model, self._model = self._model, None
            self._signature = None
            self._info = None
            if getattr(model, "backend", None) == "vllm":
                _shutdown_vllm(model.model)
            del model
            self._free_memory()

    @staticmethod
    def _free_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------ inference
    def transcribe(
        self,
        audio: str | tuple[np.ndarray, int],
        language: str | None = None,
        context: str = "",
        return_timestamps: bool = True,
        *,
        align_batch: int | None = None,
        on_event: EventCallback | None = None,
        cancel: CancelToken | None = None,
        resume: dict[int, ChunkResult] | None = None,
        save: Callable[[list[ChunkResult]], None] | None = None,
    ) -> RawASRResult:
        """Transcribe one audio window by window (see scriba.core.windowed).

        `cancel` stops before the next window; `resume` skips chunks already processed; `save` is
        called after every window. A stopped run returns `completed=False` with all finished chunks.
        """
        from qwen_asr.inference.utils import SAMPLE_RATE, merge_languages, normalize_audios

        with self._lock:
            if self._model is None:
                raise ModelLoadError("Model is not loaded")
            if return_timestamps and self._model.forced_aligner is None:
                raise AlignmentError("Timestamps requested but the forced aligner is not loaded")
            asr_batch = self._model.max_inference_batch_size
            asr_batch = 32 if not asr_batch or asr_batch <= 0 else asr_batch
            runner = WindowedRunner(self._model, asr_batch=asr_batch,
                                    align_batch=align_batch or max(1, asr_batch // 2))
            try:
                wav = normalize_audios(audio)[0]  # mono float32 at 16 kHz, same as qwen-asr
                chunks, completed, stats = runner.run(
                    wav, language=language, context=context, timestamps=return_timestamps, done=resume,
                    save=save, on_event=on_event, cancelled=(lambda: cancel is not None and cancel.cancelled),
                )
            except ScribaError:
                raise
            except Exception as exc:
                self._free_memory()
                if _is_oom(exc):
                    raise OutOfMemoryError(f"Out of memory during transcription: {exc}") from exc
                raise ScribaError(f"Transcription failed: {exc}") from exc

        tokens = None
        if return_timestamps:
            tokens = [AlignedToken(text, start, end) for c in chunks for text, start, end in (c.tokens or [])]
        processed = (chunks[-1].offset + chunks[-1].duration) if chunks else 0.0
        return RawASRResult(
            language=merge_languages([c.language for c in chunks]) if chunks else "",
            text="".join(c.text for c in chunks),  # qwen-asr joins chunk texts the same way
            tokens=tokens,
            completed=completed,
            processed_seconds=None if completed else round(processed, 3),
            notes=stats.notes,
            asr_batch=runner.asr_batch,
            align_batch=runner.align_batch,
        )


class CancelToken:
    """Thread-safe cancellation flag checked between windows."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()




CUDA_MEMORY_FRACTION = 0.94  # keep a little VRAM free for the driver and other processes


def _cap_cuda_memory(device: str) -> None:
    """Make PyTorch raise an out-of-memory error instead of letting the driver spill into system RAM.

    On Windows/WSL2 the NVIDIA driver can silently fall back to shared system memory when VRAM is
    full, which keeps the GPU "busy" but tens of times slower. With a cap, the windowed runner gets
    a real OOM and halves the batch for that window instead.
    """
    try:
        import torch

        index = int(device.split(":", 1)[1]) if ":" in device else 0
        torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION, index)
    except Exception as exc:  # pragma: no cover
        log.debug("Cannot cap CUDA memory: %s", exc)


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _shutdown_vllm(llm: Any) -> None:
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "engine_core"):
            try:
                engine.engine_core.shutdown()
            except Exception:
                pass
        cleanup_dist_env_and_memory()
    except Exception as exc:  # pragma: no cover
        log.debug("vLLM cleanup failed: %s", exc)
