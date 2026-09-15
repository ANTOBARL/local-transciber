"""Qwen3-ASR model engine.

Loads the model once and keeps it resident between jobs. The concrete backend
(transformers or vLLM) is an implementation detail hidden behind `QwenASREngine`.
The only object leaving this module is `RawASRResult`, never a qwen-asr type.
"""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scriba.config import ASRSettings
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

            aligner_kwargs = None
            aligner_device = None
            if with_aligner:
                aligner_device = dev.resolve_device(
                    asr.forced_aligner.device if asr.forced_aligner.device != "auto" else device
                )
                aligner_dtype = dev.resolve_dtype_name(asr.forced_aligner.dtype, aligner_device)
                aligner_kwargs = {"dtype": dev.torch_dtype(aligner_dtype), "device_map": aligner_device}

            log.info(
                "Loading %s (backend=%s, device=%s, dtype=%s, aligner=%s)",
                asr.model, backend, device, dtype_name,
                asr.forced_aligner.model if with_aligner else "disabled",
            )
            try:
                from scriba.compat import ensure_nagisa_importable

                ensure_nagisa_importable()
                from qwen_asr import Qwen3ASRModel
            except ImportError as exc:
                raise ModelLoadError(f"qwen-asr is not importable: {exc}", hint="pip install qwen-asr") from exc

            common = dict(
                forced_aligner=asr.forced_aligner.model if with_aligner else None,
                forced_aligner_kwargs=aligner_kwargs,
                max_inference_batch_size=asr.max_inference_batch_size,
                max_new_tokens=asr.max_new_tokens,
            )
            try:
                if backend == "vllm":
                    kwargs = dict(gpu_memory_utilization=asr.gpu_memory_utilization, dtype=dtype_name)
                    kwargs.update(asr.backend_kwargs)
                    model = Qwen3ASRModel.LLM(model=asr.model, **common, **kwargs)
                else:
                    kwargs = dict(dtype=dev.torch_dtype(dtype_name), device_map=device)
                    kwargs.update(asr.backend_kwargs)
                    if isinstance(kwargs.get("dtype"), str):
                        kwargs["dtype"] = dev.torch_dtype(kwargs["dtype"])
                    model = Qwen3ASRModel.from_pretrained(asr.model, **common, **kwargs)
            except ScribaError:
                raise
            except Exception as exc:
                self._free_memory()
                if _is_oom(exc):
                    raise OutOfMemoryError(f"Out of memory while loading {asr.model}: {exc}") from exc
                if backend == "vllm":
                    raise ModelLoadError(f"vLLM initialization failed: {exc}") from exc
                raise ModelLoadError(f"Failed to load {asr.model}: {exc}") from exc

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
        on_unit: UnitCallback | None = None,
    ) -> RawASRResult:
        with self._lock:
            if self._model is None:
                raise ModelLoadError("Model is not loaded")
            if return_timestamps and self._model.forced_aligner is None:
                raise AlignmentError("Timestamps requested but the forced aligner is not loaded")
            try:
                with instrument_progress(self._model, on_unit):
                    results = self._model.transcribe(
                        audio=audio,
                        context=context or "",
                        language=language,
                        return_time_stamps=return_timestamps,
                    )
            except ScribaError:
                raise
            except Exception as exc:
                self._free_memory()
                if _is_oom(exc):
                    raise OutOfMemoryError(f"Out of memory during transcription: {exc}") from exc
                if return_timestamps and "align" in repr(exc).lower():
                    raise AlignmentError(f"Forced alignment failed: {exc}") from exc
                raise ScribaError(f"Transcription failed: {exc}") from exc

        r = results[0]
        tokens = None
        if return_timestamps and r.time_stamps is not None:
            tokens = [
                AlignedToken(text=str(it.text), start=float(it.start_time), end=float(it.end_time))
                for it in r.time_stamps.items
            ]
        elif return_timestamps:
            tokens = []
        return RawASRResult(language=r.language or "", text=r.text or "", tokens=tokens)


UnitCallback = Callable[[str, int, int, int], None]  # (phase, done, total, batch_size)


@contextmanager
def instrument_progress(model: Any, on_unit: UnitCallback | None) -> Iterator[None]:
    """Count the batches qwen-asr processes, without changing what it computes.

    Wraps, on this instance only and for the duration of one call:
      - `_infer_asr(contexts, wavs, languages)` → number of chunks to transcribe
      - the backend `generate` (transformers module or vLLM LLM) → ASR batches done
      - `forced_aligner.align(audio, text, language)` → alignment batches done
    """
    if on_unit is None:
        yield
        return

    batch = model.max_inference_batch_size
    state = {"asr_total": 0, "asr_done": 0, "align_done": 0}
    patched: list[tuple[Any, str, Any]] = []  # (object, attribute, original instance value or _MISSING)

    def patch(obj: Any, name: str, replacement: Any) -> None:
        patched.append((obj, name, vars(obj).get(name, _MISSING)))
        setattr(obj, name, replacement)

    def safe_notify(*args: Any) -> None:
        try:
            on_unit(*args)
        except Exception as exc:  # progress must never break a transcription
            log.debug("Progress callback failed: %s", exc)

    def step(total: int) -> int:
        return total if batch is None or batch <= 0 else min(batch, total)

    orig_infer = model._infer_asr

    def infer_asr(contexts, wavs, languages):
        state["asr_total"] = len(wavs)
        safe_notify("asr", 0, len(wavs), step(len(wavs)))
        return orig_infer(contexts, wavs, languages)

    backend = model.model
    orig_generate = backend.generate

    def generate(*args, **kwargs):
        out = orig_generate(*args, **kwargs)
        total = state["asr_total"]
        if args and isinstance(args[0], list):  # vLLM: list of requests
            n = len(args[0])
        elif "input_ids" in kwargs:  # transformers
            n = int(kwargs["input_ids"].shape[0])
        else:
            n = step(total)
        state["asr_done"] = min(total, state["asr_done"] + n)
        safe_notify("asr", state["asr_done"], total, step(total))
        return out

    patch(model, "_infer_asr", infer_asr)
    patch(backend, "generate", generate)

    aligner = model.forced_aligner
    if aligner is not None:
        orig_align = aligner.align

        def align(audio, text, language):
            total = state["asr_total"]
            if state["align_done"] == 0:
                safe_notify("align", 0, total, step(total))
            out = orig_align(audio=audio, text=text, language=language)
            state["align_done"] = min(total, state["align_done"] + (len(text) if isinstance(text, list) else 1))
            safe_notify("align", state["align_done"], total, step(total))
            return out

        patch(aligner, "align", align)
    try:
        yield
    finally:
        for obj, name, original in reversed(patched):
            if original is _MISSING:
                vars(obj).pop(name, None)  # class method becomes visible again
            else:
                setattr(obj, name, original)


_MISSING = object()


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
