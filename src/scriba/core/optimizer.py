"""Inference parameter optimizer.

Runs the bundled benchmark recording with increasing batch sizes on the loaded model,
measures throughput and peak VRAM, picks the fastest batch that safely fits in free memory
and stores it in the managed block of the .env file (picked up dynamically by SettingsProvider).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from scriba.config import ScribaSettings
from scriba.envfile import env_file_path, read_managed_block, write_managed_block
from scriba.errors import OutOfMemoryError, ScribaError
from scriba.utils.logging import get_logger

log = get_logger("optimizer")

BENCHMARK_AUDIO = Path(__file__).resolve().parents[1] / "assets" / "benchmark_it.opus"
BENCHMARK_LANGUAGE = "Italian"
CANDIDATES = (2, 4, 8, 12, 16, 24, 32, 48)
CPU_MAX_BATCH = 8
VRAM_SAFETY = 0.85      # use at most 85% of the VRAM that is free before the benchmark
MIN_GAIN = 1.05         # stop when a larger batch is <5% faster
PREFER_SMALLER = 0.97   # among results within 3% of the best, pick the smallest batch
CHUNK_SECONDS_TIMESTAMPS = 180
CHUNK_SECONDS_PLAIN = 1200
ENV_KEY_BATCH = "SCRIBA_ASR__MAX_INFERENCE_BATCH_SIZE"


@dataclass
class Trial:
    batch: int
    status: str                       # ok | too_slow_gain | exceeds_vram | oom | skipped | error
    audio_seconds: float = 0.0
    seconds: float = 0.0
    speed: float | None = None        # x real time
    peak_vram_mb: float | None = None
    note: str | None = None


@dataclass
class OptimizationResult:
    best_batch: int
    previous_batch: int
    trials: list[Trial]
    device: str
    model: str
    backend: str
    dtype: str
    timestamps: bool
    vram_budget_mb: float | None
    env_path: str | None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "trials": [asdict(t) for t in self.trials]}


class OptimizerProgress:
    def __init__(self, total: int):
        self._lock = threading.Lock()
        self.stage = "loading"          # loading | warmup | trial | saving | done
        self.current_batch: int | None = None
        self.done = 0
        self.total = total
        self.trials: list[Trial] = []
        self.started = time.monotonic()

    def set(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def add_trial(self, trial: Trial) -> None:
        with self._lock:
            self.trials.append(trial)
            self.done += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "stage": self.stage,
                "current_batch": self.current_batch,
                "done": self.done,
                "total": self.total,
                "fraction": round(min(1.0, self.done / self.total), 3) if self.total else 0.0,
                "elapsed_seconds": round(time.monotonic() - self.started, 1),
                "trials": [asdict(t) for t in self.trials],
            }


def _tile(wav: np.ndarray, seconds: float, sr: int) -> np.ndarray:
    n = int(seconds * sr)
    reps = math.ceil(n / len(wav))
    return np.tile(wav, reps)[:n]


def _cuda_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    return int(device.split(":", 1)[1]) if ":" in device else 0


def apply_result(result: OptimizationResult, env_path: Path | None = None) -> str:
    """Save the tuned values to the managed block of the .env file (read dynamically by the app)."""
    path = env_path or env_file_path()
    comment = [
        f"applied {datetime.now():%Y-%m-%d %H:%M} | {result.device} | {result.model} | {result.backend} | {result.dtype}",
        "batch  speed(x realtime)  peak VRAM",
        *[f"{t.batch:>5}  {t.speed or 0:>8.1f}            {t.peak_vram_mb or 0:>7.0f} MB  {t.status}"
          for t in result.trials],
    ]
    result.env_path = str(write_managed_block(path, {ENV_KEY_BATCH: result.best_batch}, comment))
    log.info("Batch size %s saved to %s", result.best_batch, result.env_path)
    return result.env_path


def optimization_status(env_path: Path | None = None) -> dict[str, Any]:
    """Whether tuned parameters have been applied on this system, with a short summary."""
    import re

    path = env_path or env_file_path()
    values = read_managed_block(path)
    if ENV_KEY_BATCH not in values:
        return {"optimized": False}
    status: dict[str, Any] = {"optimized": True, "batch": int(values[ENV_KEY_BATCH]), "applied_at": None, "device": None}
    text = path.read_text(encoding="utf-8", errors="replace")
    # "# applied 2026-09-15 14:49 | GPU | ..." (older files used "·" as separator)
    match = re.search(r"# (?:applied )?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) [|·] ([^|·\n]+?) [|·]", text)
    if match:
        status["applied_at"], status["device"] = match.group(1), match.group(2).strip()

    # Benchmark table from the comment block: "#    12      48.2    13920 MB  ok"
    rows = [(int(b), float(s), int(v), st) for b, s, v, st in
            re.findall(r"#\s+(\d+)\s+([\d.]+)\s+(\d+) MB\s+(\w+)", text)]
    measured = [r for r in rows if r[1] > 0 and r[3] in ("ok", "too_slow_gain")]
    best = next((r for r in measured if r[0] == status["batch"]), None)
    if best:
        benchmark_speed = best[1]
        speed, source, jobs = benchmark_speed, "benchmark", 0
        # Real recordings are slower than the benchmark (varied chunks: a batch waits for its
        # longest transcript). Prefer the speed measured on the user's own completed jobs.
        from scriba.core.progress import ThroughputHistory, real_speed_key

        real = ThroughputHistory().entry(real_speed_key(status["device"], status["batch"], True))
        if real.get("speed") and real.get("jobs"):
            speed, source, jobs = float(real["speed"]), "real", int(real["jobs"])
        status.update({
            "speed": round(speed, 1),                         # x real time (ASR + alignment)
            "speed_source": source,                           # real | benchmark
            "real_jobs": jobs,
            "benchmark_speed": benchmark_speed,
            "peak_vram_mb": best[2],
            "minutes_per_audio_hour": round(60 / speed, 2),
            "speedup_vs_smallest": round(benchmark_speed / min(r[1] for r in measured), 1),
            "gpu_total_mb": _gpu_total_mb(status["device"]),
        })
    return status


def _gpu_total_mb(device_name: str | None) -> int | None:
    from scriba.utils.device import list_gpus

    gpus = list_gpus()
    match = next((g for g in gpus if device_name and g.name == device_name), gpus[0] if gpus else None)
    return int(match.total_memory_gb * 1024) if match else None


class InferenceOptimizer:
    def __init__(self, service: Any, candidates: tuple[int, ...] = CANDIDATES):
        self.service = service
        self.candidates = candidates

    def run(self, settings: ScribaSettings, write_env: bool = True, env_path: Path | None = None,
            progress: OptimizerProgress | None = None) -> OptimizationResult:
        from scriba.audio.preprocess import load_waveform

        asr = settings.asr
        engine = self.service.engine
        progress = progress or OptimizerProgress(len(self.candidates))

        progress.set(stage="loading")
        info = engine.ensure_loaded(asr)
        previous = asr.max_inference_batch_size
        cuda = _cuda_index(info.device)
        candidates = [b for b in self.candidates if cuda is not None or b <= CPU_MAX_BATCH]
        progress.set(total=len(candidates))

        sr = 16000
        wav = load_waveform(BENCHMARK_AUDIO, sr, 1)
        timestamps = asr.timestamps_active
        chunk_seconds = CHUNK_SECONDS_TIMESTAMPS if timestamps else CHUNK_SECONDS_PLAIN

        torch = None
        budget_mb = None
        reserved_base = 0
        if cuda is not None:
            import torch

            torch.cuda.empty_cache()
            free, _total = torch.cuda.mem_get_info(cuda)
            reserved_base = torch.cuda.memory_reserved(cuda)
            budget_mb = free * VRAM_SAFETY / 1024**2

        progress.set(stage="warmup")
        engine.set_batch_size(asr, 1)
        engine.transcribe((wav[: sr * 20], sr), language=BENCHMARK_LANGUAGE, return_timestamps=timestamps)

        trials: list[Trial] = []
        per_chunk_mb: float | None = None
        best_speed = 0.0
        try:
            for batch in candidates:
                if per_chunk_mb and budget_mb and per_chunk_mb * batch > budget_mb:
                    trial = Trial(batch, "skipped", note="predicted to exceed free VRAM")
                    trials.append(trial)
                    progress.add_trial(trial)
                    continue
                progress.set(stage="trial", current_batch=batch)
                trial = self._trial(engine, asr, batch, _tile(wav, batch * chunk_seconds, sr), sr, timestamps,
                                    torch, cuda, reserved_base, budget_mb)
                trials.append(trial)
                progress.add_trial(trial)
                log.info("Batch %s: %s speed=%s peak=%s MB", batch, trial.status, trial.speed, trial.peak_vram_mb)
                if trial.status != "ok":
                    break
                if trial.peak_vram_mb:
                    per_chunk_mb = trial.peak_vram_mb / batch
                if best_speed and trial.speed < best_speed * MIN_GAIN:
                    trial.status = "too_slow_gain"
                    trial.note = "less than 5% faster than the previous batch"
                    break
                best_speed = max(best_speed, trial.speed or 0.0)
        finally:
            usable = [t for t in trials if t.status in ("ok", "too_slow_gain") and t.speed]
            if usable:
                top = max(t.speed for t in usable)
                best = min((t for t in usable if t.speed >= top * PREFER_SMALLER), key=lambda t: t.batch).batch
            else:
                best = previous
            # Keep the loaded model on the user's current value until the result is applied.
            engine.set_batch_size(asr, best if write_env else previous)

        if not any(t.status in ("ok", "too_slow_gain") for t in trials):
            raise ScribaError("Optimization failed: no batch size completed successfully",
                              hint="Check `scriba doctor` and the processing log.")

        result = OptimizationResult(
            best_batch=best, previous_batch=previous, trials=trials, device=info.device_name, model=info.model,
            backend=info.backend, dtype=info.dtype, timestamps=timestamps, vram_budget_mb=budget_mb,
            env_path=None,
        )
        if write_env:
            progress.set(stage="saving")
            apply_result(result, env_path)
        progress.set(stage="done", current_batch=None)
        return result



    @staticmethod
    def _trial(engine, asr, batch, audio, sr, timestamps, torch, cuda, reserved_base, budget_mb) -> Trial:
        audio_seconds = len(audio) / sr
        engine.set_batch_size(asr, batch)
        if torch is not None:
            torch.cuda.reset_peak_memory_stats(cuda)
            torch.cuda.synchronize(cuda)
        started = time.perf_counter()
        try:
            engine.transcribe((audio, sr), language=BENCHMARK_LANGUAGE, return_timestamps=timestamps)
        except OutOfMemoryError as exc:
            return Trial(batch, "oom", audio_seconds, note=str(exc)[:200])
        except ScribaError as exc:
            return Trial(batch, "error", audio_seconds, note=str(exc)[:200])
        if torch is not None:
            torch.cuda.synchronize(cuda)
        seconds = time.perf_counter() - started
        peak = None
        if torch is not None:
            peak = max(0.0, (torch.cuda.max_memory_reserved(cuda) - reserved_base) / 1024**2)
            torch.cuda.empty_cache()
        trial = Trial(batch, "ok", round(audio_seconds, 1), round(seconds, 2), round(audio_seconds / seconds, 1),
                      None if peak is None else round(peak))
        if budget_mb is not None and peak is not None and peak > budget_mb:
            trial.status, trial.note = "exceeds_vram", f"peak {peak:.0f} MB > budget {budget_mb:.0f} MB"
        return trial
