import os
import time

import pytest

from scriba.config import SettingsProvider, load_settings
from scriba.core.engine import LoadedModelInfo
from scriba.core.optimizer import BENCHMARK_AUDIO, ENV_KEY_BATCH, InferenceOptimizer
from scriba.envfile import read_managed_block, write_managed_block
from scriba.errors import OutOfMemoryError


def test_benchmark_audio_is_bundled():
    assert BENCHMARK_AUDIO.is_file() and BENCHMARK_AUDIO.stat().st_size > 100_000


def test_managed_block_preserves_other_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("SCRIBA_DIARIZATION__HF_TOKEN=hf_secret\n", encoding="utf-8")
    write_managed_block(env, {ENV_KEY_BATCH: 16}, ["run 1"])
    write_managed_block(env, {ENV_KEY_BATCH: 8}, ["run 2"])
    text = env.read_text(encoding="utf-8")
    assert text.startswith("SCRIBA_DIARIZATION__HF_TOKEN=hf_secret\n")
    assert text.count("# >>> scriba optimizer") == 1 and "run 1" not in text
    assert read_managed_block(env) == {ENV_KEY_BATCH: "8"}


def test_settings_provider_reloads_env_file(tmp_path, monkeypatch):
    env = tmp_path / "scriba.env"
    monkeypatch.setenv("SCRIBA_ENV_FILE", str(env))
    monkeypatch.delenv("SCRIBA_ASR__MAX_INFERENCE_BATCH_SIZE", raising=False)
    provider = SettingsProvider()
    assert provider.get().asr.max_inference_batch_size == 32

    write_managed_block(env, {ENV_KEY_BATCH: 12})
    os.utime(env, ns=(time.time_ns(), time.time_ns() + 10_000_000))  # make sure mtime changes
    assert provider.get().asr.max_inference_batch_size == 12


class FakeEngine:
    """Throughput grows with batch until 8, then flattens; batch 16 runs out of memory."""

    def __init__(self, oom_at=16):
        self.batch = 1
        self.oom_at = oom_at
        self.calls = []

    def ensure_loaded(self, asr):
        return LoadedModelInfo(backend="fake", device="cpu", device_name="CPU", dtype="float32",
                               model=asr.model, aligner_model=None)

    def set_batch_size(self, asr, batch):
        self.batch = batch

    def transcribe(self, audio, language=None, context="", return_timestamps=True, on_unit=None):
        wav, sr = audio
        self.calls.append((self.batch, len(wav) / sr))
        if self.batch >= self.oom_at:
            raise OutOfMemoryError("CUDA out of memory")
        time.sleep(0.02 / min(self.batch, 8))


class FakeService:
    def __init__(self, engine):
        self.engine = engine


def test_optimizer_picks_fastest_and_writes_env(tmp_path):
    settings = load_settings()
    engine = FakeEngine()
    env = tmp_path / ".env"
    result = InferenceOptimizer(FakeService(engine), candidates=(2, 4, 8, 16)).run(settings, env_path=env)

    statuses = {t.batch: t.status for t in result.trials}
    assert statuses[2] == "ok" and statuses[4] == "ok"
    assert result.best_batch in (4, 8)
    assert read_managed_block(env) == {ENV_KEY_BATCH: str(result.best_batch)}
    assert engine.batch == result.best_batch  # loaded model switched to the tuned value
    # each trial transcribes batch × chunk length of (tiled) benchmark audio
    assert (2, 360.0) in engine.calls


def test_optimizer_dry_run_restores_previous_batch(tmp_path):
    settings = load_settings(overrides={"asr": {"max_inference_batch_size": 3}})
    engine = FakeEngine(oom_at=4)
    result = InferenceOptimizer(FakeService(engine), candidates=(2, 4)).run(settings, write_env=False)
    assert result.env_path is None
    assert [t.status for t in result.trials] == ["ok", "oom"]
    assert engine.batch == 3


def test_optimizer_fails_when_nothing_works():
    engine = FakeEngine(oom_at=1)
    with pytest.raises(Exception):
        InferenceOptimizer(FakeService(engine), candidates=(2,)).run(load_settings(), write_env=False)
