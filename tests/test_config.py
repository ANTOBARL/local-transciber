from pathlib import Path

import pytest

from scriba.config import dotted_to_nested, load_settings, parse_set_option


def test_defaults_are_hardware_agnostic():
    s = load_settings()
    assert s.asr.model == "Qwen/Qwen3-ASR-1.7B"
    assert s.asr.backend == "auto"
    assert s.asr.device == "auto"
    assert s.asr.dtype == "auto"
    assert s.asr.language == "Italian"
    assert s.diarization.enabled is True
    assert s.export.enabled_formats() == ["json", "txt", "markdown", "srt"]


def test_yaml_file_and_overrides(tmp_path: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("asr:\n  max_new_tokens: 1024\nexport:\n  vtt: true\n", encoding="utf-8")
    s = load_settings(cfg, dotted_to_nested({"asr.max_new_tokens": 2048, "asr.language": "auto"}))
    assert s.asr.max_new_tokens == 2048
    assert s.asr.language is None
    assert s.export.vtt is True


def test_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("SCRIBA_ASR__BACKEND", "transformers")
    assert load_settings().asr.backend == "transformers"


def test_parse_set_option_yaml_values():
    assert parse_set_option("asr.backend_kwargs={attn_implementation: sdpa}") == (
        "asr.backend_kwargs", {"attn_implementation": "sdpa"})
    assert parse_set_option("asr.forced_aligner.enabled=false") == ("asr.forced_aligner.enabled", False)


def test_validation_errors():
    with pytest.raises(Exception):
        load_settings(overrides={"asr": {"gpu_memory_utilization": 1.5}})
    with pytest.raises(Exception):
        load_settings(overrides={"diarization": {"min_speakers": 4, "max_speakers": 2}})


def test_public_dict_hides_token():
    s = load_settings(overrides={"diarization": {"hf_token": "hf_secretsecretsecret"}})
    assert s.to_public_dict()["diarization"]["hf_token"] == "***"
    # overrides must keep the real secret
    s2 = s.with_overrides({"asr": {"max_new_tokens": 100}})
    assert s2.diarization.hf_token.get_secret_value() == "hf_secretsecretsecret"
