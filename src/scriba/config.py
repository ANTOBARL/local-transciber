"""Centralized configuration.

A single `ScribaSettings` model is shared by the YAML file, environment variables,
the CLI and the UI. Precedence (lowest → highest):
    built-in defaults < YAML file < environment (SCRIBA_*) < explicit overrides.
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")

Backend = Literal["auto", "transformers", "vllm"]
DType = Literal["auto", "bfloat16", "float16", "float32"]


class AppSettings(BaseModel):
    output_root: Path = Path("./outputs")
    create_job_subfolder: bool = True
    keep_normalized_audio: bool = False
    overwrite: bool = False
    log_level: str = "INFO"


class AudioSettings(BaseModel):
    normalize: bool = True
    sample_rate: int = Field(16000, ge=8000, le=96000)
    channels: int = Field(1, ge=1, le=2)
    ffmpeg_path: str | None = None


class ForcedAlignerSettings(BaseModel):
    enabled: bool = True
    model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    dtype: DType = "auto"
    device: str = "auto"


class ASRSettings(BaseModel):
    model: str = "Qwen/Qwen3-ASR-1.7B"
    backend: Backend = "auto"
    device: str = "auto"
    dtype: DType = "auto"

    language: str | None = "Italian"
    context: str = ""   # sent with every audio chunk: keep it short (see core/vocabulary.py)
    glossary: str = ""  # one term per line; used only to correct the transcript, never sent to the model

    gpu_memory_utilization: float = Field(0.70, gt=0.0, le=1.0)
    max_inference_batch_size: int = Field(32, ge=-1)
    align_batch_size: int = Field(0, ge=0)  # 0 = automatic (half of the ASR batch); alignment needs more memory
    max_new_tokens: int = Field(4096, ge=1)
    # Audio is transcribed in pieces of at most this length. Long pieces (the 180 s maximum allowed by the
    # aligner) make Qwen3-ASR drop speech or loop on real recordings; 30 s pieces are faster and complete.
    chunk_seconds: float = Field(30.0, ge=5, le=180)

    return_timestamps: bool = True
    forced_aligner: ForcedAlignerSettings = ForcedAlignerSettings()

    backend_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("context", mode="after")
    @classmethod
    def _normalize_context(cls, v: str) -> str:
        from scriba.core.vocabulary import normalize_context

        return normalize_context(v)

    @field_validator("glossary", mode="after")
    @classmethod
    def _normalize_glossary(cls, v: str) -> str:
        from scriba.core.vocabulary import normalize_glossary

        return normalize_glossary(v)

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.lower() in {"auto", "auto detect", "none", "null"}:
            return None
        return s[:1].upper() + s[1:].lower()

    @property
    def timestamps_active(self) -> bool:
        return self.return_timestamps and self.forced_aligner.enabled


class DiarizationSettings(BaseModel):
    enabled: bool = True
    model: str = "pyannote/speaker-diarization-community-1"
    device: str = "auto"
    hf_token: SecretStr | None = None
    num_speakers: int | None = Field(None, ge=1)
    min_speakers: int | None = Field(None, ge=1)
    max_speakers: int | None = Field(None, ge=1)
    min_turn_seconds: float = Field(0.5, ge=0)          # shorter speaker turns are ignored
    min_speaker_run_seconds: float = Field(1.0, ge=0)   # shorter speaker changes inside speech are undone
    sentence_majority: float = Field(0.7, ge=0, le=1)   # a sentence goes to a speaker holding this share of words

    @model_validator(mode="after")
    def _check_bounds(self) -> "DiarizationSettings":
        if self.min_speakers and self.max_speakers and self.min_speakers > self.max_speakers:
            raise ValueError("min_speakers cannot be greater than max_speakers")
        return self


class ExportSettings(BaseModel):
    json_: bool = Field(True, alias="json")
    txt: bool = True
    markdown: bool = True
    srt: bool = True
    vtt: bool = False
    docx: bool = False
    max_segment_seconds: float = Field(12.0, gt=0)
    max_segment_chars: int = Field(160, ge=20)
    pause_split_seconds: float = Field(0.8, ge=0)

    model_config = {"populate_by_name": True}

    def enabled_formats(self) -> list[str]:
        flags = {"json": self.json_, "txt": self.txt, "markdown": self.markdown, "srt": self.srt, "vtt": self.vtt,
                 "docx": self.docx}
        return [name for name, on in flags.items() if on]


class ScribaSettings(BaseSettings):
    app: AppSettings = AppSettings()
    audio: AudioSettings = AudioSettings()
    asr: ASRSettings = ASRSettings()
    diarization: DiarizationSettings = DiarizationSettings()
    export: ExportSettings = ExportSettings()

    model_config = SettingsConfigDict(
        env_prefix="SCRIBA_",
        env_nested_delimiter="__",
        # Dotenv files are passed at load time (see load_settings) so they are re-read dynamically.
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # init kwargs carry the YAML file: environment must win over it.
        return env_settings, dotenv_settings, init_settings

    def to_public_dict(self) -> dict[str, Any]:
        """Serializable dump without secrets (safe for config.json and logs)."""
        data = self.model_dump(mode="json", by_alias=True)
        if data["diarization"].get("hf_token"):
            data["diarization"]["hf_token"] = "***"
        return data

    def with_overrides(self, overrides: dict[str, Any] | None) -> "ScribaSettings":
        """Return a new validated settings object with nested overrides applied."""
        if not overrides:
            return self
        base = self.model_dump(by_alias=True)
        if self.diarization.hf_token is not None:
            base["diarization"]["hf_token"] = self.diarization.hf_token.get_secret_value()
        merged = deep_merge(base, overrides)
        return ScribaSettings.model_validate(merged)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def dotted_to_nested(pairs: dict[str, Any]) -> dict[str, Any]:
    """{"asr.forced_aligner.enabled": False} -> {"asr": {"forced_aligner": {"enabled": False}}}"""
    nested: dict[str, Any] = {}
    for dotted, value in pairs.items():
        node = nested
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


def parse_set_option(item: str) -> tuple[str, Any]:
    """Parse `key.path=value`, interpreting the value as YAML (numbers, bools, null, dicts)."""
    if "=" not in item:
        raise ValueError(f"Invalid override '{item}', expected key=value")
    key, raw = item.split("=", 1)
    return key.strip(), yaml.safe_load(raw) if raw.strip() else ""


def load_settings(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ScribaSettings:
    """Load defaults, then the given YAML file (if any), .env files, env vars, and explicit overrides."""
    from scriba.envfile import env_files_to_read

    data: dict[str, Any] = _read_yaml(DEFAULT_CONFIG_PATH)
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        data = deep_merge(data, _read_yaml(path))
    env_files = tuple(str(p) for p in env_files_to_read() if p.is_file())
    settings = ScribaSettings(_env_file=env_files or None, **data)
    return settings.with_overrides(overrides)


class SettingsProvider:
    """Returns up-to-date settings, reloading when the YAML or .env files change on disk.

    Lets the optimizer write tuned values that running servers pick up without a restart.
    """

    def __init__(self, config_path: str | Path | None = None, overrides: dict[str, Any] | None = None):
        self.config_path = Path(config_path) if config_path else None
        self.overrides = overrides
        self._stamp: tuple | None = None
        self._settings: ScribaSettings | None = None
        self._lock = threading.Lock()

    def _current_stamp(self) -> tuple:
        from scriba.envfile import env_files_to_read

        paths = [*env_files_to_read(), *([self.config_path] if self.config_path else [])]
        return tuple((str(p), p.stat().st_mtime_ns if p.is_file() else None) for p in paths)

    def get(self) -> ScribaSettings:
        with self._lock:
            stamp = self._current_stamp()
            if self._settings is None or stamp != self._stamp:
                self._settings = load_settings(self.config_path, self.overrides)
                self._stamp = stamp
            return self._settings


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a mapping at top level")
    return loaded
