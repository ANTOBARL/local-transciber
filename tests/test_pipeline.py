"""Pipeline tests with a fake engine (no model download, no GPU)."""

import wave
from pathlib import Path

import numpy as np
import pytest

from scriba.config import load_settings
from scriba.core.aligner import align_words, build_segments
from scriba.core.engine import AlignedToken, RawASRResult
from scriba.core.pipeline import TranscriptionService, apply_speaker_names
from scriba.core.speakers import assign_speakers
from scriba.exporters import load_transcript
from scriba.models import JobStatus, SpeakerTurn


def tok(text, s, e):
    return AlignedToken(text=text, start=s, end=e)


def test_align_words_keeps_punctuation_and_handles_glued_chunks():
    text = "Ciao, sono Luca.Oggi parliamo dell'ICTF!"
    tokens = [tok("Ciao", 0.0, 0.4), tok("sono", 0.5, 0.8), tok("Luca", 0.8, 1.2),
              tok("Oggi", 1.3, 1.6), tok("parliamo", 1.6, 2.1), tok("dell'ICTF", 2.1, 2.9)]
    words = align_words(text, tokens)
    assert [w.text for w in words] == ["Ciao,", "sono", "Luca.Oggi", "parliamo", "dell'ICTF!"]
    assert words[2].start == 0.8 and words[2].end == 1.6
    assert words[-1].end == 2.9


def test_build_segments_splits_on_pause_and_sentence():
    text = "Buongiorno a tutti. Possiamo iniziare con il primo punto"
    tokens = [tok(w, i * 0.3, i * 0.3 + 0.25) for i, w in enumerate(text.replace(".", "").split())]
    tokens[3:] = [AlignedToken(t.text, t.start + 3, t.end + 3) for t in tokens[3:]]
    segs = build_segments(align_words(text, tokens), pause_split=0.8)
    assert [s.text for s in segs] == ["Buongiorno a tutti.", "Possiamo iniziare con il primo punto"]


def test_assign_speakers_max_overlap():
    from datetime import datetime, timezone

    from scriba.models import Segment, Transcript, Word

    words = [Word(text="parola", start=12.20, end=12.58), Word(text="dopo", start=13.0, end=13.4)]
    t = Transcript(id="x", source_file="a.wav", language="Italian", duration=20, model="m",
                   created_at=datetime.now(timezone.utc), text="parola dopo", has_timestamps=True,
                   segments=[Segment(text="parola dopo", start=12.2, end=13.4, words=words)])
    turns = [SpeakerTurn(start=10.0, end=12.30, speaker="SPEAKER_00"),
             SpeakerTurn(start=12.30, end=17.20, speaker="SPEAKER_01")]
    out = assign_speakers(t, turns)
    assert [w.speaker for s in out.segments for w in s.words] == ["SPEAKER_01", "SPEAKER_01"]
    assert out.speakers == ["SPEAKER_01"]


class FakeEngine:
    def __init__(self):
        self.loads = 0
        self._loaded = False
        self.info = None

    def is_loaded(self):
        return self._loaded

    def needs_reload(self, asr):
        return not self._loaded

    def ensure_loaded(self, asr):
        from scriba.core.engine import LoadedModelInfo

        if not self._loaded:
            self.loads += 1
            self._loaded = True
            self.info = LoadedModelInfo(backend="fake", device="cpu", device_name="CPU", dtype="float32",
                                        model=asr.model, aligner_model=asr.forced_aligner.model)
        return self.info

    def set_batch_size(self, asr, batch):
        self.batch = batch

    def release_cached_memory(self):
        self.released = getattr(self, "released", 0) + 1

    def transcribe(self, audio, language=None, context="", return_timestamps=True, **kwargs):
        wav, sr = audio
        assert sr == 16000 and wav.ndim == 1
        text = "Buongiorno a tutti. Possiamo iniziare."
        tokens = [tok("Buongiorno", 0.1, 0.5), tok("a", 0.5, 0.6), tok("tutti", 0.6, 0.9),
                  tok("Possiamo", 1.5, 1.8), tok("iniziare", 1.8, 1.95)]
        return RawASRResult(language="Italian", text=text, tokens=tokens if return_timestamps else None)


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    sr = 22050
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.repeat(pcm, 2).tobytes())
    return path


def test_service_end_to_end_with_fake_engine(tmp_path, wav_file):
    settings = load_settings(overrides={"app": {"output_root": str(tmp_path / "out")},
                                        "export": {"vtt": True}, "asr": {"context": "ICTF, Odoo"},
                                        "diarization": {"enabled": False}})
    service = TranscriptionService()
    service.engine = FakeEngine()
    statuses = []

    r1 = service.transcribe(wav_file, settings, progress=lambda s, m: statuses.append(s))
    r2 = service.transcribe(wav_file, settings)

    assert service.engine.loads == 1  # model loaded once across jobs
    assert r1.job_id != r2.job_id and r1.job_dir != r2.job_dir
    assert statuses[0] == JobStatus.PREPROCESSING and statuses[-1] == JobStatus.COMPLETED
    for name in ("job.json", "config.json", "source.json", "processing.log", "transcript.json",
                 "transcript.txt", "transcript.md", "transcript.srt", "transcript.vtt"):
        assert (r1.job_dir / name).is_file(), name

    tr = load_transcript(r1.job_dir / "transcript.json")
    assert tr.has_timestamps and tr.metadata["context"] == "ICTF, Odoo"
    assert abs(r1.audio.duration - 2.0) < 0.05
    job = (r1.job_dir / "job.json").read_text(encoding="utf-8")
    assert '"status": "completed"' in job


def test_diarization_failure_is_not_fatal(tmp_path, wav_file):
    settings = load_settings(overrides={"app": {"output_root": str(tmp_path)}, "diarization": {"enabled": True, "hf_token": "hf_test"}})
    service = TranscriptionService()
    service.engine = FakeEngine()

    class BrokenDiarizer:
        def diarize(self, *a, **k):
            raise RuntimeError("boom")

    service.diarizer = BrokenDiarizer()
    result = service.transcribe(wav_file, settings)
    assert (result.job_dir / "transcript.srt").is_file()
    assert "warn:diarization_failed" in result.warnings


def test_speaker_rename_regenerates_exports(tmp_path, wav_file):
    settings = load_settings(overrides={"app": {"output_root": str(tmp_path)}, "diarization": {"enabled": True, "hf_token": "hf_test"}})
    service = TranscriptionService()
    service.engine = FakeEngine()

    class FakeDiarizer:
        def diarize(self, *a, **k):
            return [SpeakerTurn(start=0, end=1.2, speaker="SPEAKER_00"),
                    SpeakerTurn(start=1.2, end=2.0, speaker="SPEAKER_01")]

    service.diarizer = FakeDiarizer()
    result = service.transcribe(wav_file, settings)
    apply_speaker_names(result.job_dir, {"SPEAKER_00": "Luca", "SPEAKER_01": "Clara"})
    txt = (result.job_dir / "transcript.txt").read_text(encoding="utf-8")
    assert "Luca" in txt and "Clara" in txt
    assert "[Luca]" in (result.job_dir / "transcript.srt").read_text(encoding="utf-8")


def test_failed_job_records_error(tmp_path, wav_file):
    settings = load_settings(overrides={"app": {"output_root": str(tmp_path)}, "diarization": {"enabled": False}})
    service = TranscriptionService()
    service.engine = FakeEngine()
    service.engine.transcribe = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
    with pytest.raises(RuntimeError):
        service.transcribe(wav_file, settings)
    job_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    assert '"status": "failed"' in (job_dir / "job.json").read_text(encoding="utf-8")


def test_diarization_without_token_is_skipped_without_network(tmp_path, wav_file, monkeypatch):
    monkeypatch.delenv("SCRIBA_DIARIZATION__HF_TOKEN", raising=False)
    settings = load_settings(overrides={"app": {"output_root": str(tmp_path)}, "diarization": {"enabled": True}})
    settings = settings.model_copy(update={"diarization": settings.diarization.model_copy(update={"hf_token": None})})
    service = TranscriptionService()
    service.engine = FakeEngine()

    class MustNotRun:
        def diarize(self, *a, **k):
            raise AssertionError("diarizer called without token")

    service.diarizer = MustNotRun()
    result = service.transcribe(wav_file, settings)
    assert "warn:diarization_no_token" in result.warnings

