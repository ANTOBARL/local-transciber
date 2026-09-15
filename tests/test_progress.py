from types import SimpleNamespace

from scriba.core.engine import instrument_progress
from scriba.core.progress import ThroughputHistory, TranscriptionProgress, format_eta


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_progress_measured_eta_and_smoothing():
    clock = FakeClock()
    p = TranscriptionProgress(audio_seconds=3600, timestamps=True, clock=clock)
    p.update("asr", 0, 80, step=32)
    assert p.snapshot()["eta_seconds"] is None  # no history, nothing measured yet

    clock.now = 100.0
    p.update("asr", 32, 80, step=32)  # 40% of ASR = 34% overall after 100 s
    snap = p.snapshot()
    assert snap["percent"] == 34
    assert abs(snap["eta_seconds"] - (100 / 0.34 - 100)) < 0.5

    clock.now = 150.0  # halfway through the next batch: bar moves, but never past the next boundary
    snap = p.snapshot()
    assert 0.34 < snap["fraction"] < 0.68

    p.finish()
    assert p.snapshot()["percent"] == 100 and p.snapshot()["eta_seconds"] == 0


def test_progress_uses_history_before_first_batch():
    clock = FakeClock()
    p = TranscriptionProgress(audio_seconds=1000, timestamps=False, prior_rate=0.05, clock=clock)
    p.update("asr", 0, 10, step=10)
    clock.now = 20.0
    snap = p.snapshot()
    assert snap["estimated_from_history"] and abs(snap["eta_seconds"] - 30.0) < 0.01
    assert 0 < snap["fraction"] < 1


def test_history_ignores_short_clips(tmp_path):
    h = ThroughputHistory(tmp_path / "t.json")
    h.record("k", 5, 20)
    assert h.get("k") is None
    h.record("k", 60, 600)
    assert abs(h.get("k") - 0.1) < 1e-9


def test_format_eta():
    assert format_eta(None) == "stima in corso…"
    assert format_eta(5, "en") == "almost done"
    assert format_eta(125) == "circa 2 min 00 s rimanenti" or format_eta(125) == "circa 2 min rimanenti"
    assert format_eta(4000, "en") == "about 1 h 06 min left"


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def generate(self, batch, sampling_params=None, use_tqdm=False):
        self.calls += 1
        return [f"out{i}" for i in range(len(batch))]


class FakeAligner:
    def align(self, audio, text, language):
        return list(text)


class FakeQwen:
    """Mimics the qwen-asr call structure: _infer_asr → batched generate, then batched align."""

    def __init__(self, batch=2):
        self.max_inference_batch_size = batch
        self.model = FakeBackend()
        self.forced_aligner = FakeAligner()

    def _infer_asr(self, contexts, wavs, languages):
        out = []
        for i in range(0, len(wavs), self.max_inference_batch_size):
            out += self.model.generate(wavs[i:i + self.max_inference_batch_size])
        return out

    def transcribe(self, n_chunks):
        texts = self._infer_asr([""] * n_chunks, list(range(n_chunks)), [None] * n_chunks)
        for i in range(0, n_chunks, self.max_inference_batch_size):
            self.forced_aligner.align(audio=[], text=texts[i:i + self.max_inference_batch_size], language=[])
        return texts


def test_instrumentation_counts_batches_and_restores_methods():
    model = FakeQwen(batch=2)
    events = []
    with instrument_progress(model, lambda *e: events.append(e)):
        assert model.transcribe(5) == [f"out{i}" for i in range(2)] * 2 + ["out0"]
    assert [e for e in events if e[0] == "asr"] == [("asr", 0, 5, 2), ("asr", 2, 5, 2), ("asr", 4, 5, 2),
                                                     ("asr", 5, 5, 2)]
    assert events[-1] == ("align", 5, 5, 2)
    # originals are back: no wrapper left on the instances
    assert "generate" not in vars(model.model) and "align" not in vars(model.forced_aligner)
    assert "_infer_asr" not in vars(model)


def test_instrumentation_never_breaks_transcription():
    model = FakeQwen(batch=3)

    def broken(*_):
        raise RuntimeError("ui exploded")

    with instrument_progress(model, broken):
        assert len(model.transcribe(4)) == 4
