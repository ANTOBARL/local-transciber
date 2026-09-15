from types import SimpleNamespace

from scriba.core.progress import ThroughputHistory, TranscriptionProgress, format_eta


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def plan(total=75, window=12):
    return SimpleNamespace(total_chunks=total, window=window)


def stats(asr_done=0, align_done=0, asr_seconds=0.0, align_seconds=0.0):
    return SimpleNamespace(asr_done=asr_done, align_done=align_done, asr_seconds=asr_seconds,
                           align_seconds=align_seconds)


def test_eta_uses_measured_seconds_per_chunk():
    """3 h 42 min recording: 75 chunks, windows of 12. After two windows: 20 s/chunk ASR, 5 s/chunk alignment."""
    clock = FakeClock()
    p = TranscriptionProgress(audio_seconds=13368, timestamps=True, clock=clock)
    p.on_event("plan", plan(), stats())
    assert p.snapshot()["eta_seconds"] is None and p.snapshot()["percent"] == 0

    clock.now = 600.0
    p.on_event("window", plan(), stats(asr_done=24, align_done=24, asr_seconds=480, align_seconds=120))
    snap = p.snapshot()
    assert abs(snap["eta_seconds"] - 51 * (20 + 5)) < 1
    assert snap["percent"] == int(24 / 75 * 100)
    assert snap["asr"] == [24, 75] and snap["align"] == [24, 75]


def test_bar_never_runs_more_than_one_batch_ahead():
    clock = FakeClock()
    p = TranscriptionProgress(audio_seconds=13368, timestamps=True, clock=clock)
    p.on_event("plan", plan(), stats())
    clock.now = 300.0
    p.on_event("window", plan(), stats(asr_done=12, align_done=12, asr_seconds=240, align_seconds=60))
    clock.now = 100_000.0  # the next window is taking far longer than expected
    snap = p.snapshot()
    assert snap["fraction"] <= (12 * 25 + 0.9 * 12 * 20) / (75 * 25) + 1e-6


def test_resumed_chunks_count_as_done_but_not_as_speed():
    clock = FakeClock()
    p = TranscriptionProgress(audio_seconds=13368, timestamps=True, clock=clock)
    p.on_event("plan", plan(), stats(asr_done=48, align_done=48))
    assert p.snapshot()["percent"] == 0 and p.snapshot()["eta_seconds"] is None  # nothing measured yet
    clock.now = 300.0
    p.on_event("window", plan(), stats(asr_done=60, align_done=60, asr_seconds=240, align_seconds=60))
    assert abs(p.snapshot()["eta_seconds"] - 15 * 25) < 1  # 20 + 5 s/chunk from the 12 new chunks only
    assert abs(p.measured_align_ratio() - 0.25) < 1e-9


def test_progress_uses_history_before_first_window():
    clock = FakeClock()
    p = TranscriptionProgress(audio_seconds=1000, timestamps=False, prior_rate=0.05, clock=clock)
    p.on_event("plan", plan(10, 10), stats())
    clock.now = 20.0
    snap = p.snapshot()
    assert snap["estimated_from_history"] and abs(snap["eta_seconds"] - 30.0) < 0.01
    assert snap["fraction"] == 0


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
