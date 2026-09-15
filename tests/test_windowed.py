"""Windowed runner, checkpoints and resume — with a fake Qwen model (no GPU, no download)."""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from scriba.core.checkpoint import SegmentStore
from scriba.core.windowed import ChunkResult, WindowedRunner, WindowPlan
from scriba.exporters.srt_exporter import render_srt
from scriba.exporters.txt_exporter import render_txt
from scriba.models import Segment, Transcript
from scriba.utils.paths import new_job_id, safe_name

CHUNK_SECONDS = 10.0


@dataclass(frozen=True)
class Item:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class AlignResult:
    items: list


class FakeAligner:
    def __init__(self, fail_above=None):
        self.calls = []
        self.fail_above = fail_above

    def align(self, audio, text, language):
        self.calls.append(len(text))
        if self.fail_above and len(text) > self.fail_above:
            raise RuntimeError("CUDA out of memory")
        return [AlignResult([Item(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(t.split())]) for t in text]


class FakeQwen:
    def __init__(self, aligner=None):
        self.asr_calls = []
        self.forced_aligner = aligner or FakeAligner()

    def _infer_asr(self, contexts, wavs, languages):
        self.asr_calls.append(len(wavs))
        return [f"parola{int(w[0])} fine{int(w[0])}." for w in wavs]


@pytest.fixture(autouse=True)
def fake_split(monkeypatch):
    """Six contiguous 10-second chunks whose first sample encodes the chunk index."""

    def split(wav, timestamps):
        wavs = [np.full(16000, float(i), dtype=np.float32) for i in range(6)]
        offsets = [i * CHUNK_SECONDS for i in range(6)]
        return wavs, WindowPlan(total_chunks=6, window=0, offsets=offsets, durations=[CHUNK_SECONDS] * 6)

    monkeypatch.setattr(WindowedRunner, "split", staticmethod(split))


def run(runner, **kw):
    return runner.run(np.zeros(16000, dtype=np.float32), language="Italian", context="", timestamps=True, **kw)


def test_windows_run_asr_then_alignment_with_their_own_batches():
    model = FakeQwen()
    events = []
    chunks, completed, stats = run(WindowedRunner(model, asr_batch=4, align_batch=2),
                                   on_event=lambda e, plan, s: events.append((e, s.asr_done, s.align_done)))
    assert completed and [c.index for c in chunks] == list(range(6))
    assert model.asr_calls == [4, 2]                     # two windows
    assert model.forced_aligner.calls == [2, 2, 2]       # alignment in smaller batches
    assert events[:4] == [("plan", 0, 0), ("asr", 4, 0), ("align", 4, 4), ("window", 4, 4)]
    assert chunks[2].tokens[0] == ("parola2", 20.0, 20.4)  # absolute timestamps


def test_cancel_stops_between_windows_and_keeps_timestamps():
    model = FakeQwen()
    state = {"windows": 0}

    def on_event(event, plan, stats):
        if event == "window":
            state["windows"] += 1

    chunks, completed, _ = run(WindowedRunner(model, asr_batch=2, align_batch=2), on_event=on_event,
                               cancelled=lambda: state["windows"] >= 2)
    assert not completed
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert all(c.tokens for c in chunks)


def test_resume_skips_saved_chunks():
    model = FakeQwen()
    done = {i: ChunkResult(i, i * CHUNK_SECONDS, CHUNK_SECONDS, "Italian", f"saved{i}", []) for i in range(4)}
    chunks, completed, stats = run(WindowedRunner(model, asr_batch=4, align_batch=4), done=done)
    assert completed and model.asr_calls == [2]
    assert chunks[0].text == "saved0" and chunks[5].text.startswith("parola5")


def test_out_of_memory_halves_the_failing_batch_and_retries_the_window():
    model = FakeQwen(FakeAligner(fail_above=2))
    runner = WindowedRunner(model, asr_batch=4, align_batch=4)
    chunks, completed, stats = run(runner)
    assert completed and len(chunks) == 6
    assert runner.align_batch == 2 and runner.asr_batch == 4
    assert any("batch reduced from 4 to 2" in n for n in stats.notes)


def test_segment_store_roundtrip_and_resume_lookup(tmp_path):
    fp = {"sha256": "abc", "model": "m"}
    job = tmp_path / "Riunione_20260915_101010"
    store = SegmentStore(job, fp)
    store.init()
    store.save([ChunkResult(0, 0.0, 10.0, "Italian", "ciao", [("ciao", 0.1, 0.4)])])
    assert store.load()[0].tokens == [("ciao", 0.1, 0.4)]

    from pathlib import Path

    assert SegmentStore.find_resumable(tmp_path, Path("Riunione.m4a"), fp) == job
    assert SegmentStore.find_resumable(tmp_path, Path("Riunione.m4a"), {**fp, "model": "other"}) is None
    assert SegmentStore(job, {**fp, "model": "other"}).load() == {}  # different settings never mix

    store.finish()
    assert SegmentStore.find_resumable(tmp_path, Path("Riunione.m4a"), fp) is None


def partial_transcript() -> Transcript:
    return Transcript(
        id="x", source_file="riunione.m4a", language="Italian", duration=3600, model="m",
        created_at=datetime.now(timezone.utc), text="Buongiorno.", has_timestamps=True,
        completed=False, processed_seconds=754.0,
        segments=[Segment(text="Buongiorno.", start=1.0, end=2.0)],
    )


def test_exports_contain_interruption_note():
    t = partial_transcript()
    txt = render_txt(t)
    note = "La trascrizione si è interrotta a 00:12:34 di 01:00:00"
    assert txt.startswith(f"*** {note} ***") and txt.rstrip().endswith(f"*** {note} ***")
    srt = render_srt(t)
    assert f"[{note}]" in srt and "00:12:34,000 --> 00:12:39,000" in srt


def test_job_folder_uses_audio_name_and_timestamp():
    when = datetime(2026, 9, 15, 14, 23, 1)
    assert new_job_id(when, "D:/rec/Seminario FII 14-09-2026 parte 1.m4a") == "Seminario_FII_14-09-2026_parte_1_20260915_142301"
    assert safe_name("../../etc/passwd") == "etcpasswd"
    assert safe_name("   ") == "audio"
