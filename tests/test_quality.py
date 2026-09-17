"""Loop detection, chunk repair and speaker smoothing — no model, no GPU."""

import numpy as np
import pytest

from scriba.core.quality import SpeechRate, check_text, collapse_loops, find_loops, join_chunk_texts
from scriba.core.speakers import assign_speakers, smooth_turns, smooth_word_speakers
from scriba.core.windowed import WindowedRunner, WindowPlan
from scriba.models import Segment, SpeakerTurn, Transcript, Word

LOOP = "Anche in termini di innovazione, innovazione, " + "innovazione di prodotto, " * 60 + "innovazione di"


# ---------------------------------------------------------------- detection
def test_long_phrase_loop_is_found_and_collapsed():
    words = LOOP.split()
    loops = find_loops(words)
    assert len(loops) == 1 and loops[0].size == 3 and loops[0].repeats >= 60
    text, removed = collapse_loops(LOOP + " e poi il discorso continua.")
    assert text == "Anche in termini di innovazione, innovazione, innovazione di prodotto, e poi il discorso continua."
    assert removed > 170


def test_ordinary_repetitions_are_not_loops():
    speech = "a fare a fare, del del del del discorso. Sì sì sì sì sì, esatto. Grazie, grazie, grazie a tutti."
    assert find_loops(speech.split()) == []
    assert collapse_loops(speech) == (speech, 0)


def test_rates_flag_runaway_and_nearly_empty_chunks():
    assert check_text(LOOP, 180).repetitive
    assert check_text(" ".join(f"w{i}" for i in range(1200)), 180).repetitive  # 6.7 words per second
    normal = check_text(" ".join(f"w{i}" for i in range(400)), 180)
    assert not normal.repetitive and not normal.sparse
    assert check_text("pausa caffè", 180).sparse
    assert not check_text("ok", 5).sparse


def test_early_stopped_chunk_is_sparse_compared_with_the_recording():
    rates = SpeechRate()
    for words in (340, 360, 300, 20):
        rates.add(check_text(" ".join(f"w{i}" for i in range(words)), 180))
    assert abs(rates.reference - 340 / 180) < 1e-9  # the 20-word chunk is not a healthy sample
    partial = " ".join(f"w{i}" for i in range(115))  # 0.64 words/s: fine in absolute terms
    assert not check_text(partial, 180).sparse
    assert check_text(partial, 180, rates.reference).sparse
    assert not check_text(" ".join(f"w{i}" for i in range(250)), 180, rates.reference).sparse


def test_chunk_texts_are_joined_with_spaces_except_cjk():
    assert join_chunk_texts(["non hanno bisogno di", "provare.", ""]) == "non hanno bisogno di provare."
    assert join_chunk_texts(["你好。", "世界"]) == "你好。世界"


def test_stretched_aligner_words_are_capped():
    from scriba.core.aligner import MAX_WORD_SECONDS, align_words
    from scriba.core.engine import AlignedToken

    words = align_words("tipo di prova", [AlignedToken("tipo", 10.0, 34.0), AlignedToken("di", 34.0, 34.2),
                                          AlignedToken("prova", 34.3, 34.8)])
    assert words[0].end == 10.0 + MAX_WORD_SECONDS
    assert (words[1].start, words[2].end) == (34.0, 34.8)


# ---------------------------------------------------------------- chunk repair
class LoopingQwen:
    """Loops on full chunks; `loop_pieces` decides whether shorter pieces loop too."""

    def __init__(self, loop_pieces=False):
        self.loop_pieces = loop_pieces
        self.calls = []
        self.forced_aligner = None

    def _infer_asr(self, contexts, wavs, languages):
        self.calls.append([len(w) for w in wavs])
        out = []
        for w in wavs:
            long = len(w) >= 100 * 16000
            if long or (self.loop_pieces and int(w[0]) == 1):
                out.append("Inizio " + "innovazione di prodotto, " * 50)
            else:
                out.append(f"frase del pezzo {int(w[0])}.")
        return out


@pytest.fixture
def one_long_chunk(monkeypatch):
    def split(wav, timestamps, max_seconds=None):
        return [np.zeros(180 * 16000, dtype=np.float32)], WindowPlan(1, 0, [60.0], [180.0])

    def split_sub(wav, seconds):
        return [(np.full(16000, float(i), dtype=np.float32), i * 60.0, 60.0) for i in range(3)]

    monkeypatch.setattr(WindowedRunner, "split", staticmethod(split))
    monkeypatch.setattr(WindowedRunner, "split_sub", staticmethod(split_sub))


def test_looping_chunk_is_re_decoded_in_pieces(one_long_chunk):
    model = LoopingQwen()
    chunks, completed, _ = WindowedRunner(model, asr_batch=4, align_batch=4).run(
        np.zeros(10), language="Italian", context="", timestamps=False)
    assert completed
    assert chunks[0].text == "frase del pezzo 0. frase del pezzo 1. frase del pezzo 2."
    assert chunks[0].issues == [{"kind": "repetition", "start": 60.0, "end": 240.0, "action": "redecoded",
                                 "words_removed": 0}]
    assert len(model.calls) == 2  # full window, then one retry with pieces


def test_piece_that_keeps_looping_is_collapsed_and_located(one_long_chunk):
    model = LoopingQwen(loop_pieces=True)
    chunks, _, _ = WindowedRunner(model, asr_batch=4, align_batch=4).run(
        np.zeros(10), language="Italian", context="", timestamps=False)
    assert chunks[0].text == "frase del pezzo 0. Inizio innovazione di prodotto, frase del pezzo 2."
    issue, = chunks[0].issues
    assert (issue["start"], issue["end"], issue["action"]) == (120.0, 180.0, "collapsed")
    assert issue["words_removed"] == 147
    assert len(model.calls) == 3  # both retry lengths were tried


def test_short_chunks_retry_only_with_shorter_pieces(monkeypatch):
    def split(wav, timestamps, max_seconds=None):
        return [np.zeros(16000, dtype=np.float32)], WindowPlan(1, 0, [0.0], [30.0])

    sizes = []

    def split_sub(wav, seconds):
        sizes.append(seconds)
        return [(np.full(16000, 5.0, dtype=np.float32), 0.0, 30.0)]

    class Loops:
        forced_aligner = None

        def _infer_asr(self, contexts, wavs, languages):
            return ["innovazione di prodotto " * 30 for _ in wavs]

    monkeypatch.setattr(WindowedRunner, "split", staticmethod(split))
    monkeypatch.setattr(WindowedRunner, "split_sub", staticmethod(split_sub))
    chunks, _, _ = WindowedRunner(Loops(), asr_batch=2, align_batch=2, chunk_seconds=30).run(
        np.zeros(10), language="Italian", context="", timestamps=False)
    assert sizes == [10.0]
    assert chunks[0].text == "innovazione di prodotto"


def test_pauses_spanning_chunks_are_reported_once():
    from scriba.core.windowed import merge_issues

    issues = [{"kind": "sparse", "start": s, "end": s + 30.0, "action": "kept", "words_removed": 0}
              for s in (0.0, 30.0, 60.0)]
    issues.append({"kind": "repetition", "start": 90.0, "end": 100.0, "action": "collapsed", "words_removed": 5})
    assert merge_issues(issues) == [
        {"kind": "sparse", "start": 0.0, "end": 90.0, "action": "kept", "words_removed": 0},
        {"kind": "repetition", "start": 90.0, "end": 100.0, "action": "collapsed", "words_removed": 5},
    ]


def test_chunk_issues_survive_checkpoints():
    from scriba.core.windowed import ChunkResult

    chunk = ChunkResult(0, 0.0, 10.0, "Italian", "ciao", None, [{"kind": "sparse"}])
    assert ChunkResult.from_dict(chunk.to_dict()).issues == [{"kind": "sparse"}]
    old = {k: v for k, v in chunk.to_dict().items() if k != "issues"}
    assert ChunkResult.from_dict(old).issues == []


# ---------------------------------------------------------------- speakers
def _words(spec):
    """spec: (text, start, end, speaker)"""
    return [Word(text=t, start=s, end=e, speaker=sp) for t, s, e, sp in spec]


def test_blip_turns_are_dropped_and_neighbours_merged():
    turns = [SpeakerTurn(start=0, end=5, speaker="A"), SpeakerTurn(start=5, end=5.2, speaker="B"),
             SpeakerTurn(start=5.2, end=9, speaker="A"), SpeakerTurn(start=9, end=12, speaker="B")]
    assert [(t.start, t.end, t.speaker) for t in smooth_turns(turns, 0.5)] == [(0, 9, "A"), (9, 12, "B")]


def test_short_flip_inside_a_sentence_is_undone():
    words = _words([("Parliamo", 0.0, 0.4, "A"), ("di", 0.5, 0.6, "B"), ("innovazione", 0.7, 1.2, "A"),
                    ("oggi.", 1.3, 1.6, "A")])
    smooth_word_speakers(words, min_run=1.0, majority=0)
    assert [w.speaker for w in words] == ["A"] * 4


def test_short_whole_sentence_from_another_speaker_is_kept():
    words = _words([("Iniziamo.", 0.0, 0.8, "A"), ("Sì.", 1.0, 1.3, "B"), ("Bene,", 1.5, 1.9, "A"),
                    ("allora.", 2.0, 2.5, "A")])
    smooth_word_speakers(words, min_run=1.0, majority=0.7)
    assert [w.speaker for w in words] == ["A", "B", "A", "A"]


def test_sentence_edges_follow_the_majority_speaker():
    spec = [("Quindi", 0.0, 0.3, "A")] + [(f"parola{i}", 1.0 + i, 1.5 + i, "B") for i in range(5)] + \
           [("fine.", 7.0, 7.4, "B")]
    words = _words(spec)
    smooth_word_speakers(words, min_run=0, majority=0.7)
    assert {w.speaker for w in words} == {"B"}


def test_long_unpunctuated_stretch_keeps_real_changes():
    spec = [(f"a{i}", i * 1.0, i * 1.0 + 0.5, "A") for i in range(20)] + \
           [(f"b{i}", 20 + i * 1.0, 20.5 + i * 1.0, "B") for i in range(4)] + [("fine.", 25, 25.5, "B")]
    words = _words(spec)
    smooth_word_speakers(words, min_run=1.0, majority=0.7)
    assert words[-1].speaker == "B" and words[21].speaker == "B"


def test_assign_speakers_smooths_only_when_asked():
    words = _words([("Parliamo", 0.0, 0.4, None), ("di", 0.5, 0.6, None), ("innovazione.", 0.7, 1.2, None)])
    transcript = Transcript(id="t", source_file="x", language="Italian", duration=2, model="m",
                            created_at="2026-01-01T00:00:00Z", text="Parliamo di innovazione.", has_timestamps=True,
                            segments=[Segment(text="Parliamo di innovazione.", start=0, end=1.2, words=words)])
    turns = [SpeakerTurn(start=0, end=0.45, speaker="A"), SpeakerTurn(start=0.45, end=0.65, speaker="B"),
             SpeakerTurn(start=0.65, end=2, speaker="A")]
    raw = assign_speakers(transcript, turns)
    assert raw.speakers == ["A", "B"]
    smoothed = assign_speakers(transcript, turns, min_turn=0.5, min_run=1.0, majority=0.7)
    assert smoothed.speakers == ["A"] and len(smoothed.segments) == 1
