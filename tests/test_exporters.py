from datetime import datetime, timezone

from scriba.exporters import export_all, load_transcript
from scriba.exporters.srt_exporter import render_srt
from scriba.exporters.txt_exporter import render_txt
from scriba.exporters.vtt_exporter import render_vtt
from scriba.models import Segment, Transcript, Word
from scriba.utils.time import format_clock, format_srt_time, format_vtt_time


def make_transcript(**kw) -> Transcript:
    words = [Word(text="Buongiorno", start=4.0, end=4.6, speaker="SPEAKER_00"),
             Word(text="a", start=4.6, end=4.7, speaker="SPEAKER_00"),
             Word(text="tutti.", start=4.7, end=5.2, speaker="SPEAKER_00")]
    words2 = [Word(text="Possiamo", start=8.0, end=8.5, speaker="SPEAKER_01"),
              Word(text="iniziare.", start=8.5, end=9.1, speaker="SPEAKER_01")]
    data = dict(
        id="20260915_142301_a83f91", source_file="meeting.m4a", language="Italian", duration=10.0,
        model="Qwen/Qwen3-ASR-1.7B", created_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
        text="Buongiorno a tutti. Possiamo iniziare.", has_timestamps=True,
        segments=[Segment(text="Buongiorno a tutti.", start=4.0, end=5.2, speaker="SPEAKER_00", words=words),
                  Segment(text="Possiamo iniziare.", start=8.0, end=9.1, speaker="SPEAKER_01", words=words2)],
        speakers=["SPEAKER_00", "SPEAKER_01"],
    )
    data.update(kw)
    return Transcript(**data)


def test_time_formats():
    assert format_srt_time(3725.5) == "01:02:05,500"
    assert format_vtt_time(0.042) == "00:00:00.042"
    assert format_clock(12600) == "03:30:00"


def test_srt_and_vtt():
    t = make_transcript()
    srt = render_srt(t)
    assert "1\n00:00:04,000 --> 00:00:05,200\n[SPEAKER_00] Buongiorno a tutti." in srt
    assert render_vtt(t).startswith("WEBVTT")
    assert "<v SPEAKER_01>Possiamo iniziare." in render_vtt(t)


def test_txt_uses_speaker_names():
    t = make_transcript(speaker_names={"SPEAKER_00": "Luca"})
    txt = render_txt(t)
    assert "[00:00:04] Luca\nBuongiorno a tutti." in txt
    assert "[00:00:08] SPEAKER_01\nPossiamo iniziare." in txt


def test_export_all_roundtrip_and_skip_timed(tmp_path):
    t = make_transcript()
    files, warnings = export_all(t, tmp_path, ["txt", "markdown", "srt", "vtt"])
    assert set(files) == {"json", "txt", "markdown", "srt", "vtt"} and not warnings
    assert load_transcript(files["json"]) == t

    plain = make_transcript(has_timestamps=False, speakers=[],
                            segments=[Segment(text="Ciao.", start=0, end=10)], text="Ciao.")
    files, warnings = export_all(plain, tmp_path, ["txt", "srt"])
    assert "srt" not in files and warnings
    assert files["txt"].read_text(encoding="utf-8") == "Ciao.\n"
