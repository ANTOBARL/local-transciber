"""Context normalisation and glossary correction (invented names only)."""

from scriba.config import load_settings
from scriba.core.vocabulary import Glossary, context_slowdown, normalize_context, parse_glossary
from scriba.models import Word
from scriba.webform import TranscriptionForm

GLOSSARY = """NeuroTech
Neurotech
CeLAB
Laura Bianchi
Torino Nord
Aurora Science Park
ProVare
Re-Start
politica industriale
Piemonte
Promex Italia
ABC
"""


def test_context_is_deduplicated_and_its_cost_estimated():
    assert normalize_context("NeuroTech\r\nNeurotech\r\n\r\n\r\nABC\nabc\n") == "NeuroTech\n\nABC"
    assert parse_glossary("A; b\nB;;c") == ["A", "b", "c"]
    assert 9 < context_slowdown(12684, 30) < 11
    assert context_slowdown(0, 30) == 1


def test_settings_normalise_context_and_glossary():
    s = load_settings(overrides={"asr": {"context": "X\nx\n", "glossary": "NeuroTech;Neurotech"}})
    assert s.asr.context == "X" and s.asr.glossary == "NeuroTech"


def test_only_proper_names_take_part():
    g = Glossary(GLOSSARY)
    assert "politica industriale" not in g.terms and "Piemonte" not in g.terms and "ABC" not in g.terms
    assert "NeuroTech" in g.terms and "Aurora Science Park" in g.terms


def test_near_miss_names_are_corrected():
    g = Glossary(GLOSSARY)
    text = "Parliamo di Neurotec, con Laura Bianci e il Celab. Poi Neuro Tech."
    assert g.correct_text(text) == "Parliamo di NeuroTech, con Laura Bianchi e il CeLAB. Poi NeuroTech."


def test_ordinary_words_and_punctuation_are_left_alone():
    g = Glossary(GLOSSARY)
    for text in (
        "dobbiamo provare le idee e fermare la fuga",    # lowercase: the model saw common words
        "il restart del sistema",
        "Torino dove si lavora",                          # word by word it is not "Torino Nord"
        "promesse, Italia e Europa",                      # never across punctuation
        "le politiche industriali del Piemonte",
        "su Aurora Science Park",                         # neighbouring words are kept
    ):
        assert g.correct_text(text) == text


def test_timed_words_keep_their_timing():
    g = Glossary(GLOSSARY)
    words = [Word(text="il", start=0, end=0.2, speaker="A"), Word(text="Neuro", start=0.3, end=0.5, speaker="A"),
             Word(text="Tech.", start=0.5, end=0.9, speaker="A")]
    out, report = g.correct_words(words)
    assert [(w.text, w.start, w.end) for w in out] == [("il", 0, 0.2), ("NeuroTech.", 0.3, 0.9)]
    assert report == [{"from": "Neuro Tech", "to": "NeuroTech", "count": 1}]


def test_form_keeps_configured_chunk_length_when_not_given():
    base = load_settings(overrides={"asr": {"chunk_seconds": 45}})
    assert TranscriptionForm(glossary="NeuroTech").to_settings(base).asr.chunk_seconds == 45
    assert TranscriptionForm(chunk_seconds=20).to_settings(base).asr.chunk_seconds == 20
    assert TranscriptionForm.from_settings(base).chunk_seconds == 45
