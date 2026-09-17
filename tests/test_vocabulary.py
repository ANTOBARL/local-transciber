"""Context normalisation and glossary correction."""

from scriba.config import load_settings
from scriba.core.vocabulary import Glossary, context_slowdown, normalize_context, parse_glossary
from scriba.models import Word
from scriba.webform import TranscriptionForm

GLOSSARY = """MedITech
Meditech
CeSMA
Caterina Meglio
Napoli Ovest
San Giovanni Innovation District
ForMare
E-Voluzione
politica industriale
Campania
Promos Italia
FII
"""


def test_context_is_deduplicated_and_its_cost_estimated():
    assert normalize_context("MedITech\r\nMeditech\r\n\r\n\r\nFII\nfii\n") == "MedITech\n\nFII"
    assert parse_glossary("A; b\nB;;c") == ["A", "b", "c"]
    assert 9 < context_slowdown(12684, 30) < 11
    assert context_slowdown(0, 30) == 1


def test_settings_normalise_context_and_glossary():
    s = load_settings(overrides={"asr": {"context": "X\nx\n", "glossary": "MedITech;Meditech"}})
    assert s.asr.context == "X" and s.asr.glossary == "MedITech"


def test_only_proper_names_take_part():
    g = Glossary(GLOSSARY)
    assert "politica industriale" not in g.terms and "Campania" not in g.terms and "FII" not in g.terms
    assert "MedITech" in g.terms and "San Giovanni Innovation District" in g.terms


def test_near_miss_names_are_corrected():
    g = Glossary(GLOSSARY)
    text = "Parliamo di Meditec, con Caterina Melio e il Cesma. Poi Medi Tech."
    assert g.correct_text(text) == "Parliamo di MedITech, con Caterina Meglio e il CeSMA. Poi MedITech."


def test_ordinary_words_and_punctuation_are_left_alone():
    g = Glossary(GLOSSARY)
    for text in (
        "dobbiamo formare i giovani e fermare la fuga",   # lowercase: the model saw common words
        "l'evoluzione del mercato",
        "Napoli dove si lavora",                          # word by word it is not "Napoli Ovest"
        "promosse, Italia e Europa",                      # never across punctuation
        "le politiche industriali della Campania",
        "su San Giovanni Innovation District",            # neighbouring words are kept
    ):
        assert g.correct_text(text) == text


def test_timed_words_keep_their_timing():
    g = Glossary(GLOSSARY)
    words = [Word(text="il", start=0, end=0.2, speaker="A"), Word(text="Medi", start=0.3, end=0.5, speaker="A"),
             Word(text="Tech.", start=0.5, end=0.9, speaker="A")]
    out, report = g.correct_words(words)
    assert [(w.text, w.start, w.end) for w in out] == [("il", 0, 0.2), ("MedITech.", 0.3, 0.9)]
    assert report == [{"from": "Medi Tech", "to": "MedITech", "count": 1}]


def test_form_keeps_configured_chunk_length_when_not_given():
    base = load_settings(overrides={"asr": {"chunk_seconds": 45}})
    assert TranscriptionForm(glossary="MedITech").to_settings(base).asr.chunk_seconds == 45
    assert TranscriptionForm(chunk_seconds=20).to_settings(base).asr.chunk_seconds == 20
    assert TranscriptionForm.from_settings(base).chunk_seconds == 45
