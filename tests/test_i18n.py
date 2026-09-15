from scriba.i18n import LANGUAGE_NAMES_IT, TEXTS, language_choices, t
from scriba.languages import SUPPORTED_LANGUAGES
from scriba.models import JobStatus


def test_languages_have_same_keys():
    assert set(TEXTS["it"]) == set(TEXTS["en"])


def test_every_progress_status_is_translated():
    for status in JobStatus:
        if status not in (JobStatus.PENDING, JobStatus.FAILED):
            for lang in TEXTS:
                assert f"status_{status.value}" in TEXTS[lang]


def test_language_choices_keep_canonical_values():
    it = language_choices("it", SUPPORTED_LANGUAGES)
    assert it[0] == ("Rilevamento automatico", "auto")
    assert ("Italiano", "Italian") in it
    assert set(LANGUAGE_NAMES_IT) == set(SUPPORTED_LANGUAGES)
    assert t("xx", "transcribe") == "TRASCRIVI"  # unknown language falls back to default
