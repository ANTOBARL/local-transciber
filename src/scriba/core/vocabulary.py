"""Context prompt and glossary.

The context is prepended to the prompt of every audio chunk, so its length multiplies the work of the
whole transcription: a 30 s chunk is about 390 tokens, a 12 000-character term list about 3 600. It is
meant for a short hint (event, speakers, a few key terms).

The glossary is never sent to the model. After transcription, words that almost match a glossary term
are replaced by it ("Neurotec" -> "NeuroTech", "Laura Bianci" -> "Laura Bianchi"). The rules are
deliberately conservative, so that ordinary words are never rewritten:

- only proper-name terms take part: a capital letter after the first character, or a digit
  ("Aurora Science Park", "NeuroTech", "AB-12"); "Piemonte" or "politica industriale" do not;
- only text the model already wrote with a capital letter is corrected ("provare" never becomes "ProVare");
- a match never spans punctuation, and must start with the same two letters as the term;
- approximate matches need a term of at least 7 letters, and a stricter similarity when the number of
  words differs.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from difflib import SequenceMatcher

from scriba.models import Word

CONTEXT_WARN_CHARS = 1000
CHARS_PER_TOKEN = 3.5
AUDIO_TOKENS_PER_SECOND = 13.0
PROMPT_TOKENS = 40

MIN_TERM_CHARS = 5        # letters and digits; shorter terms (ABC, XYZ) are too ambiguous
MIN_FUZZY_CHARS = 7       # shorter terms are only matched exactly (ignoring case and spacing)
SIMILARITY = 0.85
SIMILARITY_OTHER_SIZE = 0.92  # when the candidate has one word more or less than the term
MIN_WORD_SIMILARITY = 0.75
MAX_LENGTH_GAP = 0.2      # a candidate may differ in length by at most this share of the term

_EDGE = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)


def _lines(text: str, separators: str = "\n") -> list[str]:
    parts = re.split(f"[{re.escape(separators)}]", text.replace("\r\n", "\n").replace("\r", "\n"))
    return [p.strip() for p in parts]


def normalize_context(text: str) -> str:
    """Drop repeated lines (case-insensitive) and blank-line runs; the order is kept."""
    out: list[str] = []
    seen: set[str] = set()
    for line in _lines(text or ""):
        if not line:
            if out and out[-1]:
                out.append("")
            continue
        key = line.casefold()
        if key not in seen:
            seen.add(key)
            out.append(line)
    return "\n".join(out).strip()


def context_slowdown(chars: int, chunk_seconds: float) -> float:
    """Rough cost of the context: prompt length with it / without it."""
    base = chunk_seconds * AUDIO_TOKENS_PER_SECOND + PROMPT_TOKENS
    return (base + chars / CHARS_PER_TOKEN) / base


def parse_glossary(text: str) -> list[str]:
    """One term per line (or separated by ';'), duplicates removed."""
    terms: list[str] = []
    seen: set[str] = set()
    for term in _lines(text or "", "\n;"):
        term = " ".join(term.split())
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            terms.append(term)
    return terms


def normalize_glossary(text: str) -> str:
    return "\n".join(parse_glossary(text))


def _key(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _proper_name(term: str) -> bool:
    return any(ch.isupper() or ch.isdigit() for ch in term[1:])


def _clean_span(words: Sequence[str]) -> bool:
    """No punctuation between the words, and the model wrote a capital letter somewhere."""
    inner_ok = (all(_EDGE.match(w).group(3) == "" for w in words[:-1])
                and all(_EDGE.match(w).group(1) == "" for w in words[1:]))
    return inner_ok and any(ch.isupper() for w in words for ch in w)


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _words_align(span: Sequence[str], term: str) -> bool:
    """Same number of words: each one must resemble its counterpart ("Napoli dove" is not "Napoli Ovest")."""
    return all(_similar(_key(a), _key(b)) >= MIN_WORD_SIMILARITY for a, b in zip(span, term.split()))


class Glossary:
    def __init__(self, text: str):
        self.terms = [t for t in parse_glossary(text) if _proper_name(t) and len(_key(t)) >= MIN_TERM_CHARS]
        self._by_initial: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        for term in self.terms:
            key = _key(term)
            self._by_initial[key[0]].append((term, key, len(term.split())))
        sizes = {len(t.split()) for t in self.terms}
        self._sizes = sorted({s + d for s in sizes for d in (-1, 0, 1) if s + d >= 1}, reverse=True)

    def __bool__(self) -> bool:
        return bool(self.terms)

    def match(self, words: Sequence[str], i: int) -> tuple[int, str] | None:
        """Best glossary term for the words starting at `i`: (number of words, term)."""
        best: tuple[int, str, float] | None = None
        for size in self._sizes:
            if i + size > len(words):
                continue
            span = words[i:i + size]
            key = _key("".join(span))
            if len(key) < MIN_TERM_CHARS or not _clean_span(span):
                continue
            for term, term_key, term_size in self._by_initial.get(key[0], ()):
                if abs(term_size - size) > 1 or abs(len(term_key) - len(key)) > max(1, len(term_key) * MAX_LENGTH_GAP):
                    continue
                if key == term_key:
                    ratio = 1.0
                else:
                    threshold = SIMILARITY if term_size == size else SIMILARITY_OTHER_SIZE
                    if len(term_key) < MIN_FUZZY_CHARS or key[:2] != term_key[:2]:
                        continue
                    matcher = SequenceMatcher(None, key, term_key, autojunk=False)
                    if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                        continue
                    ratio = matcher.ratio()
                    if ratio < threshold or (term_size == size > 1 and not _words_align(span, term)):
                        continue
                if best is None or ratio > best[2] or (ratio == best[2] and size > best[0]):
                    best = (size, term, ratio)
        return (best[0], best[1]) if best else None

    def _scan(self, texts: Sequence[str]):
        """Yield (start, size, None) for kept words or (start, size, (original, term, new text))."""
        i = 0
        while i < len(texts):
            found = self.match(texts, i)
            if found is None:
                yield i, 1, None
                i += 1
                continue
            size, term = found
            joined = " ".join(texts[i:i + size])
            lead = _EDGE.match(texts[i]).group(1)
            trail = _EDGE.match(texts[i + size - 1]).group(3)
            original = joined[len(lead):len(joined) - len(trail)]
            yield i, size, (None if original == term else (original, term, lead + term + trail))
            i += size

    def correct_text(self, text: str) -> str:
        if not self:
            return text
        words = text.split()
        out = []
        for i, size, repl in self._scan(words):
            out.append(repl[2] if repl else " ".join(words[i:i + size]))
        return " ".join(out)

    def correct_words(self, words: list[Word]) -> tuple[list[Word], list[dict]]:
        """Replace near-miss spellings; a multi-word match becomes one timed word."""
        if not self:
            return words, []
        texts = [w.text for w in words]
        out: list[Word] = []
        changes: Counter[tuple[str, str]] = Counter()
        for i, size, repl in self._scan(texts):
            span = words[i:i + size]
            if repl is None:
                out.extend(span)
                continue
            original, term, new_text = repl
            out.append(Word(text=new_text, start=span[0].start, end=span[-1].end, speaker=span[0].speaker))
            changes[(original, term)] += 1
        report = [{"from": a, "to": b, "count": n} for (a, b), n in changes.most_common()]
        return out, report
