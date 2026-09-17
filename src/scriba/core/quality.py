"""Quality checks on chunk transcripts. Pure functions, no model dependency.

Qwen3-ASR decodes greedily and can get stuck repeating a phrase until it hits `max_new_tokens`
("innovazione di prodotto, innovazione di prodotto, ..."). Everything said after the loop starts is
lost for that chunk. qwen-asr's own fixer only looks at character patterns up to 20 characters, so
longer phrases slip through. The checks here work on words and are meant for space-delimited
languages.

The decoder can also stop early: a 3-minute chunk comes back with a handful of words and the rest of
the speech is silently dropped. Such chunks are recognised by their speech rate, in absolute terms and
compared with the rest of the recording.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

MAX_NGRAM = 12                # longest repeated phrase looked for, in words
MIN_LOOP_REPEATS = 5          # consecutive repetitions...
MIN_LOOP_WORDS = 20           # ...covering at least this many words ("del del del" is normal speech)
MAX_WORDS_PER_SECOND = 6.0    # fast Italian speech stays around 3.5-4.5
MIN_RATE_SECONDS = 10.0       # words-per-second is meaningless on very short audio
SPARSE_WORDS_PER_SECOND = 0.3
RELATIVE_SPARSE = 0.5         # below half the recording's typical rate, speech was probably dropped
MIN_SPARSE_SECONDS = 20.0
MIN_REFERENCE_CHUNKS = 3


def _norm(word: str) -> str:
    return "".join(ch for ch in word.lower() if unicodedata.category(ch)[0] in ("L", "N"))


@dataclass(frozen=True)
class Loop:
    start: int    # index of the first word
    size: int     # words in the repeated phrase
    repeats: int  # complete repetitions
    end: int      # index after the last repeated word (includes a trailing partial repetition)


def find_loops(words: list[str]) -> list[Loop]:
    """Runs of the same phrase repeated consecutively, long enough not to be ordinary speech."""
    norm = [_norm(w) for w in words]
    total = len(norm)
    loops: list[Loop] = []
    i = 0
    while i < total:
        best: Loop | None = None
        for size in range(1, MAX_NGRAM + 1):
            if i + 2 * size > total:
                break
            gram = norm[i:i + size]
            if not any(gram):
                continue
            repeats = 1
            while norm[i + repeats * size:i + (repeats + 1) * size] == gram:
                repeats += 1
            if repeats < MIN_LOOP_REPEATS or repeats * size < MIN_LOOP_WORDS:
                continue
            end = i + repeats * size
            partial = 0
            while partial < size and end + partial < total and norm[end + partial] == gram[partial]:
                partial += 1
            candidate = Loop(i, size, repeats, end + partial)
            if best is None or candidate.end > best.end:
                best = candidate
        if best is not None:
            loops.append(best)
            i = best.end
        else:
            i += 1
    return loops


def collapse_loops(text: str) -> tuple[str, int]:
    """Keep one occurrence of every repeated phrase. Returns (text, words removed)."""
    words = text.split()
    loops = find_loops(words)
    if not loops:
        return text, 0
    kept: list[str] = []
    cursor = 0
    for loop in loops:
        kept.extend(words[cursor:loop.start + loop.size])
        cursor = loop.end
    kept.extend(words[cursor:])
    return " ".join(kept), len(words) - len(kept)


@dataclass
class TextCheck:
    words: int
    seconds: float
    loops: list[Loop] = field(default_factory=list)
    reference: float | None = None  # typical words per second of this recording

    @property
    def words_per_second(self) -> float:
        return self.words / self.seconds if self.seconds > 0 else 0.0

    @property
    def repetitive(self) -> bool:
        too_fast = self.seconds >= MIN_RATE_SECONDS and self.words_per_second > MAX_WORDS_PER_SECOND
        return bool(self.loops) or too_fast

    @property
    def sparse(self) -> bool:
        if self.seconds < MIN_SPARSE_SECONDS:
            return False
        rate = self.words_per_second
        return rate < SPARSE_WORDS_PER_SECOND or (self.reference is not None and rate < RELATIVE_SPARSE * self.reference)


def check_text(text: str, seconds: float, reference: float | None = None) -> TextCheck:
    words = text.split()
    return TextCheck(words=len(words), seconds=seconds, loops=find_loops(words), reference=reference)


class SpeechRate:
    """Median words per second of the healthy chunks seen so far in one recording."""

    def __init__(self) -> None:
        self._rates: list[float] = []

    def add(self, check: TextCheck) -> None:
        if check.seconds >= MIN_SPARSE_SECONDS and not check.repetitive and not check.sparse:
            self._rates.append(check.words_per_second)

    @property
    def reference(self) -> float | None:
        if len(self._rates) < MIN_REFERENCE_CHUNKS:
            return None
        rates = sorted(self._rates)
        mid = len(rates) // 2
        return rates[mid] if len(rates) % 2 else (rates[mid - 1] + rates[mid]) / 2


def join_chunk_texts(texts: Iterable[str]) -> str:
    """Join consecutive chunk transcripts: a space between words, none between CJK characters."""
    from scriba.core.aligner import _is_cjk

    joined = ""
    for text in texts:
        text = text.strip()
        if not text:
            continue
        if joined and not (_is_cjk(joined[-1]) or _is_cjk(text[0]) or _is_cjk_punct(joined[-1])):
            joined += " "
        joined += text
    return joined


def _is_cjk_punct(ch: str) -> bool:
    code = ord(ch)
    return 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF
